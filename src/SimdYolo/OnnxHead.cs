using System.Numerics;

namespace SimdYolo;

/// <summary>用纯 C# 执行 ONNX 图，并在保持输入输出约定的前提下进行常量折叠和算子融合。</summary>
internal sealed partial class OnnxModel
{
    /// <summary>记录 DFL 五节点原始链和离散距离分箱数，供布局校验及回退。</summary>
    /// <param name="FirstReshape">把原始距离通道拆成四边与分箱维度的节点。</param>
    /// <param name="Transpose">把分箱维度移到 Softmax 轴的节点。</param>
    /// <param name="Softmax">沿分箱轴归一化的节点。</param>
    /// <param name="Conv">权重为分箱序号的加权卷积节点。</param>
    /// <param name="LastReshape">恢复四边距离布局的节点。</param>
    /// <param name="Bins">离散距离分布的分箱数量。</param>
    private sealed record DflPattern(OnnxNode FirstReshape, OnnxNode Transpose, OnnxNode Softmax, OnnxNode Conv, OnnxNode LastReshape, int Bins);
    /// <summary>已识别的 DFL 图链及其分箱数量。</summary>
    private readonly Dictionary<string, DflPattern> dflPatterns = [];

    /// <summary>严格匹配 Reshape→Transpose→Softmax→加权卷积→Reshape 的 DFL 链，并保留原节点供回退。</summary>
    private void FuseDfl()
    {
        if (graph.Opset < 13) return; // 更早的 Softmax 版本采用不同的展平语义，不能套用此融合。
        var producers = graph.Nodes.SelectMany(node => node.Outputs.Where(name => name != "").Select(name => (name, node))).ToDictionary(pair => pair.name, pair => pair.node);
        var uses = graph.Nodes.SelectMany(node => node.Inputs).Concat(graph.Outputs).GroupBy(name => name).ToDictionary(group => group.Key, group => group.Count());
        var remove = new HashSet<OnnxNode>();
        int sequence = 0;
        foreach (var last in graph.Nodes.Where(node => node.Op == "Reshape" && node.Inputs.Length == 2 && node.Outputs.Length == 1).ToArray())
        {
            if (!producers.TryGetValue(last.Inputs[0], out var conv) || conv.Op != "Conv" || conv.Inputs.Length is < 2 or > 3 || conv.Outputs.Length != 1 ||
                conv.Int("group", 1) != 1 || conv.Int("_silu") != 0 || conv.Text("auto_pad", "NOTSET") != "NOTSET" ||
                !conv.Ints("strides", [1, 1]).SequenceEqual([1, 1]) || !conv.Ints("pads", [0, 0, 0, 0]).SequenceEqual([0, 0, 0, 0]) || !conv.Ints("dilations", [1, 1]).SequenceEqual([1, 1]) ||
                !graph.Weights.TryGetValue(conv.Inputs[1], out var weight) || weight.Rank != 4 || weight.Shape[0] != 1 || weight.Shape[2] != 1 || weight.Shape[3] != 1 ||
                weight.Data.Where((value, index) => value != index).Any() || weight.Shape[1] < 1) continue;
            if (conv.Inputs.Length == 3 && conv.Inputs[2] != "" &&
                (!graph.Weights.TryGetValue(conv.Inputs[2], out var bias) || bias.Rank != 1 || bias.Length != 1 || bias.Data[0] != 0)) continue;
            if (!producers.TryGetValue(conv.Inputs[0], out var softmax) || softmax.Op != "Softmax" || softmax.Inputs.Length != 1 || softmax.Outputs.Length != 1 ||
                softmax.Int("axis", -1) is not (1 or -3) ||
                !producers.TryGetValue(softmax.Inputs[0], out var transpose) || transpose.Op != "Transpose" || transpose.Inputs.Length != 1 || transpose.Outputs.Length != 1 || !transpose.Ints("perm").SequenceEqual([0, 2, 1, 3]) ||
                !producers.TryGetValue(transpose.Inputs[0], out var first) || first.Op != "Reshape" || first.Inputs.Length != 2 || first.Outputs.Length != 1 ||
                new[] { first, transpose, softmax, conv }.Any(node => remove.Contains(node) || uses.GetValueOrDefault(node.Outputs[0]) != 1)) continue;
            string name;
            do { name = $"__simdyolo_dfl_{sequence++}"; } while (graph.Nodes.Any(node => node.Name == name) || dflPatterns.ContainsKey(name));
            dflPatterns[name] = new(first, transpose, softmax, conv, last, weight.Shape[1]);
            var fused = new OnnxNode(name, "_Dfl", [first.Inputs[0], first.Inputs[1], conv.Inputs[1], conv.Inputs.Length == 3 ? conv.Inputs[2] : "", last.Inputs[1]], last.Outputs, []);
            graph.Nodes[graph.Nodes.IndexOf(last)] = fused;
            remove.UnionWith([first, transpose, softmax, conv]);
        }
        graph.Nodes.RemoveAll(remove.Contains);
    }

    /// <summary>校验动态形状是否符合 YOLO 四边分箱布局，符合时融合计算，否则解释执行原链。</summary>
    /// <param name="node">当前执行或匹配的 ONNX 节点。</param>
    /// <param name="args">原距离张量、两个目标形状以及加权卷积参数。</param>
    /// <returns>最后一个 Reshape 对应的模型输出数组。</returns>
    /// <exception cref="ArgumentException">动态 Reshape 包含多个 -1，无法推断维度或元素数量不匹配。</exception>
    /// <exception cref="OverflowException">动态形状数值或维度乘积超过 Int32 范围。</exception>
    private Tensor[] ExecuteDfl(OnnxNode node, Tensor?[] args)
    {
        var pattern = dflPatterns[node.Name];
        Tensor x = args[0]!, firstDimensions = args[1]!, lastDimensions = args[4]!;
        // dimensions 为目标维度，source 为原形状，length 为元素数，allowZero 决定零维是否复制原维度。
        // 返回推断后的维度；多重 -1、不可整除或元素数改变抛出 ArgumentException，数值转换溢出则失败。
        int[] ReshapeDimensions(Tensor dimensions, int[] source, int length, bool allowZero)
        {
            int[] shape = dimensions.Data.Select(value => checked((int)value)).ToArray();
            if (!allowZero) for (int i = 0; i < shape.Length; i++) if (shape[i] == 0) shape[i] = source[i];
            int inferred = Array.IndexOf(shape, -1);
            if (inferred >= 0)
            {
                if (shape.Count(value => value == -1) != 1) throw new ArgumentException("Multiple inferred dimensions.");
                shape[inferred] = 1; int count = Tensor.Count(shape);
                if (count == 0 || length % count != 0) throw new ArgumentException("Invalid reshape.");
                shape[inferred] = length / count;
            }
            if (Tensor.Count(shape) != length) throw new ArgumentException("Tensor shape does not match data.");
            return shape;
        }
        // Reshape 参数可能依赖动态形状甚至输入数据。
        // 使用融合内核之前，必须核对实际四边与分箱的逻辑维度布局。
        int[] firstShape = ReshapeDimensions(firstDimensions, x.Shape, x.Length, pattern.FirstReshape.Int("allowzero") != 0);
        if (x.Rank == 3 && x.Shape[1] == 4 * pattern.Bins && firstShape.SequenceEqual([x.Shape[0], 4, pattern.Bins, x.Shape[2]]))
        {
            int[] convShape = [x.Shape[0], 1, 4, x.Shape[2]];
            int[] shape = ReshapeDimensions(lastDimensions, convShape, Tensor.Count(convShape), pattern.LastReshape.Int("allowzero") != 0);
            if (shape.SequenceEqual([x.Shape[0], 4, x.Shape[2]]))
            {
                if (DeviceScope.For(x) is { } owner)
                    return [owner.Compute(34, [x], shape, DeviceOps.Parameters((uint)Tensor.Count(shape), (uint)x.Shape[2], (uint)pattern.Bins), Tensor.Count(shape))];
                return [DflCpu(x, pattern.Bins, ops.Backend is CpuBackend cpu ? cpu.Parallel : new ParallelOptions())];
            }
        }
        // 相同语法链也可能用于其他张量布局。此时执行保留的原节点，
        // 避免仅凭算子排列就把一般 ONNX 子图误当成 YOLO 检测头。
        Tensor value = Execute(pattern.FirstReshape, [x, firstDimensions])[0];
        value = Execute(pattern.Transpose, [value])[0];
        value = Execute(pattern.Softmax, [value])[0];
        value = Execute(pattern.Conv, [value, args[2], args[3]])[0];
        return Execute(pattern.LastReshape, [value, lastDimensions]);
    }

    /// <summary>对四边离散距离执行稳定 Softmax 和分箱加权期望，使用 CPU SIMD 或标量尾部。</summary>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="bins">每条边离散距离分布的分箱数量。</param>
    /// <param name="parallel">当前 CPU 后端的并行配置；null 使用该方法的串行回退。</param>
    /// <returns>形状为 [批量,4,锚点数] 的距离张量。</returns>
    private static Tensor DflCpu(Tensor x, int bins, ParallelOptions parallel)
    {
        int spatial = x.Shape[2], count = x.Shape[0] * 4;
        var result = Tensor.Temporary(x.Shape[0], 4, spatial);
        float[] source = x.Data, destination = result.Data;
        if (CpuDfl.Enabled)
        {
            CpuDfl.Decode(x, destination, bins, spatial, parallel);
            return result;
        }
        CpuScheduling.For(0, count, parallel, side =>
        {
            int offset = side * bins * spatial, output = side * spatial, anchor = 0;
            for (; anchor <= spatial - Vector<float>.Count; anchor += Vector<float>.Count)
            {
                var maximum = new Vector<float>(float.NegativeInfinity);
                for (int bin = 0; bin < bins; bin++) maximum = Vector.Max(maximum, new Vector<float>(source, offset + bin * spatial + anchor));
                var sum = Vector<float>.Zero; var numerator = Vector<float>.Zero;
                for (int bin = 0; bin < bins; bin++)
                {
                    var exp = CpuActivation.Exp(new Vector<float>(source, offset + bin * spatial + anchor) - maximum);
                    sum += exp; numerator += exp * new Vector<float>(bin);
                }
                (numerator / sum).CopyTo(destination, output + anchor);
            }
            for (; anchor < spatial; anchor++)
            {
                float maximum = float.NegativeInfinity, sum = 0, numerator = 0;
                for (int bin = 0; bin < bins; bin++) maximum = MathF.Max(maximum, source[offset + bin * spatial + anchor]);
                for (int bin = 0; bin < bins; bin++) { float exp = MathF.Exp(source[offset + bin * spatial + anchor] - maximum); sum += exp; numerator += exp * bin; }
                destination[output + anchor] = numerator / sum;
            }
        });
        return result;
    }
}
