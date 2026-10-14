namespace SimdYolo;

/// <summary>用纯 C# 执行 ONNX 图，并在保持输入输出约定的前提下进行常量折叠和算子融合。</summary>
internal sealed partial class OnnxModel
{
    /// <summary>保存已识别的 CPU 通道拼接与点卷积融合节点名称。</summary>
    private readonly HashSet<string> concatConvolutions = [];

    /// <summary>识别只有单个卷积消费者的通道拼接，在 CPU 上直接打包输入分片以减少内存复制。</summary>
    private void FuseConcatConv()
    {
        // 此融合减少 CPU 拼接复制；GPU 继续使用导出的执行图，
        // 保持分派顺序、中间张量存活期和逐节点计时行为。
        if (ops.Backend is not CpuBackend) return;
        var producers = graph.Nodes.SelectMany(node => node.Outputs.Where(name => name != "").Select(name => (name, node))).ToDictionary(pair => pair.name, pair => pair.node);
        var uses = graph.Nodes.SelectMany(node => node.Inputs).Concat(graph.Outputs).GroupBy(name => name).ToDictionary(group => group.Key, group => group.Count());
        var remove = new HashSet<OnnxNode>();
        int sequence = 0;
        foreach (var conv in graph.Nodes.Where(node => node.Op == "Conv").ToArray())
        {
            if (conv.Inputs.Length is < 2 or > 3 || conv.Outputs.Length != 1 ||
                conv.Text("auto_pad", "NOTSET") != "NOTSET" || conv.Int("group", 1) != 1 ||
                !conv.Ints("strides", [1, 1]).SequenceEqual([1, 1]) || !conv.Ints("pads", [0, 0, 0, 0]).SequenceEqual([0, 0, 0, 0]) || !conv.Ints("dilations", [1, 1]).SequenceEqual([1, 1]) ||
                !graph.Weights.TryGetValue(conv.Inputs[1], out var weights) || !weights.IsConstant || weights.Rank != 4 || weights.Shape[2] != 1 || weights.Shape[3] != 1 ||
                !producers.TryGetValue(conv.Inputs[0], out var concat) || concat.Op != "Concat" || concat.Inputs.Length < 2 || concat.Outputs.Length != 1 ||
                concat.Int("axis") is not (1 or -3) || uses.GetValueOrDefault(concat.Outputs[0]) != 1 || remove.Contains(concat)) continue;
            string name;
            do { name = $"__simdyolo_concat_conv_{sequence++}"; } while (graph.Nodes.Any(node => node.Name == name) || concatConvolutions.Contains(name));
            var attributes = new Dictionary<string, Proto>(conv.Attributes);
            var count = new Proto(); count.Fields[3] = [(ulong)concat.Inputs.Length]; attributes["_concat_count"] = count;
            var axis = new Proto(); axis.Fields[3] = [unchecked((ulong)concat.Int("axis"))]; attributes["_concat_axis"] = axis;
            // 融合节点依次消费特征分片、常量权重与可选偏置。
            // 卷积几何与此前融合的 SiLU 标记保持原语义。
            var inputs = concat.Inputs.Concat(conv.Inputs.Skip(1)).ToArray();
            graph.Nodes[graph.Nodes.IndexOf(conv)] = conv with { Name = name, Op = "_ConcatConv", Inputs = inputs, Attributes = attributes };
            concatConvolutions.Add(name); remove.Add(concat);
        }
        graph.Nodes.RemoveAll(remove.Contains);
    }

    /// <summary>执行融合拼接卷积，并在形状不满足快路径时回退正常 Concat 与 Conv。</summary>
    /// <param name="node">当前执行或匹配的 ONNX 节点。</param>
    /// <param name="args">依次为特征分片、常量权重和可选偏置。</param>
    /// <returns>融合节点的卷积输出张量。</returns>
    /// <exception cref="ArgumentException">必需的特征输入或卷积权重缺失。</exception>
    /// <exception cref="OverflowException">节点几何属性不能转换为 Int32。</exception>
    private Tensor ExecuteConcatConv(OnnxNode node, Tensor?[] args)
    {
        int count = checked((int)node.Int("_concat_count"));
        Tensor[] inputs = args.Take(count).Select(tensor => tensor ?? throw new ArgumentException("Required input missing.")).ToArray();
        var weights = args[count] ?? throw new ArgumentException("Required weights missing.");
        Tensor? bias = args.Length > count + 1 ? args[count + 1] : null;
        int[] strides = node.Ints("strides", [1, 1]), pads = node.Ints("pads", [0, 0, 0, 0]), dilations = node.Ints("dilations", [1, 1]);
        int groups = checked((int)node.Int("group", 1)); bool silu = node.Int("_silu") == 1;
        // ConvConcat 在运行时确认 CPU、常量 1×1 卷积和单批量条件。
        // 其他合法形状以及 GPU 使用正常的 Concat+Conv 回退。
        if (inputs.All(tensor => tensor.Rank == 4)) return ops.ConvConcat(inputs, weights, bias, strides, pads, dilations, groups, silu);
        // 对一般非 NCHW 或错误秩的输入仍按原规则校验轴，
        // 防止把负轴一律当作 NCHW 的通道轴。
        return ops.Conv(TensorOps.Concat(inputs, checked((int)node.Int("_concat_axis"))), weights, bias, strides, pads, dilations, groups, silu);
    }
}
