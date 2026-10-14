using System.Numerics;

namespace SimdYolo;

/// <summary>用纯 C# 执行 ONNX 图，并在保持输入输出约定的前提下进行常量折叠和算子融合。</summary>
internal sealed partial class OnnxModel
{
    /// <summary>记录每个融合拼接节点中哪些输入需要执行 Sigmoid。</summary>
    private readonly Dictionary<string, bool[]> sigmoidConcats = [];

    /// <summary>把只有一个拼接消费者的 Sigmoid 合并到 Concat，保留未激活分片。</summary>
    private void FuseSigmoidConcat()
    {
        var producers = graph.Nodes.SelectMany(node => node.Outputs.Where(name => name != "").Select(name => (name, node))).ToDictionary(pair => pair.name, pair => pair.node);
        var uses = graph.Nodes.SelectMany(node => node.Inputs).Concat(graph.Outputs).GroupBy(name => name).ToDictionary(group => group.Key, group => group.Count());
        var remove = new HashSet<OnnxNode>();
        int sequence = 0;
        foreach (var concat in graph.Nodes.Where(node => node.Op == "Concat" && node.Inputs.Length > 0).ToArray())
        {
            bool[] sigmoid = new bool[concat.Inputs.Length];
            string[] inputs = (string[])concat.Inputs.Clone();
            for (int i = 0; i < inputs.Length; i++)
            {
                if (!producers.TryGetValue(inputs[i], out var activation) || activation.Op != "Sigmoid" || activation.Inputs.Length != 1 || activation.Outputs.Length != 1 ||
                    remove.Contains(activation) || uses.GetValueOrDefault(activation.Outputs[0]) != 1) continue;
                sigmoid[i] = true; inputs[i] = activation.Inputs[0]; remove.Add(activation);
            }
            if (!sigmoid.Any(value => value)) continue;
            string name;
            do { name = $"__simdyolo_sigmoid_concat_{sequence++}"; } while (graph.Nodes.Any(node => node.Name == name) || sigmoidConcats.ContainsKey(name));
            sigmoidConcats[name] = sigmoid;
            graph.Nodes[graph.Nodes.IndexOf(concat)] = concat with { Name = name, Op = "_SigmoidConcat", Inputs = inputs };
        }
        graph.Nodes.RemoveAll(remove.Contains);
    }

    /// <summary>CPU 直接把 Sigmoid 结果写入拼接布局；设备路径保留已有常驻激活与拼接操作。</summary>
    /// <param name="node">当前执行或匹配的 ONNX 节点。</param>
    /// <param name="args">按原 Concat 输入顺序排列的分片张量。</param>
    /// <returns>完成指定分片激活后的拼接张量。</returns>
    /// <exception cref="ArgumentException">必需输入缺失，或输入张量的非拼接维度不一致。</exception>
    /// <exception cref="OverflowException">合并后的拼接维度超出 Int32 范围。</exception>
    private Tensor ExecuteSigmoidConcat(OnnxNode node, Tensor?[] args)
    {
        Tensor[] inputs = args.Select(value => value ?? throw new ArgumentException("Required input missing.")).ToArray();
        bool[] sigmoid = sigmoidConcats[node.Name];
        // GPU 保留已有的设备激活与拼接操作。
        // CPU 把 Sigmoid 结果直接写入目标拼接布局，省去激活中间张量。
        if (DeviceScope.For(inputs) is not null)
            return TensorOps.Concat(inputs.Select((tensor, index) => sigmoid[index] ? TensorOps.Activation(tensor, false) : tensor).ToArray(), (int)node.Int("axis"));
        int axis = Tensor.Axis((int)node.Int("axis"), inputs[0].Rank);
        int[] shape = (int[])inputs[0].Shape.Clone(); shape[axis] = 0;
        foreach (var tensor in inputs)
        {
            if (tensor.Rank != shape.Length || tensor.Shape.Where((dimension, index) => index != axis && dimension != shape[index]).Any()) throw new ArgumentException("Concat shape mismatch.");
            shape[axis] = checked(shape[axis] + tensor.Shape[axis]);
        }
        var output = Tensor.Temporary(shape);
        int inner = Tensor.Count(shape[(axis + 1)..]), outer = Tensor.Count(shape[..axis]), destinationStride = shape[axis] * inner, destinationOffset = 0;
        float[] destination = output.Data;
        for (int inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
        {
            Tensor tensor = inputs[inputIndex]; float[] source = tensor.Data;
            int sourceStride = tensor.Shape[axis] * inner, offset = destinationOffset;
            if (!sigmoid[inputIndex])
                for (int row = 0; row < outer; row++) Array.Copy(source, row * sourceStride, destination, row * destinationStride + offset, sourceStride);
            else if (source.Length > 0)
            {
                // 融合 Sigmoid 拼接时，每个调度块覆盖的源元素数量。
                const int chunk = 16384;
                // block 为源元素块编号；跨行时重新计算目标偏移，把 Sigmoid 直接写入拼接输出。
                void Block(int block)
                {
                    int position = block * chunk, end = Math.Min(position + chunk, source.Length);
                    while (position < end)
                    {
                        int row = position / sourceStride, limit = Math.Min(end, (row + 1) * sourceStride), target = row * destinationStride + offset + position % sourceStride;
                        for (; position <= limit - Vector<float>.Count; position += Vector<float>.Count, target += Vector<float>.Count)
                            CpuActivation.Sigmoid(new Vector<float>(source, position)).CopyTo(destination, target);
                        for (; position < limit; position++, target++) destination[target] = TensorOps.Sigmoid(source[position]);
                    }
                }
                int blocks = (source.Length + chunk - 1) / chunk;
                if (source.Length >= 65536 && blocks > 1 && ops.Backend is CpuBackend cpu) CpuScheduling.For(0, blocks, cpu.Parallel, Block);
                else for (int block = 0; block < blocks; block++) Block(block);
            }
            destinationOffset += sourceStride;
        }
        return output;
    }
}
