namespace SimdYolo;

/// <summary>用纯 C# 执行 ONNX 图，并在保持输入输出约定的前提下进行常量折叠和算子融合。</summary>
internal sealed partial class OnnxModel
{
    /// <summary>CPU 图中被移除的正零 Add 节点数量。</summary>
    private int cpuZeroAddCount;

    /// <summary>仅移除完整 SiLU 菱形之前、无公开中间值的正零常量 Add，以恢复 CPU 卷积激活融合。</summary>
    private void FuseCpuZeroAdd()
    {
        if (ops.Backend is not CpuBackend || trace is not null ||
            Environment.GetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD") == "0") return;

        var producers = graph.Nodes.SelectMany(node => node.Outputs.Where(name => name != "").Select(name => (name, node))).ToDictionary(pair => pair.name, pair => pair.node);
        var users = graph.Nodes.SelectMany(node => node.Inputs.Distinct().Select(name => (name, node))).GroupBy(pair => pair.name).ToDictionary(group => group.Key, group => group.Select(pair => pair.node).ToArray());
        var publicOutputs = new HashSet<string>(graph.Outputs);
        var remove = new HashSet<OnnxNode>();

        foreach (var add in graph.Nodes.Where(node => node.Op == "Add" && node.Inputs.Length == 2 && node.Outputs.Length == 1).ToArray())
        {
            OnnxNode? conv = null; Tensor? zero = null;
            for (int side = 0; side < 2; side++)
            {
                if (producers.TryGetValue(add.Inputs[side], out var candidate) && candidate.Op == "Conv" &&
                    graph.Weights.TryGetValue(add.Inputs[1 - side], out var constant))
                { conv = candidate; zero = constant; break; }
            }
            if (conv is null || zero is null || !zero.IsConstant || conv.Outputs.Length != 1 ||
                conv.Inputs.Length is < 2 or > 3 || conv.Int("_silu") != 0 ||
                !graph.Weights.TryGetValue(conv.Inputs[1], out var weights) || weights.Rank != 4 ||
                !PositiveZeroBroadcast(zero, weights.Shape[0]) ||
                publicOutputs.Contains(conv.Outputs[0]) || publicOutputs.Contains(add.Outputs[0]) ||
                !users.TryGetValue(conv.Outputs[0], out var convUsers) || convUsers.Length != 1 || convUsers[0] != add ||
                !users.TryGetValue(add.Outputs[0], out var addUsers) || addUsers.Length != 2) continue;

            // 只移除完整 SiLU 菱形之前的正零 Add；公开中间输出和计时回调
            // 仍使用原始节点。随后由 SiLU 融合将激活合并到 CPU 卷积后处理，
            // 不把其他非零偏置误绑定到该卷积。
            var sigmoid = addUsers.FirstOrDefault(node => node.Op == "Sigmoid" && node.Inputs.Length == 1 && node.Outputs.Length == 1);
            var mul = addUsers.FirstOrDefault(node => node.Op == "Mul" && node.Inputs.Length == 2 && node.Outputs.Length == 1);
            if (sigmoid is null || mul is null || sigmoid.Inputs[0] != add.Outputs[0] ||
                !mul.Inputs.Contains(add.Outputs[0]) || !mul.Inputs.Contains(sigmoid.Outputs[0]) ||
                publicOutputs.Contains(sigmoid.Outputs[0]) ||
                !users.TryGetValue(sigmoid.Outputs[0], out var sigmoidUsers) || sigmoidUsers.Length != 1 || sigmoidUsers[0] != mul) continue;

            graph.Nodes[graph.Nodes.IndexOf(conv)] = conv with { Outputs = add.Outputs };
            remove.Add(add); cpuZeroAddCount++;
        }
        graph.Nodes.RemoveAll(remove.Contains);
    }

    /// <summary>检查常量是否具有标量或 NCHW 通道广播形状，且所有元素的位模式均为正零。</summary>
    /// <param name="tensor">待检查广播形状和正零位模式的常量张量。</param>
    /// <param name="channels">卷积输出通道数量。</param>
    /// <returns>可作为该卷积零加分支处理时返回 true。</returns>
    private static bool PositiveZeroBroadcast(Tensor tensor, int channels)
    {
        // Conv 使用 NCHW。末尾 [C] 向量会沿宽度广播，
        // 即使值全为零，也不能当作通道常量删除。
        bool scalar = tensor.Rank <= 4 && tensor.Shape.All(dimension => dimension == 1);
        bool channel = tensor.Rank == 3 && tensor.Shape.SequenceEqual([channels, 1, 1]) ||
            tensor.Rank == 4 && tensor.Shape.SequenceEqual([1, channels, 1, 1]);
        return (scalar || channel) && tensor.Length > 0 && tensor.Data.All(value => BitConverter.SingleToInt32Bits(value) == 0);
    }
}
