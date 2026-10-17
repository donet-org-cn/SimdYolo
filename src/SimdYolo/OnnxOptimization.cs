namespace SimdYolo;

/// <summary>用纯 C# 执行 ONNX 图，并在保持输入输出约定的前提下进行常量折叠和算子融合。</summary>
internal sealed partial class OnnxModel
{
    /// <summary>记录固定输入形状下的常量节点结果与稳定形状，不缓存神经网络数据输出。</summary>
    /// <param name="graph">需要读取、优化或创建计划的 ONNX 图。</param>
    /// <param name="retained">此形状计划是否保留到后续帧。</param>
    private sealed class ShapeExecutionPlan(OnnxGraph graph, bool retained)
    {
        /// <summary>此计划是否跨帧持久保留，决定缓存张量的常量标记。</summary>
        public readonly bool Retained = retained;
        /// <summary>按优化后节点序号保存可复用的形状或常量输出数组。</summary>
        public readonly Tensor[]?[] Cached = new Tensor[]?[graph.Nodes.Count];
        /// <summary>节点的常量与形状依赖已在此输入形状下分析完毕；后续帧不再遍历名称集合。</summary>
        public readonly bool[] Analyzed = new bool[graph.Nodes.Count];
        /// <summary>当前计划中已知不依赖图像内容的张量名称。</summary>
        public readonly HashSet<string> Constants = new(graph.Weights.Keys);
        /// <summary>当前输入形状下，张量形状已确认稳定的名称集合。</summary>
        public readonly HashSet<string> StableShapes = new(graph.Weights.Keys.Concat(graph.Inputs));
    }

    /// <summary>查找当前输入形状的执行计划，最多持久保留 8 个形状。</summary>
    /// <param name="input">当前节点或模块的输入。</param>
    /// <returns>已缓存计划或只用于当前调用的新计划。</returns>
    private ShapeExecutionPlan ShapePlanFor(Tensor input)
    {
        string shapeKey = string.Join(',', input.Shape);
        if (shapePlans.TryGetValue(shapeKey, out var plan)) return plan;
        // 跨帧最多保留 8 个输入形状的计划；只缓存形状和常量锚点，不缓存图像特征。
        bool retain = shapePlans.Count < 8;
        plan = new(graph, retain);
        if (retain) shapePlans[shapeKey] = plan;
        return plan;
    }

    /// <summary>判断节点是否只依赖初始化常量，或读取已经确认稳定的形状。</summary>
    /// <param name="node">当前执行或匹配的 ONNX 节点。</param>
    /// <param name="plan">当前输入形状的常量与稳定形状执行计划。</param>
    /// <returns>可安全缓存本节点结果时返回 true。</returns>
    private static bool CanCache(OnnxNode node, ShapeExecutionPlan plan)
        => node.Op == "Shape" && node.Inputs.Length == 1 && plan.StableShapes.Contains(node.Inputs[0]) ||
            Foldable.Contains(node.Op) && node.Inputs.All(name => name == "" || plan.Constants.Contains(name));

    /// <summary>检查节点输出形状是否由稳定输入和常量形状参数决定。</summary>
    /// <param name="node">当前执行或匹配的 ONNX 节点。</param>
    /// <param name="plan">当前输入形状的常量与稳定形状执行计划。</param>
    /// <returns>本次输入形状下输出维度稳定时返回 true。</returns>
    private static bool HasStableShape(OnnxNode node, ShapeExecutionPlan plan)
    {
        if (node.Outputs.All(name => name == "" || plan.Constants.Contains(name))) return true;
        if (node.Inputs.Any(name => name != "" && !plan.StableShapes.Contains(name))) return false;
        // index 为可选形状参数序号；输入缺失或属于已知常量时返回 true。
        bool ConstantInput(int index) => node.Inputs.Length <= index || node.Inputs[index] == "" || plan.Constants.Contains(node.Inputs[index]);
        return node.Op switch
        {
            "Constant" or "Identity" or "Cast" or "Conv" or "_ConcatConv" or "ConvTranspose" or "BatchNormalization" or
            "Sigmoid" or "Relu" or "LeakyRelu" or "Tanh" or "Softmax" or "Add" or "Sub" or "Mul" or "Div" or "Mod" or "Pow" or
            "Sqrt" or "Exp" or "Neg" or "Abs" or "Max" or "Min" or "MatMul" or "Gemm" or "MaxPool" or "AveragePool" or
            "GlobalAveragePool" or "Concat" or "Transpose" or "Flatten" or "Shape" or "Gather" or "GatherElements" or
            "Equal" or "Greater" or "Less" or "Where" or "_SigmoidConcat" or "_OnnxEndToEnd" => true,
            "Reshape" or "Expand" or "Tile" or "Split" or "TopK" or "Unsqueeze" or "Squeeze" or
            "ReduceMean" or "ReduceSum" or "ReduceMax" => ConstantInput(1),
            "Slice" => Enumerable.Range(1, node.Inputs.Length - 1).All(ConstantInput),
            "Resize" => Enumerable.Range(1, node.Inputs.Length - 1).All(ConstantInput),
            "ConstantOfShape" or "Range" => node.Inputs.All(name => name == "" || plan.Constants.Contains(name)),
            "_Dfl" => ConstantInput(1) && ConstantInput(4),
            _ => false
        };
    }

    // 此集合仅含可由初始化常量独立计算的纯算子。
    // 常量锚点及 reshape/split 参数在加载时准备一次，
    // 设备后端可通过持久权重缓存上传，避免逐帧重复处理。
    /// <summary>允许在所有输入均为常量时提前执行的纯算子集合。</summary>
    private static readonly HashSet<string> Foldable = new(
        "Constant Identity Cast Shape Gather Slice Unsqueeze Squeeze Concat Split Reshape Transpose Flatten Add Sub Mul Div Mod Pow Sqrt Exp Neg Abs Max Min Equal Greater Less Where Range Expand Tile ConstantOfShape ReduceMean ReduceSum ReduceMax".Split(' '));

    /// <summary>在无设备范围和无临时池的加载阶段计算常量子图，保存结果并移除已折叠节点。</summary>
    /// <exception cref="InvalidDataException">常量节点执行失败，或节点返回数量与图声明不一致。</exception>
    private void FoldConstants()
    {
        var previousDevice = DeviceScope.Current;
        var previousPool = CpuTensorPool.Current;
        DeviceScope.Current = null;
        CpuTensorPool.Current = null;
        try
        {
            var remove = new HashSet<OnnxNode>();
            foreach (var node in graph.Nodes)
            {
                if (!Foldable.Contains(node.Op) || node.Inputs.Any(name => name != "" && !graph.Weights.ContainsKey(name))) continue;
                Tensor?[] args = node.Inputs.Select(name => name == "" ? null : graph.Weights[name]).ToArray();
                Tensor[] result;
                try { result = Execute(node, args); }
                catch (Exception error) when (error is ArgumentException or NotSupportedException or IndexOutOfRangeException or InvalidOperationException or OverflowException)
                { throw new InvalidDataException($"ONNX node '{node.Name}' ({node.Op}): {error.Message}", error); }
                if (result.Length != node.Outputs.Length) throw new InvalidDataException($"Node '{node.Name}' output count mismatch.");
                for (int i = 0; i < result.Length; i++)
                {
                    result[i].IsConstant = true;
                    if (node.Outputs[i] != "") graph.Weights[node.Outputs[i]] = result[i];
                }
                remove.Add(node);
            }
            graph.Nodes.RemoveAll(remove.Contains);
        }
        finally { DeviceScope.Current = previousDevice; CpuTensorPool.Current = previousPool; }
    }

    /// <summary>从公开图输出反向标记依赖，删除不可到达的节点和未使用初始化权重。</summary>
    /// <param name="graph">需要读取、优化或创建计划的 ONNX 图。</param>
    private static void EliminateDeadNodes(OnnxGraph graph)
    {
        var live = new HashSet<string>(graph.Outputs);
        var keep = new HashSet<OnnxNode>();
        for (int i = graph.Nodes.Count - 1; i >= 0; i--)
        {
            var node = graph.Nodes[i];
            if (!node.Outputs.Any(live.Contains)) continue;
            keep.Add(node);
            foreach (var name in node.Inputs) if (name != "") live.Add(name);
        }
        graph.Nodes.RemoveAll(node => !keep.Contains(node));
        foreach (string name in graph.Weights.Keys.Where(name => !live.Contains(name)).ToArray()) graph.Weights.Remove(name);
    }

    /// <summary>对独占 Conv 输出的推理 BN 按输出通道折叠权重和偏置，保留 BN 输出名称。</summary>
    /// <param name="graph">需要读取、优化或创建计划的 ONNX 图。</param>
    private static void FuseBatchNormalization(OnnxGraph graph)
    {
        var producers = graph.Nodes.SelectMany(node => node.Outputs.Where(name => name != "").Select(name => (name, node))).ToDictionary(pair => pair.name, pair => pair.node);
        var uses = graph.Nodes.SelectMany(node => node.Inputs).Concat(graph.Outputs).GroupBy(name => name).ToDictionary(group => group.Key, group => group.Count());
        var remove = new HashSet<OnnxNode>();
        int sequence = 0;
        foreach (var bn in graph.Nodes.Where(node => node.Op == "BatchNormalization").ToArray())
        {
            if (bn.Inputs.Length != 5 || bn.Outputs.Length != 1 || bn.Int("training_mode") != 0 ||
                !producers.TryGetValue(bn.Inputs[0], out var conv) || conv.Op != "Conv" || conv.Outputs.Length != 1 ||
                uses.GetValueOrDefault(conv.Outputs[0]) != 1 || conv.Inputs.Length is < 2 or > 3 ||
                !graph.Weights.TryGetValue(conv.Inputs[1], out var weight) || weight.Rank != 4 ||
                bn.Inputs.Skip(1).Any(name => !graph.Weights.ContainsKey(name))) continue;
            int channels = weight.Shape[0];
            if (channels == 0) continue;
            if (bn.Inputs.Skip(1).Any(name => graph.Weights[name].Rank != 1 || graph.Weights[name].Length != channels)) continue;
            Tensor? bias = null;
            if (conv.Inputs.Length == 3 && conv.Inputs[2] != "" &&
                (!graph.Weights.TryGetValue(conv.Inputs[2], out bias) || bias.Rank != 1 || bias.Length != channels)) continue;
            float[] scale = graph.Weights[bn.Inputs[1]].Data, offset = graph.Weights[bn.Inputs[2]].Data;
            float[] mean = graph.Weights[bn.Inputs[3]].Data, variance = graph.Weights[bn.Inputs[4]].Data;
            float[] source = weight.Data, fusedWeights = new float[source.Length], fusedBias = new float[channels];
            int perChannel = source.Length / channels;
            for (int channel = 0; channel < channels; channel++)
            {
                float factor = scale[channel] / MathF.Sqrt(variance[channel] + bn.Float("epsilon", 1e-5f));
                fusedBias[channel] = ((bias?.Data[channel] ?? 0) - mean[channel]) * factor + offset[channel];
                for (int k = 0; k < perChannel; k++) fusedWeights[channel * perChannel + k] = source[channel * perChannel + k] * factor;
            }
            string weightName, biasName;
            do { weightName = $"__simdyolo_bn_{sequence++}_weight"; biasName = weightName + "_bias"; }
            while (graph.Weights.ContainsKey(weightName) || graph.Weights.ContainsKey(biasName) || producers.ContainsKey(weightName) || producers.ContainsKey(biasName));
            graph.Weights[weightName] = new(fusedWeights, weight.Shape) { IsConstant = true };
            graph.Weights[biasName] = new(fusedBias, channels) { IsConstant = true };
            graph.Nodes[graph.Nodes.IndexOf(conv)] = conv with { Inputs = [conv.Inputs[0], weightName, biasName], Outputs = bn.Outputs };
            remove.Add(bn);
        }
        graph.Nodes.RemoveAll(remove.Contains);
    }

    /// <summary>比较分支几何、权重和消费者，将可居中嵌入的小卷积核合并到大卷积核。</summary>
    /// <param name="graph">需要读取、优化或创建计划的 ONNX 图。</param>
    private static void FuseConvolutionBranches(OnnxGraph graph)
    {
        // 重参数化 YOLO 块会导出并行卷积和 Add；满足相同几何时，
        // 可在加载阶段把小核居中嵌入大核。根据实际几何与消费者判断，
        // 不依赖模型或层名称。
        bool changed;
        int sequence = 0;
        do
        {
            changed = false;
            var producers = graph.Nodes.SelectMany(node => node.Outputs.Where(name => name != "").Select(name => (name, node))).ToDictionary(pair => pair.name, pair => pair.node);
            var uses = graph.Nodes.SelectMany(node => node.Inputs).Concat(graph.Outputs).GroupBy(name => name).ToDictionary(group => group.Key, group => group.Count());
            var remove = new HashSet<OnnxNode>();
            foreach (var add in graph.Nodes.Where(node => node.Op == "Add").ToArray())
            {
                if (add.Inputs.Length != 2 || add.Outputs.Length != 1 ||
                    !producers.TryGetValue(add.Inputs[0], out var a) || !producers.TryGetValue(add.Inputs[1], out var b) ||
                    a.Op != "Conv" || b.Op != "Conv" || remove.Contains(a) || remove.Contains(b) ||
                    a.Inputs.Length is < 2 or > 3 || b.Inputs.Length is < 2 or > 3 || a.Inputs[0] != b.Inputs[0] ||
                    a.Outputs.Length != 1 || b.Outputs.Length != 1 || uses.GetValueOrDefault(a.Outputs[0]) != 1 || uses.GetValueOrDefault(b.Outputs[0]) != 1 ||
                    a.Int("_silu") != 0 || b.Int("_silu") != 0 || a.Text("auto_pad", "NOTSET") != "NOTSET" || b.Text("auto_pad", "NOTSET") != "NOTSET" ||
                    a.Int("group", 1) != b.Int("group", 1) || !a.Ints("strides", [1, 1]).SequenceEqual(b.Ints("strides", [1, 1])) ||
                    !a.Ints("dilations", [1, 1]).SequenceEqual(b.Ints("dilations", [1, 1])) ||
                    !graph.Weights.TryGetValue(a.Inputs[1], out var wa) || !graph.Weights.TryGetValue(b.Inputs[1], out var wb) || wa.Rank != 4 || wb.Rank != 4 ||
                    wa.Shape[0] != wb.Shape[0] || wa.Shape[1] != wb.Shape[1]) continue;
                if (wa.Shape[2] < wb.Shape[2] || wa.Shape[3] < wb.Shape[3]) { (a, b) = (b, a); (wa, wb) = (wb, wa); }
                int dy = wa.Shape[2] - wb.Shape[2], dx = wa.Shape[3] - wb.Shape[3];
                if (dy < 0 || dx < 0 || dy % 2 != 0 || dx % 2 != 0) continue;
                int[] pa = a.Ints("pads", [0, 0, 0, 0]), pb = b.Ints("pads", [0, 0, 0, 0]), dilation = a.Ints("dilations", [1, 1]);
                if (pa.Length != 4 || pb.Length != 4 || dilation.Length != 2 ||
                    pa[0] - pb[0] != dy / 2 * dilation[0] || pa[2] - pb[2] != dy / 2 * dilation[0] ||
                    pa[1] - pb[1] != dx / 2 * dilation[1] || pa[3] - pb[3] != dx / 2 * dilation[1]) continue;
                // conv 为待融合分支；缺少偏置时构造零常量，否则返回已加载偏置，未找到时为 null。
                Tensor? Bias(OnnxNode conv)
                {
                    if (conv.Inputs.Length == 2 || conv.Inputs[2] == "") return new Tensor(wa.Shape[0]) { IsConstant = true };
                    return graph.Weights.GetValueOrDefault(conv.Inputs[2]);
                }
                Tensor? ba = Bias(a), bb = Bias(b);
                if (ba is null || bb is null || ba.Rank != 1 || bb.Rank != 1 || ba.Length != wa.Shape[0] || bb.Length != wa.Shape[0]) continue;
                var merged = KernelFusion.Merge(wa, ba, wb, bb);
                string weightName, biasName;
                do { weightName = $"__simdyolo_rep_{sequence++}_weight"; biasName = weightName + "_bias"; }
                while (graph.Weights.ContainsKey(weightName) || graph.Weights.ContainsKey(biasName) || producers.ContainsKey(weightName) || producers.ContainsKey(biasName));
                graph.Weights[weightName] = merged.Weight; graph.Weights[biasName] = merged.Bias;
                graph.Nodes[graph.Nodes.IndexOf(add)] = a with { Name = add.Name, Inputs = [a.Inputs[0], weightName, biasName], Outputs = add.Outputs };
                remove.Add(a); remove.Add(b); changed = true;
            }
            graph.Nodes.RemoveAll(remove.Contains);
        } while (changed);
    }
}
