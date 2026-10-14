namespace SimdYolo;

/// <summary>用纯 C# 执行 ONNX 图，并在保持输入输出约定的前提下进行常量折叠和算子融合。</summary>
internal sealed partial class OnnxModel
{
    /// <summary>保存两阶段 TopK 检测头原节点、数据来源、保留数量与类别数。</summary>
    /// <param name="Nodes">按拓扑顺序排列的节点。</param>
    /// <param name="Source">原检测头输出张量名称。</param>
    /// <param name="Maximum">两阶段 TopK 保留数量。</param>
    /// <param name="Classes">检测类别数量。</param>
    /// <param name="Extra">随同一锚点收集的附加字段数量。</param>
    /// <param name="BatchOffsets">原图是否使用正确的动态批次偏移。</param>
    private sealed record EndToEndPattern(OnnxNode[] Nodes, string Source, int Maximum, int Classes, int Extra = 0, bool BatchOffsets = false);
    /// <summary>描述端到端图匹配模板中的算子、符号输入输出和属性条件。</summary>
    /// <param name="Op">ONNX 算子类型。</param>
    /// <param name="Inputs">按节点或图声明顺序排列的输入名称。</param>
    /// <param name="Outputs">按节点或图声明顺序排列的输出名称。</param>
    /// <param name="Attributes">可选节点属性校验函数；null 表示不额外检查属性。</param>
    private sealed record HeadNode(string Op, string[] Inputs, string[] Outputs, Func<OnnxNode, bool>? Attributes = null);
    /// <summary>匹配成功的端到端检测头，保留原节点和外部常量。</summary>
    private readonly Dictionary<string, EndToEndPattern> endToEndPatterns = [];

    // 导出器用较大的索引子图表达两阶段 TopK。仅依据数据流、属性和常量
    // 匹配模板，不依赖模型或层名称；未匹配的变体继续执行普通 ONNX 节点。
    /// <summary>按数据流、属性和常量严格匹配两阶段 TopK 检测头，拒绝额外中间消费者和公开中间输出。</summary>
    private void FuseEndToEndHead()
    {
        bool gpu = ops.Backend is VulkanBackend;
        bool cpu = ops.Backend is CpuBackend && Environment.GetEnvironmentVariable("SIMDYOLO_CPU_ONNX_E2E") != "0";
        if (trace is not null || (!gpu && !cpu) || gpu &&
            (Environment.GetEnvironmentVariable("SIMDYOLO_GPU_TOPK") == "0" ||
             Environment.GetEnvironmentVariable("SIMDYOLO_GPU_ONNX_E2E") == "0")) return;
        // node 为模板候选；其 axis 属性存在于 allowed 中时返回 true。
        static bool Axis(OnnxNode node, params long[] allowed) => allowed.Contains(node.Int("axis"));
        // node 为 TopK 候选；检查允许轴、largest 和 sorted 属性，不依赖节点名称。
        static bool Top(OnnxNode node) => Axis(node, -1, 1) && node.Int("largest", 1) == 1 && node.Int("sorted", 1) == 1;
        HeadNode[] template =
        [
            new("Transpose", ["$source"], ["transposed"], n => n.Ints("perm").SequenceEqual([0, 2, 1])),
            new("Split", ["transposed", "$split"], ["boxes", "scores"], n => Axis(n, -1, 2)),
            new("ReduceMax", ["scores"], ["maxima"], n => n.Int("keepdims", 1) == 0 && n.Ints("axes").Length == 1 && n.Ints("axes")[0] is -1 or 2),
            new("TopK", ["maxima", "$firstK"], ["anchorValues", "anchorIndices"], Top),
            new("Unsqueeze", ["anchorIndices", "$anchorAxis"], ["anchorColumns"]),
            new("Expand", ["anchorColumns", "$boxExpand"], ["boxColumns"]),
            new("Tile", ["boxColumns", "$boxRepeat"], ["boxIndices"]),
            new("GatherElements", ["boxes", "boxIndices"], ["selectedBoxes"], n => Axis(n, 1, -2)),
            new("Expand", ["anchorColumns", "$scoreExpand"], ["scoreColumns"]),
            new("Tile", ["scoreColumns", "$scoreRepeat"], ["scoreIndices"]),
            new("GatherElements", ["scores", "scoreIndices"], ["selectedScores"], n => Axis(n, 1, -2)),
            new("Flatten", ["selectedScores"], ["flatScores"], n => n.Int("axis", 1) == 1),
            new("TopK", ["flatScores", "$secondK"], ["bestValues", "bestIndices"], Top),
            new("Div", ["bestIndices", "$divisor"], ["fractions"]),
            new("Cast", ["fractions"], ["integerRows"], n => n.Int("to") == 7),
            new("Cast", ["integerRows"], ["rowIndices"], n => n.Int("to") == 7),
            new("Shape", ["selectedBoxes"], ["boxShape"], n => n.Int("start") == 0 && n.Int("end", long.MaxValue) >= 3),
            new("Gather", ["boxShape", "$kDimension"], ["selectedCount"], n => Axis(n, 0)),
            new("Gather", ["boxShape", "$coordinateDimension"], ["coordinateCount"], n => Axis(n, 0)),
            new("Flatten", ["selectedBoxes"], ["flatBoxes"], n => n.Int("axis", 1) == 2),
            new("Mul", ["$zero", "selectedCount"], ["batchOffsets"]),
            new("Add", ["rowIndices", "batchOffsets"], ["finalRows"]),
            new("Gather", ["flatBoxes", "finalRows"], ["flatResult"], n => Axis(n, 0)),
            new("Shape", ["finalRows"], ["rowShape"], n => n.Int("start") == 0 && n.Int("end", long.MaxValue) >= 2),
            new("Concat", ["$inferred", "coordinateCount"], ["flatResultShape"], n => Axis(n, 0)),
            new("Reshape", ["flatResult", "flatResultShape"], ["coordinateRows"], n => n.Int("allowzero") == 0),
            new("Concat", ["rowShape", "coordinateCount"], ["resultShape"], n => Axis(n, 0)),
            new("Reshape", ["coordinateRows", "resultShape"], ["coordinates"], n => n.Int("allowzero") == 0),
            new("Unsqueeze", ["bestValues", "$scoreAxis"], ["confidence"]),
            new("Mod", ["bestIndices", "$modulus"], ["classIndices"], n => n.Int("fmod") == 0),
            new("Unsqueeze", ["classIndices", "$classAxis"], ["classColumn"]),
            new("Cast", ["classColumn"], ["classIds"], n => n.Int("to") == 1),
            new("Concat", ["coordinates", "confidence", "classIds"], ["result"], n => Axis(n, -1, 2))
        ];
        var descriptors = template.SelectMany(spec => spec.Outputs.Select((output, index) => (output, spec, index))).ToDictionary(x => x.output);
        var producers = graph.Nodes.SelectMany(node => node.Outputs.Where(output => output != "").Select((output, index) => (output, node, index))).ToDictionary(x => x.output);
        var removed = new HashSet<OnnxNode>();
        int sequence = 0;
        foreach (OnnxNode final in graph.Nodes.Where(node => node.Op == "Concat" && node.Outputs.Length == 1).ToArray())
        {
            var bindings = new Dictionary<string, string>();
            var matched = new HashSet<OnnxNode>();
            // symbol 为模板符号，value 为实际张量名；递归匹配生产者并绑定共享输出，冲突时返回 false。
            bool Match(string symbol, string value)
            {
                if (bindings.TryGetValue(symbol, out string? existing)) return existing == value;
                bindings[symbol] = value;
                if (symbol.StartsWith('$')) return true;
                var descriptor = descriptors[symbol];
                HeadNode spec = descriptor.spec;
                if (!producers.TryGetValue(value, out var producer) || producer.index != descriptor.index || producer.node.Op != spec.Op ||
                    producer.node.Inputs.Length != spec.Inputs.Length || producer.node.Outputs.Length != spec.Outputs.Length ||
                    spec.Attributes is not null && !spec.Attributes(producer.node) || removed.Contains(producer.node)) return false;
                matched.Add(producer.node);
                // 递归处理共享分支前先绑定全部输出，保证同一节点的多输出对应关系一致。
                for (int i = 0; i < spec.Outputs.Length; i++)
                {
                    if (bindings.TryGetValue(spec.Outputs[i], out string? bound) && bound != producer.node.Outputs[i]) return false;
                    bindings[spec.Outputs[i]] = producer.node.Outputs[i];
                }
                for (int i = 0; i < spec.Inputs.Length; i++) if (!Match(spec.Inputs[i], producer.node.Inputs[i])) return false;
                return true;
            }
            if (!Match("result", final.Outputs[0])) continue;
            // symbol 为外部模板符号；仅在初始化张量的 shape 和所有 values 都准确匹配时返回 true。
            bool Constant(string symbol, int[] shape, params float[] values)
                => graph.Weights.TryGetValue(bindings[symbol], out var tensor) && tensor.Shape.SequenceEqual(shape) && tensor.Data.SequenceEqual(values);
            // symbol 为 TopK 数量常量；检查它是一维单元素有限正整数，成功时通过 value 返回数值。
            bool ScalarInteger(string symbol, out int value)
            {
                value = 0;
                if (!graph.Weights.TryGetValue(bindings[symbol], out var tensor) || tensor.Length != 1 || tensor.Rank != 1 || tensor.Shape[0] != 1 ||
                    !float.IsFinite(tensor.Data[0]) || tensor.Data[0] != MathF.Truncate(tensor.Data[0]) || tensor.Data[0] is <= 0 or > 65536) return false;
                value = (int)tensor.Data[0]; return true;
            }
            if (!ScalarInteger("$firstK", out int maximum) || !Constant("$secondK", [1], maximum) ||
                !graph.Weights.TryGetValue(bindings["$split"], out var split) || split.Rank != 1 || split.Length != 2 || split.Data[0] != 4 ||
                !float.IsFinite(split.Data[1]) || split.Data[1] != MathF.Truncate(split.Data[1]) || split.Data[1] is <= 0 or > 65536) continue;
            int classes = (int)split.Data[1];
            if (!Constant("$anchorAxis", [1], -1) || !Constant("$boxExpand", [3], 1, 1, 1) || !Constant("$scoreExpand", [3], 1, 1, 1) ||
                !Constant("$boxRepeat", [3], 1, 1, 4) || !Constant("$scoreRepeat", [3], 1, 1, classes) ||
                !Constant("$divisor", [], classes) || !Constant("$modulus", [], classes) || !Constant("$zero", [1, 1], 0) ||
                !Constant("$kDimension", [1], 1) || !Constant("$coordinateDimension", [1], 2) || !Constant("$inferred", [1], -1) ||
                !Constant("$scoreAxis", [1], 2) || !Constant("$classAxis", [1], 2)) continue;
            var intermediates = new HashSet<string>(matched.SelectMany(node => node.Outputs).Where(output => output != final.Outputs[0]));
            if (graph.Outputs.Any(intermediates.Contains) || graph.Nodes.Any(node => !matched.Contains(node) && node.Inputs.Any(intermediates.Contains))) continue;
            OnnxNode[] nodes = graph.Nodes.Where(matched.Contains).ToArray();
            var produced = new HashSet<string>(nodes.SelectMany(node => node.Outputs));
            string source = bindings["$source"];
            string[] external = nodes.SelectMany(node => node.Inputs).Where(input => input != "" && !produced.Contains(input) && input != source).Distinct().ToArray();
            if (external.Any(input => !graph.Weights.ContainsKey(input)) || produced.Contains(source)) continue;
            string name;
            do { name = $"__simdyolo_onnx_end_to_end_{sequence++}"; } while (graph.Nodes.Any(node => node.Name == name));
            endToEndPatterns[name] = new(nodes, source, maximum, classes);
            graph.Nodes[graph.Nodes.IndexOf(final)] = new(name, "_OnnxEndToEnd", [source, .. external], final.Outputs, []);
            matched.Remove(final); removed.UnionWith(matched);
        }
        graph.Nodes.RemoveAll(removed.Contains);
        FuseYolo26EndToEndHead();
    }

    /// <summary>对匹配的检测头选择 CPU 或 Vulkan 融合路径，并收集实例字段；布局和整数精度不符合时执行原图。</summary>
    /// <param name="node">当前执行或匹配的 ONNX 节点。</param>
    /// <param name="args">原检测头张量与匹配模板使用的外部常量。</param>
    /// <returns>融合或原图计算所得的检测输出数组。</returns>
    private Tensor[] ExecuteEndToEndHead(OnnxNode node, Tensor?[] args)
    {
        EndToEndPattern pattern = endToEndPatterns[node.Name];
        Tensor input = args[0]!;
        // 零偏移模板仅融合批量 1；使用 Range 的模板逐批收集独立锚点。
        // 其他批量或布局执行原始节点，保留形状、索引检查及错误语义。
        if (input.Rank == 3 && input.Shape[0] >= 1 && (pattern.BatchOffsets || input.Shape[0] == 1) &&
            input.Shape[1] == pattern.Classes + 4 + pattern.Extra && input.Shape[2] >= pattern.Maximum &&
            DeviceScope.For(input) is VulkanBackend vulkan && vulkan.TryEndToEndTopKOnnx(input, pattern.Maximum, pattern.Classes) is { } output)
            return [output];
        // 普通解释器在 FP32 张量中保存整数索引。超过连续整数精确范围时，
        // 必须执行原节点转换，避免融合算法悄悄提高索引精度而改变结果。
        if (ops.Backend is CpuBackend && input.Rank == 3 && input.Shape[0] >= 1 && (pattern.BatchOffsets || input.Shape[0] == 1) &&
            input.Shape[1] == pattern.Classes + 4 + pattern.Extra && input.Shape[2] >= pattern.Maximum && input.Shape[2] <= 16777216 &&
            (long)pattern.Maximum * pattern.Classes <= 16777216 && (long)input.Shape[0] * pattern.Maximum <= 16777216)
            return [CpuEndToEnd.SelectOnnx(input, pattern.Maximum, pattern.Classes)];
        var values = new Dictionary<string, Tensor>();
        for (int i = 0; i < node.Inputs.Length; i++) if (args[i] is { } value) values[node.Inputs[i]] = value;
        foreach (OnnxNode original in pattern.Nodes)
        {
            Tensor[] result = Execute(original, original.Inputs.Select(name => name == "" ? null : values[name]).ToArray());
            for (int i = 0; i < original.Outputs.Length; i++) if (original.Outputs[i] != "") values[original.Outputs[i]] = result[i];
        }
        return node.Outputs.Select(name => values[name]).ToArray();
    }
}
