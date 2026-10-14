namespace SimdYolo;

/// <summary>严格匹配采用最终锚点 GatherElements 的端到端检测、分割与姿态导出图。</summary>
internal sealed partial class OnnxModel
{
    /// <summary>将两阶段索引图替换为设备端选择；共享值、公开中间输出或改变的常量均保留原图。</summary>
    private void FuseYolo26EndToEndHead()
    {
        static bool Axis(OnnxNode node, params long[] allowed) => allowed.Contains(node.Int("axis"));
        static bool Top(OnnxNode node) => Axis(node, -1, 1) && node.Int("largest", 1) == 1 && node.Int("sorted", 1) == 1;
        static bool Shape(OnnxNode node, int rank) => node.Int("start") == 0 && node.Int("end", long.MaxValue) >= rank;
        var producers = graph.Nodes.SelectMany(node => node.Outputs.Select((output, index) => (output, node, index)))
            .Where(item => item.output != "").ToDictionary(item => item.output);
        var removed = new HashSet<OnnxNode>();
        int sequence = 0;
        foreach (OnnxNode final in graph.Nodes.Where(node => node.Op == "Concat" && node.Inputs.Length is 3 or 4 && node.Outputs.Length == 1).ToArray())
        {
            bool hasExtra = final.Inputs.Length == 4;
            // 先检查三个固定出口，避免为主干的普通拼接节点分配完整模板。
            if (!Axis(final, -1, 2) || !producers.TryGetValue(final.Inputs[0], out var coordinates) || coordinates.node.Op != "GatherElements" ||
                !producers.TryGetValue(final.Inputs[1], out var confidence) || confidence.node.Op != "Unsqueeze" ||
                !producers.TryGetValue(final.Inputs[2], out var classesColumn) || classesColumn.node.Op != "Cast" ||
                hasExtra && (!producers.TryGetValue(final.Inputs[3], out var extraColumn) || extraColumn.node.Op != "GatherElements")) continue;
            var template = new List<HeadNode>
            {
                new("Transpose", ["$source"], ["transposed"], n => n.Ints("perm").SequenceEqual([0, 2, 1])),
                new("Split", ["transposed", "$split"], hasExtra ? ["boxes", "scores", "extra"] : ["boxes", "scores"], n => Axis(n, -1, 2)),
                new("ReduceMax", ["scores"], ["maxima"], n => n.Int("keepdims", 1) == 0 &&
                    (n.Ints("axes").SequenceEqual([-1]) || n.Ints("axes").SequenceEqual([2]))),
                new("TopK", ["maxima", "$firstK"], ["anchorValues", "anchorIndices"], Top),
                new("Unsqueeze", ["anchorIndices", "$anchorAxis"], ["anchorColumns"]),
                new("Expand", ["anchorColumns", "$scoreExpand"], ["scoreColumns"]),
                new("Tile", ["scoreColumns", "$scoreRepeat"], ["scoreIndices"]),
                new("GatherElements", ["scores", "scoreIndices"], ["selectedScores"], n => Axis(n, 1, -2)),
                new("Flatten", ["selectedScores"], ["flatScores"], n => n.Int("axis", 1) == 1),
                new("TopK", ["flatScores", "$secondK"], ["bestValues", "bestIndices"], Top),
                new("Div", ["bestIndices", "$divisor"], ["fractions"]),
                new("Cast", ["fractions"], ["integerRows"], n => n.Int("to") == 7),
                new("Cast", ["integerRows"], ["rowIndices"], n => n.Int("to") == 7),
                new("Shape", ["anchorColumns"], ["anchorShape"], n => Shape(n, 3)),
                new("Gather", ["anchorShape", "$kDimension"], ["selectedCount"], n => Axis(n, 0)),
                new("Gather", ["anchorShape", "$coordinateDimension"], ["coordinateCount"], n => Axis(n, 0)),
                new("Flatten", ["anchorColumns"], ["flatAnchors"], n => n.Int("axis", 1) == 2),
                new("Mul", ["$batchRange", "selectedCount"], ["batchOffsets"]),
                new("Add", ["rowIndices", "batchOffsets"], ["finalRows"]),
                new("Gather", ["flatAnchors", "finalRows"], ["flatResult"], n => Axis(n, 0)),
                new("Shape", ["finalRows"], ["rowShape"], n => Shape(n, 2)),
                new("Concat", ["$inferred", "coordinateCount"], ["flatResultShape"], n => Axis(n, 0)),
                new("Reshape", ["flatResult", "flatResultShape"], ["coordinateRows"], n => n.Int("allowzero") == 0),
                new("Concat", ["rowShape", "coordinateCount"], ["resultShape"], n => Axis(n, 0)),
                new("Reshape", ["coordinateRows", "resultShape"], ["finalAnchorColumns"], n => n.Int("allowzero") == 0),
                new("Expand", ["finalAnchorColumns", "$boxExpand"], ["boxColumns"]),
                new("Tile", ["boxColumns", "$boxRepeat"], ["boxIndices"]),
                new("GatherElements", ["boxes", "boxIndices"], ["coordinates"], n => Axis(n, 1, -2)),
                new("Unsqueeze", ["bestValues", "$scoreAxis"], ["confidence"]),
                new("Mod", ["bestIndices", "$modulus"], ["classIndices"], n => n.Int("fmod") == 0),
                new("Unsqueeze", ["classIndices", "$classAxis"], ["classColumn"]),
                new("Cast", ["classColumn"], ["classIds"], n => n.Int("to") == 1),
                new("Concat", hasExtra ? ["coordinates", "confidence", "classIds", "selectedExtra"] : ["coordinates", "confidence", "classIds"], ["result"], n => Axis(n, -1, 2))
            };
            if (hasExtra)
            {
                template.Add(new("Expand", ["finalAnchorColumns", "$extraExpand"], ["extraColumns"]));
                template.Add(new("Tile", ["extraColumns", "$extraRepeat"], ["extraIndices"]));
                template.Add(new("GatherElements", ["extra", "extraIndices"], ["selectedExtra"], n => Axis(n, 1, -2)));
            }
            var descriptors = template.SelectMany(spec => spec.Outputs.Select((output, index) => (output, spec, index))).ToDictionary(item => item.output);
            var bindings = new Dictionary<string, string>();
            var matched = new HashSet<OnnxNode>();
            // 符号以 $ 开头时先绑定外部值；其他符号逐一核对生产者、属性和共享边。
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
                for (int i = 0; i < spec.Outputs.Length; i++)
                {
                    if (bindings.TryGetValue(spec.Outputs[i], out string? bound) && bound != producer.node.Outputs[i]) return false;
                    bindings[spec.Outputs[i]] = producer.node.Outputs[i];
                }
                for (int i = 0; i < spec.Inputs.Length; i++) if (!Match(spec.Inputs[i], producer.node.Inputs[i])) return false;
                return true;
            }
            if (!Match("result", final.Outputs[0])) continue;
            // 常量的形状与数据必须同时相同，避免把形状广播不同的图当成等价图。
            bool Constant(string value, int[] shape, params float[] data) => graph.Weights.TryGetValue(value, out Tensor? tensor) &&
                tensor.Shape.SequenceEqual(shape) && tensor.Data.SequenceEqual(data);
            bool BoundConstant(string symbol, int[] shape, params float[] data) => Constant(bindings[symbol], shape, data);
            if (!graph.Weights.TryGetValue(bindings["$split"], out Tensor? split) || split.Rank != 1 || split.Length != (hasExtra ? 3 : 2) ||
                split.Data[0] != 4 || split.Data.Skip(1).Any(value => !float.IsFinite(value) || value != MathF.Truncate(value) || value is <= 0 or > 65536) ||
                !graph.Weights.TryGetValue(bindings["$firstK"], out Tensor? firstK) || firstK.Rank != 1 || firstK.Length != 1 ||
                !float.IsFinite(firstK.Data[0]) || firstK.Data[0] != MathF.Truncate(firstK.Data[0]) || firstK.Data[0] is <= 0 or > 65536) continue;
            int classes = (int)split.Data[1], extra = hasExtra ? (int)split.Data[2] : 0, maximum = (int)firstK.Data[0];
            if (!BoundConstant("$secondK", [1], maximum) || !BoundConstant("$anchorAxis", [1], -1) ||
                !BoundConstant("$boxExpand", [3], 1, 1, 1) || !BoundConstant("$boxRepeat", [3], 1, 1, 4) ||
                !BoundConstant("$kDimension", [1], 1) || !BoundConstant("$coordinateDimension", [1], 2) ||
                !BoundConstant("$inferred", [1], -1) || !BoundConstant("$scoreAxis", [1], 2) || !BoundConstant("$classAxis", [1], 2) ||
                hasExtra && (!BoundConstant("$extraExpand", [3], 1, 1, 1) || !BoundConstant("$extraRepeat", [3], 1, 1, extra))) continue;
            // 动态辅助节点只接受一个输出，并记录在融合范围内；其他消费者会在统一检查中拒绝。
            OnnxNode? Node(string value, string operation, int inputCount, Func<OnnxNode, bool>? attributes = null)
            {
                if (!producers.TryGetValue(value, out var producer) || producer.index != 0 || producer.node.Op != operation ||
                    producer.node.Inputs.Length != inputCount || producer.node.Outputs.Length != 1 || removed.Contains(producer.node) ||
                    attributes is not null && !attributes(producer.node)) return null;
                matched.Add(producer.node); return producer.node;
            }
            bool ScoresDimension(string value, int dimension)
            {
                OnnxNode? gather = Node(value, "Gather", 2, n => Axis(n, 0));
                if (gather is null || !Constant(gather.Inputs[1], [], dimension)) return false;
                OnnxNode? shape = Node(gather.Inputs[0], "Shape", 1, n => Shape(n, 3));
                return shape is not null && shape.Inputs[0] == bindings["scores"];
            }
            bool ClassCount(string value) => Constant(value, [], classes) || ScoresDimension(value, 2);
            bool ClassVector(string value)
            {
                OnnxNode? unsqueeze = Node(value, "Unsqueeze", 2);
                return unsqueeze is not null && Constant(unsqueeze.Inputs[1], [1], 0) && ClassCount(unsqueeze.Inputs[0]);
            }
            bool ScoreRepeat(string value)
            {
                if (Constant(value, [3], 1, 1, classes)) return true;
                OnnxNode? concat = Node(value, "Concat", 3, n => Axis(n, 0));
                return concat is not null && Constant(concat.Inputs[0], [1], 1) && Constant(concat.Inputs[1], [1], 1) && ClassVector(concat.Inputs[2]);
            }
            bool ScoreExpand(string value)
            {
                if (Constant(value, [3], 1, 1, 1)) return true;
                OnnxNode? fill = Node(value, "ConstantOfShape", 1, n => n.Attributes.TryGetValue("value", out var attribute) && attribute.Byte(5) is { } bytes &&
                    OnnxGraph.ReadTensor(Proto.Parse(bytes)) is { } tensor && tensor.Shape.SequenceEqual([1]) && tensor.Data.SequenceEqual([1f]));
                if (fill is null) return false;
                OnnxNode? shape = Node(fill.Inputs[0], "Shape", 1, n => Shape(n, 1));
                return shape is not null && ScoreRepeat(shape.Inputs[0]);
            }
            bool BatchRange(string value)
            {
                OnnxNode? unsqueeze = Node(value, "Unsqueeze", 2);
                if (unsqueeze is null || !Constant(unsqueeze.Inputs[1], [1], 1)) return false;
                OnnxNode? range = Node(unsqueeze.Inputs[0], "Range", 3);
                if (range is null || !Constant(range.Inputs[0], [], 0) || !Constant(range.Inputs[2], [], 1)) return false;
                OnnxNode? cast = Node(range.Inputs[1], "Cast", 1, n => n.Int("to") == 7);
                return cast is not null && ScoresDimension(cast.Inputs[0], 0);
            }
            bool batchOffsets = !BoundConstant("$batchRange", [1, 1], 0);
            if (!ClassCount(bindings["$divisor"]) || !ClassCount(bindings["$modulus"]) || !ScoreRepeat(bindings["$scoreRepeat"]) ||
                !ScoreExpand(bindings["$scoreExpand"]) || batchOffsets && !BatchRange(bindings["$batchRange"])) continue;
            var intermediates = matched.SelectMany(node => node.Outputs).Where(output => output != final.Outputs[0]).ToHashSet();
            if (graph.Outputs.Any(intermediates.Contains) || graph.Nodes.Any(node => !matched.Contains(node) && node.Inputs.Any(intermediates.Contains))) continue;
            OnnxNode[] nodes = graph.Nodes.Where(matched.Contains).ToArray();
            var produced = nodes.SelectMany(node => node.Outputs).ToHashSet();
            string source = bindings["$source"];
            string[] external = nodes.SelectMany(node => node.Inputs).Where(input => input != "" && input != source && !produced.Contains(input)).Distinct().ToArray();
            if (produced.Contains(source) || external.Any(input => !graph.Weights.ContainsKey(input))) continue;
            string name;
            do { name = $"__simdyolo_onnx_yolo26_end_to_end_{sequence++}"; } while (graph.Nodes.Any(node => node.Name == name));
            endToEndPatterns[name] = new(nodes, source, maximum, classes, extra, batchOffsets);
            graph.Nodes[graph.Nodes.IndexOf(final)] = new(name, "_OnnxEndToEnd", [source, .. external], final.Outputs, []);
            matched.Remove(final); removed.UnionWith(matched);
        }
        graph.Nodes.RemoveAll(removed.Contains);
    }
}
