using System.Reflection;
using SimdYolo;

/// <summary>提供ONNX 两级检测头 GPU 融合的严格图匹配和多帧稳定性的回归验证。</summary>
internal static class OnnxEndToEndTests
{
    /// <summary>向测试执行器注册ONNX 两级检测头 GPU 融合的严格图匹配和多帧稳定性的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("ONNX fused end-to-end head preserves two-stage ties, NaNs and FP32 bits", () => Enabled(() => CompareFrames()));
        test("ONNX end-to-end head guards consumers, constants, callbacks and fallback shapes", () => Enabled(() => Guards()));
    }

    /// <summary>比较改变输入后的连续帧结果，并检查已发布输出没有被后续帧覆盖。</summary>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void CompareFrames()
    {
        const int classes = 7, anchors = 263, maximum = 17;
        string path = Fixture("onnx-e2e-bits", classes, maximum);
        using var cpu = new YoloDetector(path, new() { Device = ComputeDevice.Cpu, CpuThreads = 2 });
        using var gpu = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan });
        if (!Fused(gpu) || Fused(cpu)) throw new Exception("ONNX head fusion did not respect the backend.");
        Tensor? previous = null; float[]? saved = null;
        for (int frame = 0; frame < 4; frame++)
        {
            Tensor input = Input(1, classes, anchors, frame);
            Tensor expected = cpu.PredictRaw(input)[0], actual = gpu.PredictRaw(input)[0];
            Equal(actual, expected);
            if (previous is not null)
                for (int i = 0; i < previous.Length; i++) Exact(previous.Data[i], saved![i]);
            previous = actual; saved = (float[])actual.Data.Clone();
        }
        if (gpu.ReplayedFrames < 2) throw new Exception("Fused ONNX head still blocks full-frame replay.");
    }

    /// <summary>验证图模式保护条件，确保共享消费者、修改常量或不支持形状保留回退。</summary>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Guards()
    {
        const int classes = 5, maximum = 17;
        foreach (var (variant, shared, change) in new[] { ("shared", true, ""), ("tile", false, "tile"), ("divisor", false, "divisor"), ("k", false, "k"), ("sorted", false, "sorted") })
        {
            string rejected = Fixture("onnx-e2e-guard-" + variant, classes, maximum, shared, change);
            using var detector = new YoloDetector(rejected, new() { Device = ComputeDevice.Vulkan });
            if (Fused(detector)) throw new Exception("ONNX head fusion accepted a changed consumer/constant/attribute: " + variant);
        }
        string path = Fixture("onnx-e2e-fallback", classes, maximum);
        int callbacks = 0;
        using (var traced = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan, LayerCompleted = _ => callbacks++ }))
        {
            if (Fused(traced)) throw new Exception("ONNX fusion changed per-layer callback behavior.");
            traced.PredictRaw(Input(1, classes, 67, 0));
            if (callbacks < 20 || traced.ReplayedFrames != 0) throw new Exception("ONNX callbacks were skipped.");
        }
        using var cpu = new YoloDetector(path, new() { Device = ComputeDevice.Cpu, CpuThreads = 2 });
        using var gpu = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan });
        // 该导出器使用零批次偏移；批次为二时仍保留导出图的既有检测头语义。
        Tensor input = Input(2, classes, 67, 3);
        Equal(gpu.PredictRaw(input)[0], cpu.PredictRaw(input)[0]);
        if (gpu.ReplayedFrames != 0) throw new Exception("Unsupported batch shape was captured as a fused frame.");
        foreach (YoloDetector detector in new[] { cpu, gpu })
        {
            try { detector.PredictRaw(Input(1, classes, maximum - 1, 0)); }
            catch (InvalidDataException) { continue; }
            throw new Exception("ONNX TopK silently clamped an invalid k.");
        }
        // 回退后改变形状，再验证捕获/重放使用新输入，且不会复用上一批次的公开输出。
        for (int frame = 0; frame < 4; frame++)
        {
            input = Input(1, classes, 131, frame);
            Equal(gpu.PredictRaw(input)[0], cpu.PredictRaw(input)[0]);
        }
        if (gpu.ReplayedFrames < 2) throw new Exception("ONNX replay did not recover after a shape fallback.");
    }

    /// <summary>通过反射检查模型中是否存在目标检测头融合节点。</summary>
    /// <param name="detector">已加载测试图的检测器会话。</param>
    /// <returns>目标融合节点是否存在。</returns>
    internal static bool Fused(YoloDetector detector)
    {
        object model = typeof(YoloDetector).GetField("model", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(detector)!;
        object graph = model.GetType().GetField("graph", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var nodes = (System.Collections.IEnumerable)graph.GetType().GetProperty("Nodes")!.GetValue(graph)!;
        return nodes.Cast<object>().Any(node => (string)node.GetType().GetProperty("Op")!.GetValue(node)! == "_OnnxEndToEnd");
    }

    /// <summary>生成包含特殊分数、同分值和可变帧坐标的检测头测试输入。</summary>
    /// <param name="batch">测试批次数量或当前批次索引。</param>
    /// <param name="classes">检测类别数量。</param>
    /// <param name="anchors">候选框数量。</param>
    /// <param name="frame">用于改变输入和覆盖多帧行为的帧索引。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    internal static Tensor Input(int batch, int classes, int anchors, int frame)
    {
        var result = new Tensor(batch, classes + 4, anchors);
        for (int b = 0; b < batch; b++)
        {
            int offset = b * (classes + 4) * anchors;
            for (int c = 0; c < 4; c++) for (int a = 0; a < anchors; a++) result.Data[offset + c * anchors + a] = b * 10000 + c * 1000 + a + frame * 100000;
            for (int c = 0; c < classes; c++) for (int a = 0; a < anchors; a++)
                result.Data[offset + (c + 4) * anchors + a] = ((a * 7 + c * 3 + frame) % 17 - 8) / 10f;
            for (int c = 0; c < classes; c++)
            {
                int start = offset + (c + 4) * anchors;
                result.Data[start] = BitConverter.Int32BitsToSingle(0x7fc00001 + c);
                result.Data[start + 1] = float.PositiveInfinity;
                result.Data[start + 2] = float.NegativeInfinity;
                result.Data[start + 3] = 0;
                result.Data[start + 4] = BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
                result.Data[start + 5] = float.Epsilon;
                result.Data[start + 6] = -float.Epsilon;
            }
        }
        // 非平凡转置模拟主干后常驻设备的特征头。
        // 恒等转置会成为主机 reshape，按预期阻止设备重放。
        return TensorOps.Transpose(result, [0, 2, 1]);
    }

    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Equal(Tensor actual, Tensor expected)
    {
        if (!actual.Shape.SequenceEqual(expected.Shape)) throw new Exception("ONNX end-to-end output shape changed.");
        for (int i = 0; i < actual.Length; i++) Exact(actual.Data[i], expected.Data[i]);
    }
    /// <summary>核对两个浮点值的精确表示，包括 NaN 类别和符号零约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Exact(float actual, float expected)
    {
        if (BitConverter.SingleToInt32Bits(actual) != BitConverter.SingleToInt32Bits(expected))
            throw new Exception($"ONNX end-to-end changed value bits: {actual} vs {expected}.");
    }
    /// <summary>暂时启用专项优化开关，执行验证后恢复原有环境设置。</summary>
    /// <param name="action">在临时测试设置或设备帧作用域内执行的操作。</param>
    private static void Enabled(Action action)
    {
        string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_ONNX_E2E");
        string? previousTopK = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_TOPK");
        string? previousCpu = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_ONNX_E2E");
        Environment.SetEnvironmentVariable("SIMDYOLO_GPU_ONNX_E2E", "1");
        Environment.SetEnvironmentVariable("SIMDYOLO_GPU_TOPK", "1");
        Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_E2E", "0");
        try { action(); }
        finally
        {
            Environment.SetEnvironmentVariable("SIMDYOLO_GPU_ONNX_E2E", previous);
            Environment.SetEnvironmentVariable("SIMDYOLO_GPU_TOPK", previousTopK);
            Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_E2E", previousCpu);
        }
    }

    /// <summary>构造指定数据流、常量和消费者关系的最小 ONNX 测试模型。</summary>
    /// <param name="name">测试文件、张量或 ONNX 属性的名称。</param>
    /// <param name="classes">检测类别数量。</param>
    /// <param name="maximum">两级检测头允许输出的最大候选数量。</param>
    /// <param name="shared">是否为被匹配图添加共享消费者。</param>
    /// <param name="changed">需要改变的模式常量或节点属性。</param>
    /// <returns>写入的 ONNX 测试文件路径。</returns>
    internal static string Fixture(string name, int classes, int maximum, bool shared = false, string changed = "")
    {
        byte[][] Axis(int axis) => [OnnxFixtures.Integer("axis", axis)];
        byte[][] Top() => [OnnxFixtures.Integer("axis", -1), OnnxFixtures.Integer("largest", 1), OnnxFixtures.Integer("sorted", changed == "sorted" ? 0 : 1)];
        OnnxFixtures.Node[] nodes =
        [
            new("Transpose", ["images"], ["head"], [OnnxFixtures.Integers("perm", 0, 2, 1)]),
            new("Transpose", ["head"], ["transposed"], [OnnxFixtures.Integers("perm", 0, 2, 1)]),
            new("Split", ["transposed", "split"], ["boxes", "scores"], Axis(-1)),
            new("ReduceMax", ["scores"], ["maxima"], [OnnxFixtures.Integers("axes", -1), OnnxFixtures.Integer("keepdims", 0)]),
            new("TopK", ["maxima", "firstK"], ["anchorValues", "anchorIndices"], Top()),
            new("Unsqueeze", ["anchorIndices", "anchorAxis"], ["anchorColumns"]),
            new("Expand", ["anchorColumns", "boxExpand"], ["boxColumns"]),
            new("Tile", ["boxColumns", "boxRepeat"], ["boxIndices"]),
            new("GatherElements", ["boxes", "boxIndices"], ["selectedBoxes"], Axis(1)),
            new("Expand", ["anchorColumns", "scoreExpand"], ["scoreColumns"]),
            new("Tile", ["scoreColumns", "scoreRepeat"], ["scoreIndices"]),
            new("GatherElements", ["scores", "scoreIndices"], ["selectedScores"], Axis(1)),
            new("Flatten", ["selectedScores"], ["flatScores"], Axis(1)),
            new("TopK", ["flatScores", "secondK"], ["bestValues", "bestIndices"], Top()),
            new("Div", ["bestIndices", "divisor"], ["fractions"]),
            new("Cast", ["fractions"], ["integerRows"], [OnnxFixtures.Integer("to", 7)]),
            new("Cast", ["integerRows"], ["rowIndices"], [OnnxFixtures.Integer("to", 7)]),
            new("Shape", ["selectedBoxes"], ["boxShape"]),
            new("Gather", ["boxShape", "kDimension"], ["selectedCount"], Axis(0)),
            new("Gather", ["boxShape", "coordinateDimension"], ["coordinateCount"], Axis(0)),
            new("Flatten", ["selectedBoxes"], ["flatBoxes"], Axis(2)),
            new("Mul", ["zero", "selectedCount"], ["batchOffsets"]),
            new("Add", ["rowIndices", "batchOffsets"], ["finalRows"]),
            new("Gather", ["flatBoxes", "finalRows"], ["flatResult"], Axis(0)),
            new("Shape", ["finalRows"], ["rowShape"]),
            new("Concat", ["inferred", "coordinateCount"], ["flatResultShape"], Axis(0)),
            new("Reshape", ["flatResult", "flatResultShape"], ["coordinateRows"]),
            new("Concat", ["rowShape", "coordinateCount"], ["resultShape"], Axis(0)),
            new("Reshape", ["coordinateRows", "resultShape"], ["coordinates"]),
            new("Unsqueeze", ["bestValues", "scoreAxis"], ["confidence"]),
            new("Mod", ["bestIndices", "modulus"], ["classIndices"], [OnnxFixtures.Integer("fmod", 0)]),
            new("Unsqueeze", ["classIndices", "classAxis"], ["classColumn"]),
            new("Cast", ["classColumn"], ["classIds"], [OnnxFixtures.Integer("to", 1)]),
            new("Concat", ["coordinates", "confidence", "classIds"], ["output"], Axis(-1))
        ];
        OnnxFixtures.Weight[] weights =
        [
            new("split", [4, classes], [2], 7), new("firstK", [maximum], [1], 7), new("secondK", [changed == "k" ? maximum - 1 : maximum], [1], 7),
            new("anchorAxis", [-1], [1], 7), new("boxExpand", [1, 1, 1], [3], 7), new("scoreExpand", [1, 1, 1], [3], 7),
            new("boxRepeat", [1, 1, changed == "tile" ? 3 : 4], [3], 7), new("scoreRepeat", [1, 1, classes], [3], 7),
            new("divisor", [changed == "divisor" ? classes + 1 : classes], [], 7), new("modulus", [classes], [], 7),
            new("kDimension", [1], [1], 7), new("coordinateDimension", [2], [1], 7), new("zero", [0], [1, 1], 7),
            new("inferred", [-1], [1], 7), new("scoreAxis", [2], [1], 7), new("classAxis", [2], [1], 7)
        ];
        var outputs = new List<(string, int[], int)> { ("output", [-1, maximum, 6], 1) };
        if (shared) outputs.Add(("selectedScores", [-1, maximum, classes], 1));
        return OnnxFixtures.Write(name, [-1, -1, classes + 4], nodes, weights, outputs.ToArray());
    }
}
