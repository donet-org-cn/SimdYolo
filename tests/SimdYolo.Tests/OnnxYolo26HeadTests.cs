using SimdYolo;

/// <summary>验证 YOLO26 两阶段选择融合的附加字段、动态批次、精确位模式及图回退条件。</summary>
internal static class OnnxYolo26HeadTests
{
    /// <summary>注册 CPU 和可选 Vulkan 的融合正确性、重放及非法索引回归。</summary>
    /// <param name="test">命名测试的执行回调。</param>
    /// <param name="vulkan">是否验证本机 Vulkan 后端。</param>
    internal static void Run(Action<string, Action> test, bool vulkan)
    {
        test("YOLO26 ONNX CPU head preserves extra fields, batch offsets and FP32 bits", () => Settings(() => Frames(ComputeDevice.Cpu)));
        test("YOLO26 ONNX head rejects shared values, modified indices and invalid K", () => Settings(Guards));
        if (vulkan) test("YOLO26 ONNX Vulkan head preserves extra fields, dynamic batches and replay", () => Settings(() => Frames(ComputeDevice.Vulkan)));
    }

    /// <summary>建立明确启用或关闭融合的会话，供同一解释器原图与优化图比较。</summary>
    /// <param name="path">最小 ONNX 图文件。</param>
    /// <param name="device">需要验证的计算设备。</param>
    /// <param name="fusion">是否启用检测头融合。</param>
    /// <param name="callback">用于核对逐层回调的可选回调。</param>
    /// <returns>配置后的测试会话。</returns>
    private static YoloDetector Load(string path, ComputeDevice device, bool fusion, Action<LayerTiming>? callback = null)
    {
        Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_E2E", fusion ? "1" : "0");
        Environment.SetEnvironmentVariable("SIMDYOLO_GPU_ONNX_E2E", fusion ? "1" : "0");
        return new(path, new() { Device = device, CpuThreads = 2, LayerCompleted = callback });
    }

    /// <summary>比较有符号零、NaN、分数并列及附加字段，并检查跨帧公开输出和设备重放。</summary>
    /// <param name="device">要验证的 CPU 或 Vulkan 后端。</param>
    /// <exception cref="Exception">融合、位模式或多帧结果不符合原图。</exception>
    private static void Frames(ComputeDevice device)
    {
        foreach (int extra in new[] { 0, 32, 51 })
            foreach (bool dynamic in new[] { false, true })
            {
                const int classes = 7, maximum = 17;
                string path = Fixture($"yolo26-head-{device}-{extra}-{dynamic}", classes, maximum, extra, dynamic);
                using var reference = Load(path, ComputeDevice.Cpu, false);
                using var optimized = Load(path, device, true);
                if (OnnxEndToEndTests.Fused(reference) || !OnnxEndToEndTests.Fused(optimized)) throw new Exception("YOLO26 fusion did not match the complete graph.");
                Tensor? previous = null; float[]? saved = null;
                foreach (int batch in dynamic ? new[] { 1, 2, 3 } : new[] { 1, 2 })
                {
                    long replayed = optimized.ReplayedFrames;
                    for (int frame = 0; frame < 4; frame++)
                    {
                        Tensor input = Input(batch, classes, 263, extra, frame);
                        OnnxEndToEndTests.Equal(optimized.PredictRaw(input)[0], reference.PredictRaw(input)[0]);
                        if (previous is not null) OnnxEndToEndTests.Equal(previous, new Tensor(saved!, previous.Shape));
                        previous = optimized.PredictRaw(input)[0]; saved = (float[])previous.Data.Clone();
                    }
                    if (device == ComputeDevice.Vulkan && (dynamic || batch == 1) && optimized.ReplayedFrames <= replayed)
                        throw new Exception("YOLO26 fused head still prevented full-frame Vulkan replay.");
                }
            }
    }

    /// <summary>验证共享消费者、公开中间值、常量变更、回调和非法选择数量均保持保护条件。</summary>
    /// <exception cref="Exception">修改后的模式被错误融合或原图异常被吞掉。</exception>
    private static void Guards()
    {
        const int classes = 5, maximum = 17, extra = 32;
        foreach (bool dynamic in new[] { false, true })
        {
            foreach (string change in new[] { "shared", "consumer", "tile", "divisor", "k", "sorted", "extra", "offset", "dimension" })
            {
                string path = Fixture($"yolo26-head-guard-{dynamic}-{change}", classes, maximum, extra, dynamic, change);
                using var detector = Load(path, ComputeDevice.Cpu, true);
                if (OnnxEndToEndTests.Fused(detector)) throw new Exception("YOLO26 fusion accepted altered data flow: " + change);
            }
            string valid = Fixture("yolo26-head-invalid-k-" + dynamic, classes, maximum, extra, dynamic);
            int callbacks = 0;
            using (var traced = Load(valid, ComputeDevice.Cpu, true, _ => callbacks++))
            {
                if (OnnxEndToEndTests.Fused(traced)) throw new Exception("YOLO26 fusion changed tracing behavior.");
                traced.PredictRaw(Input(1, classes, 67, extra, 0));
                if (callbacks < 25) throw new Exception("YOLO26 skipped original callbacks.");
            }
            using var reference = Load(valid, ComputeDevice.Cpu, false);
            using var optimized = Load(valid, ComputeDevice.Cpu, true);
            foreach (YoloDetector detector in new[] { reference, optimized })
            {
                try { detector.PredictRaw(Input(1, classes, maximum - 1, extra, 0)); }
                catch (InvalidDataException) { continue; }
                throw new Exception("YOLO26 invalid TopK was silently clamped.");
            }
        }
    }

    /// <summary>生成含特殊分数和可追踪锚点附加字段的输入，并转置以模拟设备驻留的前序节点。</summary>
    /// <param name="batch">输入批次数量。</param>
    /// <param name="classes">类别数量。</param>
    /// <param name="anchors">候选锚点数量。</param>
    /// <param name="extra">每个锚点的实例附加字段数量。</param>
    /// <param name="frame">用于改变输入的帧序号。</param>
    /// <returns>形状为 [批次,锚点,4+类别数+附加字段] 的张量。</returns>
    private static Tensor Input(int batch, int classes, int anchors, int extra, int frame)
    {
        Tensor basis = OnnxEndToEndTests.Input(batch, classes, anchors, frame);
        var input = new Tensor(batch, anchors, classes + 4 + extra);
        for (int b = 0; b < batch; b++) for (int a = 0; a < anchors; a++)
        {
            Array.Copy(basis.Data, (b * anchors + a) * (classes + 4), input.Data, (b * anchors + a) * input.Shape[2], classes + 4);
            for (int field = 0; field < extra; field++) input.Data[(b * anchors + a) * input.Shape[2] + classes + 4 + field] = b * 100000 + field * 1000 + a + frame;
        }
        return input;
    }

    /// <summary>临时配置融合测试开关，结束时恢复调用方环境。</summary>
    /// <param name="action">在专项环境下执行的测试。</param>
    private static void Settings(Action action)
    {
        string[] names = ["SIMDYOLO_CPU_ONNX_E2E", "SIMDYOLO_GPU_ONNX_E2E", "SIMDYOLO_GPU_TOPK"];
        string?[] previous = names.Select(Environment.GetEnvironmentVariable).ToArray();
        Environment.SetEnvironmentVariable(names[2], "1");
        try { action(); }
        finally { for (int i = 0; i < names.Length; i++) Environment.SetEnvironmentVariable(names[i], previous[i]); }
    }

    /// <summary>独立编码采用最终锚点 GatherElements 的最小 ONNX 导出图。</summary>
    /// <param name="name">夹具文件名称。</param>
    /// <param name="classes">类别数量。</param>
    /// <param name="maximum">两阶段选择数量。</param>
    /// <param name="extra">实例附加字段数量。</param>
    /// <param name="dynamic">是否采用动态类别维度及批偏移。</param>
    /// <param name="change">需要人为修改的保护条件。</param>
    /// <returns>写入后的 ONNX 文件路径。</returns>
    internal static string Fixture(string name, int classes, int maximum, int extra, bool dynamic, string change = "")
    {
        byte[][] Axis(int axis) => [OnnxFixtures.Integer("axis", axis)];
        byte[][] Top() => [OnnxFixtures.Integer("axis", -1), OnnxFixtures.Integer("largest", 1), OnnxFixtures.Integer("sorted", change == "sorted" ? 0 : 1)];
        var nodes = new List<OnnxFixtures.Node>
        {
            new("Transpose", ["images"], ["head"], [OnnxFixtures.Integers("perm", 0, 2, 1)]),
            new("Transpose", ["head"], ["transposed"], [OnnxFixtures.Integers("perm", 0, 2, 1)]),
            new("Split", ["transposed", "split"], extra > 0 ? ["boxes", "scores", "extra"] : ["boxes", "scores"], Axis(-1)),
            new("ReduceMax", ["scores"], ["maxima"], [OnnxFixtures.Integers("axes", -1), OnnxFixtures.Integer("keepdims", 0)]),
            new("TopK", ["maxima", "firstK"], ["anchorValues", "anchorIndices"], Top()),
            new("Unsqueeze", ["anchorIndices", "anchorAxis"], ["anchorColumns"]),
            new("Expand", ["anchorColumns", "ones"], ["scoreColumns"]),
        };
        var weights = new List<OnnxFixtures.Weight>
        {
            new("split", extra > 0 ? [4, classes, extra] : [4, classes], [extra > 0 ? 3 : 2], 7),
            new("firstK", [maximum], [1], 7), new("secondK", [change == "k" ? maximum - 1 : maximum], [1], 7),
            new("anchorAxis", [-1], [1], 7), new("ones", [1, 1, 1], [3], 7),
            new("boxRepeat", [1, 1, change == "tile" ? 3 : 4], [3], 7), new("extraRepeat", [1, 1, change == "extra" ? extra + 1 : extra], [3], 7),
            new("kDimension", [1], [1], 7), new("coordinateDimension", [2], [1], 7),
            new("inferred", [-1], [1], 7), new("finalAxis", [2], [1], 7)
        };
        if (dynamic)
        {
            nodes.AddRange(new OnnxFixtures.Node[]
            {
                new("Shape", ["scores"], ["scoresShape"]), new("Gather", ["scoresShape", "classDimension"], ["divisor"], Axis(0)),
                new("Unsqueeze", ["divisor", "axisZero"], ["classVector"]), new("Concat", ["one", "one", "classVector"], ["scoreRepeat"], Axis(0)),
                new("Gather", ["scoresShape", "batchDimension"], ["batchCount"], Axis(0)), new("Cast", ["batchCount"], ["integerBatch"], [OnnxFixtures.Integer("to", 7)]),
                new("Range", ["zeroScalar", "integerBatch", "oneScalar"], ["range"]), new("Unsqueeze", ["range", "axisOne"], ["batchRange"])
            });
            weights.AddRange(new OnnxFixtures.Weight[]
            {
                new("classDimension", [change is "divisor" or "dimension" ? 1 : 2], [], 7), new("batchDimension", [0], [], 7),
                new("axisZero", [0], [1], 7), new("axisOne", [1], [1], 7), new("zeroScalar", [change == "offset" ? 1 : 0], [], 7),
                new("oneScalar", [1], [], 7), new("one", [1], [1], 7)
            });
        }
        else weights.AddRange(new OnnxFixtures.Weight[]
        {
            new("divisor", [change == "divisor" ? classes + 1 : classes], [], 7), new("scoreRepeat", [1, 1, classes], [3], 7),
            new("batchRange", [change == "offset" ? 1 : 0], [1, 1], 7)
        });
        if (!dynamic && change == "dimension") weights[weights.FindIndex(weight => weight.Name == "coordinateDimension")] = new("coordinateDimension", [1], [1], 7);
        nodes.AddRange(new OnnxFixtures.Node[]
        {
            new("Tile", ["scoreColumns", "scoreRepeat"], ["scoreIndices"]), new("GatherElements", ["scores", "scoreIndices"], ["selectedScores"], Axis(1)),
            new("Flatten", ["selectedScores"], ["flatScores"], Axis(1)), new("TopK", ["flatScores", "secondK"], ["bestValues", "bestIndices"], Top()),
            new("Div", ["bestIndices", "divisor"], ["fractions"]), new("Cast", ["fractions"], ["integerRows"], [OnnxFixtures.Integer("to", 7)]),
            new("Cast", ["integerRows"], ["rowIndices"], [OnnxFixtures.Integer("to", 7)]), new("Shape", ["anchorColumns"], ["anchorShape"]),
            new("Gather", ["anchorShape", "kDimension"], ["selectedCount"], Axis(0)), new("Gather", ["anchorShape", "coordinateDimension"], ["coordinateCount"], Axis(0)),
            new("Flatten", ["anchorColumns"], ["flatAnchors"], Axis(2)), new("Mul", ["batchRange", "selectedCount"], ["batchOffsets"]),
            new("Add", ["rowIndices", "batchOffsets"], ["finalRows"]), new("Gather", ["flatAnchors", "finalRows"], ["flatResult"], Axis(0)),
            new("Shape", ["finalRows"], ["rowShape"]), new("Concat", ["inferred", "coordinateCount"], ["flatResultShape"], Axis(0)),
            new("Reshape", ["flatResult", "flatResultShape"], ["coordinateRows"]), new("Concat", ["rowShape", "coordinateCount"], ["resultShape"], Axis(0)),
            new("Reshape", ["coordinateRows", "resultShape"], ["finalAnchorColumns"]), new("Expand", ["finalAnchorColumns", "ones"], ["boxColumns"]),
            new("Tile", ["boxColumns", "boxRepeat"], ["boxIndices"]), new("GatherElements", ["boxes", "boxIndices"], ["coordinates"], Axis(1)),
            new("Unsqueeze", ["bestValues", "finalAxis"], ["confidence"]), new("Mod", ["bestIndices", "divisor"], ["classIndices"], [OnnxFixtures.Integer("fmod", 0)]),
            new("Unsqueeze", ["classIndices", "finalAxis"], ["classColumn"]), new("Cast", ["classColumn"], ["classIds"], [OnnxFixtures.Integer("to", 1)])
        });
        if (extra > 0) nodes.AddRange(new OnnxFixtures.Node[]
        {
            new("Expand", ["finalAnchorColumns", "ones"], ["extraColumns"]), new("Tile", ["extraColumns", "extraRepeat"], ["extraIndices"]),
            new("GatherElements", ["extra", "extraIndices"], ["selectedExtra"], Axis(1))
        });
        nodes.Add(new("Concat", extra > 0 ? ["coordinates", "confidence", "classIds", "selectedExtra"] : ["coordinates", "confidence", "classIds"], ["output"], Axis(-1)));
        var outputs = new List<(string, int[], int)> { ("output", [-1, maximum, 6 + extra], 1) };
        if (change == "shared") outputs.Add(("anchorIndices", [-1, maximum], 7));
        if (change == "consumer") { nodes.Add(new("Identity", ["finalAnchorColumns"], ["extraOutput"])); outputs.Add(("extraOutput", [-1, maximum, 1], 1)); }
        return OnnxFixtures.Write(name, [-1, -1, classes + 4 + extra], nodes.ToArray(), weights.ToArray(), outputs.ToArray());
    }
}
