using SimdYolo;

/// <summary>提供ONNX 检测头融合的两级选择、同分规则和图形状回退的回归验证。</summary>
internal static class CpuOnnxEndToEndTests
{
    /// <summary>向测试执行器注册ONNX 检测头融合的两级选择、同分规则和图形状回退的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU ONNX fused end-to-end preserves two-stage index order and FP32 bits", () => Enabled(Frames));
        test("CPU ONNX fused end-to-end rejects changed constants and preserves shape fallback", () => Enabled(Guards));
    }

    /// <summary>使用指定融合开关和计时回调加载检测器测试会话。</summary>
    /// <param name="path">已生成的模型或测试夹具文件路径。</param>
    /// <param name="enabled">是否启用本次验证所需的优化或池化作用域。</param>
    /// <param name="trace">可选的逐层计时回调，用于核对追踪条件下的融合行为。</param>
    /// <returns>创建并配置完成的检测器测试会话。</returns>
    private static YoloDetector Load(string path, bool enabled, Action<LayerTiming>? trace = null)
    {
        Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_E2E", enabled ? "1" : "0");
        return new(path, new() { Device = ComputeDevice.Cpu, CpuThreads = 2, LayerCompleted = trace });
    }

    /// <summary>比较改变输入后的连续帧结果，并检查已发布输出没有被后续帧覆盖。</summary>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Frames()
    {
        const int classes = 7, maximum = 17;
        string path = OnnxEndToEndTests.Fixture("cpu-onnx-e2e-bits", classes, maximum);
        using var reference = Load(path, false);
        using var optimized = Load(path, true);
        if (OnnxEndToEndTests.Fused(reference) || !OnnxEndToEndTests.Fused(optimized)) throw new Exception("CPU ONNX head flag did not control fusion.");
        Tensor? previous = null;
        float[]? saved = null;
        foreach (int anchors in new[] { 263, 67, 17, 131 })
            for (int frame = 0; frame < 4; frame++)
            {
                Tensor input = OnnxEndToEndTests.Input(1, classes, anchors, frame);
                if (frame == 3)
                {
                    // 让得分排名不同于原候选顺序，并在不同排名间构造大量同分类得分。
                    for (int a = 0; a < anchors; a++)
                        for (int c = 0; c < classes; c++)
                            input.Data[a * (classes + 4) + c + 4] = c == 0 ? (a % 7) * .1f : .25f;
                    for (int a = 0; a < Math.Min(anchors, maximum); a++)
                        input.Data[a * (classes + 4) + (a % classes) + 4] = BitConverter.Int32BitsToSingle(0x7fc00001 + a);
                }
                Tensor expected = reference.PredictRaw(input)[0];
                Tensor actual = optimized.PredictRaw(input)[0];
                OnnxEndToEndTests.Equal(actual, expected);
                if (previous is not null)
                    for (int i = 0; i < previous.Length; i++)
                        if (BitConverter.SingleToInt32Bits(previous.Data[i]) != BitConverter.SingleToInt32Bits(saved![i])) throw new Exception("CPU ONNX head reused a public output buffer.");
                previous = actual;
                saved = (float[])actual.Data.Clone();
            }
    }

    /// <summary>验证图模式保护条件，确保共享消费者、修改常量或不支持形状保留回退。</summary>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Guards()
    {
        const int classes = 5, maximum = 17;
        foreach (var (variant, shared, change) in new[] { ("shared", true, ""), ("tile", false, "tile"), ("divisor", false, "divisor"), ("k", false, "k"), ("sorted", false, "sorted") })
        {
            string path = OnnxEndToEndTests.Fixture("cpu-onnx-e2e-guard-" + variant, classes, maximum, shared, change);
            using var detector = Load(path, true);
            if (OnnxEndToEndTests.Fused(detector)) throw new Exception("CPU ONNX head accepted a changed pattern: " + variant);
        }
        string fallback = OnnxEndToEndTests.Fixture("cpu-onnx-e2e-fallback", classes, maximum);
        int callbacks = 0;
        using (var traced = Load(fallback, true, _ => callbacks++))
        {
            if (OnnxEndToEndTests.Fused(traced)) throw new Exception("CPU ONNX head changed callback behavior.");
            traced.PredictRaw(OnnxEndToEndTests.Input(1, classes, 67, 0));
            if (callbacks < 20) throw new Exception("CPU ONNX head skipped callbacks.");
        }
        using var reference = Load(fallback, false);
        using var optimized = Load(fallback, true);
        Tensor batchTwo = OnnxEndToEndTests.Input(2, classes, 67, 3);
        OnnxEndToEndTests.Equal(optimized.PredictRaw(batchTwo)[0], reference.PredictRaw(batchTwo)[0]);
        foreach (var detector in new[] { reference, optimized })
        {
            try { detector.PredictRaw(OnnxEndToEndTests.Input(1, classes, maximum - 1, 0)); }
            catch (InvalidDataException) { continue; }
            throw new Exception("CPU ONNX TopK clamped an invalid selection count.");
        }
        for (int frame = 0; frame < 4; frame++)
        {
            Tensor input = OnnxEndToEndTests.Input(1, classes, 131, frame);
            OnnxEndToEndTests.Equal(optimized.PredictRaw(input)[0], reference.PredictRaw(input)[0]);
        }
    }

    /// <summary>暂时启用专项优化开关，执行验证后恢复原有环境设置。</summary>
    /// <param name="action">在临时测试设置或设备帧作用域内执行的操作。</param>
    private static void Enabled(Action action)
    {
        string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_ONNX_E2E");
        try { action(); }
        finally { Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_E2E", previous); }
    }
}
