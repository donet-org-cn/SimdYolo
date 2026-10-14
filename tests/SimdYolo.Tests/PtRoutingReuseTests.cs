using SimdYolo;

/// <summary>验证复用 PT 路由工作区在回调异常、重入拒绝和跨帧输出保留后仍可恢复。</summary>
internal static class PtRoutingReuseTests
{
    /// <summary>注册使用仓库官方 nano 权重的会话恢复测试。</summary>
    /// <param name="test">执行命名断言的测试回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("PT routing reuse: callback exceptions, rejected reentry and detached outputs recover", () =>
        {
            string path = "models/pt/yolov8n.pt"; if (!File.Exists(path)) { Console.WriteLine("SKIP optional PT routing model."); return; }
            bool fail = true, reenter = false; YoloDetector? session = null;
            var input = new Tensor(Enumerable.Range(0, 3 * 128 * 128).Select(i => (i % 251) / 250f).ToArray(), 1, 3, 128, 128);
            using var expectedSession = new YoloDetector(path, new() { CpuThreads = 1 });
            var expected = expectedSession.PredictRaw(input);
            using var detector = new YoloDetector(path, new() { CpuThreads = 1, LayerCompleted = _ =>
            {
                if (fail) { fail = false; throw new InvalidOperationException("Deliberate callback failure."); }
                if (reenter)
                {
                    reenter = false;
                    try { session!.PredictRaw(input); }
                    catch (InvalidOperationException) { return; }
                    throw new Exception("Reentrant inference was accepted.");
                }
            } });
            session = detector;
            try { detector.PredictRaw(input); throw new Exception("Callback failure was swallowed."); }
            catch (InvalidOperationException error) when (error.Message == "Deliberate callback failure.") { }
            reenter = true; var actual = detector.PredictRaw(input);
            // 独立会话的数组地址可选择不同 SIMD 面板，采用与官方 oracle 相同的 FP32 容差。
            for (int i = 0; i < actual.Length; i++)
                for (int j = 0; j < actual[i].Length; j++)
                    if (!float.IsFinite(actual[i].Data[j]) || MathF.Abs(actual[i].Data[j] - expected[i].Data[j]) > .003f + .0003f * MathF.Abs(expected[i].Data[j])) throw new Exception("Routing was not restored after failure.");
            float[][] saved = actual.Select(t => (float[])t.Data.Clone()).ToArray(); input.Data[0] += .2f; detector.PredictRaw(input);
            for (int i = 0; i < actual.Length; i++) if (!actual[i].Data.SequenceEqual(saved[i])) throw new Exception("A later frame overwrote detached output.");
        });
    }
}
