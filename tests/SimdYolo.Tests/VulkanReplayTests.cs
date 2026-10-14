using SimdYolo;

/// <summary>提供Vulkan 命令重放的输入更新、公开输出分离和形状重新记录的回归验证。</summary>
internal static class VulkanReplayTests
{
    /// <summary>向测试执行器注册Vulkan 命令重放的输入更新、公开输出分离和形状重新记录的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("Vulkan command replay updates input, detaches all outputs and re-records changed shapes", () =>
        {
            string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_VULKAN_REPLAY");
            Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_REPLAY", "1");
            try
            {
                string path = OnnxFixtures.Write("replay-multiple-outputs", [-1, 16, -1, -1],
                    [new("Conv", ["images", "w"], ["conv"]), new("Sigmoid", ["conv"], ["sigmoid"])],
                    [new("w", Enumerable.Range(0, 32 * 16).Select(i => MathF.Cos(i) * .03f).ToArray(), [32, 16, 1, 1])],
                    ("conv", [-1, 32, -1, -1], 1), ("sigmoid", [-1, 32, -1, -1], 1));
                using var cpu = new YoloDetector(path);
                using var gpu = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan });
                Tensor[]? saved = null; float[][]? snapshot = null;
                foreach (int[] shape in new int[][] { [2, 16, 9, 11], [1, 16, 7, 13], [2, 16, 9, 11] })
                {
                    var input = new Tensor(shape);
                    long replayed = gpu.ReplayedFrames;
                    for (int frame = 0; frame < 5; frame++)
                    {
                        if (frame == 3) input = new Tensor(shape);
                        for (int i = 0; i < input.Length; i++) input.Data[i] = MathF.Sin(i * .19f + frame * .37f);
                        Tensor[] expected = cpu.PredictRaw(input), actual = gpu.PredictRaw(input);
                        Compare(actual, expected);
                        if (saved is not null)
                            for (int i = 0; i < saved.Length; i++)
                                if (!saved[i].Data.AsSpan().SequenceEqual(snapshot![i])) throw new Exception("Replay overwrote a previously returned output.");
                        saved ??= actual; snapshot ??= actual.Select(t => (float[])t.Data.Clone()).ToArray();
                    }
                    if (gpu.ReplayedFrames <= replayed) throw new Exception("A GPU-only graph was never replayed.");
                }
                int callbacks = 0;
                using var traced = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan, LayerCompleted = _ => callbacks++ });
                var sample = new Tensor(1, 16, 7, 13);
                for (int i = 0; i < 4; i++) traced.PredictRaw(sample);
                if (traced.ReplayedFrames != 0 || callbacks != 8) throw new Exception("Command replay skipped layer callbacks.");
                Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_REPLAY", "0");
                using var disabled = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan });
                for (int i = 0; i < 4; i++) disabled.PredictRaw(sample);
                if (disabled.ReplayedFrames != 0) throw new Exception("The replay disable switch was ignored.");
            }
            finally { Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_REPLAY", previous); }
        });
        test("Vulkan command replay rejects CPU value-dependent slice indices", () =>
        {
            string path = OnnxFixtures.Write("replay-dynamic-values", [1, 1, 1, 512],
                [new("Gather", ["images", "zero"], ["index"], [OnnxFixtures.Integer("axis", 3)]),
                 new("Reshape", ["index", "vector"], ["start"]), new("Sigmoid", ["images"], ["sigmoid"]),
                 new("Slice", ["sigmoid", "start", "end", "axis"], ["output"])],
                [new("zero", [0], [], 7), new("vector", [1], [1], 7), new("end", [32], [1], 7), new("axis", [3], [1], 7)],
                ("output", [1, 1, 1, -1], 1));
            using var cpu = new YoloDetector(path);
            using var gpu = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan });
            var input = new Tensor(Enumerable.Range(0, 512).Select(i => i * .01f).ToArray(), 1, 1, 1, 512);
            for (int frame = 1; frame <= 5; frame++)
            {
                input.Data[0] = frame;
                Compare(gpu.PredictRaw(input), cpu.PredictRaw(input));
            }
            if (gpu.ReplayedFrames != 0) throw new Exception("CPU-dependent indexing was incorrectly replayed.");
        });
    }

    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Compare(Tensor[] actual, Tensor[] expected)
    {
        if (actual.Length != expected.Length) throw new Exception("Replay output count differs.");
        for (int output = 0; output < actual.Length; output++)
        {
            if (!actual[output].Shape.SequenceEqual(expected[output].Shape)) throw new Exception("Replay output shape differs.");
            for (int i = 0; i < expected[output].Length; i++)
            {
                float value = actual[output].Data[i], reference = expected[output].Data[i];
                if (!float.IsFinite(value) || MathF.Abs(value - reference) > 1e-4f * MathF.Max(1, MathF.Abs(reference))) throw new Exception("Replay output differs.");
            }
        }
    }
}
