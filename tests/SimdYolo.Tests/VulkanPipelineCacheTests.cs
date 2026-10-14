using SimdYolo;

/// <summary>提供Vulkan 管线与选核缓存的恢复、损坏拒绝和不可写目录回退的回归验证。</summary>
internal static class VulkanPipelineCacheTests
{
    /// <summary>向测试执行器注册Vulkan 管线与选核缓存的恢复、损坏拒绝和不可写目录回退的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("Vulkan disk caches restore pipelines and tiles, reject corruption and tolerate unwritable paths", () =>
        {
            string? previousDirectory = Environment.GetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE_DIR");
            string? previousEnabled = Environment.GetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE");
            string directory = Path.GetFullPath(Path.Combine(".cache", "tests", "vulkan-pipelines-" + Guid.NewGuid().ToString("N")));
            Directory.CreateDirectory(directory);
            try
            {
                Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE_DIR", directory);
                Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE", "1");
                float[] weight = Enumerable.Range(0, 32 * 16 * 9).Select(i => MathF.Cos(i) * .01f).ToArray();
                string path = OnnxFixtures.Write("disk-cache", [1, 16, 13, 15],
                    [new("Conv", ["images", "w"], ["output"], [OnnxFixtures.Integers("strides", 2, 2), OnnxFixtures.Integers("pads", 1, 1, 1, 1), OnnxFixtures.Integer("group", 2)])],
                    [new("w", weight, [64, 8, 3, 3])], ("output", [1, 64, 7, 8], 1));
                var input = new Tensor(Enumerable.Range(0, 16 * 13 * 15).Select(i => MathF.Sin(i)).ToArray(), 1, 16, 13, 15);
                using var reference = new YoloDetector(path);
                float[] expected = reference.PredictRaw(input)[0].Data;
                void Infer(bool? loaded, bool expectTiles)
                {
                    using var detector = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan });
                    if (loaded is { } hit && detector.PipelineCacheLoaded != hit) throw new Exception("Unexpected pipeline cache hit status.");
                    if (expectTiles && detector.CachedConvolutionShapes is not > 0) throw new Exception("Convolution tuning metadata was not restored.");
                    float[] actual = detector.PredictRaw(input)[0].Data;
                    for (int i = 0; i < expected.Length; i++)
                        if (!float.IsFinite(actual[i]) || MathF.Abs(actual[i] - expected[i]) > 1e-4f * MathF.Max(1, MathF.Abs(expected[i]))) throw new Exception("Cache changed inference output.");
                }
                Infer(false, false); Infer(true, true);
                string cache = Directory.GetFiles(directory, "*.bin").Single();
                byte[] corrupt = File.ReadAllBytes(cache); corrupt[^1] ^= 0x55; File.WriteAllBytes(cache, corrupt);
                Infer(false, true);
                foreach (string metadata in Directory.GetFiles(directory, "*.tiles.json")) File.WriteAllText(metadata, "not valid JSON");
                Infer(true, false);
                Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE", "0");
                Infer(false, false);
                string obstructed = Path.Combine(directory, "file-not-directory"); File.WriteAllText(obstructed, "cache directory cannot be created");
                Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE", "1");
                Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE_DIR", obstructed);
                Infer(false, false);
            }
            finally
            {
                Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE_DIR", previousDirectory);
                Environment.SetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE", previousEnabled);
            }
        });
    }
}
