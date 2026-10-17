using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;

/// <summary>在独立进程中验证真实模型连续推理、输出所有权、回收后内存和反复创建释放。</summary>
internal static class Program
{
    /// <summary>执行稳定性检查；本工具不提供速度排名，数值扫描和强制 GC 不计入性能对照。</summary>
    /// <param name="args">模型、设备、640×640 FP32输入、输出JSON，以及可选帧数。</param>
    /// <returns>全部检查通过为0，否则为1。</returns>
    private static int Main(string[] args)
    {
        try
        {
            if (args.Length is < 4 or > 5) throw new ArgumentException("SessionStability MODEL cpu|vulkan INPUT.f32 OUTPUT.json [frames=500]");
            int frames = args.Length == 5 ? int.Parse(args[4]) : 500;
            if (frames < 100) throw new ArgumentOutOfRangeException(nameof(args), "At least 100 frames are required.");
            ComputeDevice device = Enum.Parse<ComputeDevice>(args[1], true);
            if (device is not (ComputeDevice.Cpu or ComputeDevice.Vulkan)) throw new ArgumentException("Use an explicit CPU or Vulkan backend.");
            byte[] raw = File.ReadAllBytes(args[2]);
            if (!BitConverter.IsLittleEndian || raw.Length != 3 * 640 * 640 * sizeof(float)) throw new InvalidDataException("Expected little-endian 640x640 RGB FP32 input.");
            var input = new Tensor(MemoryMarshal.Cast<byte, float>(raw).ToArray(), 1, 3, 640, 640);
            using var process = Process.GetCurrentProcess();
            var cycles = new List<object>();
            for (int cycle = 0; cycle < 4; cycle++)
            {
                long loadStart = Stopwatch.GetTimestamp();
                var detector = new YoloDetector(args[0], new() { Device = device, CpuThreads = 12 });
                double loadMs = Stopwatch.GetElapsedTime(loadStart).TotalMilliseconds;
                if (detector.FallbackReason is not null) throw new InvalidOperationException(detector.FallbackReason);
                try
                {
                    Tensor[] first = detector.PredictRaw(input); float[][] saved = first.Select(t => (float[])t.Data.Clone()).ToArray(); Validate(first, saved);
                    for (int i = 0; i < 40; i++) detector.PredictRaw(input);
                    long before = GC.GetTotalMemory(true); process.Refresh(); long peak = process.WorkingSet64;
                    int count = cycle == 0 ? frames : 8;
                    for (int i = 0; i < count; i++)
                    {
                        Tensor[] result = detector.PredictRaw(input);
                        Validate(result, saved);
                        if (i % 25 == 0 || i == count - 1)
                        {
                            process.Refresh(); peak = Math.Max(peak, process.WorkingSet64);
                            for (int j = 0; j < first.Length; j++)
                                if (!MemoryMarshal.AsBytes(first[j].Data.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(saved[j].AsSpan()))) throw new InvalidDataException("A subsequent frame overwrote retained output.");
                        }
                    }
                    long after = GC.GetTotalMemory(true);
                    if (after - before > 32L * 1024 * 1024) throw new InvalidDataException("Live managed memory grew more than 32 MiB after warmup and full GC.");
                    cycles.Add(new { Cycle = cycle, Frames = count, ModelLoadMs = loadMs, ManagedBeforeBytes = before, ManagedAfterBytes = after, ManagedGrowthBytes = after - before, PeakWorkingSetMiB = peak / 1048576.0, detector.ReplayedFrames });
                }
                finally { detector.Dispose(); detector.Dispose(); }
                try { detector.PredictRaw(input); throw new InvalidOperationException("Disposed session accepted inference."); }
                catch (ObjectDisposedException) { }
            }
            string output = Path.GetFullPath(args[3]); Directory.CreateDirectory(Path.GetDirectoryName(output)!);
            var record = new { Status = "passed", Version = typeof(YoloDetector).Assembly.GetName().Version!.ToString(3), Model = Path.GetFileName(args[0]), Device = device.ToString(), Frames = frames + 24, SessionCycles = 4, WarmupPerCycle = 41, Tolerance = "abs 0.003 + rel 0.0003; retained outputs bit-exact", LiveMemoryLimitMiB = 32, Cycles = cycles,
                DllSha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(YoloDetector).Assembly.Location))), InputSha256 = Convert.ToHexStringLower(SHA256.HashData(raw)) };
            File.WriteAllText(output, JsonSerializer.Serialize(record, new JsonSerializerOptions { WriteIndented = true }) + "\n");
            Console.WriteLine($"PASS {record.Model} {device}: {record.Frames} checked frames, 4 load/dispose cycles.");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }

    /// <summary>比较当前输出与首帧副本，检查输出数量、长度、有限性和 FP32 容差。</summary>
    /// <param name="actual">本次返回的原始张量。</param>
    /// <param name="expected">独立保存的首帧数值。</param>
    /// <exception cref="InvalidDataException">输出结构或数值变化超过容差。</exception>
    private static void Validate(Tensor[] actual, float[][] expected)
    {
        if (actual.Length != expected.Length) throw new InvalidDataException("Output count changed.");
        for (int j = 0; j < actual.Length; j++)
        {
            if (actual[j].Length != expected[j].Length) throw new InvalidDataException("Output length changed.");
            for (int k = 0; k < actual[j].Length; k++)
                if (!float.IsFinite(actual[j].Data[k]) || MathF.Abs(actual[j].Data[k] - expected[j][k]) > .003f + .0003f * MathF.Abs(expected[j][k])) throw new InvalidDataException($"Output {j}/{k} changed across frames.");
        }
    }
}
