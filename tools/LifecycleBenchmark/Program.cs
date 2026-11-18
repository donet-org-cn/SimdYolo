using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;

/// <summary>在单独进程中测量会话构造、首帧和稳态；只依赖 .NET 与待比较的托管主库。</summary>
internal static class Program
{
    /// <summary>执行一个生命周期窗口；首条托管语句开始的计时不包含 dotnet 宿主及进入方法前的入口 JIT。</summary>
    /// <param name="args">模型、输入、后端、输出前缀及可选预热、秒数、尺寸、线程数、最少帧数。</param>
    /// <returns>成功为 0，参数、模型、数值或文件验证失败为 1。</returns>
    private static int Main(string[] args)
    {
        long managedEntry = Stopwatch.GetTimestamp();
        try
        {
            if (args.Length == 1 && args[0] is "--help" or "-h")
            {
                Console.WriteLine("LifecycleBenchmark MODEL INPUT cpu|vulkan|metal OUTPUT_PREFIX [warmup=30] [duration=3] [size=640] [threads=12] [minFrames=40] [rgbWidth rgbHeight]");
                return 0;
            }
            if (args.Length is < 4 or 10 or > 11) throw new ArgumentException("Use --help for lifecycle worker arguments.");
            string model = Path.GetFullPath(args[0]), inputPath = Path.GetFullPath(args[1]), prefix = Path.GetFullPath(args[3]);
            var requested = Enum.Parse<ComputeDevice>(args[2], ignoreCase: true);
            if (requested is not (ComputeDevice.Cpu or ComputeDevice.Vulkan or ComputeDevice.Metal)) throw new ArgumentException("Choose an explicit backend; fallback would invalidate the comparison.");
            int warmup = args.Length > 4 ? int.Parse(args[4], CultureInfo.InvariantCulture) : 30;
            double duration = args.Length > 5 ? double.Parse(args[5], CultureInfo.InvariantCulture) : 3;
            int size = args.Length > 6 ? int.Parse(args[6], CultureInfo.InvariantCulture) : 640;
            int threads = args.Length > 7 ? int.Parse(args[7], CultureInfo.InvariantCulture) : 12;
            int minFrames = args.Length > 8 ? int.Parse(args[8], CultureInfo.InvariantCulture) : 40;
            if (warmup < 1 || !double.IsFinite(duration) || duration < 0 || size < 32 || size % 32 != 0 || threads < 1 || minFrames < 1) throw new ArgumentException("Invalid lifecycle window settings.");
            using var process = Process.GetCurrentProcess();
            string? affinity = Environment.GetEnvironmentVariable("SIMDYOLO_BENCHMARK_AFFINITY");
            if (!string.IsNullOrEmpty(affinity))
            {
                if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Benchmark affinity requires Windows.");
                process.ProcessorAffinity = (nint)ulong.Parse(affinity, NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            }
            bool highQos = ConfigureWindowsQos(process);
            Directory.CreateDirectory(Path.GetDirectoryName(prefix)!);
            byte[] inputBytes = File.ReadAllBytes(inputPath);
            int imageWidth = 0, imageHeight = 0;
            Tensor? tensorInput = null;
            if (args.Length == 11)
            {
                imageWidth = int.Parse(args[9], CultureInfo.InvariantCulture);
                imageHeight = int.Parse(args[10], CultureInfo.InvariantCulture);
                if (imageWidth < 1 || imageHeight < 1 || inputBytes.Length != checked(3 * imageWidth * imageHeight))
                    throw new InvalidDataException("Image input must contain the declared dimensions as tightly packed RGB24.");
            }
            else
            {
                int elements = checked(3 * size * size);
                if (inputBytes.Length != checked(elements * sizeof(float)))
                    throw new InvalidDataException("Input bytes must contain one FP32 NCHW RGB tensor of the requested size.");
                if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("FP32 I/O uses little-endian byte order.");
                tensorInput = new Tensor(MemoryMarshal.Cast<byte, float>(inputBytes).ToArray(), 1, 3, size, size);
            }
            long baselineWorkingSet = process.WorkingSet64;
            long loadAllocatedStart = GC.GetTotalAllocatedBytes(precise: true), loadStart = Stopwatch.GetTimestamp();
            using var detector = new YoloDetector(model, new() { Device = requested, CpuThreads = threads, Width = size, Height = size });
            double loadMs = Stopwatch.GetElapsedTime(loadStart).TotalMilliseconds;
            long loadAllocated = GC.GetTotalAllocatedBytes(precise: true) - loadAllocatedStart;
            long firstAllocatedStart = GC.GetTotalAllocatedBytes(precise: true), firstStart = Stopwatch.GetTimestamp();
            object Predict() => tensorInput is null ? detector.Predict(inputBytes, imageWidth, imageHeight) : detector.PredictRaw(tensorInput);
            object first = Predict();
            double firstMs = Stopwatch.GetElapsedTime(firstStart).TotalMilliseconds;
            double readyMs = Stopwatch.GetElapsedTime(managedEntry).TotalMilliseconds;
            long firstAllocated = GC.GetTotalAllocatedBytes(precise: true) - firstAllocatedStart;
            // warmup 包含刚完成的首帧，30 表示首帧加 29 次额外调用。
            for (int i = 1; i < warmup; i++) Predict();
            long replayBefore = detector.ReplayedFrames;
            var samples = new List<double>(Math.Max(4096, minFrames));
            TimeSpan cpuBefore = process.TotalProcessorTime;
            int gen2Before = GC.CollectionCount(2);
            long allocatedBefore = GC.GetTotalAllocatedBytes(precise: true), telemetryBytes = 0;
            long steadyStart = Stopwatch.GetTimestamp(); object last = first;
            do
            {
                long inferenceStart = Stopwatch.GetTimestamp(); last = Predict();
                double elapsed = Stopwatch.GetElapsedTime(inferenceStart).TotalMilliseconds;
                // 采样数组扩容不属于推理分配；按调用线程实际分配量扣除，避免每帧内增加 GC 观测开销。
                if (samples.Count == samples.Capacity)
                {
                    long growthStart = GC.GetAllocatedBytesForCurrentThread();
                    samples.EnsureCapacity(checked(samples.Capacity * 2));
                    telemetryBytes += GC.GetAllocatedBytesForCurrentThread() - growthStart;
                }
                samples.Add(elapsed);
            } while (samples.Count < minFrames || Stopwatch.GetElapsedTime(steadyStart).TotalSeconds < duration);
            double seconds = Stopwatch.GetElapsedTime(steadyStart).TotalSeconds;
            long allocated = GC.GetTotalAllocatedBytes(precise: true) - allocatedBefore;
            double cpuSeconds = (process.TotalProcessorTime - cpuBefore).TotalSeconds;
            int gen2Collections = GC.CollectionCount(2) - gen2Before;
            process.Refresh();
            long workingSet = process.WorkingSet64, peakWorkingSet = process.PeakWorkingSet64, privateBytes = process.PrivateMemorySize64;
            double? vulkanFrameBufferCapacityMiB = GetVulkanFrameBufferCapacityMiB(detector);
            // 数值扫描、原始输出写盘、SHA256 和 JSON 均发生在全部计时窗口之后。
            var firstOutputs = SaveResults(first, prefix + ".first"); var outputs = SaveResults(last, prefix);
            double[] ordered = samples.Order().ToArray();
            double median = ordered.Length % 2 == 0 ? (ordered[ordered.Length / 2 - 1] + ordered[ordered.Length / 2]) * .5 : ordered[ordered.Length / 2];
            var record = new
            {
                Version = typeof(YoloDetector).Assembly.GetName().Version!.ToString(3), Model = Path.GetFileName(model), ModelPath = model,
                RequestedBackend = requested.ToString(), Backend = detector.BackendName, detector.FallbackReason,
                InputMode = tensorInput is null ? "RGB24" : "FP32", ImageWidth = imageWidth, ImageHeight = imageHeight, Affinity = affinity,
                Size = size, CpuThreads = threads, LogicalProcessors = Environment.ProcessorCount, WindowsHighQos = highQos,
                WarmupFrames = warmup, MinimumFrames = minFrames, DurationSeconds = duration,
                ModelLoadMs = loadMs, FirstInferenceMs = firstMs, ReadyToFirstOutputMs = readyMs,
                LoadManagedBytes = loadAllocated, FirstInferenceManagedBytes = firstAllocated,
                MedianMs = median, MeanMs = samples.Average(), P95Ms = ordered[(int)Math.Ceiling(.95 * (ordered.Length - 1))], Frames = samples.Count, Seconds = seconds,
                CpuPercent = cpuSeconds / seconds / Environment.ProcessorCount * 100, CpuCorePercent = cpuSeconds / seconds * 100,
                CpuSeconds = cpuSeconds, CpuMillisecondsPerFrame = cpuSeconds * 1000 / samples.Count, Gen2Collections = gen2Collections,
                AllocatedBytesPerFrame = (double)(allocated - telemetryBytes) / samples.Count, RawManagedAllocatedBytes = allocated, TelemetryManagedBytes = telemetryBytes,
                BaselineWorkingSetMiB = baselineWorkingSet / 1048576.0, WorkingSetMiB = workingSet / 1048576.0,
                PeakWorkingSetMiB = peakWorkingSet / 1048576.0, PrivateMemoryMiB = privateBytes / 1048576.0,
                VulkanFrameBufferCapacityMiB = vulkanFrameBufferCapacityMiB,
                detector.PipelineCacheLoaded, ReplayedFramesBeforeSteady = replayBefore, ReplayedFrames = detector.ReplayedFrames, ReplayedSteadyFrames = detector.ReplayedFrames - replayBefore,
                OutputPrefix = prefix, FirstOutputPrefix = prefix + ".first", Outputs = outputs, FirstOutputs = firstOutputs, SamplesMs = samples,
                Runtime = RuntimeInformation.FrameworkDescription, Architecture = RuntimeInformation.ProcessArchitecture.ToString()
            };
            File.WriteAllText(prefix + ".json", JsonSerializer.Serialize(record, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine(JsonSerializer.Serialize(new { record.Model, record.Backend, record.ModelLoadMs, record.FirstInferenceMs, record.MedianMs, record.ReplayedSteadyFrames }));
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }

    // Only the benchmark worker opts into HighQoS. The library keeps the caller's
    // OS scheduling policy; this option neither changes the system power plan nor CPU affinity.
    private static bool ConfigureWindowsQos(Process process)
    {
        if (Environment.GetEnvironmentVariable("SIMDYOLO_BENCHMARK_HIGH_QOS") != "1") return false;
        if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("HighQoS benchmarking requires Windows.");
        PowerThrottling state = new() { Version = 1, ControlMask = 1, StateMask = 0 };
        if (!SetProcessInformation(process.Handle, 4, ref state, 12))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastPInvokeError(), "Could not set benchmark HighQoS.");
        return true;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PowerThrottling { internal uint Version, ControlMask, StateMask; }

    [DllImport("kernel32.dll", ExactSpelling = true, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetProcessInformation(nint process, int informationClass, ref PowerThrottling state, uint bytes);

    // Read buffer capacities after timing so the same worker can inspect the published DLL.
    private static double? GetVulkanFrameBufferCapacityMiB(YoloDetector detector)
    {
        const BindingFlags instanceFields = BindingFlags.Instance | BindingFlags.NonPublic;
        if (typeof(YoloDetector).GetField("backend", instanceFields)?.GetValue(detector) is not VulkanBackend backend) return null;
        var type = typeof(VulkanBackend);
        var capacityField = type.GetNestedType("GpuBuffer", BindingFlags.NonPublic)?.GetField("Capacity");
        if (capacityField is null || type.GetField("arenas", instanceFields)?.GetValue(backend) is not System.Collections.IEnumerable arenas)
            return null;
        ulong bytes = 0;
        foreach (var arena in arenas) bytes += (ulong)capacityField.GetValue(arena)!;
        if (type.GetField("winogradProducts", instanceFields)?.GetValue(backend) is object winograd)
            bytes += (ulong)capacityField.GetValue(winograd)!;
        if (type.GetField("winogradInputSlots", instanceFields)?.GetValue(backend) is System.Collections.IEnumerable inputs)
            foreach (var slot in inputs)
                if (slot.GetType().GetField("Buffer", instanceFields)?.GetValue(slot) is object input)
                    bytes += (ulong)capacityField.GetValue(input)!;
        return bytes / 1048576.0;
    }

    private static object[] SaveResults(object result, string prefix)
    {
        if (result is Tensor[] tensors) return SaveOutputs(tensors, prefix);
        var instances = (IReadOnlyList<YoloInstance>)result;
        var data = JsonSerializer.SerializeToUtf8Bytes(instances.Select(instance => new
        {
            instance.Box, instance.Keypoints,
            Mask = instance.Mask is { } mask ? new { mask.X, mask.Y, mask.Width, mask.Height } : null
        }));
        string path = prefix + ".instances.json";
        File.WriteAllBytes(path, data);
        var saved = new List<object>
        {
            new { Index = 0, Shape = new[] { instances.Count }, Path = path, Bytes = data.Length,
                  Sha256 = Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant(), Format = "json" }
        };
        for (int i = 0; i < instances.Count; i++)
            if (instances[i].Mask is { } mask)
                saved.Add(SaveFloats(mask.Probabilities, [mask.Height, mask.Width], prefix, i + 1));
        return [.. saved];
    }

    /// <summary>保存首帧或稳态原始输出，并核对有限数值；文件写入和哈希不计入性能窗口。</summary>
    /// <param name="outputs">已经 materialize 且可持有的公开输出张量。</param>
    /// <param name="prefix">原始文件输出前缀。</param>
    /// <returns>按输出顺序保存的形状、文件名、字节数和哈希。</returns>
    /// <exception cref="InvalidDataException">输出为空或包含非有限值。</exception>
    /// <exception cref="IOException">输出文件无法创建或写入。</exception>
    private static object[] SaveOutputs(Tensor[] outputs, string prefix)
    {
        if (outputs.Length == 0) throw new InvalidDataException("Model returned no outputs.");
        var records = new object[outputs.Length];
        for (int i = 0; i < outputs.Length; i++)
            records[i] = SaveFloats(outputs[i].Data, outputs[i].Shape, prefix, i);
        return records;
    }

    private static object SaveFloats(float[] values, int[] shape, string prefix, int index)
    {
        foreach (float value in values) if (!float.IsFinite(value)) throw new InvalidDataException("Model returned a non-finite value.");
        string path = prefix + $".{index}.f32";
        ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(values.AsSpan());
        using (var file = File.Create(path)) file.Write(bytes);
        File.WriteAllText(prefix + $".{index}.shape.json", JsonSerializer.Serialize(shape));
        return new { Index = index, Shape = shape, Path = path, Bytes = bytes.Length, Sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(), Format = "f32" };
    }
}
