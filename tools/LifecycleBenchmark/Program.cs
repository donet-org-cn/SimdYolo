using System.Diagnostics;
using System.Globalization;
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
                Console.WriteLine("LifecycleBenchmark MODEL INPUT.f32 cpu|vulkan|metal OUTPUT_PREFIX [warmup=30] [duration=3] [size=640] [threads=12] [minFrames=40]");
                return 0;
            }
            if (args.Length is < 4 or > 9) throw new ArgumentException("Use --help for lifecycle worker arguments.");
            string model = Path.GetFullPath(args[0]), inputPath = Path.GetFullPath(args[1]), prefix = Path.GetFullPath(args[3]);
            var requested = Enum.Parse<ComputeDevice>(args[2], ignoreCase: true);
            if (requested is not (ComputeDevice.Cpu or ComputeDevice.Vulkan or ComputeDevice.Metal)) throw new ArgumentException("Choose an explicit backend; fallback would invalidate the comparison.");
            int warmup = args.Length > 4 ? int.Parse(args[4], CultureInfo.InvariantCulture) : 30;
            double duration = args.Length > 5 ? double.Parse(args[5], CultureInfo.InvariantCulture) : 3;
            int size = args.Length > 6 ? int.Parse(args[6], CultureInfo.InvariantCulture) : 640;
            int threads = args.Length > 7 ? int.Parse(args[7], CultureInfo.InvariantCulture) : 12;
            int minFrames = args.Length > 8 ? int.Parse(args[8], CultureInfo.InvariantCulture) : 40;
            if (warmup < 1 || !double.IsFinite(duration) || duration < 0 || size < 32 || size % 32 != 0 || threads < 1 || minFrames < 1) throw new ArgumentException("Invalid lifecycle window settings.");
            Directory.CreateDirectory(Path.GetDirectoryName(prefix)!);
            byte[] inputBytes = File.ReadAllBytes(inputPath);
            int elements = checked(3 * size * size);
            if (inputBytes.Length != checked(elements * sizeof(float))) throw new InvalidDataException("Input bytes must contain one FP32 NCHW RGB tensor of the requested size.");
            float[] values = new float[elements];
            if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("The benchmark input and output contract uses little-endian FP32.");
            MemoryMarshal.Cast<byte, float>(inputBytes).CopyTo(values);
            var input = new Tensor(values, 1, 3, size, size);
            using var process = Process.GetCurrentProcess();
            long baselineWorkingSet = process.WorkingSet64;
            long loadAllocatedStart = GC.GetTotalAllocatedBytes(precise: true), loadStart = Stopwatch.GetTimestamp();
            using var detector = new YoloDetector(model, new() { Device = requested, CpuThreads = threads, Width = size, Height = size });
            double loadMs = Stopwatch.GetElapsedTime(loadStart).TotalMilliseconds;
            long loadAllocated = GC.GetTotalAllocatedBytes(precise: true) - loadAllocatedStart;
            long firstAllocatedStart = GC.GetTotalAllocatedBytes(precise: true), firstStart = Stopwatch.GetTimestamp();
            Tensor[] first = detector.PredictRaw(input);
            double firstMs = Stopwatch.GetElapsedTime(firstStart).TotalMilliseconds;
            double readyMs = Stopwatch.GetElapsedTime(managedEntry).TotalMilliseconds;
            long firstAllocated = GC.GetTotalAllocatedBytes(precise: true) - firstAllocatedStart;
            // warmup 包含刚完成的首帧，30 表示首帧加 29 次额外调用。
            for (int i = 1; i < warmup; i++) detector.PredictRaw(input);
            long replayBefore = detector.ReplayedFrames;
            var samples = new List<double>(Math.Max(4096, minFrames));
            TimeSpan cpuBefore = process.TotalProcessorTime;
            long allocatedBefore = GC.GetTotalAllocatedBytes(precise: true), telemetryBytes = 0;
            long steadyStart = Stopwatch.GetTimestamp(); Tensor[] last = first;
            do
            {
                long inferenceStart = Stopwatch.GetTimestamp(); last = detector.PredictRaw(input);
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
            process.Refresh(); long workingSet = process.WorkingSet64;
            // 数值扫描、原始输出写盘、SHA256 和 JSON 均发生在全部计时窗口之后。
            var firstOutputs = SaveOutputs(first, prefix + ".first"); var outputs = SaveOutputs(last, prefix);
            double[] ordered = samples.Order().ToArray();
            double median = ordered.Length % 2 == 0 ? (ordered[ordered.Length / 2 - 1] + ordered[ordered.Length / 2]) * .5 : ordered[ordered.Length / 2];
            var record = new
            {
                Version = typeof(YoloDetector).Assembly.GetName().Version!.ToString(3), Model = Path.GetFileName(model), ModelPath = model,
                RequestedBackend = requested.ToString(), Backend = detector.BackendName, detector.FallbackReason,
                Size = size, CpuThreads = threads, LogicalProcessors = Environment.ProcessorCount, WarmupFrames = warmup, MinimumFrames = minFrames, DurationSeconds = duration,
                ModelLoadMs = loadMs, FirstInferenceMs = firstMs, ReadyToFirstOutputMs = readyMs,
                LoadManagedBytes = loadAllocated, FirstInferenceManagedBytes = firstAllocated,
                MedianMs = median, MeanMs = samples.Average(), P95Ms = ordered[(int)Math.Ceiling(.95 * (ordered.Length - 1))], Frames = samples.Count, Seconds = seconds,
                CpuPercent = cpuSeconds / seconds / Environment.ProcessorCount * 100, CpuCorePercent = cpuSeconds / seconds * 100,
                AllocatedBytesPerFrame = (double)(allocated - telemetryBytes) / samples.Count, RawManagedAllocatedBytes = allocated, TelemetryManagedBytes = telemetryBytes,
                BaselineWorkingSetMiB = baselineWorkingSet / 1048576.0, WorkingSetMiB = workingSet / 1048576.0,
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
        {
            var tensor = outputs[i]; foreach (float value in tensor.Data) if (!float.IsFinite(value)) throw new InvalidDataException("Model returned a non-finite value.");
            string path = prefix + $".{i}.f32";
            ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(tensor.Data.AsSpan());
            using (var file = File.Create(path)) file.Write(bytes);
            File.WriteAllText(prefix + $".{i}.shape.json", JsonSerializer.Serialize(tensor.Shape));
            records[i] = new { Index = i, Shape = tensor.Shape, Path = path, Bytes = bytes.Length, Sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant() };
        }
        return records;
    }
}
