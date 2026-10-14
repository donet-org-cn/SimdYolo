using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;

// 此工具只用于 ONNX Runtime 基准对照，不是 SimdYolo 运行库依赖。
// 参数顺序为模型、cpu/cuda、正方形尺寸、最低帧数、线程数、预热数、FP32 输入、最低秒数、同步目录、输出前缀、CUDA DLL 目录、ORT 原生目录。
// 输入为 [1,3,size,size] 连续 FP32；输出为统计 JSON 和每个模型输出的 .f32 数据、.json 维度文件。
// 配置或输入长度错误抛出 ArgumentException；数值转换、原生库加载、ORT 执行、文件读写等异常统一输出并返回退出码 1。
try
{
    if (args.Length < 12) throw new ArgumentException("MODEL cpu|cuda SIZE ITERATIONS THREADS WARMUP INPUT.f32 DURATION SIGNAL_DIR OUTPUT_PREFIX CUDA_DLL_DIR ORT_NATIVE_DIR");
    string modelPath = args[0], backend = args[1], inputPath = args[6], signalDirectory = args[8], outputPrefix = args[9];
    int size = int.Parse(args[2]), iterations = int.Parse(args[3]), threads = int.Parse(args[4]), warmup = int.Parse(args[5]);
    double duration = double.Parse(args[7], CultureInfo.InvariantCulture);
    if (backend is not ("cpu" or "cuda") || size <= 0 || iterations <= 0 || threads <= 0 || warmup <= 0 || duration < 2) throw new ArgumentException("Invalid benchmark configuration.");
    // 在加载原生库和模型前记录工作集；随后启动耗时包含 ORT/CUDA 初始化、输入准备和全部预热。
    long baselineRss = Environment.WorkingSet;
    var startup = Stopwatch.StartNew();
    string nativeDirectory = Path.GetFullPath(args[11]);

    // 使用与 Python GPU 对照进程相同的官方 ORT 原生构建。
    // 当时测试的 NuGet 原生构建在本机 GTX 1050 Ti 的 QuickGelu 上失败，因此通过指定目录解析 DLL。
    NativeLibrary.SetDllImportResolver(typeof(OrtEnv).Assembly, (name, assembly, searchPath) =>
        name.StartsWith("onnxruntime", StringComparison.Ordinal) ? NativeLibrary.Load(Path.Combine(nativeDirectory, "onnxruntime.dll")) : IntPtr.Zero);
    // CUDA 对照显式加载指定目录中的依赖库；缺失的可选 DLL 跳过，实际加载失败由外层报告。
    if (backend == "cuda" && args.Length > 10)
    {
        string dllDirectory = Path.GetFullPath(args[10]);
        Environment.SetEnvironmentVariable("PATH", dllDirectory + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH"));
        foreach (string name in new[] { "cublasLt64_12.dll", "cublas64_12.dll", "nvrtc64_120_0.dll", "cudart64_12.dll", "cufft64_11.dll", "cudnn_engines_runtime_compiled64_9.dll", "cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll", "cudnn_adv64_9.dll", "cudnn_graph64_9.dll", "cudnn64_9.dll" })
        {
            string path = Path.Combine(dllDirectory, name);
            if (File.Exists(path)) NativeLibrary.Load(path);
        }
    }
    // 两种后端都启用全部 ORT 图优化；算子内并行使用指定 threads，算子间保持单线程顺序执行。
    using var options = new SessionOptions { IntraOpNumThreads = threads, InterOpNumThreads = 1, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL, GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
    using var cudaOptions = new OrtCUDAProviderOptions();
    if (backend == "cuda")
    {
        // 禁用 TF32，使用穷举 cuDNN 算法选择及最大工作区；算法搜索位于预热阶段，不计入正式窗口。
        cudaOptions.UpdateOptions(new Dictionary<string, string> { ["use_tf32"] = "0", ["cudnn_conv_algo_search"] = "EXHAUSTIVE", ["cudnn_conv_use_max_workspace"] = "1" });
        options.AppendExecutionProvider_CUDA(cudaOptions);
    }
    using var session = new InferenceSession(modelPath, options);
    // 各实现共用同一 FP32 输入文件，工具本身不执行图像预处理；长度必须准确匹配三个通道。
    byte[] bytes = File.ReadAllBytes(inputPath);
    if (bytes.Length != 3 * size * size * sizeof(float)) throw new ArgumentException("Input byte count mismatch.");
    float[] input = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    using var inputValue = OrtValue.CreateTensorValueFromMemory(input, new long[] { 1, 3, size, size });
    var inputs = new Dictionary<string, OrtValue> { [session.InputNames[0]] = inputValue };
    using var runOptions = new RunOptions();
    // 一轮 Infer 执行 ORT，并把所有输出复制到独立托管数组，返回数据数组和对应维度。
    // 这使计时包含可供调用方长期持有的主机结果；CUDA 到主机的同步和复制也在此边界内。
    // ORT 输出不是 FP32、推理失败或数据访问失败时会抛出 ORT/托管异常，由顶层统一处理。
    // outputs 在返回前释放，返回数组与 ORT 分配器存储没有别名。
    (float[][] Data, long[][] Shapes) Infer()
    {
        using var outputs = session.Run(runOptions, inputs, session.OutputNames);
        var data = new float[outputs.Count][];
        var shapes = new long[outputs.Count][];
        for (int i = 0; i < outputs.Count; i++)
        {
            data[i] = outputs[i].GetTensorDataAsSpan<float>().ToArray();
            var info = outputs[i].GetTensorTypeAndShape(); shapes[i] = info.Shape;
        }
        return (data, shapes);
    }
    // 正式窗口之前预热 warmup 次，首次预热另记 firstMs；startupMs 截止于最后一次预热结束。
    double firstMs = 0;
    for (int i = 0; i < warmup; i++)
    {
        var watch = Stopwatch.StartNew(); Infer(); if (i == 0) firstMs = watch.Elapsed.TotalMilliseconds;
    }
    double startupMs = startup.Elapsed.TotalMilliseconds;
    // ready.json 告知外部资源采样器初始化完成；等待 go 的时间及文件写入不进入正式计时。
    Directory.CreateDirectory(signalDirectory);
    string provider = backend == "cuda" ? "CUDAExecutionProvider" : "CPUExecutionProvider";
    File.WriteAllText(Path.Combine(signalDirectory, "ready.json"), JsonSerializer.Serialize(new { Backend = provider, StartupMs = startupMs, FirstInferenceMs = firstMs, BaselineWorkingSetMiB = baselineRss / 1048576.0 }));
    // 同步等待超过两分钟抛出 TimeoutException；该等待不修改推理性能样本。
    var signalTimeout = Stopwatch.StartNew();
    while (!File.Exists(Path.Combine(signalDirectory, "go")))
    {
        if (signalTimeout.Elapsed.TotalMinutes > 2) throw new TimeoutException("Benchmark start signal timed out.");
        Thread.Sleep(5);
    }
    // 使用本进程全部线程累计 CPU 时间测量窗口消耗，而不是仅计算调用线程时间。
    using var process = Process.GetCurrentProcess();
    var beforeCpu = process.TotalProcessorTime;
    var samples = new List<double>();
    (float[][] Data, long[][] Shapes) result = (Array.Empty<float[]>(), Array.Empty<long[]>());
    // 单帧样本只包围 Infer；完整窗口还包含计时和样本记录开销。
    // 帧数与持续秒数都达到下限才结束，避免短样本与资源采样窗口不一致。
    var window = Stopwatch.StartNew();
    while (samples.Count < iterations || window.Elapsed.TotalSeconds < duration)
    {
        var watch = Stopwatch.StartNew(); result = Infer(); samples.Add(watch.Elapsed.TotalMilliseconds);
    }
    double seconds = window.Elapsed.TotalSeconds, cpuSeconds = (process.TotalProcessorTime - beforeCpu).TotalSeconds;
    // 完成标记、最后一帧结果保存、样本排序和统计 JSON 序列化全部位于计时窗口之外。
    File.WriteAllText(Path.Combine(signalDirectory, "done"), "done");
    for (int i = 0; i < result.Data.Length; i++)
    {
        File.WriteAllBytes(outputPrefix + $".{i}.f32", MemoryMarshal.AsBytes(result.Data[i].AsSpan()).ToArray());
        File.WriteAllText(outputPrefix + $".{i}.json", JsonSerializer.Serialize(result.Shapes[i]));
    }
    double[] sorted = samples.Order().ToArray();
    // p 为 0 到 1 的分位位置；在非空 sorted 样本上做线性插值，返回延迟毫秒。
    // 调用方只传入 0.5 或 0.95，本局部函数不额外验证范围。
    double Percentile(double p) { double index = p * (sorted.Length - 1); int lo = (int)index, hi = (int)Math.Ceiling(index); return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo); }
    // CpuPercent 按逻辑处理器总容量归一化；CpuMsPerFrame 使用全部线程累计 CPU 时间除以帧数。
    // ThroughputFps 按完整窗口计算，FramesPerSecond 使用 1000/中位延迟，均保留原始 SamplesMs 供复核。
    Console.WriteLine(JsonSerializer.Serialize(new { Model = Path.GetFileName(modelPath), Backend = provider, Version = OrtEnv.Instance().GetVersionString(), Host = ".NET", Outputs = result.Shapes, Shape = new[] { 1, 3, size, size }, Iterations = samples.Count, Warmup = warmup, CpuThreads = threads, StartupMs = startupMs, FirstInferenceMs = firstMs, WindowSeconds = seconds, CpuSeconds = cpuSeconds, CpuPercent = cpuSeconds / seconds / Environment.ProcessorCount * 100, CpuMsPerFrame = cpuSeconds * 1000 / samples.Count, ThroughputFps = samples.Count / seconds, MedianMs = Percentile(.5), P95Ms = Percentile(.95), MeanMs = samples.Average(), MinMs = sorted[0], FramesPerSecond = 1000 / Percentile(.5), SamplesMs = samples }, new JsonSerializerOptions { WriteIndented = true }));
}
catch (Exception error) { Console.Error.WriteLine(error); Environment.ExitCode = 1; }
