using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using SimdYolo;

// 命令行入口接收 inspect、labels、bench、profile、raw、detect、predict 和 preprocess 命令。
// 模型和输入由文件提供，结果写入标准输出及指定文件；所有实际异常统一打印到标准错误并返回退出码 1。
// 参数不足、数值格式错误、无效配置、输入数据损坏、模型或后端不支持以及文件读写错误都由下方 catch 报告。
try
{
    if (args.Length == 0) { Console.WriteLine("SimdYolo: inspect MODEL | labels MODEL | bench MODEL [cpu|vulkan|metal|auto] [size=640] [iterations=5] [threads=0] [warmup=1] [input.f32] [durationSeconds=0] [signalDirectory] [outputPrefix] | profile MODEL [backend] [size] | raw MODEL INPUT.f32 OUTPUT_PREFIX [backend] [size] [batch=1] | detect MODEL IMAGE.ppm [backend] | predict MODEL IMAGE.ppm [backend] [outputDir] [size]"); return; }
    if (args.Length < 2) throw new ArgumentException("Model path required.");
    string command = args[0], path = args[1];
    // inspect 和 labels 只读取元数据，不加载权重存储，也不创建 CPU/GPU 后端。
    if (command is "inspect" or "labels")
    {
        var info = ModelInfo.Read(path);
        Console.WriteLine(JsonSerializer.Serialize(command == "labels" ? (object)info.ClassNames : info, new JsonSerializerOptions { WriteIndented = true, Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() } })); return;
    }
    // preprocess 的 path 是 P6 图像；args[2] 是原始 FP32 输出路径，args[3] 可指定正方形目标尺寸。
    // 输出按 NCHW 保存，同时写入缩放和填边 JSON，便于各实现复用完全相同的预处理输入。
    if (command == "preprocess")
    {
        var (pixels, w, h) = ReadPpm(path); int target = args.Length > 3 ? int.Parse(args[3]) : 640;
        var (tensor, transform) = ImagePreprocessor.Letterbox(pixels, w, h, target, target);
        File.WriteAllBytes(args[2], MemoryMarshal.AsBytes(tensor.Data.AsSpan()).ToArray()); File.WriteAllText(args[2] + ".json", JsonSerializer.Serialize(transform)); return;
    }
    // 启动耗时从创建模型前开始，到全部预热完成为止，包含模型加载和基准输入准备。
    // 基线工作集在模型加载之前记录；它是当前进程驻留内存，不代表托管堆或峰值内存。
    var startup = Stopwatch.StartNew();
    long baselineWorkingSet = Environment.WorkingSet;
    int deviceIndex = command == "raw" ? 4 : command is "detect" or "predict" ? 3 : 2;
    var device = args.Length > deviceIndex ? Enum.Parse<ComputeDevice>(args[deviceIndex], true) : ComputeDevice.Cpu;
    int sizeIndex = command is "raw" or "predict" ? 5 : command is "detect" or "inspect" ? int.MaxValue : 3;
    int size = args.Length > sizeIndex ? int.Parse(args[sizeIndex]) : 640;
    int threads = command == "bench" && args.Length > 5 ? int.Parse(args[5]) : 0;
    // using 保证命令结束时释放后端和 CPU 工作线程。profile 回调会同步 GPU，属于诊断模式。
    long loadStarted = Stopwatch.GetTimestamp();
    using var detector = new YoloDetector(path, new() { Device = device, Width = size, Height = size, CpuThreads = threads, LayerCompleted = command == "profile" ? t => Console.WriteLine($"{t.Name} {t.Operation} {t.Milliseconds:F3} ms") : null });
    double modelLoadMs = Stopwatch.GetElapsedTime(loadStarted).TotalMilliseconds;
    Console.WriteLine(detector.BackendName); if (detector.FallbackReason is { } reason) Console.WriteLine("Fallback: " + reason);
    // detect：PPM 解码位于计时外；耗时包含填边、推理和检测后处理，输出为原图坐标 JSON。
    if (command == "detect")
    {
        var (pixels, w, h) = ReadPpm(args[2]); var watch = Stopwatch.StartNew(); var detections = detector.Detect(pixels, w, h);
        Console.WriteLine(JsonSerializer.Serialize(new { ElapsedMs = watch.Elapsed.TotalMilliseconds, Detections = detections }, new JsonSerializerOptions { WriteIndented = true })); return;
    }
    // predict：计时同样包含图像预处理、模型执行和任务解码；掩码与 JSON 写文件发生在取样之后。
    // 分割掩码以 PGM 灰度保存，实例 JSON 记录其位置、尺寸及文件名，姿态结果保留关键点。
    if (command == "predict")
    {
        var (pixels, w, h) = ReadPpm(args[2]); var watch = Stopwatch.StartNew(); var instances = detector.Predict(pixels, w, h); double elapsed = watch.Elapsed.TotalMilliseconds;
        string directory = args.Length > 4 ? args[4] : "predictions"; Directory.CreateDirectory(directory);
        var results = instances.Select((instance, i) =>
        {
            string? maskPath = null;
            if (instance.Mask is { } mask)
            {
                maskPath = $"mask-{i}.pgm"; using var f = File.Create(Path.Combine(directory, maskPath));
                f.Write(System.Text.Encoding.ASCII.GetBytes($"P5\n{mask.Width} {mask.Height}\n255\n")); f.Write(mask.Probabilities.Select(p => (byte)Math.Clamp((int)MathF.Round(p * 255), 0, 255)).ToArray());
            }
            return new { instance.Box, ClassName = detector.ClassNames.GetValueOrDefault(instance.Box.ClassId), instance.Keypoints, Mask = instance.Mask is { } m ? new { m.X, m.Y, m.Width, m.Height, File = maskPath } : null };
        }).ToArray();
        string json = JsonSerializer.Serialize(new { Task = detector.Task.ToString(), ElapsedMs = elapsed, Instances = results }, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(directory, "predictions.json"), json); Console.WriteLine($"{instances.Count} instances; {elapsed:F2} ms; {Path.GetFullPath(Path.Combine(directory, "predictions.json"))}"); return;
    }
    int[] shape = detector.InputShape; int height = shape[2] > 0 ? shape[2] : size, width = shape[3] > 0 ? shape[3] : size;
    int batch = command == "raw" && args.Length > 6 ? int.Parse(args[6]) : 1;
    if (batch <= 0) throw new ArgumentException("Batch must be positive.");
    var input = new Tensor(batch, 3, height, width);
    // raw：读取已预处理的连续 FP32 文件，其字节数必须与 [1,3,H,W] 完全一致。
    // 每个原始输出分别保存 .f32 数据与 .json 形状；字节数错误抛出 ArgumentException。
    if (command == "raw")
    {
        var bytes = File.ReadAllBytes(args[2]); if (bytes.Length != input.Length * 4) throw new ArgumentException("Input raw byte count mismatch.");
        MemoryMarshal.Cast<byte, float>(bytes).CopyTo(input.Data); var output = detector.PredictRaw(input);
        for (int i = 0; i < output.Length; i++) { File.WriteAllBytes(args[3] + $".{i}.f32", MemoryMarshal.AsBytes(output[i].Data.AsSpan()).ToArray()); File.WriteAllText(args[3] + $".{i}.json", JsonSerializer.Serialize(output[i].Shape)); Console.WriteLine(output[i]); }
        return;
    }
    // profile 用常量输入执行两帧并打印逐层耗时；逐层回调和同步会影响总耗时，不用于正式排名。
    if (command == "profile") { Array.Fill(input.Data, .5f); detector.PredictRaw(input); detector.PredictRaw(input); return; }
    if (command != "bench") throw new ArgumentException("Unknown command.");
    // bench 参数依次指定最低迭代数、CPU 线程数、预热数、可选 FP32 输入、最低计时秒数和同步目录。
    // 不提供输入文件时使用 0.5 常量。迭代数非正、预热负数或输入长度错误会抛出 ArgumentException。
    Array.Fill(input.Data, 0.5f); int iterations = args.Length > 4 ? int.Parse(args[4]) : 5;
    int warmup = args.Length > 6 ? int.Parse(args[6]) : 1;
    if (iterations < 1 || warmup < 0) throw new ArgumentException("Iterations must be positive and warmup nonnegative.");
    if (args.Length > 7)
    {
        var bytes = File.ReadAllBytes(args[7]); if (bytes.Length != input.Length * sizeof(float)) throw new ArgumentException("Benchmark input byte count mismatch.");
        MemoryMarshal.Cast<byte, float>(bytes).CopyTo(input.Data);
    }
    // 计时秒数必须有限且非负；0 表示仅按最低迭代数运行。outputPrefix 指定最后一帧原始结果的导出前缀。
    double durationSeconds = args.Length > 8 ? double.Parse(args[8], System.Globalization.CultureInfo.InvariantCulture) : 0;
    string? signalDirectory = args.Length > 9 ? args[9] : null;
    string? outputPrefix = args.Length > 10 ? args[10] : null;
    if (durationSeconds < 0 || !double.IsFinite(durationSeconds)) throw new ArgumentException("Duration must be finite and nonnegative.");
    // 预热不纳入正式延迟样本；FirstInferenceMs 只记录首次预热，未预热时保持 0。
    double firstInferenceMs = 0, readyToFirstOutputMs = 0;
    var warmupWatch = Stopwatch.StartNew();
    for (int i = 0; i < warmup; i++)
    {
        var firstWatch = Stopwatch.StartNew(); detector.PredictRaw(input);
        if (i == 0) { firstInferenceMs = firstWatch.Elapsed.TotalMilliseconds; readyToFirstOutputMs = startup.Elapsed.TotalMilliseconds; }
    }
    double warmupMs = warmupWatch.Elapsed.TotalMilliseconds;
    double startupMs = startup.Elapsed.TotalMilliseconds;
    // 写 ready.json 后等待外部 go 标记，使资源采样器与推理窗口同步。
    // 等待与 ready 文件写入均在正式计时之外；超过两分钟抛出 TimeoutException。
    if (signalDirectory is not null)
    {
        Directory.CreateDirectory(signalDirectory);
        File.WriteAllText(Path.Combine(signalDirectory, "ready.json"), JsonSerializer.Serialize(new { Backend = detector.BackendName, StartupMs = startupMs, ModelLoadMs = modelLoadMs, FirstInferenceMs = firstInferenceMs, ReadyToFirstOutputMs = readyToFirstOutputMs, WarmupMs = warmupMs, BaselineWorkingSetMiB = baselineWorkingSet / 1048576.0 }));
        var signalTimeout = Stopwatch.StartNew();
        while (!File.Exists(Path.Combine(signalDirectory, "go")))
        {
            if (signalTimeout.Elapsed.TotalMinutes > 2) throw new TimeoutException("Benchmark start signal timed out.");
            Thread.Sleep(5);
        }
    }
    // 每帧延迟只包围 PredictRaw，包含设备推理结果同步到可供调用方持有的输出。
    // 整体窗口与进程 CPU 时间还包含计时、列表记录和循环开销；窗口同时满足最低迭代数与最低秒数才结束。
    var times = new List<double>(); Tensor[]? result = null;
    using var process = Process.GetCurrentProcess();
    var cpuStarted = process.TotalProcessorTime;
    var window = Stopwatch.StartNew();
    while (times.Count < iterations || window.Elapsed.TotalSeconds < durationSeconds) { var watch = Stopwatch.StartNew(); result = detector.PredictRaw(input); times.Add(watch.Elapsed.TotalMilliseconds); }
    double windowSeconds = window.Elapsed.TotalSeconds;
    double cpuSeconds = (process.TotalProcessorTime - cpuStarted).TotalSeconds;
    // 在窗口与 CPU 时间已取样后写 done 标记；原始结果导出、排序和 JSON 序列化也不计入推理窗口。
    if (signalDirectory is not null) File.WriteAllText(Path.Combine(signalDirectory, "done"), "done");
    if (outputPrefix is not null)
        for (int i = 0; i < result!.Length; i++)
        {
            File.WriteAllBytes(outputPrefix + $".{i}.f32", MemoryMarshal.AsBytes(result[i].Data.AsSpan()).ToArray());
            File.WriteAllText(outputPrefix + $".{i}.json", JsonSerializer.Serialize(result[i].Shape));
        }
    var samples = times.ToArray(); times.Sort();
    // p 是 0 到 1 的分位位置；对已排序的非空样本做线性插值，返回毫秒值。
    // 当前调用只使用 0.5 与 0.95；此局部函数不另外校验 p。
    double Percentile(double p) { double index = p * (times.Count - 1); int lo = (int)index, hi = (int)Math.Ceiling(index); return times[lo] + (times[hi] - times[lo]) * (index - lo); }
    double median = Percentile(.5);
    // CpuPercent 按逻辑处理器总容量归一化；CpuMsPerFrame 是进程全部线程累计 CPU 时间除以帧数。
    // ThroughputFps 按完整窗口计算，FramesPerSecond 是 1000/中位延迟；二者定义不同。
    Console.WriteLine(JsonSerializer.Serialize(new { Version = "SimdYolo " + typeof(YoloDetector).Assembly.GetName().Version!.ToString(3), Model = Path.GetFileName(path), Shape = input.Shape, Outputs = result!.Select(t => t.Shape), Iterations = samples.Length, Warmup = warmup, CpuThreads = threads > 0 ? threads : Environment.ProcessorCount, StartupMs = startupMs, ModelLoadMs = modelLoadMs, FirstInferenceMs = firstInferenceMs, ReadyToFirstOutputMs = readyToFirstOutputMs, WarmupMs = warmupMs, detector.PipelineCacheLoaded, detector.CachedConvolutionShapes, detector.ReplayedFrames, WindowSeconds = windowSeconds, CpuSeconds = cpuSeconds, CpuPercent = cpuSeconds / windowSeconds / Environment.ProcessorCount * 100, CpuMsPerFrame = cpuSeconds * 1000 / samples.Length, ThroughputFps = samples.Length / windowSeconds, MedianMs = median, P95Ms = Percentile(.95), MinMs = times[0], MeanMs = times.Average(), FramesPerSecond = 1000 / median, SamplesMs = samples, WorkingSetMB = Environment.WorkingSet / 1048576.0 }, new JsonSerializerOptions { WriteIndented = true }));
}
catch (Exception e) { Console.Error.WriteLine(e.ToString()); Environment.ExitCode = 1; }

// 读取 path 指定的 P6 二进制 RGB PPM；返回独立像素字节数组和正数宽高。
// 仅支持最大通道值 255；格式不支持抛出 NotSupportedException，头部或数据长度损坏抛出 InvalidDataException。
// 宽高数字转换可抛出 FormatException/OverflowException，受检像素数量乘法可溢出，文件读取异常原样上抛。
static (byte[] Pixels, int Width, int Height) ReadPpm(string path)
{
    byte[] data = File.ReadAllBytes(path); int pos = 0;
    // 从当前 pos 跳过空白和 # 注释，返回下一个 ASCII 头部字段并推进位置；没有字段时返回空字符串。
    string Token()
    {
        while (pos < data.Length)
        {
            if (data[pos] == '#') { while (pos < data.Length && data[pos] != '\n') pos++; }
            else if (char.IsWhiteSpace((char)data[pos])) pos++; else break;
        }
        int start = pos; while (pos < data.Length && !char.IsWhiteSpace((char)data[pos])) pos++;
        return System.Text.Encoding.ASCII.GetString(data, start, pos - start);
    }
    if (Token() != "P6") throw new NotSupportedException("CLI accepts P6 RGB PPM. Library accepts RGB/BGR bytes from any decoder.");
    int w = int.Parse(Token()), h = int.Parse(Token()); if (w <= 0 || h <= 0 || Token() != "255") throw new InvalidDataException("Invalid PPM header.");
    if (pos >= data.Length || !char.IsWhiteSpace((char)data[pos])) throw new InvalidDataException("PPM separator missing.");
    byte separator = data[pos++]; if (separator == '\r' && pos < data.Length && data[pos] == '\n') pos++;
    int count = checked(w * h * 3); if (data.Length - pos != count) throw new InvalidDataException("PPM byte count mismatch.");
    return (data.AsSpan(pos, count).ToArray(), w, h);
}
