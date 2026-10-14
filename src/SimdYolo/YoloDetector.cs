namespace SimdYolo;

/// <summary>统一 PT 与 ONNX 模型的输入元数据和前向推理接口。</summary>
internal interface IModel
{
    /// <summary>模型要求的输入维度；负维度表示允许由本次输入决定大小。</summary>
    /// <value>模型要求的输入维度；负维度表示允许由本次输入决定大小。</value>
    int[] InputShape { get; }
    /// <summary>模型提供的类别编号与名称映射。</summary>
    /// <value>模型提供的类别编号与名称映射。</value>
    IReadOnlyDictionary<int, string> Names { get; }
    /// <summary>检测头的类别通道数量，未知时为 0。</summary>
    /// <value>由模型结构或元数据推导的类别数量。</value>
    int ClassCount { get; }
    /// <summary>模型是否直接返回经过两阶段筛选的端到端检测结果。</summary>
    /// <value>模型是否直接返回经过两阶段筛选的端到端检测结果。</value>
    bool EndToEnd { get; }
    /// <summary>模型对应的目标检测、实例分割或姿态任务。</summary>
    /// <value>模型对应的目标检测、实例分割或姿态任务。</value>
    YoloTask Task { get; }
    /// <summary>姿态关键点的数量和每点维度，通常为 [17, 3]。</summary>
    /// <value>姿态关键点的数量和每点维度，通常为 [17, 3]。</value>
    int[] KeypointShape { get; }
    /// <summary>模型解码时优先采用的输出维度布局。</summary>
    /// <value>模型解码时优先采用的输出维度布局。</value>
    OutputLayout PreferredLayout { get; }
    /// <summary>执行一轮模型前向推理，不执行图像预处理与检测后处理。</summary>
    /// <param name="input">当前节点或模块的输入。</param>
    /// <returns>按模型输出顺序排列的张量。</returns>
    Tensor[] Run(Tensor input);
}

/// <summary>选择纯托管 CPU 或 Vulkan、Metal 计算后端。</summary>
public enum ComputeDevice
{
    /// <summary>使用纯 C# SIMD 和托管线程执行 CPU 推理。</summary>
    Cpu,
    /// <summary>使用 Vulkan 计算后端。</summary>
    Vulkan,
    /// <summary>使用 macOS Metal 计算后端。</summary>
    Metal,
    /// <summary>优先使用平台 GPU 后端，已知初始化失败时回退 CPU。</summary>
    Auto
}
/// <summary>指定检测头的通道排列或端到端输出格式。</summary>
public enum OutputLayout
{
    /// <summary>依据模型元数据和输出形状推断布局；存在歧义时需要显式设置。</summary>
    Auto,
    /// <summary>使用 [批量,字段,锚点] 通道优先布局。</summary>
    ChannelsFirst,
    /// <summary>使用 [批量,锚点,字段] 通道末尾布局。</summary>
    ChannelsLast,
    /// <summary>使用 [批量,保留数,6+附加字段]，前六字段依次为 xyxy、分数、类别，后接掩码系数或关键点。</summary>
    EndToEnd
}
/// <summary>指定目标检测、实例分割或姿态估计任务。</summary>
public enum YoloTask
{
    /// <summary>根据模型元数据选择任务。</summary>
    Auto,
    /// <summary>仅解码目标检测框和类别。</summary>
    Detection,
    /// <summary>解码目标检测框及实例分割掩码。</summary>
    Segmentation,
    /// <summary>解码目标检测框及姿态关键点。</summary>
    Pose
}
/// <summary>模型会话、图像预处理与结果解码的不可变配置。</summary>
public sealed record DetectorOptions
{
    /// <summary>计算设备；Auto 优先创建平台 GPU 后端，初始化失败时允许回退 CPU。</summary>
    /// <value>计算设备；Auto 优先创建平台 GPU 后端，初始化失败时允许回退 CPU。</value>
    public ComputeDevice Device { get; init; } = ComputeDevice.Cpu;
    /// <summary>动态输入模型的目标图像宽度；静态 ONNX 输入优先使用模型维度。</summary>
    /// <value>动态输入模型的目标图像宽度；静态 ONNX 输入优先使用模型维度。</value>
    public int Width { get; init; } = 640;
    /// <summary>动态输入模型的目标图像高度；静态 ONNX 输入优先使用模型维度。</summary>
    /// <value>动态输入模型的目标图像高度；静态 ONNX 输入优先使用模型维度。</value>
    public int Height { get; init; } = 640;
    /// <summary>保留检测候选的最低置信度，取值范围为 0 到 1。</summary>
    /// <value>保留检测候选的最低置信度，取值范围为 0 到 1。</value>
    public float Confidence { get; init; } = 0.25f;
    /// <summary>非极大值抑制使用的交并比阈值，取值范围为 0 到 1。</summary>
    /// <value>非极大值抑制使用的交并比阈值，取值范围为 0 到 1。</value>
    public float IouThreshold { get; init; } = 0.45f;
    /// <summary>每张图像最多返回的检测实例数量。</summary>
    /// <value>每张图像最多返回的检测实例数量。</value>
    public int MaxDetections { get; init; } = 300;
    /// <summary>按置信度排序后参与非极大值抑制的候选数量上限。</summary>
    /// <value>按置信度排序后参与非极大值抑制的候选数量上限。</value>
    public int MaxCandidates { get; init; } = 30000;
    /// <summary>是否允许不同类别的检测框相互抑制。</summary>
    /// <value>是否允许不同类别的检测框相互抑制。</value>
    public bool ClassAgnosticNms { get; init; }
    /// <summary>检测结果张量的布局；Auto 使用模型元数据及张量维度推断。</summary>
    /// <value>检测结果张量的布局；Auto 使用模型元数据及张量维度推断。</value>
    public OutputLayout OutputLayout { get; init; } = OutputLayout.Auto;
    /// <summary>CPU 最大并行度；0 使用 CPU 后端的默认线程配置。</summary>
    /// <value>CPU 最大并行度；0 使用 CPU 后端的默认线程配置。</value>
    public int CpuThreads { get; init; }
    /// <summary>结果解码的任务类型；Auto 使用模型声明的任务。</summary>
    /// <value>结果解码的任务类型；Auto 使用模型声明的任务。</value>
    public YoloTask Task { get; init; } = YoloTask.Auto;
    /// <summary>类别数量；0 优先读取模型名称映射或由输出维度推断。</summary>
    /// <value>类别数量；0 优先读取模型名称映射或由输出维度推断。</value>
    public int ClassCount { get; init; }
    /// <summary>显式姿态任务解码时的关键点数量，默认 17。</summary>
    /// <value>显式姿态任务解码时的关键点数量，默认 17。</value>
    public int KeypointCount { get; init; } = 17;
    /// <summary>每个关键点的维度；2 表示坐标，3 另含可见性分数。</summary>
    /// <value>每个关键点的维度；2 表示坐标，3 另含可见性分数。</value>
    public int KeypointDimensions { get; init; } = 3;
    /// <summary>可选逐层耗时回调；启用时 GPU 会同步完成该层，耗时不能作为无回调性能基准。</summary>
    /// <value>可选逐层耗时回调；启用时 GPU 会同步完成该层，耗时不能作为无回调性能基准。</value>
    public Action<LayerTiming>? LayerCompleted { get; init; }
}
/// <summary>记录一次模型层执行的名称、算子类型与耗时。</summary>
/// <param name="Name">记录或节点名称。</param>
/// <param name="Operation">本层的模块或算子名称。</param>
/// <param name="Milliseconds">本次执行耗时，单位为毫秒。</param>
public readonly record struct LayerTiming(string Name, string Operation, double Milliseconds);
/// <summary>保存类别、置信度以及左上角坐标和宽高形式的检测框。</summary>
/// <param name="ClassId">从 0 开始的类别编号。</param>
/// <param name="Confidence">模型输出的类别置信度。</param>
/// <param name="X">检测框左上角横坐标。</param>
/// <param name="Y">检测框左上角纵坐标。</param>
/// <param name="Width">检测框宽度。</param>
/// <param name="Height">检测框高度。</param>
public readonly record struct Detection(int ClassId, float Confidence, float X, float Y, float Width, float Height);
/// <summary>保存等比例缩放和填边参数，用于把模型坐标还原到原图。</summary>
/// <param name="SourceWidth">原图像素宽度。</param>
/// <param name="SourceHeight">原图像素高度。</param>
/// <param name="TargetWidth">目标像素宽度。</param>
/// <param name="TargetHeight">目标像素高度。</param>
/// <param name="ScaleX">实际缩放后的横向比例。</param>
/// <param name="ScaleY">实际缩放后的纵向比例。</param>
/// <param name="PadX">左侧填边像素数。</param>
/// <param name="PadY">顶部填边像素数。</param>
public readonly record struct Letterbox(int SourceWidth, int SourceHeight, int TargetWidth, int TargetHeight, float ScaleX, float ScaleY, int PadX, int PadY);

/// <summary>加载 PT 或 ONNX 的推理会话；同一会话串行执行推理并负责释放后端资源。</summary>
public sealed class YoloDetector : IDisposable
{
    /// <summary>由本会话独占并负责释放的计算后端。</summary>
    private readonly IComputeBackend backend;
    /// <summary>包含模型元数据、常量权重和前向执行逻辑的模型实例。</summary>
    private readonly IModel model;
    /// <summary>加载时确认的预处理、推理与解码配置。</summary>
    private readonly DetectorOptions options;
    /// <summary>协调推理和释放操作的会话锁。</summary>
    private readonly object gate = new();
    /// <summary>仅属于本会话的 CPU 临时张量池。</summary>
    private readonly CpuTensorPool tensorPool = new();
    /// <summary>会话是否已经释放。</summary>
    private bool disposed;
    /// <summary>标记当前正在执行推理，用于拒绝回调触发的重入。</summary>
    private bool inferenceRunning;
    /// <summary>实际使用的计算后端名称。</summary>
    /// <value>实际使用的计算后端名称。</value>
    public string BackendName => backend.Name;
    /// <summary>Vulkan 驱动管线缓存是否恢复成功；其他后端返回 null。</summary>
    /// <value>Vulkan 驱动管线缓存是否恢复成功；其他后端返回 null。</value>
    public bool? PipelineCacheLoaded => backend is VulkanBackend vulkan ? vulkan.PipelineCacheLoaded : null;
    /// <summary>从 Vulkan 缓存恢复的卷积形状选择数量；其他后端返回 null。</summary>
    /// <value>从 Vulkan 缓存恢复的卷积形状选择数量；其他后端返回 null。</value>
    public int? CachedConvolutionShapes => backend is VulkanBackend vulkan ? vulkan.CachedConvolutionShapes : null;
    /// <summary>使用已验证 Vulkan 命令序列重放的帧数。</summary>
    /// <value>使用已验证 Vulkan 命令序列重放的帧数。</value>
    public long ReplayedFrames => backend is VulkanBackend vulkan ? vulkan.ReplayedFrames : 0;
    /// <summary>Auto 模式从 GPU 回退到 CPU 的初始化原因；未回退时为 null。</summary>
    /// <value>Auto 模式从 GPU 回退到 CPU 的初始化原因；未回退时为 null。</value>
    public string? FallbackReason { get; }
    /// <summary>模型保存的类别编号与名称映射。</summary>
    /// <value>模型保存的类别编号与名称映射。</value>
    public IReadOnlyDictionary<int, string> ClassNames => model.Names;
    /// <summary>已加载模型的不可变元数据快照；仅读取文件可调用 ModelInfo.Read。</summary>
    /// <value>包含类别名称、通道数量、任务及端到端标记的元数据。</value>
    public ModelInfo Info { get; }
    /// <summary>返回模型输入形状的副本，避免调用方修改模型内部维度。</summary>
    /// <value>返回模型输入形状的副本，避免调用方修改模型内部维度。</value>
    public int[] InputShape => (int[])model.InputShape.Clone();
    /// <summary>最终采用的解码任务；显式配置优先于模型元数据。</summary>
    /// <value>最终采用的解码任务；显式配置优先于模型元数据。</value>
    public YoloTask Task => options.Task == YoloTask.Auto ? model.Task : options.Task;
    /// <summary>校验选项，创建计算后端，并加载 .pt、.pth 或 .onnx 模型；加载失败会释放已创建后端。</summary>
    /// <param name="path">模型文件路径，扩展名决定 PT 或 ONNX 解析方式。</param>
    /// <param name="options">推理和解码配置；null 使用默认配置。</param>
    /// <exception cref="ArgumentException">设备、尺寸、阈值、线程数或任务选项不合法。</exception>
    /// <exception cref="NotSupportedException">文件扩展名不支持，或模型解析器不支持检查点或算子。</exception>
    /// <exception cref="InvalidDataException">模型数据损坏，路由或算子输入输出约定无效。</exception>
    /// <exception cref="IOException">模型文件无法打开或读取。</exception>
    /// <exception cref="PlatformNotSupportedException">明确选择的 GPU 后端不支持当前平台。</exception>
    /// <exception cref="InvalidOperationException">明确指定的 GPU 后端无法创建设备、队列或计算管线，或模型结构包含不唯一的必要记录。</exception>
    /// <exception cref="DllNotFoundException">明确指定的 GPU 后端缺少所需原生驱动库。</exception>
    /// <exception cref="EntryPointNotFoundException">明确指定的 GPU 后端驱动库缺少所需入口。</exception>
    /// <exception cref="FormatException">模型保存的数值文本或外部权重偏移长度不是有效数字。</exception>
    /// <exception cref="OverflowException">模型元数据、权重尺寸或偏移长度超过受支持整数范围。</exception>
    public YoloDetector(string path, DetectorOptions? options = null)
    {
        this.options = options ?? new();
        if (this.options.Width <= 0 || this.options.Height <= 0 || this.options.MaxDetections <= 0 || this.options.MaxCandidates <= 0 || this.options.CpuThreads < 0 || this.options.ClassCount < 0 || this.options.KeypointCount <= 0 || this.options.KeypointDimensions is not (2 or 3) || !Enum.IsDefined(this.options.Device) || !Enum.IsDefined(this.options.Task) || !Enum.IsDefined(this.options.OutputLayout) || !float.IsFinite(this.options.Confidence) || this.options.Confidence is < 0 or > 1 || !float.IsFinite(this.options.IouThreshold) || this.options.IouThreshold is < 0 or > 1) throw new ArgumentException("Invalid detector options.");
        backend = ComputeBackends.Create(this.options.Device, this.options.CpuThreads, out var fallback); FallbackReason = fallback;
        if (backend is CpuBackend cpuBackend) tensorPool.Parallel = cpuBackend.Parallel;
        try
        {
            var ops = new TensorOps(backend);
            model = Path.GetExtension(path).ToLowerInvariant() switch { ".onnx" => new OnnxModel(path, ops, this.options.LayerCompleted), ".pt" or ".pth" => new PtModel(path, ops, this.options.LayerCompleted), _ => throw new NotSupportedException("Use .pt/.pth YOLO checkpoint or .onnx.") };
            Info = new(Path.GetExtension(path).Equals(".onnx", StringComparison.OrdinalIgnoreCase) ? "ONNX" : "PT", model.Names, model.ClassCount, model.Task, model.EndToEnd, model.InputShape, model.KeypointShape);
        }
        catch { backend.Dispose(); throw; }
    }
    /// <summary>对 NCHW 张量执行前向推理，返回可由调用方持有的原始输出。</summary>
    /// <param name="nchw">已预处理的连续 FP32 NCHW 图像输入。</param>
    /// <returns>按模型输出顺序排列的张量；后续推理不会把这些输出作为临时缓冲区回收。</returns>
    /// <exception cref="ObjectDisposedException">会话已经释放。</exception>
    /// <exception cref="ArgumentException">输入形状与数据长度不一致，或不满足模型输入约定。</exception>
    /// <exception cref="InvalidOperationException">同一会话发生重入，或计算后端提交、同步或设备存储校验失败。</exception>
    /// <exception cref="InvalidDataException">模型路由数据缺失，或 ONNX 节点执行及输出数量校验失败。</exception>
    /// <exception cref="NotSupportedException">PT 模块的运行参数不受支持，或设备算子与分派尺寸超过支持范围。</exception>
    /// <exception cref="OverflowException">输入维度乘积、运行时形状或索引转换超过受支持整数范围。</exception>
    public Tensor[] PredictRaw(Tensor nchw)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (Tensor.Count(nchw.Shape) != nchw.Length) throw new ArgumentException("Input tensor shape does not match its storage.", nameof(nchw));
            if (inferenceRunning) throw new InvalidOperationException("Reentrant inference on the same session is not supported.");
            inferenceRunning = true;
            try
            {
                if (backend is not IDeviceBackend device)
                {
                    var previous = CpuTensorPool.Current; var previousDevice = DeviceScope.Current;
                    CpuTensorPool.Current = tensorPool; DeviceScope.Current = null;
                    // 把模型输出从临时池中分离后交给调用方；帧结束只能回收中间值。
                    try { return tensorPool.DetachOutputs(model.Run(nchw)); }
                    finally { CpuTensorPool.Current = previous; DeviceScope.Current = previousDevice; tensorPool.ReturnAll(); }
                }
                var outerPool = CpuTensorPool.Current; var outerDevice = DeviceScope.Current;
                CpuTensorPool.Current = null;
                try
                {
                    // 只有无逐层回调且输入约定匹配时才能重放已验证的 Vulkan 命令。
                    if (backend is VulkanBackend replay && options.LayerCompleted is null && replay.TryReplay(nchw, out var cached)) return cached;
                    device.BeginFrame();
                    try
                    {
                        if (backend is VulkanBackend capture && options.LayerCompleted is null) capture.BeginReplayCapture(nchw);
                        var outputs = model.Run(nchw); device.MaterializeOutputs(outputs); return outputs;
                    }
                    finally { device.EndFrame(); }
                }
                finally { CpuTensorPool.Current = outerPool; DeviceScope.Current = outerDevice; }
            }
            finally { inferenceRunning = false; }
        }
    }
    /// <summary>对 RGB24 或 BGR24 图像进行填边、推理和检测框解码。</summary>
    /// <param name="pixels">按行排列的 RGB24 或 BGR24 三通道字节数据。</param>
    /// <param name="width">原图宽度，以像素计。</param>
    /// <param name="height">原图高度，以像素计。</param>
    /// <param name="bgr">true 表示输入为 BGR 通道顺序，false 表示 RGB。</param>
    /// <param name="rowStride">源图像每行占用的字节数；0 按 width × 3 处理。</param>
    /// <returns>已还原到原图坐标并完成筛选的检测框列表。</returns>
    /// <exception cref="ArgumentException">图像尺寸、跨度或输出解码配置不合法。</exception>
    /// <exception cref="ObjectDisposedException">会话已经释放。</exception>
    /// <exception cref="InvalidOperationException">同一会话发生重入，或计算后端提交和同步失败。</exception>
    /// <exception cref="NotSupportedException">模型输出不是受支持的单图检测头布局，或任务、算子及设备尺寸不受支持。</exception>
    /// <exception cref="InvalidDataException">模型路由或节点执行失败。</exception>
    /// <exception cref="OverflowException">图像缓冲区尺寸、运行时形状或类别编号超过受支持整数范围。</exception>
    public IReadOnlyList<Detection> Detect(ReadOnlySpan<byte> pixels, int width, int height, bool bgr = false, int rowStride = 0)
    {
        int tw = model.InputShape.ElementAtOrDefault(3) > 0 ? model.InputShape[3] : options.Width;
        int th = model.InputShape.ElementAtOrDefault(2) > 0 ? model.InputShape[2] : options.Height;
        var (input, transform) = ImagePreprocessor.Letterbox(pixels, width, height, tw, th, bgr, rowStride);
        var outputs = PredictRaw(input);
        return VisionDecoder.Decode(outputs, transform, EffectiveOptions(outputs), decodeMasks: false).Select(i => i.Box).ToArray();
    }
    /// <summary>对图像执行完整推理，并解码检测框、分割掩码或姿态关键点。</summary>
    /// <param name="pixels">按行排列的 RGB24 或 BGR24 三通道字节数据。</param>
    /// <param name="width">原图宽度，以像素计。</param>
    /// <param name="height">原图高度，以像素计。</param>
    /// <param name="bgr">true 表示输入为 BGR 通道顺序，false 表示 RGB。</param>
    /// <param name="rowStride">源图像每行占用的字节数；0 按 width × 3 处理。</param>
    /// <returns>已还原到原图坐标的实例结果列表。</returns>
    /// <exception cref="ArgumentException">图像尺寸、跨度或任务输出布局不合法。</exception>
    /// <exception cref="ObjectDisposedException">会话已经释放。</exception>
    /// <exception cref="InvalidOperationException">同一会话发生重入，或计算后端提交和同步失败。</exception>
    /// <exception cref="NotSupportedException">模型输出不是受支持的单图布局，端到端分割或姿态解码不受支持，或算子及设备尺寸不受支持。</exception>
    /// <exception cref="InvalidDataException">模型路由或节点执行失败。</exception>
    /// <exception cref="OverflowException">图像缓冲区尺寸、关键点数量、运行时形状或类别编号超过受支持整数范围。</exception>
    public IReadOnlyList<YoloInstance> Predict(ReadOnlySpan<byte> pixels, int width, int height, bool bgr = false, int rowStride = 0)
    {
        int tw = model.InputShape.ElementAtOrDefault(3) > 0 ? model.InputShape[3] : options.Width;
        int th = model.InputShape.ElementAtOrDefault(2) > 0 ? model.InputShape[2] : options.Height;
        var (input, transform) = ImagePreprocessor.Letterbox(pixels, width, height, tw, th, bgr, rowStride);
        var outputs = PredictRaw(input); return VisionDecoder.Decode(outputs, transform, EffectiveOptions(outputs));
    }
    /// <summary>用模型元数据补齐自动任务、类别数、关键点维度和输出布局。</summary>
    /// <param name="outputs">本次推理输出数组。</param>
    /// <returns>供本次输出解码使用的配置副本。</returns>
    private DetectorOptions EffectiveOptions(Tensor[] outputs)
    {
        int[] kp = model.KeypointShape;
        return options with
        {
            Task = Task,
            ClassCount = options.ClassCount > 0 ? options.ClassCount : model.ClassCount,
            KeypointCount = options.Task == YoloTask.Auto ? kp[0] : options.KeypointCount,
            KeypointDimensions = options.Task == YoloTask.Auto ? kp[1] : options.KeypointDimensions,
            OutputLayout = options.OutputLayout == OutputLayout.Auto ? model.EndToEnd ? OutputLayout.EndToEnd : model.PreferredLayout : options.OutputLayout
        };
    }
    /// <summary>等待正在执行的推理离开会话锁，释放计算后端并清空 CPU 缓冲区池；重复调用无副作用。</summary>
    public void Dispose() { lock (gate) { if (disposed) return; disposed = true; backend.Dispose(); tensorPool.Clear(); } }
}

/// <summary>按设备选项创建计算后端，并处理 Auto 模式的 GPU 初始化回退。</summary>
public static class ComputeBackends
{
    /// <summary>创建指定设备后端；Auto 在支持的平台优先使用 GPU，已知初始化失败时回退 CPU。</summary>
    /// <param name="device">需要创建的 CPU、Vulkan、Metal 或 Auto 设备。</param>
    /// <param name="threads">CPU 最大并行度；0 使用后端默认值。</param>
    /// <param name="fallback">返回 Auto 模式的回退原因；未发生回退时为 null。</param>
    /// <returns>由调用方负责释放的计算后端。</returns>
    /// <exception cref="PlatformNotSupportedException">明确指定的 GPU 后端不支持当前平台或进程位宽。</exception>
    /// <exception cref="InvalidOperationException">明确指定的 GPU 后端设备、队列或计算管线初始化失败。</exception>
    /// <exception cref="DllNotFoundException">明确指定的 GPU 后端所需驱动库不存在。</exception>
    /// <exception cref="EntryPointNotFoundException">明确指定的 GPU 后端所需驱动入口不存在。</exception>
    public static IComputeBackend Create(ComputeDevice device, int threads, out string? fallback)
    {
        fallback = null;
        if (device == ComputeDevice.Cpu) return new CpuBackend(threads);
        if (device == ComputeDevice.Vulkan) return new VulkanBackend();
        if (device == ComputeDevice.Metal) return new MetalBackend();
        try { return OperatingSystem.IsMacOS() ? new MetalBackend() : new VulkanBackend(); }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or PlatformNotSupportedException or InvalidOperationException || e is TypeInitializationException { InnerException: DllNotFoundException or EntryPointNotFoundException })
        { fallback = e.Message; return new CpuBackend(threads); }
    }
}

/// <summary>把已有解码器提供的三通道图像转换为归一化 NCHW 输入。</summary>
public static class ImagePreprocessor
{
    /// <summary>按目标尺寸等比例缩放图像，以 114 灰度填边，并用双线性插值写入 RGB FP32 平面。</summary>
    /// <param name="pixels">按行排列的 RGB24 或 BGR24 三通道字节数据。</param>
    /// <param name="width">原图宽度，以像素计。</param>
    /// <param name="height">原图高度，以像素计。</param>
    /// <param name="targetWidth">模型目标输入宽度，以像素计。</param>
    /// <param name="targetHeight">模型目标输入高度，以像素计。</param>
    /// <param name="bgr">true 表示输入为 BGR 通道顺序，false 表示 RGB。</param>
    /// <param name="rowStride">源图像每行占用的字节数；0 按 width × 3 处理。</param>
    /// <returns>归一化输入张量以及对应的缩放和填边变换。</returns>
    /// <exception cref="ArgumentException">图像或目标尺寸不是正数、行跨度不足，或图像缓冲区长度不够。</exception>
    /// <exception cref="OverflowException">图像尺寸或字节跨度计算超过 Int32 范围。</exception>
    public static unsafe (Tensor Input, Letterbox Transform) Letterbox(ReadOnlySpan<byte> pixels, int width, int height, int targetWidth, int targetHeight, bool bgr = false, int rowStride = 0)
    {
        if (width <= 0 || height <= 0 || targetWidth <= 0 || targetHeight <= 0) throw new ArgumentException("Image dimensions must be positive.");
        rowStride = rowStride == 0 ? checked(width * 3) : rowStride;
        if (rowStride < checked(width * 3) || pixels.Length < checked((height - 1) * rowStride + width * 3)) throw new ArgumentException("Image buffer too short.");
        float gain = MathF.Min((float)targetWidth / width, (float)targetHeight / height);
        int rw = Math.Clamp((int)MathF.Round(width * gain), 1, targetWidth), rh = Math.Clamp((int)MathF.Round(height * gain), 1, targetHeight);
        int px = (targetWidth - rw) / 2, py = (targetHeight - rh) / 2, plane = checked(targetWidth * targetHeight); var input = new Tensor(1, 3, targetHeight, targetWidth);
        Array.Fill(input.Data, 114f / 255f);
        var x0Map = new int[rw]; var x1Map = new int[rw]; var fractions = new float[rw];
        for (int x = 0; x < rw; x++) { float sx = Math.Clamp((x + .5f) * width / rw - .5f, 0, width - 1); x0Map[x] = (int)sx; x1Map[x] = Math.Min(x0Map[x] + 1, width - 1); fractions[x] = sx - x0Map[x]; }
        fixed (byte* source = pixels) fixed (float* destination = input.Data)
        {
            nint sourceAddress = (nint)source, targetAddress = (nint)destination;
            // 处理缩放后的第 y 行；双线性取样写入三个 RGB 平面，不产生返回值。
            void Row(int y)
            {
                byte* src = (byte*)sourceAddress; float* dst = (float*)targetAddress;
                float sy = Math.Clamp((y + .5f) * height / rh - .5f, 0, height - 1); int y0 = (int)sy, y1 = Math.Min(y0 + 1, height - 1); float fy = sy - y0;
                for (int x = 0; x < rw; x++)
                {
                    int x0 = x0Map[x], x1 = x1Map[x]; float fx = fractions[x];
                    for (int c = 0; c < 3; c++)
                    {
                        int channel = bgr ? 2 - c : c; float top = src[y0 * rowStride + x0 * 3 + channel] * (1 - fx) + src[y0 * rowStride + x1 * 3 + channel] * fx;
                        float bottom = src[y1 * rowStride + x0 * 3 + channel] * (1 - fx) + src[y1 * rowStride + x1 * 3 + channel] * fx;
                        dst[c * plane + (y + py) * targetWidth + x + px] = (top * (1 - fy) + bottom * fy) / 255f;
                    }
                }
            }
            if ((long)rw * rh >= 32768) Parallel.For(0, rh, Row); else for (int y = 0; y < rh; y++) Row(y);
        }
        return (input, new(width, height, targetWidth, targetHeight, (float)rw / width, (float)rh / height, px, py));
    }
}

/// <summary>解码单张图像的 YOLO 检测输出，并执行置信度筛选与非极大值抑制。</summary>
public static class DetectionDecoder
{
    /// <summary>解码检测输出并把保留的检测框映射到原图坐标。</summary>
    /// <param name="output">单图检测头的三维输出张量。</param>
    /// <param name="transform">预处理记录的缩放、填边和原图尺寸。</param>
    /// <param name="options">推理和解码配置；null 使用默认配置。</param>
    /// <returns>按置信度排序并裁剪到原图边界的检测框；原始检测头执行 NMS，端到端输出保留筛选结果而不再执行 NMS。</returns>
    /// <exception cref="ArgumentException">变换或候选配置不合法、输出字段数或类别数不匹配，或自动布局存在歧义。</exception>
    /// <exception cref="NotSupportedException">输出不是批量为 1 的三维张量。</exception>
    /// <exception cref="OverflowException">端到端输出中的类别编号超过 Int32 范围。</exception>
    public static IReadOnlyList<Detection> Decode(Tensor output, Letterbox transform, DetectorOptions? options = null)
        => DecodeIndexed(output, transform, options).Select(d => d.Box).ToArray();
    /// <summary>解码检测框并保留原始锚点编号，供分割和姿态结果关联使用。</summary>
    /// <param name="output">单图检测头的三维输出张量。</param>
    /// <param name="transform">预处理记录的缩放、填边和原图尺寸。</param>
    /// <param name="options">推理和解码配置；null 使用默认配置。</param>
    /// <returns>检测框与对应输出锚点编号的列表。</returns>
    /// <exception cref="ArgumentException">变换、候选配置、类别数量或输出布局不合法，或自动布局存在歧义。</exception>
    /// <exception cref="NotSupportedException">输出不是批量为 1 的三维张量。</exception>
    /// <exception cref="OverflowException">端到端类别编号超出 Int32 范围。</exception>
    internal static IReadOnlyList<(Detection Box, int Anchor)> DecodeIndexed(Tensor output, Letterbox transform, DetectorOptions? options = null)
    {
        options ??= new();
        if (transform.SourceWidth <= 0 || transform.SourceHeight <= 0 || transform.TargetWidth <= 0 || transform.TargetHeight <= 0 || !float.IsFinite(transform.ScaleX) || !float.IsFinite(transform.ScaleY) || transform.ScaleX <= 0 || transform.ScaleY <= 0 || options.MaxDetections <= 0 || options.MaxCandidates <= 0 || options.ClassCount < 0) throw new ArgumentException("Invalid decode transform/options.");
        if (output.Rank != 3 || output.Shape[0] != 1) throw new NotSupportedException("Decode expects a single-image rank-3 tensor.");
        var layout = options.OutputLayout;
        if (layout == OutputLayout.Auto)
        {
            int expected = options.ClassCount > 0 ? 4 + options.ClassCount : 0;
            // [1,N,6] 等布局可能与原始检测头维度产生歧义，显式布局选项可消除歧义。
            if (expected > 0 && output.Shape[1] == expected && output.Shape[2] != expected) layout = OutputLayout.ChannelsFirst;
            else if (expected > 0 && output.Shape[2] == expected && output.Shape[1] != expected) layout = OutputLayout.ChannelsLast;
            else if (output.Shape[2] == 6 && output.Shape[1] != 6) layout = OutputLayout.EndToEnd;
            else if (output.Shape[1] >= 5 && output.Shape[1] < output.Shape[2]) layout = OutputLayout.ChannelsFirst;
            else if (output.Shape[2] >= 5 && output.Shape[2] < output.Shape[1]) layout = OutputLayout.ChannelsLast;
            else throw new ArgumentException("Ambiguous YOLO output layout. Set OutputLayout explicitly.");
        }
        bool end2end = layout == OutputLayout.EndToEnd, channelsFirst = layout == OutputLayout.ChannelsFirst;
        int count = channelsFirst ? output.Shape[2] : output.Shape[1], fields = channelsFirst ? output.Shape[1] : output.Shape[2];
        if (fields < 5 || end2end && fields != 6) throw new ArgumentException("Invalid detection output.");
        // a 为原输出锚点编号，c 为字段编号；按当前布局返回对应字段值。
        float At(int a, int c) => output.Data[channelsFirst ? c * count + a : a * fields + c];
        var candidates = new List<(Detection Box, int Anchor)>();
        int classFields = options.ClassCount > 0 ? 4 + options.ClassCount : fields;
        if (!end2end && (classFields < 5 || classFields > fields)) throw new ArgumentException("ClassCount does not match output.");
        for (int a = 0; a < count; a++)
        {
            int cls = 0; float confidence;
            if (end2end)
            { confidence = At(a, 4); float cv = At(a, 5); if (!float.IsFinite(cv) || cv < 0 || cv != MathF.Truncate(cv)) continue; cls = checked((int)cv); }
            else
            { confidence = float.NegativeInfinity; for (int c = 4; c < classFields; c++) if (At(a, c) > confidence) { confidence = At(a, c); cls = c - 4; } }
            if (!float.IsFinite(confidence) || confidence < options.Confidence) continue;
            float x = At(a, 0), y = At(a, 1), w = At(a, 2), h = At(a, 3);
            if (!float.IsFinite(x) || !float.IsFinite(y) || !float.IsFinite(w) || !float.IsFinite(h)) continue;
            if (end2end) { w -= x; h -= y; } else { x -= w * .5f; y -= h * .5f; }
            if (w <= 0 || h <= 0) continue;
            // 先在模型坐标中执行 NMS，再还原并裁剪到原图边界，保持标准 YOLO 后处理顺序。
            candidates.Add((new(cls, confidence, x, y, w, h), a));
        }
        candidates = candidates.OrderByDescending(d => d.Box.Confidence).Take(options.MaxCandidates).ToList();
        var kept = new List<(Detection Box, int Anchor)>();
        foreach (var d in candidates)
        {
            if (!end2end && kept.Any(k => (options.ClassAgnosticNms || k.Box.ClassId == d.Box.ClassId) && Iou(k.Box, d.Box) > options.IouThreshold)) continue;
            kept.Add(d); if (kept.Count >= options.MaxDetections) break;
        }
        return kept.Select(item =>
        {
            var d = item.Box;
            float x1 = Math.Clamp((d.X - transform.PadX) / transform.ScaleX, 0, transform.SourceWidth), y1 = Math.Clamp((d.Y - transform.PadY) / transform.ScaleY, 0, transform.SourceHeight);
            float x2 = Math.Clamp((d.X + d.Width - transform.PadX) / transform.ScaleX, 0, transform.SourceWidth), y2 = Math.Clamp((d.Y + d.Height - transform.PadY) / transform.ScaleY, 0, transform.SourceHeight);
            return (Box: d with { X = x1, Y = y1, Width = x2 - x1, Height = y2 - y1 }, item.Anchor);
        }).Where(d => d.Box.Width > 0 && d.Box.Height > 0).ToArray();
    }
    /// <summary>计算两个左上角坐标和宽高形式检测框的交并比。</summary>
    /// <param name="a">第一个原图或模型坐标检测框。</param>
    /// <param name="b">处于相同坐标系的第二个检测框。</param>
    /// <returns>交集面积与并集面积的比值；分母至少为 float.Epsilon。</returns>
    public static float Iou(Detection a, Detection b)
    {
        float w = MathF.Max(0, MathF.Min(a.X + a.Width, b.X + b.Width) - MathF.Max(a.X, b.X)), h = MathF.Max(0, MathF.Min(a.Y + a.Height, b.Y + b.Height) - MathF.Max(a.Y, b.Y));
        float intersection = w * h; return intersection / MathF.Max(float.Epsilon, a.Width * a.Height + b.Width * b.Height - intersection);
    }
}
