using System.IO.Compression;
using System.Text;

namespace SimdYolo;

/// <summary>用纯 C# 读取完整 YOLO torch.save 检查点并解释执行支持的模块，不导入或运行 Python。</summary>
internal sealed partial class PtModel : IModel
{
    /// <summary>按模型顺序保存的顶层 YOLO 模块。</summary>
    private readonly PyObject[] layers;
    /// <summary>把卷积和矩阵乘法分派到当前计算后端的算子入口。</summary>
    private readonly TensorOps ops;
    /// <summary>可选模块耗时回调。</summary>
    private readonly Action<LayerTiming>? trace;
    /// <summary>加载时规范化为绝对层号的路由；-1 表示图像输入。</summary>
    private readonly int[][] from;
    /// <summary>每个中间层输出的后续使用次数，用于及时回收临时张量。</summary>
    private readonly int[] uses;
    /// <summary>同一会话串行推理时复用的路由结果槽；每帧结束清除张量引用。</summary>
    private readonly object?[] routedValues;
    /// <summary>逐帧恢复的引用计数工作区。</summary>
    private readonly int[] remainingUses;
    /// <summary>多输入层的复用参数数组，避免每帧创建路由数组和 LINQ 闭包。</summary>
    private readonly object[][] routedArguments;
    /// <summary>CPU 张量池扫描的复用根集合，末尾存放当前层结果。</summary>
    private readonly object?[] retainedValues;
    /// <summary>加载时折叠 BN 或融合 Rep 分支后的常量卷积权重与偏置。</summary>
    private readonly Dictionary<PyObject, (Tensor Weight, Tensor Bias)> fused = [];
    /// <summary>PT 模型接受的 NCHW RGB 输入模板，空间尺寸由运行时决定。</summary>
    /// <value>PT 模型接受的 NCHW RGB 输入模板，空间尺寸由运行时决定。</value>
    public int[] InputShape => trainedImageSize > 0 ? [1, 3, trainedImageSize, trainedImageSize] : [1, 3, -1, -1];
    /// <summary>原生训练归档声明的固定推理尺寸；普通 PT 为零并保持动态尺寸。</summary>
    private readonly int trainedImageSize;
    /// <summary>检查点保存的类别编号与名称。</summary>
    /// <value>检查点保存的类别编号与名称。</value>
    public IReadOnlyDictionary<int, string> Names { get; }
    /// <summary>最终检测头声明的类别通道数量。</summary>
    /// <value>检测头 nc，缺失时使用标签最大编号加一。</value>
    public int ClassCount { get; }
    /// <summary>由 v10Detect、头状态、融合分支或 YAML 识别的端到端模式。</summary>
    /// <value>最终检测、分割或姿态头是否使用 one2one 分支。</value>
    public bool EndToEnd { get; }
    /// <summary>根据最后一层识别出的检测、分割或姿态任务。</summary>
    /// <value>根据最后一层识别出的检测、分割或姿态任务。</value>
    public YoloTask Task { get; }
    /// <summary>检查点的姿态关键点形状；未声明时使用 [17, 3]。</summary>
    /// <value>检查点的姿态关键点形状；未声明时使用 [17, 3]。</value>
    public int[] KeypointShape { get; }
    /// <summary>端到端模型采用六字段及可选实例数据，其他模型采用通道优先布局。</summary>
    /// <value>模型的实例行布局或通道优先布局。</value>
    public OutputLayout PreferredLayout => EndToEnd ? OutputLayout.EndToEnd : OutputLayout.ChannelsFirst;
    /// <summary>允许解释执行的 Torch 和 YOLO 模块名称白名单。</summary>
    private static readonly HashSet<string> Supported = new("Sequential ModuleList Conv2d ConvTranspose2d BatchNorm2d SiLU ReLU LeakyReLU Identity Upsample MaxPool2d Conv DWConv Bottleneck RepBottleneck RepNBottleneck CIB C2f C2fCIB C2fPSA C3k2 C3 C3k RepCSP RepNCSP SPPF SPP SPPELAN Concat RepNCSPELAN4 ELAN1 ADown AConv SCDown RepConv RepConvN RepVGGDW PSA C2PSA PSABlock Attention Detect Segment Segment26 Pose Pose26 Proto Proto26 DDetect DualDDetect TripleDDetect v10Detect DFL CBLinear CBFuse Silence".Split(' '));
    /// <summary>读取 Torch ZIP 检查点的符号对象和权重，校验层路由，并在加载阶段完成卷积参数预融合。</summary>
    /// <param name="path">包含完整 YOLO 模块架构与权重的 Torch ZIP 检查点路径。</param>
    /// <param name="ops">使用当前计算后端的张量算子入口。</param>
    /// <param name="trace">可选逐层或逐节点耗时回调。</param>
    /// <exception cref="NotSupportedException">检查点不是完整的受支持 YOLO 模型、使用 TorchScript、旧 Pickle 或大端存储。</exception>
    /// <exception cref="InvalidDataException">元数据超限、模型为空、路由不合法或存储内容损坏。</exception>
    public PtModel(string path, TensorOps ops, Action<LayerTiming>? trace = null)
        : this(File.OpenRead(path), ops, trace, null) { }

    /// <summary>从拥有所有权的 PT 流读取模型，可在融合前替换检测头训练参数。</summary>
    /// <param name="source">本构造函数负责关闭的 PT ZIP 流。</param>
    /// <param name="ops">前向算子集合。</param>
    /// <param name="trace">可选层计时回调。</param>
    /// <param name="trainingState">可选的检测头训练快照。</param>
    internal PtModel(Stream source, TensorOps ops, Action<LayerTiming>? trace, TrainingModelState? trainingState)
    {
        this.ops = ops;
        this.trace = trace;
        trainedImageSize = trainingState?.Manifest.ImageSize ?? 0;
        using var stream = source; using var zip = new ZipArchive(stream, ZipArchiveMode.Read);
        var data = zip.Entries.SingleOrDefault(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal)) ?? throw new NotSupportedException("PT requires torch.save ZIP checkpoint with data.pkl. Legacy pickle and TorchScript are unsupported.");
        string prefix = data.FullName[..^"data.pkl".Length];
        if (zip.Entries.Any(e => e.FullName.StartsWith(prefix + "code/", StringComparison.Ordinal))) throw new NotSupportedException("TorchScript .pt is not a YOLO nn.Module checkpoint.");
        if (zip.GetEntry(prefix + "byteorder") is { } order)
        { using var reader = new StreamReader(order.Open()); if (reader.ReadToEnd().Trim() != "little") throw new NotSupportedException("Big-endian PT storage unsupported."); }
        if (data.Length > 64 * 1024 * 1024) throw new InvalidDataException("PT metadata exceeds 64 MiB.");
        byte[] bytes = new byte[(int)data.Length]; using (var s = data.Open()) s.ReadExactly(bytes);
        object? root = new PickleReader(zip, prefix).Load(bytes);
        if (root is Dictionary<object, object?> checkpoint) root = checkpoint.GetValueOrDefault("ema") ?? checkpoint.GetValueOrDefault("model");
        if (root is not PyObject model || !model.Has("model")) throw new NotSupportedException("PT must contain a YOLO detection nn.Module. A state_dict alone has no executable architecture; export ONNX or save the full model.");
        layers = model.Child("model").Children;
        if (layers.Length == 0) throw new InvalidDataException("PT model has no layers.");
        trainingState?.Apply(layers[^1]);
        foreach (var layer in layers) Validate(layer);
        from = layers.Select(l => l.Get("f") switch { List<object?> list => list.Select(Convert.ToInt32).ToArray(), object?[] tuple => tuple.Select(Convert.ToInt32).ToArray(), null => new[] { -1 }, var f => new[] { Convert.ToInt32(f) } }).ToArray();
        uses = new int[layers.Length];
        for (int i = 0; i < layers.Length; i++) for (int j = 0; j < from[i].Length; j++)
        {
            int f = from[i][j], idx = f < 0 ? i + f : f;
            if (idx < -1 || idx >= i) throw new InvalidDataException("Invalid YOLO layer routing.");
            from[i][j] = idx; if (idx >= 0) uses[idx]++;
        }
        routedValues = new object?[layers.Length]; remainingUses = new int[layers.Length]; retainedValues = new object?[layers.Length + 1];
        routedArguments = from.Select(route => route.Length > 1 ? new object[route.Length] : Array.Empty<object>()).ToArray();
        Names = trainingState is null ? ModelInfo.ReadNames(model.Get("names")) : trainingState.Manifest.ClassNames.Select((name, index) => (name, index)).ToDictionary(p => p.index, p => p.name);
        ClassCount = layers[^1].Int("nc", ModelInfo.LabelClassCount(Names));
        EndToEnd = ModelInfo.IsEndToEnd(layers[^1], model);
        layers[^1].State["_end2end"] = EndToEnd;
        Task = layers[^1].Type.Name switch { "Segment" or "Segment26" => YoloTask.Segmentation, "Pose" or "Pose26" => YoloTask.Pose, _ => YoloTask.Detection };
        KeypointShape = layers[^1].Get("kpt_shape") is IEnumerable<object?> kpts ? kpts.Select(Convert.ToInt32).ToArray() : [17, 3];
        // 在加载阶段一次性折叠 BN 和重参数化分支，推理过程中只读取融合权重。
        foreach (var layer in layers) Prepare(layer);
    }
    /// <summary>递归校验模块来源和类型是否属于当前解释器支持范围。</summary>
    /// <param name="p">当前符号模块的参数与状态。</param>
    /// <exception cref="NotSupportedException">模块类型或模块来源不在支持范围内。</exception>
    private static void Validate(PyObject p)
    {
        bool trustedType = p.Type.Module.StartsWith("torch.nn.modules.", StringComparison.Ordinal) || p.Type.Module.StartsWith("ultralytics.nn.modules", StringComparison.Ordinal) || p.Type.Module is "models.common" or "models.yolo";
        if (!trustedType || !Supported.Contains(p.Type.Name)) throw new NotSupportedException($"PT module {p.Type.Module}.{p.Type.Name} unsupported.");
        foreach (var child in InferenceChildren(p)) Validate(child);
    }
    /// <summary>枚举推理会使用的子模块，忽略 YOLO26 姿态损失和分割语义辅助分支。</summary>
    /// <param name="p">要枚举子模块的符号模块。</param>
    /// <returns>需要验证及预融合的子模块，不包含已知仅训练分支。</returns>
    private static IEnumerable<PyObject> InferenceChildren(PyObject p) => p.Map("_modules")
        .Where(pair => !IsTrainingOnly(p, pair.Key))
        .Select(pair => pair.Value).OfType<PyObject>();
    /// <summary>递归预处理模块，在加载阶段折叠 BatchNorm 并融合满足条件的 Rep 卷积分支。</summary>
    /// <param name="p">当前符号模块的参数与状态。</param>
    private void Prepare(PyObject p)
    {
        foreach (var pair in p.Map("_modules"))
            if (pair.Value is PyObject child && !IsTrainingOnly(p, pair.Key) && !IsInactiveHeadBranch(p, pair.Key)) Prepare(child);
        if (p.Type.Name is "Conv" or "DWConv" && p.Has("bn"))
        {
            var conv = p.Child("conv"); var bn = p.Child("bn"); var weights = conv.Weight("weight"); var w = new Tensor((float[])weights.Data.Clone(), weights.Shape); var bias = new Tensor(weights.Shape[0]);
            var gamma = bn.Weight("weight"); var beta = bn.Weight("bias"); var mean = bn.Weight("running_mean"); var variance = bn.Weight("running_var"); var oldBias = conv.Map("_parameters").GetValueOrDefault("bias") as Tensor;
            int size = w.Length / w.Shape[0]; float epsilon = bn.Float("eps", 1e-5f);
            float[] data = w.Data, gammas = gamma.Data, betas = beta.Data, means = mean.Data, variances = variance.Data;
            for (int c = 0; c < w.Shape[0]; c++)
            {
                float scale = gammas[c] / MathF.Sqrt(variances[c] + epsilon); int start = c * size, j = 0;
                var factor = new System.Numerics.Vector<float>(scale);
                for (; j <= size - System.Numerics.Vector<float>.Count; j += System.Numerics.Vector<float>.Count)
                    (new System.Numerics.Vector<float>(data, start + j) * factor).CopyTo(data, start + j);
                for (; j < size; j++) data[start + j] *= scale;
                bias.Data[c] = betas[c] + ((oldBias?.Data[c] ?? 0) - means[c]) * scale;
            }
            w.IsConstant = bias.IsConstant = true; fused[p] = (w, bias);
        }
        if (p.Type.Name is "RepConv" or "RepConvN" && !p.Has("conv") && p.Has("conv1") && p.Has("conv2") && fused.TryGetValue(p.Child("conv1"), out var large) && fused.TryGetValue(p.Child("conv2"), out var small))
        {
            var merged = KernelFusion.Merge(large.Weight, large.Bias, small.Weight, small.Bias);
            if (p.Has("bn"))
            {
                var bn = p.Child("bn"); int channels = merged.Weight.Shape[0], icg = merged.Weight.Shape[1], kh = merged.Weight.Shape[2], kw = merged.Weight.Shape[3];
                for (int c = 0; c < channels; c++)
                {
                    float scale = bn.Weight("weight").Data[c] / MathF.Sqrt(bn.Weight("running_var").Data[c] + bn.Float("eps", 1e-5f));
                    merged.Weight.Data[((c * icg + c % icg) * kh + kh / 2) * kw + kw / 2] += scale;
                    merged.Bias.Data[c] += bn.Weight("bias").Data[c] - bn.Weight("running_mean").Data[c] * scale;
                }
            }
            fused[p] = merged;
        }
        if (p.Type.Name == "RepVGGDW" && p.Has("conv1") && fused.TryGetValue(p.Child("conv"), out large) && fused.TryGetValue(p.Child("conv1"), out small)) fused[p] = KernelFusion.Merge(large.Weight, large.Bias, small.Weight, small.Bias);
    }
    /// <summary>判断已知仅训练分支，不改变推理模块的来源和类型验证。</summary>
    /// <param name="module">父模块。</param>
    /// <param name="key">子模块键。</param>
    /// <returns>是否为姿态损失或语义辅助分支。</returns>
    private static bool IsTrainingOnly(PyObject module, object key) =>
        module.Type.Name == "Pose26" && key is "flow_model" or "cv4_sigma" or "one2one_cv4_sigma" || module.Type.Name == "Proto26" && key is "semseg";

    /// <summary>判断当前端到端模式不使用的检测头分支，省去其 BN 克隆与预融合。</summary>
    /// <param name="module">包含 one2one 分支的检测、分割或姿态头。</param>
    /// <param name="key">子模块键。</param>
    /// <returns>当前推理不会执行该分支时为 true。</returns>
    private static bool IsInactiveHeadBranch(PyObject module, object key)
    {
        if (!module.Has("one2one_cv2") || key is not string name) return false;
        return ModelInfo.IsEndToEnd(module) ? name is "cv2" or "cv3" or "cv4" or "cv4_kpts" : name.StartsWith("one2one_", StringComparison.Ordinal);
    }
    /// <summary>按路由关系解释执行顶层模块，并在最后一次使用后释放中间结果。</summary>
    /// <param name="input">当前节点或模块的输入。</param>
    /// <returns>检测张量或分割检测张量与原型张量组成的模型输出数组。</returns>
    /// <exception cref="ArgumentException">输入不是 RGB NCHW，或高宽不是 32 的整数倍。</exception>
    /// <exception cref="InvalidDataException">路由所需的中间张量不存在。</exception>
    public Tensor[] Run(Tensor input)
    {
        if (input.Rank != 4 || input.Shape[1] != 3 || input.Shape[2] % 32 != 0 || input.Shape[3] % 32 != 0) throw new ArgumentException("PT YOLO expects NCHW RGB, height/width multiples of 32.");
        if (trainedImageSize > 0 && (input.Shape[2] != trainedImageSize || input.Shape[3] != trainedImageSize)) throw new ArgumentException("训练归档推理尺寸必须与保存时一致。");
        object?[] values = routedValues; int[] remaining = remainingUses; Array.Copy(uses, remaining, uses.Length); object result = input;
        try
        {
            for (int i = 0; i < layers.Length; i++)
            {
                // index 为加载时规范化的绝对路由；-1 读取本帧图像，缺失结果表示路由损坏。
                object LayerInput(int index) => index < 0 ? input : values[index] ?? throw new InvalidDataException("Missing routed tensor.");
                object argument;
                if (from[i].Length == 1) argument = LayerInput(from[i][0]);
                else
                {
                    object[] args = routedArguments[i]; for (int j = 0; j < args.Length; j++) args[j] = LayerInput(from[i][j]); argument = args;
                }
                long started = trace is not null ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                if (extractingFeatures && i == layers.Length - 1)
                    return ((object[])argument).Cast<Tensor>().ToArray();
                result = Forward(layers[i], argument); values[i] = result;
                if (trace is not null && ops.Backend is IDeviceBackend device) device.Flush();
                trace?.Invoke(new(i.ToString(), layers[i].Type.Name, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds));
                foreach (int index in from[i]) if (index >= 0 && --remaining[index] == 0) values[index] = null;
                if (CpuTensorPool.Current is { } pool)
                {
                    for (int j = 0; j < values.Length; j++) retainedValues[j] = remaining[j] > 0 ? values[j] : null;
                    retainedValues[^1] = result; pool.RetainOnly(retainedValues);
                }
            }
            return result is Tensor t ? [t] : (Tensor[])result;
        }
        finally
        {
            Array.Clear(values); Array.Clear(retainedValues); foreach (object[] args in routedArguments) Array.Clear(args);
        }
    }
    /// <summary>执行只返回单个张量的模块，并把解释器结果转换为张量。</summary>
    /// <param name="m">要执行的符号模块。</param>
    /// <param name="x">参与计算的输入张量。</param>
    /// <returns>模块前向输出张量。</returns>
    private Tensor F(PyObject m, Tensor x) => (Tensor)Forward(m, x);
    /// <summary>按名称取出一个子模块并执行其前向计算。</summary>
    /// <param name="m">要执行的符号模块。</param>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="x">参与计算的输入张量。</param>
    /// <returns>指定子模块的输出张量。</returns>
    private Tensor C(PyObject m, string key, Tensor x) => F(m.Child(key), x);
    /// <summary>执行通道拼接后的子模块卷积，满足 CPU 常量 1×1 条件时直接消费分片以省去拼接复制。</summary>
    /// <param name="m">要执行的符号模块。</param>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="inputs">按路由或节点声明顺序排列的输入张量。</param>
    /// <returns>子模块卷积输出；不符合融合条件时采用正常拼接路径。</returns>
    private Tensor CConcat(PyObject m, string key, IReadOnlyList<Tensor> inputs)
    {
        var module = m.Child(key);
        string activation = module.Has("act") ? module.Child("act").Type.Name : "Identity";
        if (ops.Backend is CpuBackend && module.Type.Name == "Conv" && (activation is "SiLU" or "Identity") && fused.TryGetValue(module, out var parameters))
        {
            var conv = module.Child("conv");
            if (conv.Get("padding_mode") is null or "zeros")
            {
                int[] pads = conv.Pair("padding", 0);
                return ops.ConvConcat(inputs, parameters.Weight, parameters.Bias, conv.Pair("stride"), [pads[0], pads[1], pads[0], pads[1]], conv.Pair("dilation"), conv.Int("groups", 1), activation == "SiLU");
            }
        }
        return C(m, key, TensorOps.Concat(inputs, 1));
    }
    /// <summary>按广播规则相加两个 FP32 张量。</summary>
    /// <param name="a">左侧输入张量。</param>
    /// <param name="b">右侧输入张量。</param>
    /// <returns>广播后逐元素相加的张量。</returns>
    private static Tensor Add(Tensor a, Tensor b) => TensorOps.Arithmetic(a, b, 10);
    /// <summary>沿通道轴把张量拆成两个等长分片。</summary>
    /// <param name="t">待拆分的输入张量。</param>
    /// <returns>两个等长通道分片组成的数组；设备分片可共享底层存储，不保证拥有独立数据。</returns>
    private static Tensor[] Chunk(Tensor t) => TensorOps.Split(t, 1, [t.Shape[1] / 2, t.Shape[1] / 2]);
    /// <summary>根据符号模块类型执行 YOLO 块、检测头、分割头或姿态头的前向计算。</summary>
    /// <param name="p">当前符号模块的参数与状态。</param>
    /// <param name="input">单个张量或由层路由产生的输入对象数组。</param>
    /// <returns>模块输出张量、张量数组或传给下一模块的中间对象。</returns>
    /// <exception cref="ArgumentException">模块需要张量输入，但当前路由输入不是张量。</exception>
    /// <exception cref="NotSupportedException">模块、上采样模式或池化变体尚未实现。</exception>
    private object Forward(PyObject p, object input)
    {
        // 返回当前模块的单张量输入；路由对象类型错误时抛出 ArgumentException。
        Tensor X() => input as Tensor ?? throw new ArgumentException($"{p.Type.Name} requires tensor.");
        // x 为待激活张量；存在 act 子模块时执行它，否则返回原张量。
        Tensor Act(Tensor x) => p.Has("act") ? C(p, "act", x) : x;
        switch (p.Type.Name)
        {
            case "Identity": case "Silence": return input;
            case "Sequential": case "ModuleList": foreach (var child in p.Children) input = Forward(child, input); return input;
            case "Conv2d": return Conv(p, X());
            case "ConvTranspose2d": var tp = p.Pair("padding", 0); return TensorOps.ConvTranspose(X(), p.Weight("weight"), p.Map("_parameters").GetValueOrDefault("bias") as Tensor, p.Pair("stride"), [tp[0], tp[1], tp[0], tp[1]], p.Pair("dilation"), p.Pair("output_padding", 0), p.Int("groups", 1));
            case "Conv":
            case "DWConv":
                if (fused.TryGetValue(p, out var f)) return p.Has("act") && p.Child("act").Type.Name == "SiLU" ? Conv(p.Child("conv"), X(), f.Weight, f.Bias, true) : Act(Conv(p.Child("conv"), X(), f.Weight, f.Bias));
                return Act(C(p, "conv", X()));
            case "BatchNorm2d":
                var ybn = new Tensor(X().Shape); int spatial = Tensor.Count(X().Shape[2..]), channels = X().Shape[1];
                var gamma = p.Weight("weight"); var beta = p.Weight("bias"); var mean = p.Weight("running_mean"); var variance = p.Weight("running_var");
                for (int b = 0; b < X().Shape[0]; b++) for (int c = 0; c < channels; c++)
                { float scale = gamma.Data[c] / MathF.Sqrt(variance.Data[c] + p.Float("eps", 1e-5f)); for (int i = 0; i < spatial; i++) { int index = (b * channels + c) * spatial + i; ybn.Data[index] = (X().Data[index] - mean.Data[c]) * scale + beta.Data[c]; } }
                return ybn;
            case "SiLU": return TensorOps.Activation(X(), true);
            case "ReLU": return TensorOps.Unary(X(), x => MathF.Max(0, x));
            case "LeakyReLU": return TensorOps.Unary(X(), x => x >= 0 ? x : p.Float("negative_slope", 0.01f) * x);
            case "Upsample":
                if (p.Get("mode") is not "nearest") throw new NotSupportedException("PT Upsample requires nearest.");
                float[] factors = p.Get("scale_factor") is object?[] pair ? pair.Select(Convert.ToSingle).ToArray() : [p.Float("scale_factor", 2), p.Float("scale_factor", 2)];
                var size = p.Get("size") is object?[] fixedSize ? fixedSize.Select(Convert.ToInt32).ToArray() : new[] { (int)(X().Shape[2] * factors[0]), (int)(X().Shape[3] * factors[1]) };
                return TensorOps.ResizeNearest(X(), size[0], size[1]);
            case "MaxPool2d": if (p.Bool("ceil_mode") || p.Pair("dilation").Any(d => d != 1)) throw new NotSupportedException("PT pool variant unsupported."); return Pool(p, X());
            case "Bottleneck":
            case "RepBottleneck":
            case "RepNBottleneck":
                var bneck = C(p, "cv2", C(p, "cv1", X())); return p.Bool("add") ? Add(X(), bneck) : bneck;
            case "CIB": var cib = C(p, "cv1", X()); return p.Bool("add") ? Add(X(), cib) : cib;
            case "SCDown": return C(p, "cv2", C(p, "cv1", X()));
            case "C2f":
            case "C3k2":
            case "C2fCIB":
            case "C2fPSA":
                var ys = Chunk(C(p, "cv1", X())).ToList(); foreach (var m in p.Child("m").Children) ys.Add(F(m, ys[^1])); return CConcat(p, "cv2", ys);
            case "C3": case "C3k": case "RepCSP": case "RepNCSP": return CConcat(p, "cv3", [C(p, "m", C(p, "cv1", X())), C(p, "cv2", X())]);
            case "SPPF":
                var sp = new List<Tensor> { C(p, "cv1", X()) }; for (int i = 0; i < p.Int("n", 3); i++) sp.Add(C(p, "m", sp[^1]));
                var pooled = CConcat(p, "cv2", sp); return p.Bool("add") ? Add(pooled, X()) : pooled;
            case "SPP": var first = C(p, "cv1", X()); return CConcat(p, "cv2", new[] { first }.Concat(p.Child("m").Children.Select(m => F(m, first))).ToArray());
            case "SPPELAN": var elan = new List<Tensor> { C(p, "cv1", X()) }; foreach (string key in new[] { "cv2", "cv3", "cv4" }) elan.Add(C(p, key, elan[^1])); return CConcat(p, "cv5", elan);
            case "Concat": return TensorOps.Concat(((object[])input).Cast<Tensor>().ToArray(), p.Int("d", 1));
            case "RepNCSPELAN4": case "ELAN1": var re = Chunk(C(p, "cv1", X())).ToList(); re.Add(C(p, "cv2", re[^1])); re.Add(C(p, "cv3", re[^1])); return CConcat(p, "cv4", re);
            case "AConv": return C(p, "cv1", TensorOps.Pool(X(), [2, 2], [1, 1], [0, 0, 0, 0], true));
            case "ADown": var ad = Chunk(TensorOps.Pool(X(), [2, 2], [1, 1], [0, 0, 0, 0], true)); return TensorOps.Concat([C(p, "cv1", ad[0]), C(p, "cv2", TensorOps.Pool(ad[1], [3, 3], [2, 2], [1, 1, 1, 1]))], 1);
            case "RepConv":
            case "RepConvN":
                if (fused.TryGetValue(p, out var repKernel))
                {
                    bool silu = p.Child("act").Type.Name == "SiLU";
                    var repOutput = Conv(p.Child("conv1").Child("conv"), X(), repKernel.Weight, repKernel.Bias, silu);
                    return silu ? repOutput : Act(repOutput);
                }
                if (p.Has("conv")) return Act(C(p, "conv", X()));
                var rep = Add(C(p, "conv1", X()), C(p, "conv2", X())); if (p.Has("bn")) rep = Add(rep, C(p, "bn", X())); return Act(rep);
            case "RepVGGDW":
                if (fused.TryGetValue(p, out var dwKernel))
                {
                    bool silu = p.Child("act").Type.Name == "SiLU";
                    var repOutput = Conv(p.Child("conv").Child("conv"), X(), dwKernel.Weight, dwKernel.Bias, silu);
                    return silu ? repOutput : Act(repOutput);
                }
                return Act(p.Has("conv1") ? Add(C(p, "conv", X()), C(p, "conv1", X())) : C(p, "conv", X()));
            case "C2PSA": var ps = Chunk(C(p, "cv1", X())); ps[1] = C(p, "m", ps[1]); return CConcat(p, "cv2", ps);
            case "PSA": ps = Chunk(C(p, "cv1", X())); ps[1] = Add(ps[1], C(p, "attn", ps[1])); ps[1] = Add(ps[1], C(p, "ffn", ps[1])); return CConcat(p, "cv2", ps);
            case "PSABlock": var att = C(p, "attn", X()); if (p.Bool("add", true)) att = Add(X(), att); var feed = C(p, "ffn", att); return p.Bool("add", true) ? Add(att, feed) : feed;
            case "Attention": return Attention(p, X());
            case "Detect": case "DDetect": case "DualDDetect": case "TripleDDetect": case "v10Detect": return Detect(p, ((object[])input).Cast<Tensor>().ToArray());
            case "Proto": return C(p, "cv3", C(p, "cv2", C(p, "upsample", C(p, "cv1", X()))));
            case "Proto26":
                var protoFeatures = ((object[])input).Cast<Tensor>().ToArray(); var refined = protoFeatures[0]; var refineBranches = p.Child("feat_refine").Children;
                for (int i = 0; i < refineBranches.Length; i++)
                {
                    var higher = F(refineBranches[i], protoFeatures[i + 1]);
                    refined = Add(refined, TensorOps.ResizeNearest(higher, higher.Shape[2] * (1 << (i + 1)), higher.Shape[3] * (1 << (i + 1))));
                }
                return C(p, "cv3", C(p, "cv2", C(p, "upsample", C(p, "cv1", C(p, "feat_fuse", refined)))));
            case "Segment":
            case "Segment26":
                var features = ((object[])input).Cast<Tensor>().ToArray();
                var proto = p.Child("proto").Type.Name == "Proto26" ? (Tensor)Forward(p.Child("proto"), features.Cast<object>().ToArray()) : C(p, "proto", features[0]);
                bool segmentEndToEnd = ModelInfo.IsEndToEnd(p); var maskBranches = p.Child(segmentEndToEnd ? "one2one_cv4" : "cv4").Children;
                var coefficients = features.Select((t, i) => F(maskBranches[i], t).Reshape(t.Shape[0], p.Int("nm"), t.Shape[2] * t.Shape[3])).ToArray();
                var maskHead = TensorOps.Concat([Detect(p, features, false), TensorOps.Concat(coefficients, 2)], 1);
                return new[] { segmentEndToEnd ? EndToEndTopK(maskHead, p.Int("max_det", 300), p.Int("nc")) : maskHead, proto };
            case "Pose":
            case "Pose26":
                features = ((object[])input).Cast<Tensor>().ToArray(); bool poseEndToEnd = ModelInfo.IsEndToEnd(p), pose26 = p.Type.Name == "Pose26";
                string posePrefix = poseEndToEnd ? "one2one_" : ""; var keypointBranches = p.Child(posePrefix + "cv4").Children;
                var projections = pose26 ? p.Child(posePrefix + "cv4_kpts").Children : [];
                int[] kpShape = ((IEnumerable<object?>)p.Get("kpt_shape")!).Select(Convert.ToInt32).ToArray(); int nk = kpShape[0] * kpShape[1];
                var pointLevels = new List<Tensor>(); var stride = (Tensor)p.Get("stride")!;
                for (int level = 0; level < features.Length; level++)
                {
                    var t = features[level]; int kpSpatial = t.Shape[2] * t.Shape[3]; var rawPoints = F(keypointBranches[level], t);
                    if (pose26) rawPoints = F(projections[level], rawPoints);
                    if (DeviceScope.For(rawPoints) is not null) { pointLevels.Add(DeviceOps.DecodePose(rawPoints, kpShape[0], kpShape[1], stride.Data[level], pose26)); continue; }
                    if (CpuPose.Enabled && kpShape[1] is 2 or 3)
                    {
                        pointLevels.Add(CpuPose.Decode(rawPoints, kpShape[0], kpShape[1], stride.Data[level], ops.Backend is CpuBackend poseCpu ? poseCpu.Parallel : CpuScheduling.Options, pose26));
                        continue;
                    }
                    var logits = rawPoints.Reshape(t.Shape[0], nk, kpSpatial);
                    for (int b = 0; b < t.Shape[0]; b++) for (int k = 0; k < kpShape[0]; k++) for (int a = 0; a < kpSpatial; a++)
                    {
                        int off = (b * nk + k * kpShape[1]) * kpSpatial + a;
                        logits.Data[off] = (pose26 ? logits.Data[off] + (a % t.Shape[3] + .5f) : logits.Data[off] * 2 + a % t.Shape[3]) * stride.Data[level];
                        logits.Data[off + kpSpatial] = (pose26 ? logits.Data[off + kpSpatial] + (a / t.Shape[3] + .5f) : logits.Data[off + kpSpatial] * 2 + a / t.Shape[3]) * stride.Data[level];
                        if (kpShape[1] == 3) logits.Data[off + 2 * kpSpatial] = TensorOps.Sigmoid(logits.Data[off + 2 * kpSpatial]);
                    }
                    pointLevels.Add(logits);
                }
                var poseHead = TensorOps.Concat([Detect(p, features, false), TensorOps.Concat(pointLevels, 2)], 1);
                return poseEndToEnd ? EndToEndTopK(poseHead, p.Int("max_det", 300), p.Int("nc")) : poseHead;
            case "DFL":
                int bins = p.Int("c1", 16); int batch = X().Shape[0], anchors = X().Shape[2]; var dfl = TensorOps.Softmax(X().Reshape(batch, 4, bins, anchors), 2); var distance = new Tensor(batch, 4, anchors);
                for (int b = 0; b < batch; b++) for (int c = 0; c < 4; c++) for (int a = 0; a < anchors; a++) for (int r = 0; r < bins; r++) distance.Data[(b * 4 + c) * anchors + a] += dfl.Data[((b * 4 + c) * bins + r) * anchors + a] * r;
                return distance;
            case "CBLinear": return TensorOps.Split(C(p, "conv", X()), 1, ((IEnumerable<object?>)p.Get("c2s")!).Select(Convert.ToInt32).ToArray());
            case "CBFuse":
                var objects = (object[])input; var last = (Tensor)objects[^1]; var idxs = ((IEnumerable<object?>)p.Get("idx")!).Select(Convert.ToInt32).ToArray();
                for (int i = 0; i < objects.Length - 1; i++) last = Add(last, TensorOps.ResizeNearest(((Tensor[])objects[i])[idxs[i]], last.Shape[2], last.Shape[3])); return last;
            default: throw new NotSupportedException(p.Type.Name);
        }
    }
    /// <summary>读取 Torch 卷积几何和参数，并调用后端卷积，支持预融合偏置与 SiLU。</summary>
    /// <param name="p">当前符号模块的参数与状态。</param>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="weight">替代模块参数的融合权重；null 读取模块自身权重。</param>
    /// <param name="bias">替代模块参数的融合偏置；null 读取模块自身偏置。</param>
    /// <param name="silu">是否在卷积后处理阶段执行 SiLU。</param>
    /// <returns>按卷积几何生成的输出张量。</returns>
    /// <exception cref="NotSupportedException">卷积使用非零填充模式。</exception>
    private Tensor Conv(PyObject p, Tensor x, Tensor? weight = null, Tensor? bias = null, bool silu = false)
    {
        if (p.Get("padding_mode") is not (null or "zeros")) throw new NotSupportedException("Nonzero Conv padding mode unsupported.");
        int[] pads = p.Pair("padding", 0); return ops.Conv(x, weight ?? p.Weight("weight"), bias ?? p.Map("_parameters").GetValueOrDefault("bias") as Tensor, p.Pair("stride"), [pads[0], pads[1], pads[0], pads[1]], p.Pair("dilation"), p.Int("groups", 1), silu);
    }
    /// <summary>按 Torch 模块声明执行二维最大池化。</summary>
    /// <param name="p">当前符号模块的参数与状态。</param>
    /// <param name="x">参与计算的输入张量。</param>
    /// <returns>池化后的张量。</returns>
    private static Tensor Pool(PyObject p, Tensor x)
    { var pads = p.Pair("padding", 0); return TensorOps.Pool(x, p.Pair("kernel_size"), p.Pair("stride"), [pads[0], pads[1], pads[0], pads[1]]); }
    /// <summary>计算多头 Q/K/V 注意力、位置编码残差和输出投影。</summary>
    /// <param name="p">当前符号模块的参数与状态。</param>
    /// <param name="x">参与计算的输入张量。</param>
    /// <returns>恢复为原 NCHW 空间布局的注意力输出。</returns>
    private Tensor Attention(PyObject p, Tensor x)
    {
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3], n = h * w, heads = p.Int("num_heads"), kd = p.Int("key_dim"), hd = p.Int("head_dim");
        var qkv = C(p, "qkv", x).Reshape(b, heads, 2 * kd + hd, n); var parts = TensorOps.Split(qkv, 2, [kd, kd, hd]);
        var q = TensorOps.Transpose(parts[0], [0, 1, 3, 2]); var scores = ops.MatMul(q, parts[1]);
        scores = TensorOps.Softmax(TensorOps.Scale(scores, p.Float("scale", 1f / MathF.Sqrt(kd))), -1);
        var attended = ops.MatMul(parts[2], TensorOps.Transpose(scores, [0, 1, 3, 2])).Reshape(b, c, h, w);
        return C(p, "proj", Add(attended, C(p, "pe", parts[2].Reshape(b, c, h, w))));
    }
    /// <summary>计算各特征层的边框分布与分类分支，并按步长解码和拼接检测结果。</summary>
    /// <param name="p">当前符号模块的参数与状态。</param>
    /// <param name="inputs">按路由或节点声明顺序排列的输入张量。</param>
    /// <param name="selectEndToEnd">是否立即执行 TopK；拼接实例附加字段前设为 false。</param>
    /// <returns>通道优先检测结果，或端到端两阶段筛选后的六字段结果。</returns>
    /// <exception cref="InvalidDataException">检测分支数量不匹配，或缺少特征层步长。</exception>
    private Tensor Detect(PyObject p, Tensor[] inputs, bool selectEndToEnd = true)
    {
        string boxKey = "cv2", classKey = "cv3"; int nl = p.Int("nl", inputs.Length);
        if (p.Type.Name == "DualDDetect") { inputs = inputs[^nl..]; boxKey = "cv4"; classKey = "cv5"; }
        if (p.Type.Name == "TripleDDetect") { inputs = inputs[^nl..]; boxKey = "cv6"; classKey = "cv7"; }
        bool end2end = ModelInfo.IsEndToEnd(p);
        if (end2end) { boxKey = "one2one_cv2"; classKey = "one2one_cv3"; }
        var boxes = p.Child(boxKey).Children; var classes = p.Child(classKey).Children; int nc = p.Int("nc"), bins = p.Int("reg_max", 16), batch = inputs[0].Shape[0];
        if (inputs.Length != boxes.Length || classes.Length != boxes.Length) throw new InvalidDataException("Detection branches mismatch.");
        var stride = p.Get("stride") as Tensor ?? throw new InvalidDataException("Detect stride missing.");
        if (ops.Backend is IDeviceBackend { FrameActive: true })
        {
            var levels = inputs.Select((feature, i) => DeviceOps.DecodeHead(F(boxes[i], feature), F(classes[i], feature), bins, stride.Data[i], end2end)).ToArray();
            var decoded = TensorOps.Concat(levels, 2); return end2end && selectEndToEnd ? EndToEndTopK(decoded, p.Int("max_det", 300), nc) : decoded;
        }
        var decodedLevels = inputs.Select((feature, i) => CpuHead.Decode(F(boxes[i], feature), F(classes[i], feature), bins, stride.Data[i], end2end, ops.Backend is CpuBackend cpuHead ? cpuHead.Parallel : null)).ToArray();
        var output = TensorOps.Concat(decodedLevels, 2);
        return end2end && selectEndToEnd ? EndToEndTopK(output, p.Int("max_det", 300), nc) : output;
    }
    /// <summary>先筛选最大类别分数最高的锚点，再筛选锚点与类别组合，保持确定的相同分数顺序。</summary>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="maximum">端到端检测头允许保留的最大结果数量。</param>
    /// <param name="nc">类别数量。</param>
    /// <returns>形状为 [批量,保留数,6+附加字段] 的结果，附加字段与边框使用同一锚点。</returns>
    private static Tensor EndToEndTopK(Tensor x, int maximum, int nc)
    {
        if (DeviceScope.For(x) is OnnxTraceBackend trace) return trace.EndToEnd(x, maximum, nc);
        if (DeviceScope.For(x) is VulkanBackend vulkan && vulkan.TryEndToEndTopK(x, maximum, nc) is { } deviceResult) return deviceResult;
        if (CpuEndToEnd.PtEnabled && DeviceScope.For(x) is null && x.Rank == 3 && x.Shape[1] >= nc + 4 && nc > 0 && maximum > 0 && x.Shape[2] > 0)
            return CpuEndToEnd.SelectPt(x, maximum, nc);
        int anchors = x.Shape[2], k = Math.Min(maximum, anchors), extra = x.Shape[1] - nc - 4, fields = 6 + extra; var result = new Tensor(x.Shape[0], k, fields);
        for (int b = 0; b < x.Shape[0]; b++)
        {
            int off = b * x.Shape[1] * anchors;
            float[] data = x.Data; var maxima = new float[anchors]; Array.Fill(maxima, float.NegativeInfinity);
            for (int c = 0; c < nc; c++)
            {
                int src = off + (4 + c) * anchors, a = 0;
                for (; a <= anchors - System.Numerics.Vector<float>.Count; a += System.Numerics.Vector<float>.Count)
                    System.Numerics.Vector.Max(new System.Numerics.Vector<float>(maxima, a), new System.Numerics.Vector<float>(data, src + a)).CopyTo(maxima, a);
                for (; a < anchors; a++) maxima[a] = MathF.Max(maxima[a], data[src + a]);
            }
            var anchorQueue = new PriorityQueue<int, (float Score, int Tie)>();
            for (int a = 0; a < anchors; a++) { if (anchorQueue.Count < k) anchorQueue.Enqueue(a, (maxima[a], -a)); else anchorQueue.EnqueueDequeue(a, (maxima[a], -a)); }
            var classQueue = new PriorityQueue<(int Anchor, int Class, float Score), (float Score, int Anchor, int Class)>();
            foreach (var item in anchorQueue.UnorderedItems)
            {
                int a = item.Element;
                for (int c = 0; c < nc; c++)
                {
                    float score = data[off + (4 + c) * anchors + a];
                    if (classQueue.Count < k) classQueue.Enqueue((a, c, score), (score, -a, -c));
                    else classQueue.EnqueueDequeue((a, c, score), (score, -a, -c));
                }
            }
            var best = classQueue.UnorderedItems.Select(v => v.Element).OrderByDescending(v => v.Score).ThenBy(v => v.Anchor).ThenBy(v => v.Class).ToArray();
            for (int i = 0; i < best.Length; i++)
            {
                int dst = (b * k + i) * fields; for (int j = 0; j < 4; j++) result.Data[dst + j] = x.Data[off + j * anchors + best[i].Anchor];
                result.Data[dst + 4] = best[i].Score; result.Data[dst + 5] = best[i].Class;
                for (int j = 0; j < extra; j++) result.Data[dst + 6 + j] = x.Data[off + (4 + nc + j) * anchors + best[i].Anchor];
            }
        }
        return result;
    }
}
