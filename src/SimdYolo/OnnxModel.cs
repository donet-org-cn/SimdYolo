namespace SimdYolo;

/// <summary>用纯 C# 执行 ONNX 图，并在保持输入输出约定的前提下进行常量折叠和算子融合。</summary>
internal sealed partial class OnnxModel : IModel
{
    /// <summary>本会话加载并完成优化的 ONNX 图。</summary>
    private readonly OnnxGraph graph;
    /// <summary>将卷积与矩阵乘法分派到当前计算后端的入口。</summary>
    private readonly TensorOps ops;
    /// <summary>可选逐节点计时回调。</summary>
    private readonly Action<LayerTiming>? trace;
    /// <summary>加载时建立的整数路由和会话独占工作区。</summary>
    private readonly OnnxExecutionPlan execution;
    /// <summary>以输入形状为键的有限形状执行计划缓存。</summary>
    private readonly Dictionary<string, ShapeExecutionPlan> shapePlans = [];
    /// <summary>图像输入维度的副本；非正静态维度按动态输入处理。</summary>
    /// <value>图像输入维度的副本；非正静态维度按动态输入处理。</value>
    public int[] InputShape => (int[])graph.InputShape.Clone();
    /// <summary>模型元数据中声明的类别名称映射。</summary>
    /// <value>模型元数据中声明的类别名称映射。</value>
    public IReadOnlyDictionary<int, string> Names { get; }
    /// <summary>模型提供或由原始检测头维度推导的类别通道数量。</summary>
    /// <value>实际类别通道数量；无法确定时为 0。</value>
    public int ClassCount { get; }
    /// <summary>模型元数据声明的检测、分割或姿态任务。</summary>
    /// <value>模型元数据声明的检测、分割或姿态任务。</value>
    public YoloTask Task { get; }
    /// <summary>模型声明的姿态关键点数量与维度；未声明时采用 [17, 3]。</summary>
    /// <value>模型声明的姿态关键点数量与维度；未声明时采用 [17, 3]。</value>
    public int[] KeypointShape { get; }
    /// <summary>端到端模型使用实例行布局，普通输出继续由解码器识别通道轴。</summary>
    /// <value>端到端布局或 Auto。</value>
    public OutputLayout PreferredLayout => EndToEnd ? OutputLayout.EndToEnd : OutputLayout.Auto;
    /// <summary>根据 end2end/args 元数据或无歧义输出形状识别端到端模式。</summary>
    /// <value>模型是否输出六字段及可选实例附加字段。</value>
    public bool EndToEnd { get; }
    /// <summary>普通 ONNX 解释器支持的算子名称集合。</summary>
    private static readonly HashSet<string> Supported = new("Constant Identity Conv ConvTranspose BatchNormalization Sigmoid Relu LeakyRelu Tanh Softmax Add Sub Mul Div Mod Pow Sqrt Exp Neg Abs Max Min MatMul Gemm MaxPool AveragePool GlobalAveragePool Concat Split Reshape Transpose Flatten Resize Shape Gather GatherElements Slice Unsqueeze Squeeze Cast ConstantOfShape Expand Tile ReduceMean ReduceSum ReduceMax Range TopK Equal Greater Less Where".Split(' '));
    /// <summary>读取并优化 ONNX 图，解析任务元数据，建立使用次数与融合模式表。</summary>
    /// <param name="path">模型文件路径，扩展名决定 PT 或 ONNX 解析方式。</param>
    /// <param name="ops">使用当前计算后端的张量算子入口。</param>
    /// <param name="trace">可选逐层或逐节点耗时回调。</param>
    /// <exception cref="NotSupportedException">加载后的 ONNX 图仍包含未实现的算子。</exception>
    /// <exception cref="InvalidDataException">模型读取或常量子图执行失败。</exception>
    public OnnxModel(string path, TensorOps ops, Action<LayerTiming>? trace = null)
    {
        graph = OnnxGraph.Load(path); this.ops = ops;
        this.trace = trace;
        var info = ModelInfo.FromOnnx(graph.Metadata, graph.InputShape, graph.OutputShapes);
        Task = info.Task; KeypointShape = info.Task == YoloTask.Pose ? info.KeypointShape : [17, 3];
        Names = info.ClassNames; ClassCount = info.ClassCount; EndToEnd = info.EndToEnd;
        EliminateDeadNodes(graph);
        FoldConstants();
        FuseBatchNormalization(graph);
        FuseConvolutionBranches(graph);
        FuseCpuZeroAdd();
        FuseSilu(graph);
        FuseConcatConv();
        FuseDfl();
        FuseSigmoidConcat();
        FuseEndToEndHead();
        EliminateDeadNodes(graph);
        foreach (var n in graph.Nodes) if (!Supported.Contains(n.Op) &&
            !(n.Op == "_Dfl" && dflPatterns.ContainsKey(n.Name)) &&
            !(n.Op == "_ConcatConv" && concatConvolutions.Contains(n.Name)) &&
            !(n.Op == "_OnnxEndToEnd" && endToEndPatterns.ContainsKey(n.Name)) &&
            !(n.Op == "_SigmoidConcat" && sigmoidConcats.ContainsKey(n.Name))) throw new NotSupportedException($"ONNX node '{n.Name}': operator '{n.Op}' unsupported.");
        execution = new OnnxExecutionPlan(graph);
    }
    /// <summary>执行优化后的 ONNX 图，缓存仅依赖常量和稳定形状的节点，并按使用次数释放中间值。</summary>
    /// <param name="input">当前节点或模块的输入。</param>
    /// <returns>按图输出顺序返回的张量；常量输出会复制以保护持久缓存。</returns>
    /// <exception cref="ArgumentException">输入秩或静态维度与模型输入不一致。</exception>
    /// <exception cref="InvalidDataException">节点执行失败，或实际输出数量与节点声明不一致；消息包含节点名称。</exception>
    public Tensor[] Run(Tensor input)
    {
        if (input.Rank != graph.InputShape.Length || input.Shape.Where((d, i) => graph.InputShape[i] > 0 && d != graph.InputShape[i]).Any()) throw new ArgumentException($"Expected image input [{string.Join(',', graph.InputShape)}].");
        ShapeExecutionPlan plan = ShapePlanFor(input);
        Tensor?[] values = execution.Values; execution.Begin(input);
        try
        {
            for (int nodeIndex = 0; nodeIndex < graph.Nodes.Count; nodeIndex++)
            {
                var node = graph.Nodes[nodeIndex];
                var route = execution.Nodes[nodeIndex]; Tensor?[] args = route.Arguments; Tensor[] outputs;
                for (int i = 0; i < args.Length; i++) args[i] = route.Inputs[i] < 0 ? null : values[route.Inputs[i]];
                long started = trace is not null ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                try
                {
                    if (plan.Cached[nodeIndex] is { } cached) outputs = cached;
                    else if (!plan.Analyzed[nodeIndex] && CanCache(node, plan))
                    {
                        var previousDevice = DeviceScope.Current;
                        try
                        {
                            DeviceScope.Current = null;
                            // 常量结果的生命周期超过当前帧，需要复制到独立数组，
                            // 避免形状缓存引用已经归还并被下一帧覆盖的池缓冲区。
                            outputs = Execute(node, args).Select(tensor => new Tensor((float[])tensor.Data.Clone(), tensor.Shape) { IsConstant = plan.Retained }).ToArray();
                        }
                        finally { DeviceScope.Current = previousDevice; }
                        plan.Cached[nodeIndex] = outputs;
                        foreach (string name in node.Outputs) if (name != "") plan.Constants.Add(name);
                    }
                    else outputs = Execute(node, args);
                    if (!plan.Analyzed[nodeIndex])
                    {
                        if (HasStableShape(node, plan)) foreach (string name in node.Outputs) if (name != "") plan.StableShapes.Add(name);
                        plan.Analyzed[nodeIndex] = true;
                    }
                }
                catch (Exception e) when (e is ArgumentException or NotSupportedException or IndexOutOfRangeException or InvalidOperationException or OverflowException)
                { throw new InvalidDataException($"ONNX node '{node.Name}' ({node.Op}): {e.Message}", e); }
                if (trace is not null && ops.Backend is IDeviceBackend device) device.Flush();
                trace?.Invoke(new(node.Name, node.Op, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds));
                if (outputs.Length != node.Outputs.Length) throw new InvalidDataException($"Node '{node.Name}' output count mismatch.");
                for (int i = 0; i < outputs.Length; i++) if (route.Outputs[i] >= 0) values[route.Outputs[i]] = outputs[i];
                if (CpuTensorPool.Current is { } pool)
                {
                    object?[] roots = execution.Roots;
                    for (int i = 0; i < route.Live.Length; i++) roots[i] = values[route.Live[i]];
                    Array.Clear(roots, route.Live.Length, roots.Length - route.Live.Length);
                    pool.RetainOnly(roots);
                }
            }
            var result = new Tensor[execution.Outputs.Length];
            for (int i = 0; i < result.Length; i++)
            {
                var tensor = values[execution.Outputs[i]] ?? throw new InvalidDataException($"ONNX output '{graph.Outputs[i]}' is missing.");
                // 公开张量允许修改，因此初始化权重和形状缓存输出要复制，
                // 防止调用方改写模型持久常量并影响后续推理。
                result[i] = tensor.IsConstant || plan.Constants.Contains(graph.Outputs[i])
                    ? new Tensor((float[])tensor.Data.Clone(), tensor.Shape) : tensor;
            }
            return result;
        }
        finally { execution.Clear(); }
    }
    /// <summary>根据算子和属性分派单个 ONNX 节点，必要时选择设备或托管实现。</summary>
    /// <param name="n">当前 ONNX 节点，包含算子类型与属性。</param>
    /// <param name="a">按节点输入顺序排列的张量；空名称输入对应 null。</param>
    /// <returns>与节点声明输出数量对应的张量数组。</returns>
    /// <exception cref="ArgumentException">必需输入缺失，或轴、尺寸、步长及广播条件不合法。</exception>
    /// <exception cref="NotSupportedException">算子或其属性组合不受支持。</exception>
    /// <exception cref="OverflowException">形状、索引转换或张量长度超过 Int32 范围。</exception>
    private Tensor[] Execute(OnnxNode n, Tensor?[] a)
    {
        // i 为节点输入序号；返回必需输入，缺失时抛出 ArgumentException。
        Tensor X(int i = 0) => a[i] ?? throw new ArgumentException("Required input missing.");
        // i 为整数参数输入序号；受检转换每个 FP32 值，超出 Int32 时抛出 OverflowException。
        int[] I(int i) => X(i).Data.Select(v => checked((int)v)).ToArray();
        // t 为单个节点结果；将它包装成与解释器协议一致的单元素输出数组。
        Tensor[] One(Tensor t) => [t];
        // x 为输入，operation 为设备算子编号，function 为 CPU 标量实现；返回逐元素计算结果。
        Tensor Unary(Tensor x, int operation, Func<float, float> function, float scale = 1)
            => DeviceScope.For(x) is { } owner ? DeviceOps.Unary(owner, x, operation, scale) : TensorOps.Unary(x, function);
        // x、y 为广播输入；维度允许且有设备范围时提交设备内核，否则使用托管 function。
        Tensor Binary(Tensor x, Tensor y, int operation, Func<float, float, float> function)
            => Math.Max(x.Rank, y.Rank) <= 4 && DeviceScope.For(x, y) is { } owner ? DeviceOps.Binary(owner, x, y, operation) : TensorOps.Binary(x, y, function);
        switch (n.Op)
        {
            case "_OnnxEndToEnd": return ExecuteEndToEndHead(n, a);
            case "_Dfl": return ExecuteDfl(n, a);
            case "_ConcatConv": return [ExecuteConcatConv(n, a)];
            case "_SigmoidConcat": return [ExecuteSigmoidConcat(n, a)];
            case "Constant":
                if (n.Attributes.TryGetValue("value", out var v)) return One(OnnxGraph.ReadTensor(Proto.Parse(v.Byte(5)!)));
                if (n.Attributes.TryGetValue("value_float", out v)) return One(Tensor.Scalar(v.Floats(2)[0]));
                if (n.Attributes.TryGetValue("value_int", out v)) return One(Tensor.Scalar(v.Int(3)));
                if (n.Attributes.TryGetValue("value_ints", out v)) { var d = v.Ints(8).Select(x => (float)x).ToArray(); return One(new(d, d.Length)); }
                throw new NotSupportedException("Constant attribute unsupported.");
            case "Identity":
            case "Cast":
                if (n.Op == "Cast" && n.Int("to", 1) is not (1 or 6 or 7 or 9 or 10 or 11)) throw new NotSupportedException("Cast target unsupported.");
                return One(n.Op != "Cast" ? X() : n.Int("to") switch { 6 or 7 => TensorOps.Unary(X(), MathF.Truncate), 9 => TensorOps.Unary(X(), x => x != 0 ? 1 : 0), 10 => TensorOps.Unary(X(), x => (float)(Half)x), _ => X() });
            case "Conv":
                if (n.Text("auto_pad", "NOTSET") != "NOTSET") throw new NotSupportedException("Conv auto_pad unsupported; export explicit pads.");
                return One(ops.Conv(X(), X(1), a.Length > 2 ? a[2] : null, n.Ints("strides", [1, 1]), n.Ints("pads", [0, 0, 0, 0]), n.Ints("dilations", [1, 1]), (int)n.Int("group", 1), n.Int("_silu") == 1));
            case "ConvTranspose":
                if (n.Text("auto_pad", "NOTSET") != "NOTSET" || n.Attributes.ContainsKey("output_shape")) throw new NotSupportedException("ConvTranspose variant unsupported.");
                return One(TensorOps.ConvTranspose(X(), X(1), a.Length > 2 ? a[2] : null, n.Ints("strides", [1, 1]), n.Ints("pads", [0, 0, 0, 0]), n.Ints("dilations", [1, 1]), n.Ints("output_padding", [0, 0]), (int)n.Int("group", 1)));
            case "BatchNormalization":
                if (n.Int("training_mode") != 0 || n.Outputs.Length != 1) throw new NotSupportedException("Inference BatchNormalization only.");
                var shape = Enumerable.Repeat(1, X().Rank).ToArray(); shape[1] = X().Shape[1];
                var scale = TensorOps.Binary(X(1), X(4), (s, variance) => s / MathF.Sqrt(variance + n.Float("epsilon", 1e-5f))).Reshape(shape);
                return One(TensorOps.Arithmetic(TensorOps.Arithmetic(TensorOps.Arithmetic(X(), X(3).Reshape(shape), 11), scale, 12), X(2).Reshape(shape), 10));
            case "Sigmoid": return One(TensorOps.Activation(X(), false));
            case "Relu": return One(Unary(X(), 3, x => MathF.Max(x, 0)));
            case "LeakyRelu": return One(Unary(X(), 9, x => x >= 0 ? x : x * n.Float("alpha", 0.01f), n.Float("alpha", 0.01f)));
            case "Tanh": return One(TensorOps.Unary(X(), MathF.Tanh));
            case "Sqrt": return One(Unary(X(), 6, MathF.Sqrt));
            case "Exp": return One(Unary(X(), 5, MathF.Exp));
            case "Neg": return One(Unary(X(), 7, x => -x));
            case "Abs": return One(Unary(X(), 8, MathF.Abs));
            case "Softmax":
                int softAxis = Tensor.Axis((int)n.Int("axis", graph.Opset < 13 ? 1 : -1), X().Rank);
                return One(graph.Opset < 13 ? TensorOps.Softmax(X().Reshape(Tensor.Count(X().Shape[..softAxis]), Tensor.Count(X().Shape[softAxis..])), 1).Reshape(X().Shape) : TensorOps.Softmax(X(), softAxis));
            case "Add": return One(TensorOps.Arithmetic(X(), X(1), 10));
            case "Sub": return One(TensorOps.Arithmetic(X(), X(1), 11));
            case "Mul": return One(TensorOps.Arithmetic(X(), X(1), 12));
            case "Div": return One(TensorOps.Arithmetic(X(), X(1), 13));
            case "Mod": return One(TensorOps.Binary(X(), X(1), (x, y) => n.Int("fmod") != 0 ? x % y : x - MathF.Floor(x / y) * y));
            case "Pow": return One(Binary(X(), X(1), 14, MathF.Pow));
            case "Equal": return One(TensorOps.Binary(X(), X(1), (x, y) => x == y ? 1 : 0));
            case "Greater": return One(TensorOps.Binary(X(), X(1), (x, y) => x > y ? 1 : 0));
            case "Less": return One(TensorOps.Binary(X(), X(1), (x, y) => x < y ? 1 : 0));
            case "Max": case "Min": return One(a.Skip(1).Aggregate(X(), (r, t) => Binary(r, t!, n.Op == "Max" ? 15 : 16, n.Op == "Max" ? MathF.Max : MathF.Min)));
            case "MatMul": return One(ops.MatMul(X(), X(1)));
            case "Gemm":
                var gemm = ops.MatMul(n.Int("transA") != 0 ? TensorOps.Transpose(X()) : X(), n.Int("transB") != 0 ? TensorOps.Transpose(X(1)) : X(1));
                gemm = TensorOps.Scale(gemm, n.Float("alpha", 1));
                return One(a.Length > 2 && a[2] is not null ? TensorOps.Arithmetic(gemm, TensorOps.Scale(X(2), n.Float("beta", 1)), 10) : gemm);
            case "MaxPool":
            case "AveragePool":
                if (n.Int("ceil_mode") != 0 || n.Text("auto_pad", "NOTSET") != "NOTSET" || n.Ints("dilations", [1, 1]).Any(d => d != 1) || n.Outputs.Length != 1) throw new NotSupportedException("Pool variant unsupported.");
                return One(TensorOps.Pool(X(), n.Ints("kernel_shape"), n.Ints("strides", [1, 1]), n.Ints("pads", [0, 0, 0, 0]), n.Op == "AveragePool", n.Int("count_include_pad") != 0));
            case "GlobalAveragePool": return One(TensorOps.Pool(X(), X().Shape[2..], [1, 1], [0, 0, 0, 0], true));
            case "Concat": return One(TensorOps.Concat(a.Select(t => t!).ToArray(), (int)n.Int("axis")));
            case "Split":
                int axis = Tensor.Axis((int)n.Int("axis"), X().Rank); int[] sizes = a.Length > 1 && a[1] is not null ? I(1) : n.Ints("split");
                if (sizes.Length == 0)
                {
                    int chunks = n.Outputs.Length, chunk = (X().Shape[axis] + chunks - 1) / chunks;
                    sizes = Enumerable.Range(0, chunks).Select(i => Math.Max(0, Math.Min(chunk, X().Shape[axis] - i * chunk))).ToArray();
                }
                return TensorOps.Split(X(), axis, sizes);
            case "Reshape":
                var dims = I(1); if (n.Int("allowzero") == 0) for (int i = 0; i < dims.Length; i++) if (dims[i] == 0) dims[i] = X().Shape[i];
                return One(X().Reshape(dims));
            case "Flatten":
                int fa = (int)n.Int("axis", 1); if (fa < 0) fa += X().Rank;
                return One(X().Reshape(Tensor.Count(X().Shape[..fa]), Tensor.Count(X().Shape[fa..])));
            case "Transpose": return One(TensorOps.Transpose(X(), n.Attributes.ContainsKey("perm") ? n.Ints("perm") : null));
            case "Resize":
                if (n.Text("mode", "nearest") != "nearest" || n.Text("coordinate_transformation_mode", "half_pixel") != "asymmetric" || n.Text("nearest_mode", "round_prefer_floor") != "floor" || n.Int("antialias") != 0 || n.Attributes.ContainsKey("axes")) throw new NotSupportedException("Resize supports nearest/asymmetric/floor (standard YOLO export).");
                int[] rs = a.Length > 3 && a[3] is { Length: > 0 } ? I(3) : X().Shape.Select((d, i) => (int)MathF.Floor(d * X(2).Data[i])).ToArray();
                if (rs.Length != 4 || rs[0] != X().Shape[0] || rs[1] != X().Shape[1]) throw new NotSupportedException("Resize batch/channel resizing unsupported.");
                return One(TensorOps.ResizeNearest(X(), rs[2], rs[3]));
            case "Shape":
                int start = (int)n.Int("start"), end = (int)n.Int("end", X().Rank); if (start < 0) start += X().Rank; if (end < 0) end += X().Rank;
                var sd = X().Shape[Math.Clamp(start, 0, X().Rank)..Math.Clamp(end, 0, X().Rank)].Select(i => (float)i).ToArray(); return One(new(sd, sd.Length));
            case "Gather": return One(Gather(X(), X(1), (int)n.Int("axis")));
            case "GatherElements": return One(GatherElements(X(), X(1), (int)n.Int("axis")));
            case "Slice":
                return One(Slice(X(), X(1).Data, X(2).Data, a.Length > 3 && a[3] is not null ? I(3) : Enumerable.Range(0, X(1).Length).ToArray(), a.Length > 4 && a[4] is not null ? I(4) : Enumerable.Repeat(1, X(1).Length).ToArray()));
            case "Unsqueeze":
            case "Squeeze":
                var axes = a.Length > 1 && a[1] is not null ? I(1) : n.Ints("axes"); var sh = X().Shape.ToList();
                if (n.Op == "Unsqueeze")
                {
                    int rank = sh.Count + axes.Length; var normalized = axes.Select(i => Tensor.Axis(i, rank)).Order().ToArray();
                    if (normalized.Distinct().Count() != normalized.Length) throw new ArgumentException("Duplicate axes.");
                    foreach (int ax in normalized) sh.Insert(ax, 1);
                }
                else
                {
                    if (axes.Length == 0) sh.RemoveAll(d => d == 1);
                    else foreach (int ax in axes.Select(i => Tensor.Axis(i, sh.Count)).Distinct().OrderDescending()) { if (sh[ax] != 1) throw new ArgumentException("Squeeze dimension must be one."); sh.RemoveAt(ax); }
                }
                return One(X().Reshape(sh.ToArray()));
            case "ConstantOfShape":
                var filled = new Tensor(I(0)); float value = n.Attributes.TryGetValue("value", out var cv) ? OnnxGraph.ReadTensor(Proto.Parse(cv.Byte(5)!)).Data[0] : 0;
                Array.Fill(filled.Data, value); return One(filled);
            case "Expand": return One(TensorOps.Binary(X(), new Tensor(I(1)), (x, _) => x));
            case "Tile":
                var repeats = I(1); if (repeats.Length != X().Rank || repeats.Any(r => r < 0)) throw new ArgumentException("Invalid Tile repeats.");
                var tiled = new Tensor(X().Shape.Select((d, i) => checked(d * repeats[i])).ToArray()); var tiledStrides = Tensor.Strides(X().Shape);
                for (int i = 0; i < tiled.Length; i++) { int t = i, src = 0; for (int d = X().Rank - 1; d >= 0; d--) { src += t % tiled.Shape[d] % X().Shape[d] * tiledStrides[d]; t /= tiled.Shape[d]; } tiled.Data[i] = X().Data[src]; }
                return One(tiled);
            case "ReduceMean":
            case "ReduceSum":
            case "ReduceMax":
                var ra = a.Length > 1 && a[1] is not null ? I(1) : n.Ints("axes");
                if (ra.Length == 0 && n.Int("noop_with_empty_axes") != 0) return One(X());
                return One(Reduce(X(), ra.Length > 0 ? ra : Enumerable.Range(0, X().Rank).ToArray(), n.Int("keepdims", 1) != 0, n.Op, ops.Backend is CpuBackend cpuReduce ? cpuReduce.Parallel : null));
            case "Range":
                float from = X().Data[0], to = X(1).Data[0], step = X(2).Data[0]; if (step == 0) throw new ArgumentException("Range step is zero.");
                int length = Math.Max(0, checked((int)MathF.Ceiling((to - from) / step))); var range = new Tensor(length); for (int i = 0; i < length; i++) range.Data[i] = from + i * step; return One(range);
            case "TopK": return TopK(X(), checked((int)X(1).Data[0]), (int)n.Int("axis", -1), n.Int("largest", 1) != 0);
            case "Where":
                var ws = TensorOps.BroadcastShape(TensorOps.BroadcastShape(X().Shape, X(1).Shape), X(2).Shape);
                var condition = TensorOps.Binary(X(), new Tensor(ws), (x, _) => x); var yes = TensorOps.Binary(X(1), new Tensor(ws), (x, _) => x); var no = TensorOps.Binary(X(2), new Tensor(ws), (x, _) => x);
                for (int i = 0; i < yes.Length; i++) yes.Data[i] = condition.Data[i] != 0 ? yes.Data[i] : no.Data[i]; return One(yes);
            default: throw new NotSupportedException(n.Op);
        }
    }
    /// <summary>识别 Conv→Sigmoid 与 Mul 的 SiLU 数据流，在无额外消费者时合并到卷积后处理。</summary>
    /// <param name="graph">需要读取、优化或创建计划的 ONNX 图。</param>
    private static void FuseSilu(OnnxGraph graph)
    {
        var producers = graph.Nodes.SelectMany(n => n.Outputs.Where(o => o != "").Select(o => (o, n))).ToDictionary(p => p.o, p => p.n);
        var counts = graph.Nodes.SelectMany(n => n.Inputs).Concat(graph.Outputs).GroupBy(s => s).ToDictionary(g => g.Key, g => g.Count());
        var remove = new HashSet<OnnxNode>();
        foreach (var mul in graph.Nodes.Where(n => n.Op == "Mul").ToArray())
        {
            if (mul.Inputs.Length != 2 || !producers.TryGetValue(mul.Inputs[0], out var left) || !producers.TryGetValue(mul.Inputs[1], out var right)) continue;
            var conv = left.Op == "Conv" ? left : right.Op == "Conv" ? right : null; var sigmoid = left.Op == "Sigmoid" ? left : right.Op == "Sigmoid" ? right : null;
            if (conv is null || sigmoid is null || conv.Outputs.Length != 1 || sigmoid.Inputs[0] != conv.Outputs[0] || counts.GetValueOrDefault(conv.Outputs[0]) != 2 || counts.GetValueOrDefault(sigmoid.Outputs[0]) != 1) continue;
            var flag = new Proto(); flag.Fields[3] = [(ulong)1]; conv.Attributes["_silu"] = flag;
            int index = graph.Nodes.IndexOf(conv); graph.Nodes[index] = conv with { Outputs = mul.Outputs }; remove.Add(sigmoid); remove.Add(mul);
        }
        graph.Nodes.RemoveAll(remove.Contains);
    }
    /// <summary>沿指定轴按索引数组选取整块张量，支持负索引。</summary>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="indices">使用 FP32 保存索引值的张量，支持负索引。</param>
    /// <param name="axis">操作轴；负数表示从最后一维反向计数。</param>
    /// <returns>将索引形状插入所选轴位置的输出张量。</returns>
    /// <exception cref="ArgumentException">规范负索引后仍超出所选轴范围。</exception>
    /// <exception cref="ArgumentOutOfRangeException">轴编号超出输入维度范围。</exception>
    /// <exception cref="OverflowException">FP32 索引不能转换为 Int32。</exception>
    private static Tensor Gather(Tensor x, Tensor indices, int axis)
    {
        axis = Tensor.Axis(axis, x.Rank); var shape = x.Shape[..axis].Concat(indices.Shape).Concat(x.Shape[(axis + 1)..]).ToArray(); var y = new Tensor(shape);
        int inner = Tensor.Count(x.Shape[(axis + 1)..]), outer = Tensor.Count(x.Shape[..axis]);
        for (int o = 0; o < outer; o++) for (int j = 0; j < indices.Length; j++)
        {
            int idx = checked((int)indices.Data[j]); if (idx < 0) idx += x.Shape[axis]; if ((uint)idx >= x.Shape[axis]) throw new ArgumentException("Gather index out of range.");
            Array.Copy(x.Data, (o * x.Shape[axis] + idx) * inner, y.Data, (o * indices.Length + j) * inner, inner);
        }
        return y;
    }
    /// <summary>逐元素沿指定轴按索引选取数据，在 Vulkan 支持范围内避免主特征图下载。</summary>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="indices">使用 FP32 保存索引值的张量，支持负索引。</param>
    /// <param name="axis">操作轴；负数表示从最后一维反向计数。</param>
    /// <returns>与索引张量形状相同的选取结果。</returns>
    /// <exception cref="ArgumentException">输入与索引秩不同，或索引及非所选轴的坐标越界。</exception>
    /// <exception cref="ArgumentOutOfRangeException">轴编号超出输入维度范围。</exception>
    /// <exception cref="OverflowException">FP32 索引不能转换为 Int32。</exception>
    private static Tensor GatherElements(Tensor x, Tensor indices, int axis)
    {
        axis = Tensor.Axis(axis, x.Rank); if (x.Rank != indices.Rank) throw new ArgumentException("GatherElements ranks mismatch.");
        if (OnnxDeviceOperations.Enabled && OnnxDeviceOperations.CanGather(x, indices) && DeviceScope.For(x) is VulkanBackend owner)
            return OnnxDeviceOperations.GatherElements(owner, x, indices, axis);
        var y = new Tensor(indices.Shape); var stride = Tensor.Strides(x.Shape);
        for (int i = 0; i < y.Length; i++)
        {
            int t = i, src = 0;
            for (int d = x.Rank - 1; d >= 0; d--)
            {
                int c = t % indices.Shape[d]; t /= indices.Shape[d];
                if (d == axis) { c = checked((int)indices.Data[i]); if (c < 0) c += x.Shape[d]; }
                if ((uint)c >= x.Shape[d]) throw new ArgumentException("GatherElements index out of range."); src += c * stride[d];
            }
            y.Data[i] = x.Data[src];
        }
        return y;
    }
    /// <summary>按起止位置、轴和步长切片；连续切片优先使用设备复制或整块数组复制。</summary>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="starts">各切片轴的起始位置。</param>
    /// <param name="ends">各切片轴的结束位置，不包含该位置。</param>
    /// <param name="axes">需要操作的轴编号。</param>
    /// <param name="steps">各切片轴的非零步长，支持负向切片。</param>
    /// <returns>按照切片规则生成的张量。</returns>
    /// <exception cref="ArgumentException">参数数量不一致、轴重复，或切片步长为 0。</exception>
    /// <exception cref="ArgumentOutOfRangeException">切片轴超出输入维度范围。</exception>
    private static Tensor Slice(Tensor x, float[] starts, float[] ends, int[] axes, int[] steps)
    {
        var shape = (int[])x.Shape.Clone(); int[] begin = new int[x.Rank], step = Enumerable.Repeat(1, x.Rank).ToArray();
        if (starts.Length != ends.Length || axes.Length != starts.Length || steps.Length != axes.Length || axes.Distinct().Count() != axes.Length) throw new ArgumentException("Slice inputs invalid.");
        for (int i = 0; i < axes.Length; i++)
        {
            int ax = Tensor.Axis(axes[i], x.Rank), size = x.Shape[ax], s = steps[i]; if (s == 0) throw new ArgumentException("Slice step zero.");
            // v 为切片位置，end 区分结束边界；返回按步长方向规范负索引并裁剪后的边界。
            int Bound(float v, bool end)
            {
                if (v <= int.MinValue) return s > 0 ? 0 : -1;
                if (v >= int.MaxValue) return s > 0 ? size : size - 1;
                int b = (int)v; if (b < 0) b += size; return Math.Clamp(b, s > 0 ? 0 : end ? -1 : 0, s > 0 ? size : size - 1);
            }
            begin[ax] = Bound(starts[i], false); int e = Bound(ends[i], true); step[ax] = s;
            shape[ax] = Math.Max(0, (int)Math.Ceiling((double)(e - begin[ax]) / s));
        }
        // YOLO 通道拆分常被导出为 Slice。连续切片优先在设备上复制，
        // 避免访问 Data 引发同步，使每个 C2f 块都下载完整特征图。
        if (step.All(s => s == 1))
        {
            Tensor result = x;
            for (int ax = 0; ax < shape.Length; ax++)
            {
                if (begin[ax] == 0 && shape[ax] == result.Shape[ax]) continue;
                int[] target = (int[])result.Shape.Clone(); target[ax] = shape[ax];
                if (DeviceScope.For(result) is { } device && target.Length <= 4)
                    result = device.DeviceSlice(result, target, ax, begin[ax]);
                else
                {
                    var copy = Tensor.Temporary(target);
                    int inner = Tensor.Count(result.Shape[(ax + 1)..]), outer = Tensor.Count(result.Shape[..ax]);
                    int count = target[ax] * inner;
                    float[] source = result.Data, destination = copy.Data;
                    for (int o = 0; o < outer; o++) Array.Copy(source, (o * result.Shape[ax] + begin[ax]) * inner, destination, o * count, count);
                    result = copy;
                }
            }
            return result;
        }
        var y = Tensor.Temporary(shape); var stride = Tensor.Strides(x.Shape);
        for (int i = 0; i < y.Length; i++) { int t = i, src = 0; for (int d = x.Rank - 1; d >= 0; d--) { src += (begin[d] + t % shape[d] * step[d]) * stride[d]; t /= shape[d]; } y.Data[i] = x.Data[src]; }
        return y;
    }
    /// <summary>沿指定轴执行求和、均值或最大值归约，按要求保留或移除归约维度。</summary>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="axes">需要操作的轴编号。</param>
    /// <param name="keep">是否把归约轴保留为大小为 1 的维度。</param>
    /// <param name="op">ReduceSum、ReduceMean 或 ReduceMax 算子名称。</param>
    /// <param name="parallel">当前 CPU 后端的并行配置；null 使用该方法的串行回退。</param>
    /// <returns>归约后的张量。</returns>
    private static Tensor Reduce(Tensor x, int[] axes, bool keep, string op, ParallelOptions? parallel = null)
    {
        if (op == "ReduceMax" && axes.Length == 1 && Tensor.Axis(axes[0], x.Rank) == x.Rank - 1) return ReduceMaxLast(x, keep, parallel);
        var reduced = axes.Select(ax => Tensor.Axis(ax, x.Rank)).ToHashSet(); int[] shape = x.Shape.Select((d, i) => reduced.Contains(i) ? 1 : d).ToArray(); var y = new Tensor(shape);
        if (op == "ReduceMax") Array.Fill(y.Data, float.NegativeInfinity); var stride = Tensor.Strides(shape);
        for (int i = 0; i < x.Length; i++)
        {
            int t = i, dst = 0; for (int d = x.Rank - 1; d >= 0; d--) { int c = t % x.Shape[d]; t /= x.Shape[d]; if (!reduced.Contains(d)) dst += c * stride[d]; }
            y.Data[dst] = op == "ReduceMax" ? MathF.Max(y.Data[dst], x.Data[i]) : y.Data[dst] + x.Data[i];
        }
        if (op == "ReduceMean") { int count = Tensor.Count(axes.Select(ax => x.Shape[Tensor.Axis(ax, x.Rank)])); for (int i = 0; i < y.Length; i++) y.Data[i] /= count; }
        return keep ? y : y.Reshape(shape.Where((_, i) => !reduced.Contains(i)).ToArray());
    }
    /// <summary>使用固定大小堆筛选前 k 个值，保持 NaN 优先以及相同值按原索引排序的语义。</summary>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="k">每个轴切片需要保留的元素数量。</param>
    /// <param name="axis">操作轴；负数表示从最后一维反向计数。</param>
    /// <param name="largest">true 筛选最大值，false 筛选最小值。</param>
    /// <returns>先为筛选值、后为 FP32 存储索引的两个张量。</returns>
    /// <exception cref="ArgumentException">k 为负数或超过所选轴长度。</exception>
    /// <exception cref="ArgumentOutOfRangeException">轴编号超出输入维度范围。</exception>
    private static Tensor[] TopK(Tensor x, int k, int axis, bool largest)
    {
        axis = Tensor.Axis(axis, x.Rank); int size = x.Shape[axis]; if (k < 0 || k > size) throw new ArgumentException("TopK k invalid.");
        if (axis == x.Rank - 1 && DeviceScope.For(x) is VulkanBackend device && device.TryTopKLastAxis(x, k, largest) is { } deviceOutputs) return deviceOutputs;
        var shape = (int[])x.Shape.Clone(); shape[axis] = k; var values = Tensor.Temporary(shape); var indices = Tensor.Temporary(shape);
        if (values.Length == 0) return [values, indices];
        int inner = Tensor.Count(x.Shape[(axis + 1)..]), outer = Tensor.Count(x.Shape[..axis]);
        float[] data = x.Data, valueData = values.Data, indexData = indices.Data;
        var selected = new int[k];
        for (int o = 0; o < outer; o++) for (int j = 0; j < inner; j++)
        {
            int offset = o * size * inner + j;
            // left、right 为原轴索引；比较分数后比较索引，NaN 和正负零遵循普通 ONNX 路径。
            int Compare(int left, int right)
            {
                float a = data[offset + left * inner], b = data[offset + right * inner];
                // 保持先按值、再按原索引排序的规则。NaN 优先，
                // 相等值和正负零由原索引确定稳定顺序。
                int order;
                if (float.IsNaN(a)) order = float.IsNaN(b) ? 0 : -1;
                else if (float.IsNaN(b)) order = 1;
                else { order = a.CompareTo(b); if (largest) order = -order; }
                return order == 0 ? left.CompareTo(right) : order;
            }
            // root 为待下沉堆节点，count 为堆长度；维护最差入选项位于根的固定大小堆。
            void Down(int root, int count)
            {
                while (true)
                {
                    int child = root * 2 + 1; if (child >= count) return;
                    if (child + 1 < count && Compare(selected[child + 1], selected[child]) > 0) child++;
                    if (Compare(selected[root], selected[child]) >= 0) return;
                    (selected[root], selected[child]) = (selected[child], selected[root]); root = child;
                }
            }
            for (int index = 0; index < k; index++) selected[index] = index;
            for (int root = k / 2 - 1; root >= 0; root--) Down(root, k);
            for (int candidate = k; candidate < size; candidate++)
                if (Compare(candidate, selected[0]) < 0) { selected[0] = candidate; Down(0, k); }
            // 仅对入选的 k 个索引排序，相同值的次序仍保持确定。
            Array.Sort(selected, Compare);
            for (int t = 0; t < k; t++) { int dst = (o * k + t) * inner + j; valueData[dst] = data[offset + selected[t] * inner]; indexData[dst] = selected[t]; }
        }
        return [values, indices];
    }
}
