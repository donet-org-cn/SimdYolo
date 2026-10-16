namespace SimdYolo;

/// <summary>定义帧内设备张量的分配、计算、同步和主机数据物化契约。</summary>
internal interface IDeviceBackend
{
    /// <summary>取得当前后端是否仍在执行活动推理帧。</summary>
    /// <value>帧开始后为 true，结束或失败清理后为 false。</value>
    bool FrameActive { get; }
    /// <summary>取得当前设备推理帧编号，用于校验设备区段有效期。</summary>
    /// <value>每次 BeginFrame 递增的帧标识。</value>
    long FrameId { get; }
    /// <summary>开始一帧设备执行并建立有效的帧编号与设备作用域。</summary>
    /// <exception cref="ObjectDisposedException">后端已经释放。</exception>
    /// <exception cref="InvalidOperationException">当前已有帧在录制，或 Vulkan 描述符、查询池、命令操作失败。</exception>
    void BeginFrame();
    /// <summary>完成当前帧并清理设备作用域；调用方应在 finally 中执行。</summary>
    void EndFrame();
    /// <summary>提交尚未完成的设备命令并同步，供 CPU 读取或跨算子计时使用。</summary>
    void Flush();
    /// <summary>通知后端当前张量的主机数据即将被读取；默认实现不做处理。</summary>
    /// <param name="tensor">待上传或观察主机读取行为的张量。</param>
    void ObserveHostData(Tensor tensor) { }
    /// <summary>校验设备张量所属帧，录制下载拷贝并同步返回独立主机数组。</summary>
    /// <param name="storage">待下载的设备区段与帧归属。</param>
    /// <returns>下载完成后的独立 FP32 数组，不借用 GPU 映射内存。</returns>
    /// <exception cref="InvalidOperationException">设备视图不属于当前后端活动帧，或 Vulkan 下载提交失败。</exception>
    float[] Download(DeviceStorage storage);
    /// <summary>在帧结束前物化输出主机数组；默认实现逐个读取 Data，后端可覆盖为批量下载。</summary>
    /// <param name="outputs">当前帧的最终输出张量数组。</param>
    /// <exception cref="InvalidOperationException">输出属于其他后端或已结束的帧，或下载提交失败。</exception>
    /// <exception cref="OverflowException">合并下载区的字节数溢出。</exception>
    /// <remarks>每次推理的公开输出使用独立主机数组；后续帧复用上传/下载区不会改写已返回的数据。</remarks>
    void MaterializeOutputs(Tensor[] outputs) { foreach (var output in outputs) _ = output.Data; }
    /// <summary>按算子编号与指定输出形状调度设备计算，返回帧内设备结果张量。</summary>
    /// <param name="operation">与设备着色器特化常量约定一致的算子编号。</param>
    /// <param name="inputs">操作输入张量，最多使用三个数据输入。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="parameters">与着色器约定一致的 32 个 uint 推送常量槽位。</param>
    /// <param name="workItems">着色器需要覆盖的逻辑工作项数。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    Tensor Compute(int operation, Tensor[] inputs, int[] shape, uint[] parameters, int workItems);
    /// <summary>按指定 NCHW 形状与卷积参数计算设备结果，允许融合偏置和 SiLU。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="stride">高、宽两个方向的卷积步长。</param>
    /// <param name="pads">上、左、下、右方向的填充量。</param>
    /// <param name="dilation">高、宽方向的卷积膨胀系数。</param>
    /// <param name="groups">卷积分组数，输入和输出通道须满足该分组布局。</param>
    /// <param name="silu">是否在偏置相加后融合 SiLU 激活。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <exception cref="InvalidOperationException">通用卷积的输入、权重或偏置未对齐，或 Vulkan 管线与设备调度失败。</exception>
    Tensor DeviceConv(Tensor x, Tensor w, Tensor? bias, int[] shape, int[] stride, int[] pads, int[] dilation, int groups, bool silu);
    /// <summary>执行最后两维为矩阵、前置维度允许广播的设备矩阵乘法。</summary>
    /// <param name="a">左侧矩阵张量，最后两维为 m × k。</param>
    /// <param name="b">右侧矩阵张量，最后两维为 k × n。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <exception cref="NotSupportedException">任一矩阵维度为 0，或调度超过可移植线程组上限。</exception>
    Tensor DeviceMatMul(Tensor a, Tensor b, int[] shape);
    /// <summary>沿指定轴按输入顺序拼接设备张量。</summary>
    /// <param name="tensors">沿同一轴按顺序拼接的输入张量集合。</param>
    /// <param name="axis">已经归一化的处理轴索引。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    Tensor DeviceConcat(IReadOnlyList<Tensor> tensors, int axis, int[] shape);
    /// <summary>切取指定轴的连续区段；可安全对齐时借用原设备区段，否则调度切片着色器。</summary>
    /// <param name="x">当前运算的输入张量，元素按连续行主序存储。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="axis">已经归一化的处理轴索引。</param>
    /// <param name="start">切片沿指定轴的起点。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <remarks>借用视图共享底层缓冲区，不单独释放；字节偏移不等于 FP32 元素偏移。</remarks>
    Tensor DeviceSlice(Tensor x, int[] shape, int axis, int start);
}

/// <summary>描述设备缓冲区中的 FP32 区段；只保存借用视图，底层资源由所属后端释放。</summary>
/// <param name="Owner">分配和释放底层资源的设备后端。</param>
/// <param name="Buffer">后端持有的设备缓冲区对象；此视图不取得独立所有权。</param>
/// <param name="Offset">缓冲区内的字节偏移。</param>
/// <param name="Length">区段中的 FP32 元素数。</param>
/// <param name="FrameId">此设备视图有效的推理帧编号。</param>
internal sealed record DeviceStorage(IDeviceBackend Owner, object Buffer, ulong Offset, int Length, long FrameId);
/// <summary>保存当前线程的设备执行上下文，并根据张量所属帧选择可用后端。</summary>
internal static class DeviceScope
{
    /// <summary>保存当前线程的活动设备后端；不会自动传递给工作线程。</summary>
    [ThreadStatic] public static IDeviceBackend? Current;
    /// <summary>优先取得张量当前帧的所属后端，否则在当前线程的活动设备作用域中选择后端。</summary>
    /// <param name="tensors">待检查帧归属的输入张量。</param>
    /// <returns>可用于当前帧的设备后端；没有可用设备作用域时为 null。</returns>
    public static IDeviceBackend? For(params Tensor[] tensors)
    {
        foreach (var tensor in tensors) if (tensor.Storage is { } s && s.Owner.FrameActive && s.FrameId == s.Owner.FrameId) return s.Owner;
        return Current is { FrameActive: true } owner && tensors.Any(t => t.Length >= 256) ? owner : null;
    }
}

/// <summary>把张量算子的形状、广播步长和 FP32 常量编码为设备着色器参数。</summary>
internal static class DeviceOps
{
    /// <summary>把操作参数复制到固定 32 个 uint 的推送常量区，未指定的槽位保持零。</summary>
    /// <param name="values">需要复制到参数区的 uint 值。</param>
    /// <returns>长度为 32 的新参数数组。</returns>
    /// <exception cref="ArgumentException">传入参数数量超过 32，无法复制到固定参数区。</exception>
    public static uint[] Parameters(params uint[] values)
    { var result = new uint[32]; values.CopyTo(result, 0); return result; }
    /// <summary>保留 FP32 的位模式，把标量编码为着色器 uint 参数。</summary>
    /// <param name="value">需要保留位模式的 FP32 标量。</param>
    /// <returns>与输入浮点数逐位相同的无符号整数。</returns>
    public static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);
    /// <summary>按输入形状调度设备一元运算，并编码可选缩放因子。</summary>
    /// <param name="owner">拥有设备资源且当前帧处于活动状态的后端。</param>
    /// <param name="x">当前运算的输入张量，元素按连续行主序存储。</param>
    /// <param name="operation">与设备着色器特化常量约定一致的算子编号。</param>
    /// <param name="scale">设备一元操作的缩放系数。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    public static Tensor Unary(IDeviceBackend owner, Tensor x, int operation, float scale = 1)
        => owner.Compute(operation, [x], x.Shape, Parameters((uint)x.Length, Bits(scale)), x.Length);
    /// <summary>计算广播形状，把最多四维的广播步长编码为设备二元运算参数。</summary>
    /// <param name="owner">拥有设备资源且当前帧处于活动状态的后端。</param>
    /// <param name="a">参与广播的第一输入张量。</param>
    /// <param name="b">参与广播的第二输入张量。</param>
    /// <param name="operation">与设备着色器特化常量约定一致的算子编号。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <exception cref="NotSupportedException">广播后的张量秩超过 4。</exception>
    public static Tensor Binary(IDeviceBackend owner, Tensor a, Tensor b, int operation)
    {
        int[] shape = TensorOps.BroadcastShape(a.Shape, b.Shape);
        if (shape.Length > 4) throw new NotSupportedException("Device broadcast rank > 4.");
        var p = Parameters((uint)Tensor.Count(shape));
        var ad = Enumerable.Repeat(1, 4 - a.Rank).Concat(a.Shape).ToArray();
        var bd = Enumerable.Repeat(1, 4 - b.Rank).Concat(b.Shape).ToArray();
        var yd = Enumerable.Repeat(1, 4 - shape.Length).Concat(shape).ToArray();
        int[] ast = Tensor.Strides(ad), bst = Tensor.Strides(bd);
        for (int d = 0; d < 4; d++) { p[1 + d] = (uint)yd[d]; p[5 + d] = ad[d] == 1 ? 0 : (uint)ast[d]; p[9 + d] = bd[d] == 1 ? 0 : (uint)bst[d]; }
        return owner.Compute(operation, [a, b], shape, p, Tensor.Count(shape));
    }
    /// <summary>沿指定轴调度设备 Softmax，以其余维度的乘积作为独立归一化工作项数。</summary>
    /// <param name="owner">拥有设备资源且当前帧处于活动状态的后端。</param>
    /// <param name="x">当前运算的输入张量，元素按连续行主序存储。</param>
    /// <param name="axis">已经归一化的处理轴索引。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    public static Tensor Softmax(IDeviceBackend owner, Tensor x, int axis)
    {
        int inner = Tensor.Count(x.Shape[(axis + 1)..]), outer = Tensor.Count(x.Shape[..axis]);
        return owner.Compute(25, [x], x.Shape, Parameters((uint)(inner * outer), (uint)x.Shape[axis], (uint)inner), inner * outer);
    }
    /// <summary>把最多四维的维度置换编码为输入步长，并调度设备转置。</summary>
    /// <param name="owner">拥有设备资源且当前帧处于活动状态的后端。</param>
    /// <param name="x">当前运算的输入张量，元素按连续行主序存储。</param>
    /// <param name="perm">输出各轴对应的输入轴排列。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <exception cref="NotSupportedException">输入张量秩超过 4。</exception>
    public static Tensor Transpose(IDeviceBackend owner, Tensor x, int[] perm, int[] shape)
    {
        if (owner is OnnxTraceBackend trace) return trace.Transpose(x, perm, shape);
        if (x.Rank > 4) throw new NotSupportedException("Device transpose rank > 4.");
        var p = Parameters((uint)x.Length); int lead = 4 - x.Rank; int[] strides = Tensor.Strides(x.Shape);
        for (int i = 0; i < 4; i++) { p[1 + i] = i < lead ? 1 : (uint)shape[i - lead]; p[5 + i] = i < lead ? 0 : (uint)strides[perm[i - lead]]; }
        return owner.Compute(23, [x], shape, p, x.Length);
    }
    /// <summary>在设备端执行 DFL 边框解码并合并类别分数，生成批次、字段、锚点布局。</summary>
    /// <param name="boxes">含四个边距分布的 NCHW 回归输出。</param>
    /// <param name="classes">NCHW 类别分数张量，与边框输出具有相同空间尺寸。</param>
    /// <param name="bins">每个边距使用的 DFL 分布桶数。</param>
    /// <param name="stride">特征图相对于网络输入图像的采样步长。</param>
    /// <param name="xyxy">是否输出左右上下坐标；false 表示中心坐标与宽高。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <remarks>调用前必须存在活动设备作用域；此内部入口不负责建立或验证作用域。</remarks>
    public static Tensor DecodeHead(Tensor boxes, Tensor classes, int bins, float stride, bool xyxy)
    {
        var owner = DeviceScope.For(boxes, classes)!; int n = boxes.Shape[0], h = boxes.Shape[2], w = boxes.Shape[3], nc = classes.Shape[1];
        return owner.Compute(30, [boxes, classes], [n, nc + 4, h * w], Parameters((uint)(n * h * w), (uint)(h * w), (uint)w, (uint)bins, (uint)nc, Bits(stride), xyxy ? 1u : 0u), n * h * w);
    }
    /// <summary>在设备端恢复关键点坐标和可选置信度，输出按关键点字段展平的张量。</summary>
    /// <param name="x">按 [批次, 关键点数 × 字段数, 高, 宽] 排列的姿态头张量。</param>
    /// <param name="count">每个检测的关键点数量。</param>
    /// <param name="dimensions">每个关键点字段数，通常为 2 个坐标或 3 个含置信度字段。</param>
    /// <param name="stride">特征图相对于网络输入图像的采样步长。</param>
    /// <param name="yolo26">是否使用 YOLO26 的单倍偏移和半像素锚点。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <remarks>调用前必须存在活动设备作用域；此内部入口不负责建立或验证作用域。</remarks>
    public static Tensor DecodePose(Tensor x, int count, int dimensions, float stride, bool yolo26 = false)
    {
        int spatial = x.Shape[2] * x.Shape[3];
        return DeviceScope.For(x)!.Compute(31, [x], [x.Shape[0], count * dimensions, spatial], Parameters((uint)x.Length, (uint)spatial, (uint)x.Shape[3], (uint)dimensions, Bits(stride), yolo26 ? 1u : 0u), x.Length);
    }
}
