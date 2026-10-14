namespace SimdYolo;

/// <summary>连续行优先 FP32 张量，可持有托管数据或等待下载的设备存储。</summary>
public sealed class Tensor
{
    /// <summary>托管数据缓冲区；仅持有设备存储时为 null。</summary>
    private float[]? data;
    /// <summary>当前设备存储；访问 Data 下载到主机后清除该引用。</summary>
    /// <value>当前设备存储；访问 Data 下载到主机后清除该引用。</value>
    internal DeviceStorage? Storage { get; private set; }
    /// <summary>返回托管数据数组；设备张量首次访问时同步下载，返回的数组允许调用方修改。</summary>
    /// <value>返回托管数据数组；设备张量首次访问时同步下载，返回的数组允许调用方修改。</value>
    /// <exception cref="InvalidOperationException">设备数据下载时帧已结束，或存储的所有者与帧编号不符合当前活动设备帧。</exception>
    public float[] Data
    {
        get
        {
            DeviceScope.Current?.ObserveHostData(this);
            if (data is null) { data = Storage!.Owner.Download(Storage); Storage = null; }
            return data;
        }
    }
    /// <summary>张量维度数组；构造时复制传入形状，调用方应保持它与实际存储一致。</summary>
    /// <value>张量维度数组；构造时复制传入形状，调用方应保持它与实际存储一致。</value>
    public int[] Shape { get; }
    /// <summary>张量逻辑元素数量。</summary>
    /// <value>张量逻辑元素数量。</value>
    public int Length { get; }
    /// <summary>张量维度数量。</summary>
    /// <value>张量维度数量。</value>
    public int Rank => Shape.Length;
    /// <summary>标记不可由推理修改的权重或形状常量，供预打包缓存与存活期管理使用。</summary>
    /// <value>标记不可由推理修改的权重或形状常量，供预打包缓存与存活期管理使用。</value>
    internal bool IsConstant { get; set; }
    /// <summary>按指定形状创建张量；传入数据或设备存储时共享其元素存储，仅复制维度数组。</summary>
    /// <param name="data">连续 FP32 数据数组；构造函数保留该数组引用，不复制元素。</param>
    /// <param name="shape">按行优先顺序声明的张量维度。</param>
    /// <exception cref="ArgumentException">形状包含负维度，或元素总数与数据或设备存储长度不一致。</exception>
    /// <exception cref="OverflowException">维度乘积超出 Int32 范围。</exception>
    public Tensor(float[] data, params int[] shape)
    {
        if (Count(shape) != data.Length) throw new ArgumentException("Tensor shape does not match data.");
        this.data = data; Shape = (int[])shape.Clone(); Length = data.Length;
    }
    /// <summary>按指定形状创建张量；传入数据或设备存储时共享其元素存储，仅复制维度数组。</summary>
    /// <param name="storage">当前后端拥有的设备存储。</param>
    /// <param name="shape">按行优先顺序声明的张量维度。</param>
    /// <exception cref="ArgumentException">形状包含负维度，或元素总数与数据或设备存储长度不一致。</exception>
    /// <exception cref="OverflowException">维度乘积超出 Int32 范围。</exception>
    internal Tensor(DeviceStorage storage, int[] shape)
    { if (Count(shape) != storage.Length) throw new ArgumentException("Device tensor shape mismatch."); Storage = storage; Shape = (int[])shape.Clone(); Length = storage.Length; }
    /// <summary>接收后端已下载的数据，并把张量切换为主机存储。</summary>
    /// <param name="values">需要接收的数据或模板指定的常量序列。</param>
    /// <exception cref="ArgumentException">下载数据长度与张量逻辑长度不一致。</exception>
    internal void AcceptDownloadedData(float[] values)
    {
        if (values.Length != Length) throw new ArgumentException("Downloaded tensor storage mismatch.");
        data = values; Storage = null;
    }
    /// <summary>按指定形状分配独立、全零的连续 FP32 数据数组，并复制形状数组。</summary>
    /// <param name="shape">按行优先顺序声明的非负张量维度。</param>
    /// <exception cref="ArgumentException">形状包含负维度。</exception>
    /// <exception cref="OverflowException">维度乘积超出 Int32 范围。</exception>
    public Tensor(params int[] shape) : this(new float[Count(shape)], shape) { }
    /// <summary>按形状申请 CPU 临时张量，优先复用当前会话缓冲区池。</summary>
    /// <param name="shape">按行优先顺序声明的张量维度。</param>
    /// <returns>未保证清零的临时张量；内核应在读取之前写满逻辑元素。</returns>
    internal static Tensor Temporary(params int[] shape)
    { int count = Count(shape); return new(CpuTensorPool.Current?.Rent(count, clear: false) ?? new float[count], shape); }
    // 仅供会在首次读取前完整覆盖每个逻辑元素的内核使用。
    /// <summary>申请不保证清零的临时张量；仅供会完整覆盖输出的内核使用。</summary>
    /// <param name="shape">按行优先顺序声明的张量维度。</param>
    /// <returns>来自当前池或未初始化数组的张量。</returns>
    internal static Tensor TemporaryUninitialized(params int[] shape)
    { int count = Count(shape); return new(CpuTensorPool.Current?.Rent(count, clear: false) ?? GC.AllocateUninitializedArray<float>(count), shape); }
    /// <summary>创建维度数量为 0、元素数量为 1 的 FP32 标量张量。</summary>
    /// <param name="x">标量的 FP32 值。</param>
    /// <returns>包含指定值的标量张量。</returns>
    public static Tensor Scalar(float x) => new([x], []);
    /// <summary>以受检乘法计算形状各维度的元素总数；空形状表示一个标量。</summary>
    /// <param name="shape">非负维度序列；空序列表示一个标量。</param>
    /// <returns>维度乘积；任一维度为 0 时返回 0。</returns>
    /// <exception cref="ArgumentException">维度包含负数。</exception>
    /// <exception cref="OverflowException">维度乘积超出 Int32 范围。</exception>
    public static int Count(IEnumerable<int> shape)
    {
        int count = 1;
        foreach (int d in shape) { if (d < 0) throw new ArgumentException("Negative tensor dimension."); count = checked(count * d); }
        return count;
    }
    /// <summary>在不复制元素存储的情况下创建新形状视图，允许一个 -1 维度自动推断。</summary>
    /// <param name="shape">目标维度数组，允许一个 -1 根据元素总数推断。</param>
    /// <returns>与当前张量共享数据或设备存储的新视图。</returns>
    /// <exception cref="ArgumentException">存在多个 -1、无法推断尺寸，或目标形状元素数量与原张量不同。</exception>
    /// <exception cref="OverflowException">目标维度乘积超出 Int32 范围。</exception>
    public Tensor Reshape(params int[] shape)
    {
        shape = (int[])shape.Clone();
        int inferred = Array.IndexOf(shape, -1);
        if (inferred >= 0)
        {
            if (shape.Count(x => x == -1) != 1) throw new ArgumentException("Multiple inferred dimensions.");
            shape[inferred] = 1;
            int count = Count(shape);
            if (count == 0 || Length % count != 0) throw new ArgumentException("Invalid reshape.");
            shape[inferred] = Length / count;
        }
        if (Count(shape) != Length) throw new ArgumentException("Tensor shape does not match data.");
        return Storage is { } storage ? new Tensor(storage, shape) { IsConstant = IsConstant } : new Tensor(Data, shape) { IsConstant = IsConstant };
    }
    /// <summary>将负轴编号规范为非负轴编号，并检查是否处于张量维度范围。</summary>
    /// <param name="axis">操作轴；负数表示从最后一维反向计数。</param>
    /// <param name="rank">张量的维度数量。</param>
    /// <returns>0 到 rank - 1 范围内的轴编号。</returns>
    /// <exception cref="ArgumentOutOfRangeException">规范后的轴编号仍超出维度范围。</exception>
    internal static int Axis(int axis, int rank)
    {
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));
        return axis;
    }
    /// <summary>生成连续行优先布局的元素步长。</summary>
    /// <param name="shape">按行优先顺序声明的张量维度。</param>
    /// <returns>与形状维度数量相同的步长数组。</returns>
    /// <exception cref="OverflowException">步长或维度乘积超出 Int32 范围。</exception>
    internal static int[] Strides(int[] shape)
    {
        int[] s = new int[shape.Length]; int n = 1;
        for (int i = s.Length - 1; i >= 0; i--) { s[i] = n; n = checked(n * shape[i]); }
        return s;
    }
    /// <summary>把张量维度格式化为方括号中的逗号分隔文本。</summary>
    /// <returns>例如 [1,3,640,640] 的形状文本。</returns>
    public override string ToString() => $"[{string.Join(',', Shape)}]";
}
