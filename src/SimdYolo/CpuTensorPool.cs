namespace SimdYolo;

/// <summary>维护推理会话专用的精确长度缓冲区；公开输出在返回前永久脱离池以避免被后续推理覆盖。</summary>
internal sealed class CpuTensorPool
{
    /// <summary>当前推理调用线程使用的张量池；工作线程不继承该作用域。</summary>
    [ThreadStatic] internal static CpuTensorPool? Current;
    /// <summary>当前会话或后端的 CPU 并行配置。</summary>
    /// <value>当前会话或后端的 CPU 并行配置。</value>
    internal ParallelOptions Parallel { get; set; } = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };
    /// <summary>按精确数组长度分组的空闲缓冲区栈。</summary>
    private readonly Dictionary<int, Stack<float[]>> available = [];
    /// <summary>仍由推理会话拥有的租用数组集合。</summary>
    private readonly HashSet<float[]> leased = new(ReferenceEqualityComparer.Instance);
    /// <summary>图边界扫描中仍被可达张量引用的数组集合。</summary>
    private readonly HashSet<float[]> live = new(ReferenceEqualityComparer.Instance);
    /// <summary>图边界扫描后待归还的数组列表。</summary>
    private readonly List<float[]> release = [];
    /// <summary>空闲缓存数组的总字节数，最多保留 256 MiB。</summary>
    private long retainedBytes;
    /// <summary>租用精确长度的数组并登记所有权，可选择清零；匹配的已归还数组优先复用。</summary>
    /// <param name="count">所需 FP32 元素数，返回数组长度与此值完全一致。</param>
    /// <param name="clear">是否将租用数组清零。</param>
    /// <returns>满足当前布局或长度要求的连续 FP32 数组。</returns>
    public float[] Rent(int count, bool clear = true)
    {
        float[] buffer;
        if (available.TryGetValue(count, out var stack) && stack.Count > 0) { buffer = stack.Pop(); retainedBytes -= (long)count * sizeof(float); }
        else buffer = clear ? new float[count] : GC.AllocateUninitializedArray<float>(count);
        if (clear) Array.Clear(buffer); leased.Add(buffer); return buffer;
    }
    /// <summary>将公开输出永久移出租用集合，重复输出别名复制为独立数组，保护跨帧输出。</summary>
    /// <param name="tensors">需要返回或拼接的张量序列。</param>
    /// <returns>已脱离池且重复别名具有独立数组的同一个输出序列。</returns>
    public Tensor[] DetachOutputs(Tensor[] tensors)
    {
        // 计算完成的公开输出数组直接转移给调用方，并永久脱离池。
        // 后续推理因此不能覆盖已返回结果；同一输出数组的重复别名复制成独立数组。
        var transferred = new HashSet<float[]>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < tensors.Length; i++)
        {
            Tensor tensor = tensors[i]; float[] buffer = tensor.Data;
            if (leased.Remove(buffer)) transferred.Add(buffer);
            else if (transferred.Contains(buffer)) tensors[i] = new Tensor((float[])buffer.Clone(), tensor.Shape);
        }
        return tensors;
    }
    // 仅在图/层边界且全部并行内核结束后调用。
    // 以数组引用登记存活关系，不能以 Tensor 对象标识判断，因为 Reshape 可能共享底层存储。
    /// <summary>在所有内核完成的图边界扫描仍可达的张量数组，归还不再被引用的临时数组。</summary>
    /// <param name="roots">图中仍可达的张量或对象集合。</param>
    public void RetainOnly(IEnumerable<object?> roots)
    {
        live.Clear(); release.Clear();
        // 递归扫描张量及对象集合，登记仍可达的底层数组引用。
        // 参数 value：待扫描的张量、对象集合或其他对象。
        void Visit(object? value)
        {
            if (value is Tensor tensor) live.Add(tensor.Data);
            else if (value is IEnumerable<object?> items) foreach (var item in items) Visit(item);
        }
        foreach (var root in roots) Visit(root);
        foreach (var buffer in leased) if (!live.Contains(buffer)) release.Add(buffer);
        foreach (var buffer in release) { leased.Remove(buffer); Return(buffer); }
        live.Clear(); release.Clear();
    }
    /// <summary>将数组按精确长度入栈，仅在保留总量不超过 256 MiB 时缓存。</summary>
    /// <param name="buffer">需要归还的临时数组。</param>
    private void Return(float[] buffer)
    {
        long bytes = (long)buffer.Length * sizeof(float);
        if (bytes > 256L * 1024 * 1024 - retainedBytes) return;
        if (!available.TryGetValue(buffer.Length, out var stack)) available[buffer.Length] = stack = [];
        stack.Push(buffer); retainedBytes += bytes;
    }
    /// <summary>将仍由会话持有的所有临时数组归还缓存，并清空租用集合。</summary>
    public void ReturnAll()
    {
        foreach (var buffer in leased) Return(buffer);
        leased.Clear();
    }
    /// <summary>清空所有池化集合和保留字节计数，释放会话对缓存数组的引用。</summary>
    public void Clear() { available.Clear(); leased.Clear(); live.Clear(); release.Clear(); retainedBytes = 0; }
}
