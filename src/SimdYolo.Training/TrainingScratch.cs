namespace SimdYolo.Training;

/// <summary>复用长度精确匹配的卷积临时矩阵；每个训练会话最多保留 32 MiB，避免大图永久占用工作区。</summary>
internal sealed class TrainingScratch
{
    /// <summary>空闲矩阵的总保留字节上限，不限制当前计算必须使用的矩阵。</summary>
    private const long Limit = 32L * 1024 * 1024;
    /// <summary>按元素数量区分工作区，保证 Tensor 所需的精确长度。</summary>
    private readonly Dictionary<int, Stack<float[]>> available = [];
    /// <summary>当前空闲工作区的字节数。</summary>
    private long retained;

    /// <summary>借用临时矩阵；所有调用均位于会话线程，工作线程只读取或写入已分配数组。</summary>
    /// <param name="length">矩阵元素数量。</param><param name="clear">是否清零填边和尚未写入的区域。</param>
    /// <returns>退出 using 作用域时自动归还的租约。</returns>
    internal Lease Rent(int length, bool clear = false)
    {
        float[] data;
        if (available.TryGetValue(length, out var stack) && stack.TryPop(out var cached)) { data = cached; retained -= (long)length * 4; if (clear) Array.Clear(data); }
        else data = new float[length];
        return new(this, data);
    }

    /// <summary>归还临时矩阵；超过容量时交回 GC，不永久持有大图缓冲区。</summary>
    /// <param name="data">当前租约独占的数组。</param>
    private void Return(float[] data)
    {
        long bytes = (long)data.Length * 4;
        if (bytes > Limit - retained) return;
        if (!available.TryGetValue(data.Length, out var stack)) available.Add(data.Length, stack = new());
        stack.Push(data); retained += bytes;
    }

    /// <summary>仅管理临时数组的所有权，不将前向输出或参数梯度放入可回收工作区。</summary>
    /// <param name="owner">本数组所属的会话工作区。</param><param name="data">独占的临时矩阵。</param>
    internal sealed class Lease(TrainingScratch owner, float[] data) : IDisposable
    {
        /// <summary>防止重复释放导致同一数组同时被再次借用。</summary>
        private bool returned;
        /// <summary>当前租约独占的数组。</summary><value>使用范围截止第一次 Dispose。</value>
        internal float[] Data { get; } = data;
        /// <summary>将数组归还所属工作区；可重复调用。</summary>
        public void Dispose() { if (!returned) { returned = true; owner.Return(Data); } }
    }
}
