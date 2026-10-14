using System.Runtime.CompilerServices;

namespace SimdYolo;

/// <summary>按只读权重批次的数组身份缓存打包结果，避免逐帧重新查询每个 Winograd 坐标的缓存并分配结果数组。</summary>
/// <typeparam name="TMatrix">单个权重矩阵的打包结果类型。</typeparam>
internal static class CpuWeightBatch<TMatrix> where TMatrix : class
{
    /// <summary>以源批次数组为弱键保存打包结果；源批次不再使用后，缓存不会阻止其回收。</summary>
    private static readonly ConditionalWeakTable<float[][], Entry> Cache = new();

    /// <summary>保存已验证的逻辑形状、源矩阵引用快照和打包矩阵。</summary>
    /// <param name="Sources">各源矩阵的引用快照，用于发现批次内数组被替换的情况。</param>
    /// <param name="Rows">每个矩阵的逻辑行数。</param>
    /// <param name="Depth">每个矩阵的归约深度。</param>
    /// <param name="Matrices">按源批次顺序保存的打包矩阵。</param>
    /// <remarks>各构造参数对应同名自动生成属性；源矩阵的元素必须保持只读。</remarks>
    private sealed record Entry(float[][] Sources, int Rows, int Depth, TMatrix[] Matrices);

    /// <summary>取得同一只读权重批次的打包结果；逻辑形状或任一源矩阵引用变化时重新准备。</summary>
    /// <param name="sources">按变换坐标或批次顺序排列的只读权重矩阵。</param>
    /// <param name="rows">每个源矩阵的逻辑行数。</param>
    /// <param name="depth">每个源矩阵的归约深度。</param>
    /// <param name="prepare">将一个源矩阵打包为指定行数和归约深度的回调。</param>
    /// <returns>顺序与源批次一致的打包矩阵数组；调用方不得修改此共享数组。</returns>
    /// <exception cref="ArgumentException">矩阵形状或缓冲区长度不满足单矩阵打包回调的要求。</exception>
    /// <exception cref="OverflowException">单矩阵打包回调的受检查元素数量运算溢出。</exception>
    internal static TMatrix[] Prepare(float[][] sources, int rows, int depth, Func<float[], int, int, TMatrix> prepare)
    {
        if (Cache.TryGetValue(sources, out var entry) && Matches(entry, sources, rows, depth)) return entry.Matrices;
        lock (Cache)
        {
            if (Cache.TryGetValue(sources, out entry) && Matches(entry, sources, rows, depth)) return entry.Matrices;
            var matrices = new TMatrix[sources.Length];
            for (int index = 0; index < sources.Length; index++) matrices[index] = prepare(sources[index], rows, depth);
            entry = new((float[][])sources.Clone(), rows, depth, matrices);
            Cache.Remove(sources);
            Cache.Add(sources, entry);
            return matrices;
        }
    }

    /// <summary>检查逻辑形状和源矩阵引用，保证同一批次数组被重新解释或替换元素时不会返回过期结果。</summary>
    /// <param name="entry">已经缓存的源引用快照与打包矩阵。</param>
    /// <param name="sources">当前调用提供的源权重批次。</param>
    /// <param name="rows">当前逻辑行数。</param>
    /// <param name="depth">当前归约深度。</param>
    /// <returns>形状及每个源矩阵引用全部匹配时为 true。</returns>
    private static bool Matches(Entry entry, float[][] sources, int rows, int depth)
    {
        if (entry.Rows != rows || entry.Depth != depth || entry.Sources.Length != sources.Length) return false;
        for (int index = 0; index < sources.Length; index++)
            if (!ReferenceEquals(entry.Sources[index], sources[index])) return false;
        return true;
    }
}
