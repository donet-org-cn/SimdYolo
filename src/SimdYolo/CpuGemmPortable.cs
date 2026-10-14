using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SimdYolo;

/// <summary>使用 System.Numerics.Vector 实现可移植矩阵乘法，以连续 K 面板改善缓存访问。</summary>
internal static class CpuGemmPortable
{
    // 将 B 的列面板沿 K 轴连续打包。
    // 归约时若按 N 跨度反复访问原 B，容易造成缓存组冲突与 TLB 未命中。
    /// <summary>计算行主序 FP32 矩阵乘法，按连续空间面板复用输入缓存。</summary>
    /// <param name="a">行主序左矩阵 A[M,K]。</param>
    /// <param name="b">行主序右矩阵 B[K,N]；微内核方法中为已打包的 K×面板列数布局。</param>
    /// <param name="c">预分配的行主序输出矩阵 C[M,N]。</param>
    /// <param name="m">左矩阵与输出矩阵的行数 M。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="k">矩阵乘法的归约长度 K。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    public static void Multiply(float[] a, float[] b, float[] c, int m, int n, int k, ParallelOptions parallel)
    {
        if (m < 0 || n < 0 || k < 0 || a.Length < checked(m * k) || b.Length < checked(k * n) || c.Length < checked(m * n)) throw new ArgumentException("Invalid GEMM buffer.");
        if (m == 0 || n == 0) return;
        if (k == 0) { Array.Clear(c, 0, m * n); return; }
        int width = Vector<float>.Count, columns = width * 2, panels = (n + columns - 1) / columns;
        var packed = ArrayPool<float>.Shared.Rent(checked(panels * k * columns));
        try
        {
            // 将当前 B 列面板沿 K 轴连续写入打包缓冲区。
            // 参数 panel：当前输入空间面板的索引。
            void Pack(int panel)
            {
                int column = panel * columns, take = Math.Min(columns, n - column), target = panel * k * columns;
                for (int p = 0; p < k; p++)
                {
                    Array.Copy(b, p * n + column, packed, target + p * columns, take);
                    if (take < columns) Array.Clear(packed, target + p * columns + take, columns - take);
                }
            }
            if ((long)n * k > 262144) CpuScheduling.For(0, panels, parallel, Pack);
            else for (int panel = 0; panel < panels; panel++) Pack(panel);
            // 计算当前输出行块与 B 列面板的乘积，并写入对应结果矩阵区域。
            // 参数 tile：当前矩阵行块与列面板组合的任务索引。
            void Tile(int tile)
            {
                int panel = tile % panels, row = tile / panels * 4, column = panel * columns;
                MultiplyRows(a, packed, c, m, n, k, row, column, panel * k * columns, 0, 0);
            }
            int tiles = ((m + 3) / 4) * panels;
            if ((long)m * n * k > 262144) CpuScheduling.For(0, tiles, parallel, Tile);
            else for (int tile = 0; tile < tiles; tile++) Tile(tile);
        }
        finally { ArrayPool<float>.Shared.Return(packed); }
    }

    /// <summary>为已验证的共享数组批次逐列打包，直接读取各输入偏移并写入各输出偏移，保留原有四行归约内核。</summary>
    /// <param name="a">保存全部左矩阵的源数组。</param>
    /// <param name="b">保存全部右矩阵的源数组。</param>
    /// <param name="c">连续保存各批次输出的目标数组。</param>
    /// <param name="aOffsets">已经验证范围的左矩阵起始偏移。</param>
    /// <param name="bOffsets">已经验证范围的右矩阵起始偏移。</param>
    /// <param name="m">每个左矩阵和输出矩阵的行数。</param>
    /// <param name="n">每个右矩阵和输出矩阵的列数。</param>
    /// <param name="k">每个矩阵乘积的归约深度；调用方已经处理零深度。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    /// <exception cref="OverflowException">面板、工作项或临时缓冲区元素数量溢出。</exception>
    /// <exception cref="AggregateException">并行回调或临时工作区清理失败。</exception>
    /// <exception cref="OperationCanceledException">并行调度的取消令牌已请求取消。</exception>
    /// <remarks>只由 CpuGemm 在批次数、矩阵范围和输出别名全部验证后调用。</remarks>
    internal static void MultiplyOffsetsBatchCore(float[] a, float[] b, float[] c, int[] aOffsets, int[] bOffsets,
                                                   int m, int n, int k, ParallelOptions parallel)
    {
        int columns = Vector<float>.Count * 2, panels = checked((int)(((long)n + columns - 1) / columns));
        int work = checked(aOffsets.Length * panels), mn = checked(m * n);
        // 每个参与线程只保存当前 K×列面板；完整 A、B、C 都使用原数组中的范围。
        void Panel(int index, float[] packed)
        {
            int batch = index / panels, column = index % panels * columns, take = Math.Min(columns, n - column);
            for (int p = 0; p < k; p++)
            {
                Array.Copy(b, bOffsets[batch] + p * n + column, packed, p * columns, take);
                if (take < columns) Array.Clear(packed, p * columns + take, columns - take);
            }
            for (int row = 0; row < m; row += 4)
                MultiplyRows(a, packed, c, m, n, k, row, column, 0, aOffsets[batch], batch * mn);
        }
        if ((long)aOffsets.Length * m * n * k > 262144 && parallel.MaxDegreeOfParallelism != 1)
            CpuScheduling.For(0, work, parallel, () => ArrayPool<float>.Shared.Rent(checked(k * columns)),
                (index, _, packed) => { Panel(index, packed); return packed; },
                packed => ArrayPool<float>.Shared.Return(packed));
        else
        {
            var packed = ArrayPool<float>.Shared.Rent(checked(k * columns));
            try { for (int index = 0; index < work; index++) Panel(index, packed); }
            finally { ArrayPool<float>.Shared.Return(packed); }
        }
    }

    /// <summary>计算一个最多四行的输出块；逐 K 的 SIMD 乘加及尾部写入顺序与原有可移植内核相同。</summary>
    /// <param name="a">行主序左矩阵的源数组。</param>
    /// <param name="packed">按 K×双向量列宽组织的右矩阵面板。</param>
    /// <param name="c">行主序输出矩阵的目标数组。</param>
    /// <param name="m">逻辑输出行数。</param>
    /// <param name="n">输出行跨度。</param>
    /// <param name="k">递增归约的深度。</param>
    /// <param name="row">本次输出块的起始行。</param>
    /// <param name="column">本次输出块的起始列。</param>
    /// <param name="packedOffset">当前右矩阵面板在打包数组中的起始偏移。</param>
    /// <param name="aOffset">当前左矩阵在源数组中的起始偏移。</param>
    /// <param name="cOffset">当前输出矩阵在目标数组中的起始偏移。</param>
    private static void MultiplyRows(float[] a, float[] packed, float[] c, int m, int n, int k, int row, int column,
                                     int packedOffset, int aOffset, int cOffset)
    {
        int width = Vector<float>.Count, columns = width * 2, rows = Math.Min(4, m - row);
        ref float ar = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(a), aOffset);
        ref float br = ref MemoryMarshal.GetArrayDataReference(packed);
        ref float cr = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(c), cOffset);
        var c00 = Vector<float>.Zero; var c01 = c00; var c10 = c00; var c11 = c00;
        var c20 = c00; var c21 = c00; var c30 = c00; var c31 = c00;
        if (rows == 4)
        {
            for (int p = 0; p < k; p++)
            {
                var b0 = Vector.LoadUnsafe(ref br, (nuint)(packedOffset + p * columns));
                var b1 = Vector.LoadUnsafe(ref br, (nuint)(packedOffset + p * columns + width));
                var a0 = new Vector<float>(Unsafe.Add(ref ar, row * k + p));
                var a1 = new Vector<float>(Unsafe.Add(ref ar, (row + 1) * k + p));
                var a2 = new Vector<float>(Unsafe.Add(ref ar, (row + 2) * k + p));
                var a3 = new Vector<float>(Unsafe.Add(ref ar, (row + 3) * k + p));
                c00 = Vector.MultiplyAddEstimate(a0, b0, c00); c01 = Vector.MultiplyAddEstimate(a0, b1, c01);
                c10 = Vector.MultiplyAddEstimate(a1, b0, c10); c11 = Vector.MultiplyAddEstimate(a1, b1, c11);
                c20 = Vector.MultiplyAddEstimate(a2, b0, c20); c21 = Vector.MultiplyAddEstimate(a2, b1, c21);
                c30 = Vector.MultiplyAddEstimate(a3, b0, c30); c31 = Vector.MultiplyAddEstimate(a3, b1, c31);
            }
            Store(c00, c01, ref cr, row * n + column, n - column);
            Store(c10, c11, ref cr, (row + 1) * n + column, n - column);
            Store(c20, c21, ref cr, (row + 2) * n + column, n - column);
            Store(c30, c31, ref cr, (row + 3) * n + column, n - column);
        }
        else for (int r = 0; r < rows; r++)
        {
            var s0 = Vector<float>.Zero; var s1 = s0;
            for (int p = 0; p < k; p++)
            {
                var av = new Vector<float>(Unsafe.Add(ref ar, (row + r) * k + p));
                s0 = Vector.MultiplyAddEstimate(av, Vector.LoadUnsafe(ref br, (nuint)(packedOffset + p * columns)), s0);
                s1 = Vector.MultiplyAddEstimate(av, Vector.LoadUnsafe(ref br, (nuint)(packedOffset + p * columns + width)), s1);
            }
            Store(s0, s1, ref cr, (row + r) * n + column, n - column);
        }
    }
    /// <summary>仅将当前 SIMD 向量中的有效通道写入目标缓冲区。</summary>
    /// <param name="low">结果的低位 SIMD 通道。</param>
    /// <param name="high">结果的高位 SIMD 通道。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="offset">打包数据在缓冲区中的起始偏移。</param>
    /// <param name="remaining">输出中仍可写入的有效元素数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(Vector<float> low, Vector<float> high, ref float output, int offset, int remaining)
    {
        int width = Vector<float>.Count;
        if (remaining >= width * 2) { low.StoreUnsafe(ref output, (nuint)offset); high.StoreUnsafe(ref output, (nuint)(offset + width)); }
        else for (int i = 0; i < remaining; i++) Unsafe.Add(ref output, offset + i) = i < width ? low[i] : high[i - width];
    }
}
