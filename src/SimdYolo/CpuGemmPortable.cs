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
                int panel = tile % panels, row = tile / panels * 4, column = panel * columns, rows = Math.Min(4, m - row), bp = panel * k * columns;
                ref float ar = ref MemoryMarshal.GetArrayDataReference(a);
                ref float br = ref MemoryMarshal.GetArrayDataReference(packed);
                ref float cr = ref MemoryMarshal.GetArrayDataReference(c);
                var c00 = Vector<float>.Zero; var c01 = c00; var c10 = c00; var c11 = c00;
                var c20 = c00; var c21 = c00; var c30 = c00; var c31 = c00;
                if (rows == 4)
                {
                    for (int p = 0; p < k; p++)
                    {
                        var b0 = Vector.LoadUnsafe(ref br, (nuint)(bp + p * columns));
                        var b1 = Vector.LoadUnsafe(ref br, (nuint)(bp + p * columns + width));
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
                        s0 = Vector.MultiplyAddEstimate(av, Vector.LoadUnsafe(ref br, (nuint)(bp + p * columns)), s0);
                        s1 = Vector.MultiplyAddEstimate(av, Vector.LoadUnsafe(ref br, (nuint)(bp + p * columns + width)), s1);
                    }
                    Store(s0, s1, ref cr, (row + r) * n + column, n - column);
                }
            }
            int tiles = ((m + 3) / 4) * panels;
            if ((long)m * n * k > 262144) CpuScheduling.For(0, tiles, parallel, Tile);
            else for (int tile = 0; tile < tiles; tile++) Tile(tile);
        }
        finally { ArrayPool<float>.Shared.Return(packed); }
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
