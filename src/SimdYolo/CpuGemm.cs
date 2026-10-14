using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>提供按 16 列打包的行主序矩阵乘法，并根据硬件能力选择 AVX/FMA 或可移植 SIMD。</summary>
internal static class CpuGemm
{
    // 同一个输入空间面板供全部输出通道反复使用，使其尽量驻留在缓存中。
    /// <summary>计算 C[M,N]=A[M,K]×B[K,N]，根据 AVX/FMA 可用性选择微内核或可移植实现。</summary>
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
        if (Avx.IsSupported && Fma.IsSupported) MultiplyCore([a], [b], [c], m, n, k, parallel);
        else CpuGemmPortable.Multiply(a, b, c, m, n, k, parallel);
    }

    /// <summary>计算尺寸一致的一组行主序矩阵乘法，可缓存只读常量权重。</summary>
    /// <param name="a">各批次的行主序左矩阵 A[M,K]。</param>
    /// <param name="b">各批次的行主序右矩阵 B[K,N]。</param>
    /// <param name="c">各批次预分配的行主序输出矩阵 C[M,N]。</param>
    /// <param name="m">左矩阵与输出矩阵的行数 M。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="k">矩阵乘法的归约长度 K。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    /// <param name="constant">权重是否只读，从而允许缓存其打包结果。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    public static void MultiplyBatch(float[][] a, float[][] b, float[][] c, int m, int n, int k, ParallelOptions parallel, bool constant = false)
    {
        if (Avx.IsSupported && Fma.IsSupported) MultiplyCore(a, b, c, m, n, k, parallel, constant);
        else for (int i = 0; i < a.Length; i++) CpuGemmPortable.Multiply(a[i], b[i], c[i], m, n, k, parallel);
    }

    /// <summary>验证矩阵批次数与长度，按 B 的 16 列面板分配任务，并在所有输出行间复用打包输入。</summary>
    /// <param name="a">各批次的行主序左矩阵 A[M,K]。</param>
    /// <param name="b">各批次的行主序右矩阵 B[K,N]。</param>
    /// <param name="c">各批次预分配的行主序输出矩阵 C[M,N]。</param>
    /// <param name="m">左矩阵与输出矩阵的行数 M。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="k">矩阵乘法的归约长度 K。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    /// <param name="constant">权重是否只读，从而允许缓存其打包结果。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    private static void MultiplyCore(float[][] a, float[][] b, float[][] c, int m, int n, int k, ParallelOptions parallel, bool constant = false)
    {
        if (a.Length != b.Length || a.Length != c.Length || m < 0 || n < 0 || k < 0) throw new ArgumentException("Invalid GEMM buffer.");
        for (int t = 0; t < a.Length; t++)
            if (a[t].Length < checked(m * k) || b[t].Length < checked(k * n) || c[t].Length < checked(m * n)) throw new ArgumentException("Invalid GEMM buffer.");
        if (m == 0 || n == 0) return;
        if (k == 0) { foreach (var output in c) Array.Clear(output, 0, m * n); return; }
        var weights = CpuPackedGemm.Use(m) && constant ? a.Select(weight => CpuPackedGemm.Prepare(weight, m, k, true)).ToArray() : null;
        int panels = (n + 15) / 16;
        // 为当前批次打包一个 B 列面板，并计算所有输出行的乘积。
        // 参数 index：工作线程、任务或输出元素的索引。
        // 参数 packed：当前工作线程使用的连续打包缓冲区。
        void Panel(int index, float[] packed)
        {
            int batch = index / panels, column = index % panels * 16, take = Math.Min(16, n - column);
            Pack(b[batch], packed, column, take, n, k);
            if (weights is not null) CpuPackedGemm.Multiply(weights[batch], packed, c[batch], n, column, take);
            else MultiplyAvx(a[batch], packed, c[batch], m, n, k, column, take);
        }
        if ((long)a.Length * m * n * k > 262144 && parallel.MaxDegreeOfParallelism != 1)
            CpuScheduling.For(0, a.Length * panels, parallel,
                () => ArrayPool<float>.Shared.Rent(checked(k * 16)),
                (index, _, scratch) => { Panel(index, scratch); return scratch; },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        else
        {
            var scratch = ArrayPool<float>.Shared.Rent(checked(k * 16));
            try { for (int i = 0; i < a.Length * panels; i++) Panel(i, scratch); }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }
    }

    /// <summary>将 B 的指定列范围打包为 K×16 连续面板，尾部补零以供 SIMD 安全读取。</summary>
    /// <param name="b">行主序右矩阵 B[K,N]；微内核方法中为已打包的 K×面板列数布局。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="k">矩阵乘法的归约长度 K。</param>
    private static unsafe void Pack(float[] b, float[] packed, int column, int take, int n, int k)
    {
        fixed (float* source = b, target = packed)
        {
            if (take == 16)
                for (int p = 0; p < k; p++)
                {
                    Avx.Store(target + p * 16, Avx.LoadVector256(source + p * n + column));
                    Avx.Store(target + p * 16 + 8, Avx.LoadVector256(source + p * n + column + 8));
                }
            else
                for (int p = 0; p < k; p++)
                {
                    for (int j = 0; j < take; j++) target[p * 16 + j] = source[p * n + column + j];
                    for (int j = take; j < 16; j++) target[p * 16 + j] = 0;
                }
        }
    }

    /// <summary>使用 6×16 AVX/FMA 微内核计算当前空间面板，深 K 按 128 项分块并保持递增 K 的累积次序。</summary>
    /// <param name="a">行主序左矩阵 A[M,K]。</param>
    /// <param name="b">行主序右矩阵 B[K,N]；微内核方法中为已打包的 K×面板列数布局。</param>
    /// <param name="c">预分配的行主序输出矩阵 C[M,N]。</param>
    /// <param name="m">左矩阵与输出矩阵的行数 M。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="k">矩阵乘法的归约长度 K。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="bOffset">打包右矩阵在输入缓冲区中的起始偏移。</param>
    /// <param name="cOffset">结果矩阵在输出缓冲区中的起始偏移。</param>
    internal static unsafe void MultiplyAvx(float[] a, float[] b, float[] c, int m, int n, int k, int column, int take, int bOffset = 0, int cOffset = 0)
    {
        fixed (float* ap = a, bStart = b, cStart = c)
        {
            float* bp = bStart + bOffset, cp = cStart + cOffset;
            int depth = take == 16 && k >= 256 && m >= 32 ? 128 : k;
            for (int begin = 0; begin < k; begin += depth)
            {
            int end = Math.Min(k, begin + depth);
            int row = 0;
            for (; row + 6 <= m; row += 6)
            {
                var c00 = Vector256<float>.Zero; var c01 = c00; var c10 = c00; var c11 = c00;
                var c20 = c00; var c21 = c00; var c30 = c00; var c31 = c00;
                var c40 = c00; var c41 = c00; var c50 = c00; var c51 = c00;
                if (begin != 0)
                {
                    c00 = Avx.LoadVector256(cp + row * n + column); c01 = Avx.LoadVector256(cp + row * n + column + 8);
                    c10 = Avx.LoadVector256(cp + (row + 1) * n + column); c11 = Avx.LoadVector256(cp + (row + 1) * n + column + 8);
                    c20 = Avx.LoadVector256(cp + (row + 2) * n + column); c21 = Avx.LoadVector256(cp + (row + 2) * n + column + 8);
                    c30 = Avx.LoadVector256(cp + (row + 3) * n + column); c31 = Avx.LoadVector256(cp + (row + 3) * n + column + 8);
                    c40 = Avx.LoadVector256(cp + (row + 4) * n + column); c41 = Avx.LoadVector256(cp + (row + 4) * n + column + 8);
                    c50 = Avx.LoadVector256(cp + (row + 5) * n + column); c51 = Avx.LoadVector256(cp + (row + 5) * n + column + 8);
                }
                float* aa = ap + row * k;
                for (int p = begin; p < end; p++)
                {
                    var b0 = Avx.LoadVector256(bp + p * 16); var b1 = Avx.LoadVector256(bp + p * 16 + 8);
                    var av = Avx.BroadcastScalarToVector256(aa + p);
                    c00 = Fma.MultiplyAdd(av, b0, c00); c01 = Fma.MultiplyAdd(av, b1, c01);
                    av = Avx.BroadcastScalarToVector256(aa + k + p);
                    c10 = Fma.MultiplyAdd(av, b0, c10); c11 = Fma.MultiplyAdd(av, b1, c11);
                    av = Avx.BroadcastScalarToVector256(aa + k * 2 + p);
                    c20 = Fma.MultiplyAdd(av, b0, c20); c21 = Fma.MultiplyAdd(av, b1, c21);
                    av = Avx.BroadcastScalarToVector256(aa + k * 3 + p);
                    c30 = Fma.MultiplyAdd(av, b0, c30); c31 = Fma.MultiplyAdd(av, b1, c31);
                    av = Avx.BroadcastScalarToVector256(aa + k * 4 + p);
                    c40 = Fma.MultiplyAdd(av, b0, c40); c41 = Fma.MultiplyAdd(av, b1, c41);
                    av = Avx.BroadcastScalarToVector256(aa + k * 5 + p);
                    c50 = Fma.MultiplyAdd(av, b0, c50); c51 = Fma.MultiplyAdd(av, b1, c51);
                }
                Store(c00, c01, cp + row * n + column, take); Store(c10, c11, cp + (row + 1) * n + column, take);
                Store(c20, c21, cp + (row + 2) * n + column, take); Store(c30, c31, cp + (row + 3) * n + column, take);
                Store(c40, c41, cp + (row + 4) * n + column, take); Store(c50, c51, cp + (row + 5) * n + column, take);
            }
            if (row + 4 <= m)
            {
                var c00 = Vector256<float>.Zero; var c01 = c00; var c10 = c00; var c11 = c00;
                var c20 = c00; var c21 = c00; var c30 = c00; var c31 = c00; float* aa = ap + row * k;
                if (begin != 0)
                {
                    c00 = Avx.LoadVector256(cp + row * n + column); c01 = Avx.LoadVector256(cp + row * n + column + 8);
                    c10 = Avx.LoadVector256(cp + (row + 1) * n + column); c11 = Avx.LoadVector256(cp + (row + 1) * n + column + 8);
                    c20 = Avx.LoadVector256(cp + (row + 2) * n + column); c21 = Avx.LoadVector256(cp + (row + 2) * n + column + 8);
                    c30 = Avx.LoadVector256(cp + (row + 3) * n + column); c31 = Avx.LoadVector256(cp + (row + 3) * n + column + 8);
                }
                for (int p = begin; p < end; p++)
                {
                    var b0 = Avx.LoadVector256(bp + p * 16); var b1 = Avx.LoadVector256(bp + p * 16 + 8);
                    var av = Avx.BroadcastScalarToVector256(aa + p);
                    c00 = Fma.MultiplyAdd(av, b0, c00); c01 = Fma.MultiplyAdd(av, b1, c01);
                    av = Avx.BroadcastScalarToVector256(aa + k + p);
                    c10 = Fma.MultiplyAdd(av, b0, c10); c11 = Fma.MultiplyAdd(av, b1, c11);
                    av = Avx.BroadcastScalarToVector256(aa + k * 2 + p);
                    c20 = Fma.MultiplyAdd(av, b0, c20); c21 = Fma.MultiplyAdd(av, b1, c21);
                    av = Avx.BroadcastScalarToVector256(aa + k * 3 + p);
                    c30 = Fma.MultiplyAdd(av, b0, c30); c31 = Fma.MultiplyAdd(av, b1, c31);
                }
                Store(c00, c01, cp + row * n + column, take); Store(c10, c11, cp + (row + 1) * n + column, take);
                Store(c20, c21, cp + (row + 2) * n + column, take); Store(c30, c31, cp + (row + 3) * n + column, take); row += 4;
            }
            for (; row < m; row++)
            {
                var s0 = Vector256<float>.Zero; var s1 = s0;
                if (begin != 0) { s0 = Avx.LoadVector256(cp + row * n + column); s1 = Avx.LoadVector256(cp + row * n + column + 8); }
                for (int p = begin; p < end; p++)
                {
                    var av = Avx.BroadcastScalarToVector256(ap + row * k + p);
                    s0 = Fma.MultiplyAdd(av, Avx.LoadVector256(bp + p * 16), s0);
                    s1 = Fma.MultiplyAdd(av, Avx.LoadVector256(bp + p * 16 + 8), s1);
                }
                Store(s0, s1, cp + row * n + column, take);
            }
            }
        }
    }

    /// <summary>将低八通道和高八通道写入输出，仅写入当前面板实际存在的列。</summary>
    /// <param name="low">结果的低位 SIMD 通道。</param>
    /// <param name="high">结果的高位 SIMD 通道。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector256<float> low, Vector256<float> high, float* output, int take)
    {
        if (take == 16) { Avx.Store(output, low); Avx.Store(output + 8, high); }
        else for (int j = 0; j < take; j++) output[j] = j < 8 ? low.GetElement(j) : high.GetElement(j - 8);
    }
}
