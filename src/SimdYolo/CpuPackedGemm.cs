using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>缓存按六行交错的常量权重，使用 6×16 AVX/FMA 微内核减少归约中的地址计算。</summary>
internal static class CpuPackedGemm
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_PACKED_A") != "0";
    // 输出通道很少时，补齐到六行的成本大于减少寻址的收益。
    // 此类形状继续使用原有矩阵内核。
    /// <summary>检查行数与 AVX/FMA 能力，避免小矩阵因六行填充产生额外成本。</summary>
    /// <param name="rows">矩阵逻辑行数。</param>
    /// <returns>满足当前快路径的形状、模式和硬件要求时为 true；否则为 false。</returns>
    internal static bool Use(int rows) => Enabled && rows >= 6 && Avx.IsSupported && Fma.IsSupported;
    /// <summary>按常量权重数组引用保存的弱引用打包缓存。</summary>
    private static readonly ConditionalWeakTable<float[], Matrix> Cache = new();

    /// <summary>保存交错打包后的权重数据及其逻辑行数和归约深度。</summary>
    /// <param name="Data">交错打包的 FP32 权重数组。</param>
    /// <param name="Rows">逻辑权重行数。</param>
    /// <param name="Depth">逻辑归约深度 K。</param>
    /// <remarks>各构造参数对应同名自动生成属性，描述本记录保存的数据。</remarks>
    internal sealed record Matrix(float[] Data, int Rows, int Depth);

    /// <summary>按权重数组引用和逻辑形状取得六行交错权重；仅常量权重允许复用缓存。</summary>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="rows">矩阵逻辑行数。</param>
    /// <param name="depth">矩阵 K 轴的归约长度。</param>
    /// <param name="constant">权重是否只读，从而允许缓存其打包结果。</param>
    /// <returns>具有指定行数和归约深度的交错打包权重。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    /// <remarks>缓存仅对保持只读的权重有效；同一数组的逻辑形状变化时重新打包。</remarks>
    internal static Matrix Prepare(float[] weights, int rows, int depth, bool constant)
    {
        if (rows < 0 || depth < 0 || weights.Length < checked(rows * depth)) throw new ArgumentException("Invalid packed GEMM weights.");
        if (!constant) return Pack(weights, rows, depth);
        if (Cache.TryGetValue(weights, out var value) && value.Rows == rows && value.Depth == depth) return value;
        lock (Cache)
        {
            if (Cache.TryGetValue(weights, out value) && value.Rows == rows && value.Depth == depth) return value;
            value = Pack(weights, rows, depth);
            // 同一权重数组可能被重新解释为不同形状。
            // 即使元素总数相同，也不能复用逻辑行数/深度不同的打包缓存。
            Cache.Remove(weights); Cache.Add(weights, value);
            return value;
        }
    }

    /// <summary>将权重按六行、K 深度交错组织，缺少的行写零。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="rows">矩阵逻辑行数。</param>
    /// <param name="depth">矩阵 K 轴的归约长度。</param>
    /// <returns>具有指定行数和归约深度的交错打包权重。</returns>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    private static Matrix Pack(float[] source, int rows, int depth)
    {
        int blocks = rows / 6 + (rows % 6 == 0 ? 0 : 1);
        var packed = GC.AllocateUninitializedArray<float>(checked(blocks * checked(depth * 6)));
        for (int block = 0; block < blocks; block++) for (int p = 0; p < depth; p++)
        {
            int dst = (block * depth + p) * 6, first = block * 6;
            for (int r = 0; r < 6; r++) packed[dst + r] = first + r < rows ? source[(first + r) * depth + p] : 0;
        }
        return new(packed, rows, depth);
    }

    /// <summary>以 6×16 FP32 FMA 内核计算当前面板，可指定输入和输出缓冲区偏移。</summary>
    /// <param name="weights">按六行交错的只读打包权重。</param>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="bOffset">打包右矩阵在输入缓冲区中的起始偏移。</param>
    /// <param name="cOffset">结果矩阵在输出缓冲区中的起始偏移。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static unsafe void Multiply(Matrix weights, float[] input, float[] output, int n, int column, int take,
                                          int bOffset = 0, int cOffset = 0)
    {
        int m = weights.Rows, k = weights.Depth;
        if (n < 0 || column < 0 || take < 0 || take > 16 || column > n - take || bOffset < 0 || cOffset < 0 ||
            input.Length < checked(bOffset + checked(k * 16)) || output.Length < checked(cOffset + checked(m * n)))
            throw new ArgumentException("Invalid packed GEMM panel.");
        if (m == 0 || take == 0) return;
        if (k == 0)
        {
            for (int row = 0; row < m; row++) output.AsSpan(cOffset + row * n + column, take).Clear();
            return;
        }
        fixed (float* aStart = weights.Data, bStart = input, cStart = output)
        {
            float* bp = bStart + bOffset, cp = cStart + cOffset;
            int depth = take == 16 && k >= 256 && m >= 32 ? 128 : k;
            for (int begin = 0; begin < k; begin += depth)
            {
                int end = Math.Min(k, begin + depth);
                for (int row = 0, block = 0; row < m; row += 6, block++)
                {
                    var c00 = Vector256<float>.Zero; var c01 = c00; var c10 = c00; var c11 = c00;
                    var c20 = c00; var c21 = c00; var c30 = c00; var c31 = c00;
                    var c40 = c00; var c41 = c00; var c50 = c00; var c51 = c00;
                    if (begin != 0)
                    {
                        c00 = Avx.LoadVector256(cp + row * n + column); c01 = Avx.LoadVector256(cp + row * n + column + 8);
                        if (row + 1 < m) { c10 = Avx.LoadVector256(cp + (row + 1) * n + column); c11 = Avx.LoadVector256(cp + (row + 1) * n + column + 8); }
                        if (row + 2 < m) { c20 = Avx.LoadVector256(cp + (row + 2) * n + column); c21 = Avx.LoadVector256(cp + (row + 2) * n + column + 8); }
                        if (row + 3 < m) { c30 = Avx.LoadVector256(cp + (row + 3) * n + column); c31 = Avx.LoadVector256(cp + (row + 3) * n + column + 8); }
                        if (row + 4 < m) { c40 = Avx.LoadVector256(cp + (row + 4) * n + column); c41 = Avx.LoadVector256(cp + (row + 4) * n + column + 8); }
                        if (row + 5 < m) { c50 = Avx.LoadVector256(cp + (row + 5) * n + column); c51 = Avx.LoadVector256(cp + (row + 5) * n + column + 8); }
                    }
                    float* aa = aStart + (block * k + begin) * 6, bb = bp + begin * 16;
                    // 权重按六行交错后，每个 K 步骤使用固定行偏移。
                    // 相较原 Tier1 内核，每步减少五次动态行地址计算。
                    for (int p = begin; p < end; p++, aa += 6, bb += 16)
                    {
                        var b0 = Avx.LoadVector256(bb); var b1 = Avx.LoadVector256(bb + 8);
                        var av = Avx.BroadcastScalarToVector256(aa);
                        c00 = Fma.MultiplyAdd(av, b0, c00); c01 = Fma.MultiplyAdd(av, b1, c01);
                        av = Avx.BroadcastScalarToVector256(aa + 1);
                        c10 = Fma.MultiplyAdd(av, b0, c10); c11 = Fma.MultiplyAdd(av, b1, c11);
                        av = Avx.BroadcastScalarToVector256(aa + 2);
                        c20 = Fma.MultiplyAdd(av, b0, c20); c21 = Fma.MultiplyAdd(av, b1, c21);
                        av = Avx.BroadcastScalarToVector256(aa + 3);
                        c30 = Fma.MultiplyAdd(av, b0, c30); c31 = Fma.MultiplyAdd(av, b1, c31);
                        av = Avx.BroadcastScalarToVector256(aa + 4);
                        c40 = Fma.MultiplyAdd(av, b0, c40); c41 = Fma.MultiplyAdd(av, b1, c41);
                        av = Avx.BroadcastScalarToVector256(aa + 5);
                        c50 = Fma.MultiplyAdd(av, b0, c50); c51 = Fma.MultiplyAdd(av, b1, c51);
                    }
                    Store(c00, c01, cp + row * n + column, take);
                    if (row + 1 < m) Store(c10, c11, cp + (row + 1) * n + column, take);
                    if (row + 2 < m) Store(c20, c21, cp + (row + 2) * n + column, take);
                    if (row + 3 < m) Store(c30, c31, cp + (row + 3) * n + column, take);
                    if (row + 4 < m) Store(c40, c41, cp + (row + 4) * n + column, take);
                    if (row + 5 < m) Store(c50, c51, cp + (row + 5) * n + column, take);
                }
            }
        }
    }

    /// <summary>安全写出最多 16 个结果通道，尾面板只写实际有效列。</summary>
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
