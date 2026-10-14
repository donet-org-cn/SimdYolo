using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

// 四个输出行、每行三个 AVX 向量共占用 12 个累积寄存器。
// 相较六行/16 列内核，同为 96 个乘积，每个 K 步骤只需四次权重广播。
// 24 列面板也减少打包任务和调度回调，且仍按原 FP32 K 次序归约。
/// <summary>实现四行、24 列的 AVX/FMA 微内核，每步复用一次权重广播处理三个向量。</summary>
internal static class CpuPointwise24
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_POINTWISE_24") != "0";
    /// <summary>按常量权重数组引用保存的弱引用打包缓存。</summary>
    private static readonly ConditionalWeakTable<float[], Matrix> Cache = new();
    /// <summary>保存交错打包后的权重数据及其逻辑行数和归约深度。</summary>
    /// <param name="Data">交错打包的 FP32 权重数组。</param>
    /// <param name="Rows">逻辑权重行数。</param>
    /// <param name="Depth">逻辑归约深度 K。</param>
    /// <remarks>各构造参数对应同名自动生成属性，描述本记录保存的数据。</remarks>
    internal sealed record Matrix(float[] Data, int Rows, int Depth);

    /// <summary>检查常量权重、批次和空间列数是否适合 4×24 AVX/FMA 内核。</summary>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <returns>满足当前快路径的形状、模式和硬件要求时为 true；否则为 false。</returns>
    internal static bool ShouldUse(Tensor weights, Tensor output)
        => Enabled && Avx.IsSupported && Fma.IsSupported && weights.IsConstant &&
           output.Shape[1] >= 4 && (long)output.Shape[2] * output.Shape[3] >= 24;

    /// <summary>将常量权重按四行交错打包，并按数组引用与逻辑行数/深度缓存。</summary>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="rows">矩阵逻辑行数。</param>
    /// <param name="depth">矩阵 K 轴的归约长度。</param>
    /// <returns>具有指定行数和归约深度的交错打包权重。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    /// <remarks>缓存仅对保持只读的权重有效；同一数组的逻辑形状变化时重新打包。</remarks>
    internal static Matrix Prepare(float[] weights, int rows, int depth)
    {
        if (rows < 0 || depth < 0 || weights.Length < checked(rows * depth)) throw new ArgumentException("Invalid 24-column GEMM weights.");
        if (Cache.TryGetValue(weights, out var value) && value.Rows == rows && value.Depth == depth) return value;
        lock (Cache)
        {
            if (Cache.TryGetValue(weights, out value) && value.Rows == rows && value.Depth == depth) return value;
            int blocks = (rows + 3) / 4;
            var data = GC.AllocateUninitializedArray<float>(checked(blocks * checked(depth * 4)));
            for (int block = 0; block < blocks; block++) for (int p = 0; p < depth; p++)
                for (int row = 0; row < 4; row++)
                    data[(block * depth + p) * 4 + row] = block * 4 + row < rows ? weights[(block * 4 + row) * depth + p] : 0;
            value = new(data, rows, depth); Cache.Remove(weights); Cache.Add(weights, value); return value;
        }
    }

    /// <summary>直接打包输入张量的通道拼接，采用 24 列点卷积内核并融合偏置与 SiLU。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="inputs">按通道拼接顺序排列的输入张量。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static void Conv(CpuBackend backend, IReadOnlyList<Tensor> inputs, Tensor weights, Tensor? bias, Tensor output, bool silu)
    {
        if (!Avx.IsSupported || !Fma.IsSupported || !weights.IsConstant) throw new ArgumentException("Pointwise24 requires AVX/FMA and constant weights.");
        if (CpuPointwiseAligned.Enabled && (output.Shape[2] * output.Shape[3]) % 16 == 0)
        {
            CpuPointwiseAligned.Conv(backend, inputs, weights, bias, output, silu);
            return;
        }
        float[][] sources = inputs.Select(input => input.Data).ToArray();
        int[] channels = inputs.Select(input => input.Shape[1]).ToArray();
        int m = output.Shape[1], n = checked(output.Shape[2] * output.Shape[3]), k = weights.Shape[1];
        int panels = (n + 23) / 24;
        var matrix = Prepare(weights.Data, m, k); float[] target = output.Data; float[]? biases = bias?.Data;
        // 打包当前 24 列通道拼接数据，使用四行内核计算点卷积和偏置/激活。
        // 参数 panel：当前输入空间面板的索引。
        // 参数 scratch：当前线程私有的临时工作缓冲区。
        void Panel(int panel, float[] scratch)
        {
            int column = panel * 24, take = Math.Min(24, n - column);
            Pack(sources, channels, scratch, n, column, take);
            Multiply(matrix, scratch, target, n, column, take);
            Epilogue(target, biases, m, n, column, take, silu);
        }
        if ((long)m * n * k > 262144 && backend.Parallel.MaxDegreeOfParallelism != 1)
            CpuScheduling.For(0, panels, backend.Parallel,
                () => ArrayPool<float>.Shared.Rent(checked(k * 24)),
                (panel, _, scratch) => { Panel(panel, scratch); return scratch; },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        else
        {
            var scratch = ArrayPool<float>.Shared.Rent(checked(k * 24));
            try { for (int panel = 0; panel < panels; panel++) Panel(panel, scratch); }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }
    }

    /// <summary>按源张量和通道顺序打包 K×24 面板，可在工作缓冲区中指定对齐偏移。</summary>
    /// <param name="sources">按通道拼接顺序排列的源数组。</param>
    /// <param name="channels">各源张量的输入通道数量，顺序与源数组一致。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="offset">打包数据在缓冲区中的起始偏移。</param>
    internal static unsafe void Pack(float[][] sources, int[] channels, float[] packed, int n, int column, int take, int offset = 0)
    {
        fixed (float* target = packed)
        {
            int row = 0;
            for (int s = 0; s < sources.Length; s++)
                fixed (float* source = sources[s])
                    for (int ci = 0; ci < channels[s]; ci++, row++)
                    {
                        float* src = source + ci * n + column, dst = target + offset + row * 24;
                        if (take == 24)
                        {
                            Avx.Store(dst, Avx.LoadVector256(src)); Avx.Store(dst + 8, Avx.LoadVector256(src + 8));
                            Avx.Store(dst + 16, Avx.LoadVector256(src + 16));
                        }
                        else
                        {
                            for (int j = 0; j < take; j++) dst[j] = src[j];
                            for (int j = take; j < 24; j++) dst[j] = 0;
                        }
                    }
        }
    }

    /// <summary>以四行、三个 AVX 向量计算 24 列面板，保持递增 K 的 FP32 FMA 归约顺序。</summary>
    /// <param name="matrix">包含交错权重和逻辑形状的打包矩阵。</param>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="bOffset">打包右矩阵在输入缓冲区中的起始偏移。</param>
    /// <param name="cOffset">结果矩阵在输出缓冲区中的起始偏移。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static unsafe void Multiply(Matrix matrix, float[] input, float[] output, int n, int column, int take,
                                         int bOffset = 0, int cOffset = 0)
    {
        int m = matrix.Rows, k = matrix.Depth;
        if (n < 0 || column < 0 || take < 0 || take > 24 || column > n - take || bOffset < 0 || cOffset < 0 ||
            input.Length < checked(bOffset + checked(k * 24)) || output.Length < checked(cOffset + checked(m * n)))
            throw new ArgumentException("Invalid 24-column GEMM panel.");
        if (m == 0 || take == 0) return;
        if (k == 0)
        {
            for (int row = 0; row < m; row++) output.AsSpan(cOffset + row * n + column, take).Clear();
            return;
        }
        fixed (float* aStart = matrix.Data, bBase = input, cBase = output)
        {
            float* bStart = bBase + bOffset, cStart = cBase + cOffset;
            // 深度为 512 的 24 列 B 面板占 48 KiB，超过典型 AVX2 CPU 的 32 KiB L1 缓存。
            // 因此对全部输出行复用深度为 128 的片段，并保留递增 K 的 FMA 累积次序。
            // K=256 时，B 加四行 A 约占 28 KiB，仍可驻留 L1，避免不必要的部分结果存储与重载。
            int depth = k >= 384 && m >= 32 && (take is 16 or 24) ? 128 : k;
            for (int begin = 0; begin < k; begin += depth)
            {
            int end = Math.Min(k, begin + depth);
            for (int row = 0, block = 0; row < m; row += 4, block++)
            {
                var c00 = Vector256<float>.Zero; var c01 = c00; var c02 = c00;
                var c10 = c00; var c11 = c00; var c12 = c00;
                var c20 = c00; var c21 = c00; var c22 = c00;
                var c30 = c00; var c31 = c00; var c32 = c00;
                if (begin != 0)
                {
                    c00 = Avx.LoadVector256(cStart + row * n + column); c01 = Avx.LoadVector256(cStart + row * n + column + 8);
                    if (take == 24) c02 = Avx.LoadVector256(cStart + row * n + column + 16);
                    if (row + 1 < m)
                    {
                        c10 = Avx.LoadVector256(cStart + (row + 1) * n + column); c11 = Avx.LoadVector256(cStart + (row + 1) * n + column + 8);
                        if (take == 24) c12 = Avx.LoadVector256(cStart + (row + 1) * n + column + 16);
                    }
                    if (row + 2 < m)
                    {
                        c20 = Avx.LoadVector256(cStart + (row + 2) * n + column); c21 = Avx.LoadVector256(cStart + (row + 2) * n + column + 8);
                        if (take == 24) c22 = Avx.LoadVector256(cStart + (row + 2) * n + column + 16);
                    }
                    if (row + 3 < m)
                    {
                        c30 = Avx.LoadVector256(cStart + (row + 3) * n + column); c31 = Avx.LoadVector256(cStart + (row + 3) * n + column + 8);
                        if (take == 24) c32 = Avx.LoadVector256(cStart + (row + 3) * n + column + 16);
                    }
                }
                float* aa = aStart + (block * k + begin) * 4, bb = bStart + begin * 24;
                for (int p = begin; p < end; p++, aa += 4, bb += 24)
                {
                    var b0 = Avx.LoadVector256(bb); var b1 = Avx.LoadVector256(bb + 8); var b2 = Avx.LoadVector256(bb + 16);
                    var av = Avx.BroadcastScalarToVector256(aa);
                    c00 = Fma.MultiplyAdd(av, b0, c00); c01 = Fma.MultiplyAdd(av, b1, c01); c02 = Fma.MultiplyAdd(av, b2, c02);
                    av = Avx.BroadcastScalarToVector256(aa + 1);
                    c10 = Fma.MultiplyAdd(av, b0, c10); c11 = Fma.MultiplyAdd(av, b1, c11); c12 = Fma.MultiplyAdd(av, b2, c12);
                    av = Avx.BroadcastScalarToVector256(aa + 2);
                    c20 = Fma.MultiplyAdd(av, b0, c20); c21 = Fma.MultiplyAdd(av, b1, c21); c22 = Fma.MultiplyAdd(av, b2, c22);
                    av = Avx.BroadcastScalarToVector256(aa + 3);
                    c30 = Fma.MultiplyAdd(av, b0, c30); c31 = Fma.MultiplyAdd(av, b1, c31); c32 = Fma.MultiplyAdd(av, b2, c32);
                }
                Store(c00, c01, c02, cStart + row * n + column, take);
                if (row + 1 < m) Store(c10, c11, c12, cStart + (row + 1) * n + column, take);
                if (row + 2 < m) Store(c20, c21, c22, cStart + (row + 2) * n + column, take);
                if (row + 3 < m) Store(c30, c31, c32, cStart + (row + 3) * n + column, take);
            }
            }
        }
    }

    /// <summary>安全写出三个八通道向量，只写当前面板的有效列。</summary>
    /// <param name="a">待写出的低八通道结果。</param>
    /// <param name="b">待写出的中间八通道结果。</param>
    /// <param name="c">待写出的高八通道结果。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector256<float> a, Vector256<float> b, Vector256<float> c, float* output, int take)
    {
        if (take == 24) { Avx.Store(output, a); Avx.Store(output + 8, b); Avx.Store(output + 16, c); }
        else if (take == 16) { Avx.Store(output, a); Avx.Store(output + 8, b); }
        else if (take == 8) Avx.Store(output, a);
        else for (int j = 0; j < take; j++) output[j] = j < 8 ? a.GetElement(j) : j < 16 ? b.GetElement(j - 8) : c.GetElement(j - 16);
    }

    /// <summary>对当前点卷积面板补加每通道偏置并可选执行 SiLU。</summary>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="channels">要处理的通道数。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    internal static unsafe void Epilogue(float[] output, float[]? bias, int channels, int n, int column, int take, bool silu)
    {
        if (bias is null && !silu) return;
        fixed (float* ptr = output)
            for (int row = 0; row < channels; row++)
            {
                float* target = ptr + row * n + column; float bv = bias?[row] ?? 0;
                if (take == 24)
                {
                    var vb = Vector256.Create(bv); var a = Avx.LoadVector256(target) + vb;
                    var b = Avx.LoadVector256(target + 8) + vb; var c = Avx.LoadVector256(target + 16) + vb;
                    if (silu)
                    {
                        a = CpuActivation.Silu(a); b = CpuActivation.Silu(b); c = CpuActivation.Silu(c);
                    }
                    Avx.Store(target, a); Avx.Store(target + 8, b); Avx.Store(target + 16, c);
                }
                else
                {
                    int j = 0; var vb = Vector256.Create(bv);
                    for (; j + 8 <= take; j += 8)
                    {
                        var v = Avx.LoadVector256(target + j) + vb;
                        if (silu) v = CpuActivation.Silu(v);
                        Avx.Store(target + j, v);
                    }
                    for (; j < take; j++) { float v = target[j] + bv; target[j] = silu ? v * TensorOps.Sigmoid(v) : v; }
                }
            }
    }
}
