using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>融合通道拼接与 1×1 卷积，直接按源张量顺序打包 K 轴而不物化拼接结果。</summary>
internal static class CpuPointwise
{
    /// <summary>是否启用通道拼接与点卷积融合。</summary>
    internal static readonly bool ConcatEnabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_CONCAT") != "0";
    /// <summary>是否启用点卷积面板融合。</summary>
    internal static readonly bool FusionEnabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_POINTWISE_FUSION") != "0";
    /// <summary>当前处理器是否具备此路径需要的 AVX 与 FMA。</summary>
    /// <value>当前处理器是否具备此路径需要的 AVX 与 FMA。</value>
    internal static bool Supported => Avx.IsSupported && Fma.IsSupported;

    // 各源张量按顺序贡献连续输入通道。
    // K 轴打包保留该顺序，使 GEMM 的归约与先物化 NCHW 通道拼接完全一致。
    /// <summary>按源张量通道顺序直接打包 1×1 卷积输入，使用互不重叠的空间面板写出偏置与可选 SiLU 结果。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="inputs">按通道拼接顺序排列的输入张量。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static void Conv(CpuBackend backend, IReadOnlyList<Tensor> inputs, Tensor weights, Tensor? bias, Tensor output, bool silu)
    {
        if (CpuPointwise24.ShouldUse(weights, output))
        {
            CpuPointwise24.Conv(backend, inputs, weights, bias, output, silu);
            return;
        }
        float[][] sources = inputs.Select(input => input.Data).ToArray();
        int[] channels = inputs.Select(input => input.Shape[1]).ToArray();
        float[] a = weights.Data, c = output.Data; float[]? d = bias?.Data;
        int m = output.Shape[1], n = checked(output.Shape[2] * output.Shape[3]), k = weights.Shape[1], panels = (n + 15) / 16;
        var packedWeights = CpuPackedGemm.Use(m) && weights.IsConstant ? CpuPackedGemm.Prepare(a, m, k, true) : null;
        // 打包当前 16 列通道拼接数据，计算点卷积并融合偏置/激活。
        // 参数 panel：当前输入空间面板的索引。
        // 参数 packed：当前工作线程使用的连续打包缓冲区。
        void Panel(int panel, float[] packed)
        {
            int column = panel * 16, take = Math.Min(16, n - column);
            Pack(sources, channels, packed, n, column, take);
            if (packedWeights is not null) CpuPackedGemm.Multiply(packedWeights, packed, c, n, column, take);
            else CpuGemm.MultiplyAvx(a, packed, c, m, n, k, column, take);
            Epilogue(c, d, m, n, column, take, silu);
        }
        if ((long)m * n * k > 262144 && backend.Parallel.MaxDegreeOfParallelism != 1)
            CpuScheduling.For(0, panels, backend.Parallel,
                () => ArrayPool<float>.Shared.Rent(checked(k * 16)),
                (panel, _, packed) => { Panel(panel, packed); return packed; },
                packed => ArrayPool<float>.Shared.Return(packed));
        else
        {
            var packed = ArrayPool<float>.Shared.Rent(checked(k * 16));
            try { for (int panel = 0; panel < panels; panel++) Panel(panel, packed); }
            finally { ArrayPool<float>.Shared.Return(packed); }
        }
    }
    /// <summary>依次读取各源张量的输入通道，打包为 K×16 面板，尾部补零且不改变通道拼接顺序。</summary>
    /// <param name="sources">按通道拼接顺序排列的源数组。</param>
    /// <param name="channels">各源张量的输入通道数量，顺序与源数组一致。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    private static unsafe void Pack(float[][] sources, int[] channels, float[] packed, int n, int column, int take)
    {
        fixed (float* target = packed)
        {
            int row = 0;
            for (int s = 0; s < sources.Length; s++)
                fixed (float* source = sources[s])
                    for (int ci = 0; ci < channels[s]; ci++, row++)
                    {
                        float* src = source + ci * n + column; float* dst = target + row * 16;
                        if (take == 16)
                        {
                            Avx.Store(dst, Avx.LoadVector256(src)); Avx.Store(dst + 8, Avx.LoadVector256(src + 8));
                        }
                        else
                        {
                            for (int j = 0; j < take; j++) dst[j] = src[j];
                            for (int j = take; j < 16; j++) dst[j] = 0;
                        }
                    }
        }
    }
    /// <summary>对当前 16 列点卷积面板补加每个输出通道的偏置，并可选执行 SiLU。</summary>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="channels">要处理的通道数。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    private static unsafe void Epilogue(float[] output, float[]? bias, int channels, int n, int column, int take, bool silu)
    {
        if (bias is null && !silu) return;
        fixed (float* ptr = output)
            for (int c = 0; c < channels; c++)
            {
                float* dst = ptr + c * n + column; float bv = bias?[c] ?? 0;
                if (take == 16)
                {
                    var vb = Vector256.Create(bv); var low = Avx.LoadVector256(dst) + vb; var high = Avx.LoadVector256(dst + 8) + vb;
                    if (silu) { low = CpuActivation.Silu(low); high = CpuActivation.Silu(high); }
                    Avx.Store(dst, low); Avx.Store(dst + 8, high);
                }
                else for (int j = 0; j < take; j++) { float v = dst[j] + bv; dst[j] = silu ? v * TensorOps.Sigmoid(v) : v; }
            }
    }
}
