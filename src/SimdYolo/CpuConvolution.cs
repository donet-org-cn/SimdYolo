using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>通过按需打包空间面板实现通用卷积，避免为整幅图像分配 im2col 缓冲区。</summary>
internal static class CpuConvolution
{
    /// <summary>是否按实际输出地址对齐卷积任务的缓存行。</summary>
    private static readonly bool AlignOutput = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_CONV_ALIGNED") != "0";
    /// <summary>是否以较窄 SIMD 向量处理空间面板尾部。</summary>
    private static readonly bool VectorTails = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_CONV_TAIL_SIMD") != "0";
    // 仅物化微内核本次需要的 16 个空间位置。
    // 带步长的卷积不再创建和反复读取整幅图像大小的 im2col。
    /// <summary>按 16 个空间位置打包卷积输入，经矩阵乘法写入输出，并融合偏置与可选 SiLU。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="w">卷积权重，布局为 [输出通道,每组输入通道,核高,核宽]。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="stride">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    public static void Conv(CpuBackend backend, Tensor x, Tensor w, Tensor? bias, Tensor output, int[] stride, int[] pads, int[] dilation, bool silu)
    {
        int batch = x.Shape[0], ic = x.Shape[1], ih = x.Shape[2], iw = x.Shape[3], oc = w.Shape[0], kh = w.Shape[2], kw = w.Shape[3];
        int oh = output.Shape[2], ow = output.Shape[3], spatial = oh * ow, k = ic * kh * kw, panels = (spatial + 15) / 16;
        float[] input = x.Data, weights = w.Data, target = output.Data;
        var packedWeights = CpuPackedGemm.Use(oc) && w.IsConstant ? CpuPackedGemm.Prepare(weights, oc, k, true) : null;
        if (AlignOutput && batch == 1 && spatial > 16 && spatial % 16 == 0)
        {
            ConvAligned(backend, x, w, bias, output, stride, pads, dilation, silu, packedWeights);
            return;
        }
        for (int b = 0; b < batch; b++)
        {
            float[] batchTarget = batch == 1 ? target : ArrayPool<float>.Shared.Rent(oc * spatial);
            try
            {
                CpuScheduling.For(0, panels, backend.Parallel,
                    () => ArrayPool<float>.Shared.Rent(checked(k * 16)),
                    (panel, _, packed) =>
                    {
                        int start = panel * 16, take = Math.Min(16, spatial - start);
                        Pack(input, packed, b * ic * ih * iw, ic, ih, iw, kh, kw, ow, start, take, stride, pads, dilation);
                        if (packedWeights is not null) CpuPackedGemm.Multiply(packedWeights, packed, batchTarget, spatial, start, take);
                        else CpuGemm.MultiplyAvx(weights, packed, batchTarget, oc, spatial, k, start, take);
                        Epilogue(batchTarget, bias?.Data, oc, spatial, start, take, silu);
                        return packed;
                    },
                    packed => ArrayPool<float>.Shared.Return(packed));
                if (batch != 1) Array.Copy(batchTarget, 0, target, b * oc * spatial, oc * spatial);
            }
            finally { if (batch != 1) ArrayPool<float>.Shared.Return(batchTarget); }
        }
    }

    /// <summary>固定输出数组，按实际 64 字节缓存行边界分配面板，使行首与行尾碎片由同一线程写入。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="w">卷积权重，布局为 [输出通道,每组输入通道,核高,核宽]。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="stride">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <param name="packedWeights">按输出通道交错打包的可选常量权重，未准备时使用普通矩阵内核。</param>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    private static unsafe void ConvAligned(CpuBackend backend, Tensor x, Tensor w, Tensor? bias, Tensor output,
        int[] stride, int[] pads, int[] dilation, bool silu, CpuPackedGemm.Matrix? packedWeights)
    {
        int ic = x.Shape[1], ih = x.Shape[2], iw = x.Shape[3], oc = w.Shape[0], kh = w.Shape[2], kw = w.Shape[3];
        int spatial = output.Shape[2] * output.Shape[3], k = ic * kh * kw;
        float[] input = x.Data, weights = w.Data, target = output.Data;
        fixed (float* targetStart = target)
        {
            int prefix = (int)((64 - ((nuint)targetStart & 63)) & 63) / sizeof(float);
            int full = (spatial - prefix) / 16, tail = spatial - prefix - full * 16;
            int edges = prefix != 0 || tail != 0 ? 1 : 0;
            // 完成当前缓存行片段的输入打包、矩阵乘法、偏置和可选 SiLU。
            // 参数 start：当前空间面板的起始位置；out 参数返回领取区间的起点。
            // 参数 take：当前面板中实际有效的位置数。
            // 参数 packed：当前工作线程使用的连续打包缓冲区。
            void Piece(int start, int take, float[] packed)
            {
                Pack(input, packed, 0, ic, ih, iw, kh, kw, output.Shape[3], start, take, stride, pads, dilation);
                if (packedWeights is not null) CpuPackedGemm.Multiply(packedWeights, packed, target, spatial, start, take);
                else CpuGemm.MultiplyAvx(weights, packed, target, oc, spatial, k, start, take);
                Epilogue(target, bias?.Data, oc, spatial, start, take, silu);
            }
            CpuScheduling.For(0, full + edges, backend.Parallel,
                () => ArrayPool<float>.Shared.Rent(checked(k * 16)),
                (index, _, packed) =>
                {
                    // 相邻通道行的首尾碎片可能位于同一缓存行。
                    // 因此每一行的首尾碎片均交给同一个工作线程，避免缓存行争用。
                    if (edges != 0 && index == 0)
                    {
                        if (prefix != 0) Piece(0, prefix, packed);
                        if (tail != 0) Piece(prefix + full * 16, tail, packed);
                    }
                    else Piece(prefix + (index - edges) * 16, 16, packed);
                    return packed;
                }, packed => ArrayPool<float>.Shared.Return(packed));
            _ = System.Threading.Volatile.Read(ref targetStart[0]);
        }
    }

    /// <summary>将指定空间面板按输入通道、核行、核列顺序展开为 K×16 输入，填充与未使用通道写零。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="batchStart">当前批次在输入数组中的起始偏移。</param>
    /// <param name="ic">输入通道数。</param>
    /// <param name="ih">输入平面高度。</param>
    /// <param name="iw">输入平面宽度。</param>
    /// <param name="kh">卷积核高度。</param>
    /// <param name="kw">卷积核宽度。</param>
    /// <param name="ow">输出平面宽度。</param>
    /// <param name="start">当前空间面板的起始位置；out 参数返回领取区间的起点。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="stride">空间步长，顺序为高、宽。</param>
    /// <param name="pad">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    private static unsafe void Pack(float[] input, float[] packed, int batchStart, int ic, int ih, int iw, int kh, int kw, int ow, int start, int take, int[] stride, int[] pad, int[] dilation)
    {
        if (CpuConvolutionPack.ShouldUse(kh, kw, stride, pad, dilation))
        {
            CpuConvolutionPack.Pack(input, packed, batchStart, ic, ih, iw, ow, start, take, pad[0], pad[1]);
            return;
        }
        Span<int> ys = stackalloc int[16]; Span<int> xs = stackalloc int[16];
        for (int j = 0; j < take; j++) { ys[j] = (start + j) / ow * stride[0] - pad[0]; xs[j] = (start + j) % ow * stride[1] - pad[1]; }
        bool row = take == 16 && start / ow == (start + 15) / ow;
        var order = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
        fixed (float* src = input, dst = packed)
        {
            int r = 0;
            for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++, r++)
            {
                int sy = ys[0] + ky * dilation[0], sx = xs[0] + kx * dilation[1]; float* p = dst + r * 16;
                if (row && (uint)sy < ih && sx >= 0 && (long)sx + 16L * stride[1] <= iw && stride[1] is 1 or 2)
                {
                    float* source = src + batchStart + (ci * ih + sy) * iw + sx;
                    if (stride[1] == 1)
                    {
                        Avx.Store(p, Avx.LoadVector256(source)); Avx.Store(p + 8, Avx.LoadVector256(source + 8));
                    }
                    else
                    {
                        var left = Avx.Shuffle(Avx.LoadVector256(source), Avx.LoadVector256(source + 8), 0x88);
                        var right = Avx.Shuffle(Avx.LoadVector256(source + 16), Avx.LoadVector256(source + 24), 0x88);
                        Avx.Store(p, Avx2.PermuteVar8x32(left, order)); Avx.Store(p + 8, Avx2.PermuteVar8x32(right, order));
                    }
                }
                else
                {
                    for (int j = 0; j < take; j++)
                    {
                        sy = ys[j] + ky * dilation[0]; sx = xs[j] + kx * dilation[1];
                        p[j] = (uint)sy < ih && (uint)sx < iw ? src[batchStart + (ci * ih + sy) * iw + sx] : 0;
                    }
                    for (int j = take; j < 16; j++) p[j] = 0;
                }
            }
        }
    }
    /// <summary>对当前空间面板逐输出通道补加偏置并可选执行 SiLU。</summary>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="channels">要处理的通道数。</param>
    /// <param name="spatial">每个通道的空间元素数。</param>
    /// <param name="start">当前空间面板的起始位置；out 参数返回领取区间的起点。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    private static unsafe void Epilogue(float[] output, float[]? bias, int channels, int spatial, int start, int take, bool silu)
    {
        if (bias is null && !silu) return;
        fixed (float* ptr = output)
            for (int c = 0; c < channels; c++)
            {
                float* dst = ptr + c * spatial + start; float bv = bias?[c] ?? 0;
                if (take == 16)
                {
                    var vb = Vector256.Create(bv); var low = Avx.LoadVector256(dst) + vb; var high = Avx.LoadVector256(dst + 8) + vb;
                    if (silu) { low = CpuActivation.Silu(low); high = CpuActivation.Silu(high); }
                    Avx.Store(dst, low); Avx.Store(dst + 8, high);
                }
                else
                {
                    int j = 0;
                    if (VectorTails)
                    {
                        for (; j <= take - 8; j += 8)
                        {
                            var value = Avx.LoadVector256(dst + j) + Vector256.Create(bv);
                            if (silu) value = CpuActivation.Silu(value);
                            Avx.Store(dst + j, value);
                        }
                        for (; j <= take - 4; j += 4)
                        {
                            var value = Sse.LoadVector128(dst + j) + Vector128.Create(bv);
                            if (silu) value = CpuActivation.Silu(value);
                            Sse.Store(dst + j, value);
                        }
                    }
                    for (; j < take; j++) { float v = dst[j] + bv; dst[j] = silu ? v * TensorOps.Sigmoid(v) : v; }
                }
            }
    }
}
