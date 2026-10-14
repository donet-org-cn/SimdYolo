using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

// F(4×4,3×3) 使用较大的变换，换取更少的通道乘积。
// 在通道乘法占主要成本的深层卷积中，这种取舍更有收益。
/// <summary>实现 F(4×4,3×3) 卷积，以较大的输入变换换取较少的通道乘积。</summary>
internal static class CpuWinograd4
{
    /// <summary>是否启用 80 通道分类头的 F(4,3) 特定尺寸路径。</summary>
    private static readonly bool HeadShape = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_HEAD_F4") != "0";
    /// <summary>F(4,3) 通用形状扩展开关；值 2 还包含小平面的填充 SIMD 路径。</summary>
    private static readonly string ExpandedShapes = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_F4_EXPAND") ?? "2";
    /// <summary>是否为小平面构造一次零填充输入以减少标量边界计算。</summary>
    private static readonly bool PaddedSmall = ExpandedShapes == "2";
    // 较大的 YOLOv8 分类头超出原有 IC≤64 的 F(4,3) 路径。
    // 扩展仍按通道数与空间尺寸筛选；部分深层图块过少，不能抵消较大变换的开销。
    /// <summary>判断输入/输出通道和空间尺寸是否属于已启用的 F(4,3) 扩展适用范围。</summary>
    /// <param name="ic">输入通道数。</param>
    /// <param name="oc">输出通道数。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <returns>满足当前快路径的形状、模式和硬件要求时为 true；否则为 false。</returns>
    internal static bool AdditionalHotShape(int ic, int oc, int h, int width)
        => HeadShape && ic == 80 && oc == 80 && h == 80 && width == 80 ||
           ExpandedShapes is "1" or "2" && ic is >= 32 and <= 256 && oc is >= 32 and <= 256 &&
           ic + oc <= 512 && (ic >= 80 || oc > 128) && h >= (PaddedSmall ? 20 : 40) && h <= 160 &&
           width >= (PaddedSmall ? 20 : 40) && width <= 160;
    /// <summary>按常量权重张量保存的弱引用 Winograd 变换缓存。</summary>
    private static readonly ConditionalWeakTable<Tensor, float[][]> Weights = new();
    /// <summary>F(4,3) 权重变换的 6×3 常量矩阵。</summary>
    private static readonly float[] G = [ .25f, 0, 0, -1f / 6, -1f / 6, -1f / 6, -1f / 6, 1f / 6, -1f / 6, 1f / 24, 1f / 12, 1f / 6, 1f / 24, -1f / 12, 1f / 6, 0, 0, 1 ];
    /// <summary>以 F(4,3) 变换权重和输入，缓存只读权重；按图块数选择线程私有面板或按变换坐标并行的全图路径。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="w">卷积权重，布局为 [输出通道,每组输入通道,核高,核宽]。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    public static Tensor Conv(CpuBackend backend, Tensor x, Tensor w, Tensor? bias, bool silu)
    {
        int batch = x.Shape[0], ic = x.Shape[1], h = x.Shape[2], width = x.Shape[3], oc = w.Shape[0];
        int th = (h + 3) / 4, tw = (width + 3) / 4, tiles = th * tw;
        // 用 F(4,3) 权重矩阵变换每个输入/输出通道的 3×3 核，生成 36 个坐标矩阵。
        // 参数 weight：当前深度卷积或 Winograd 变换使用的权重张量。
        // 返回值：按变换坐标排列的权重或输入数组。
        float[][] Pack(Tensor weight)
        {
            var packed = Enumerable.Range(0, 36).Select(_ => new float[oc * ic]).ToArray(); float[] data = weight.Data;
            Span<float> temp = stackalloc float[18];
            for (int co = 0; co < oc; co++) for (int ci = 0; ci < ic; ci++)
            {
                int start = (co * ic + ci) * 9, dst = co * ic + ci;
                for (int r = 0; r < 6; r++) for (int c = 0; c < 3; c++)
                    temp[r * 3 + c] = G[r * 3] * data[start + c] + G[r * 3 + 1] * data[start + 3 + c] + G[r * 3 + 2] * data[start + 6 + c];
                for (int r = 0; r < 6; r++) for (int c = 0; c < 6; c++)
                    packed[r * 6 + c][dst] = temp[r * 3] * G[c * 3] + temp[r * 3 + 1] * G[c * 3 + 1] + temp[r * 3 + 2] * G[c * 3 + 2];
            }
            return packed;
        }
        float[][] weights = w.IsConstant ? Weights.GetValue(w, Pack) : Pack(w);
        if (CpuWinograd424.ShouldUse(backend, tiles, ic, oc, w.IsConstant))
            return CpuWinograd424.Conv(backend, x, weights, bias, silu);
        if (CpuWinogradPanels.ShouldUse(backend, tiles, ic, oc))
            return CpuWinogradPanels.Conv(backend, x, weights, bias, silu, true, w.IsConstant);
        var output = Tensor.TemporaryUninitialized(batch, oc, h, width); float[] input = x.Data, target = output.Data;
        var transformed = new float[36][]; var products = new float[36][];
        for (int i = 0; i < 36; i++) { transformed[i] = ArrayPool<float>.Shared.Rent(ic * tiles); products[i] = ArrayPool<float>.Shared.Rent(oc * tiles); }
        try
        {
            for (int b = 0; b < batch; b++)
            {
                if (PaddedSmall && h <= 40 && width <= 40)
                {
                    int ph = th * 4 + 2, pw = tw * 4 + 2, paddedLength = ph * pw;
                    CpuScheduling.For(0, ic, backend.Parallel, () => ArrayPool<float>.Shared.Rent(paddedLength),
                        (ci, _, padded) =>
                        {
                            padded.AsSpan(0, paddedLength).Clear();
                            for (int row = 0; row < h; row++)
                                input.AsSpan(((b * ic + ci) * h + row) * width, width).CopyTo(padded.AsSpan((row + 1) * pw + 1));
                            Span<float> patch = stackalloc float[36]; Span<float> temp = stackalloc float[36]; Span<float> result = stackalloc float[6];
                            for (int ty = 0; ty < th; ty++) for (int tx = 0; tx < tw; tx++)
                            {
                                int dst = ci * tiles + ty * tw + tx, src = ty * 4 * pw + tx * 4;
                                if (tx + 4 <= tw)
                                {
                                    CpuWinogradPanels.TransformFour4Global(padded, transformed, src, pw, dst);
                                    tx += 3; continue;
                                }
                                for (int r = 0; r < 6; r++) padded.AsSpan(src + r * pw, 6).CopyTo(patch.Slice(r * 6));
                                for (int c = 0; c < 6; c++)
                                {
                                    Transform(patch[c], patch[6 + c], patch[12 + c], patch[18 + c], patch[24 + c], patch[30 + c], result);
                                    for (int r = 0; r < 6; r++) temp[r * 6 + c] = result[r];
                                }
                                for (int r = 0; r < 6; r++)
                                {
                                    Transform(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                                    for (int c = 0; c < 6; c++) transformed[r * 6 + c][dst] = result[c];
                                }
                            }
                            return padded;
                        }, padded => ArrayPool<float>.Shared.Return(padded));
                }
                else CpuScheduling.For(0, ic, backend.Parallel, ci =>
                {
                    Span<float> patch = stackalloc float[36]; Span<float> temp = stackalloc float[36]; Span<float> result = stackalloc float[6];
                    for (int ty = 0; ty < th; ty++) for (int tx = 0; tx < tw; tx++)
                    {
                        if (Avx2.IsSupported && ty > 0 && ty * 4 + 4 < h && tx > 0 && tx + 8 < tw)
                        {
                            TransformEight(input, transformed, ((b * ic + ci) * h + ty * 4 - 1) * width + tx * 4 - 1, width, ci * tiles + ty * tw + tx);
                            tx += 7;
                            continue;
                        }
                        for (int r = 0; r < 6; r++) for (int c = 0; c < 6; c++)
                        {
                            int yy = ty * 4 - 1 + r, xx = tx * 4 - 1 + c;
                            patch[r * 6 + c] = (uint)yy < h && (uint)xx < width ? input[((b * ic + ci) * h + yy) * width + xx] : 0;
                        }
                        for (int c = 0; c < 6; c++)
                        {
                            Transform(patch[c], patch[6 + c], patch[12 + c], patch[18 + c], patch[24 + c], patch[30 + c], result);
                            for (int r = 0; r < 6; r++) temp[r * 6 + c] = result[r];
                        }
                        int dst = ci * tiles + ty * tw + tx;
                        for (int r = 0; r < 6; r++)
                        {
                            Transform(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                            for (int c = 0; c < 6; c++) transformed[r * 6 + c][dst] = result[c];
                        }
                    }
                });
                CpuGemm.MultiplyBatch(weights, transformed, products, oc, tiles, ic, backend.Parallel, w.IsConstant);
                CpuScheduling.For(0, oc, backend.Parallel, co =>
                {
                    Span<float> temp = stackalloc float[24]; Span<float> result = stackalloc float[4];
                    for (int ty = 0; ty < th; ty++) for (int tx = 0; tx < tw; tx++)
                    {
                        int tile = co * tiles + ty * tw + tx;
                        if (Avx.IsSupported && tx + 8 <= tw && (tx + 8) * 4 <= width && ty * 4 + 3 < h)
                        {
                            InverseEight(products, target, tile, ((b * oc + co) * h + ty * 4) * width + tx * 4, width, bias?.Data[co] ?? 0);
                            tx += 7;
                            continue;
                        }
                        if (PaddedSmall && Sse.IsSupported && tx + 4 <= tw && (tx + 4) * 4 <= width && ty * 4 + 3 < h)
                        {
                            InverseFour(products, target, tile, ((b * oc + co) * h + ty * 4) * width + tx * 4, width, bias?.Data[co] ?? 0);
                            tx += 3; continue;
                        }
                        for (int c = 0; c < 6; c++)
                        {
                            Inverse(products[c][tile], products[6 + c][tile], products[12 + c][tile], products[18 + c][tile], products[24 + c][tile], products[30 + c][tile], result);
                            for (int r = 0; r < 4; r++) temp[r * 6 + c] = result[r];
                        }
                        float bv = bias?.Data[co] ?? 0;
                        for (int r = 0; r < 4 && ty * 4 + r < h; r++)
                        {
                            Inverse(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                            int dst = ((b * oc + co) * h + ty * 4 + r) * width + tx * 4;
                            for (int c = 0; c < 4 && tx * 4 + c < width; c++) target[dst + c] = result[c] + bv;
                        }
                    }
                    if (silu)
                    {
                        int start = (b * oc + co) * h * width, end = start + h * width, j = start;
                        for (; j <= end - Vector<float>.Count; j += Vector<float>.Count) { var v = new Vector<float>(target, j); CpuActivation.Silu(v).CopyTo(target, j); }
                        for (; j < end; j++) target[j] *= TensorOps.Sigmoid(target[j]);
                    }
                });
            }
        }
        finally { foreach (var array in transformed.Concat(products)) ArrayPool<float>.Shared.Return(array); }
        return output;
    }
    /// <summary>应用 F(4,3) 输入变换；一维重载写入各变换分量，面板重载处理当前空间图块。</summary>
    /// <param name="d0">一维变换的第 1 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d1">一维变换的第 2 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d2">一维变换的第 3 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d3">一维变换的第 4 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d4">一维变换的第 5 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d5">一维变换的第 6 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="output">接收一维变换结果的跨度，长度须满足变换所需分量数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Transform(float d0, float d1, float d2, float d3, float d4, float d5, Span<float> output)
    {
        output[0] = 4 * d0 - 5 * d2 + d4;
        output[1] = -4 * (d1 + d2) + d3 + d4;
        output[2] = 4 * (d1 - d2) - d3 + d4;
        output[3] = -2 * d1 - d2 + 2 * d3 + d4;
        output[4] = 2 * d1 - d2 - 2 * d3 + d4;
        output[5] = 4 * d1 - 5 * d3 + d5;
    }
    /// <summary>应用 F(4,3) 逆变换；一维重载还原输出分量，面板重载写回有效空间位置。</summary>
    /// <param name="d0">一维变换的第 1 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d1">一维变换的第 2 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d2">一维变换的第 3 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d3">一维变换的第 4 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d4">一维变换的第 5 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d5">一维变换的第 6 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="output">接收一维变换结果的跨度，长度须满足变换所需分量数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Inverse(float d0, float d1, float d2, float d3, float d4, float d5, Span<float> output)
    {
        output[0] = d0 + d1 + d2 + d3 + d4;
        output[1] = d1 - d2 + 2 * (d3 - d4);
        output[2] = d1 + d2 + 4 * (d3 + d4);
        output[3] = d1 - d2 + 8 * (d3 - d4) + d5;
    }
    /// <summary>对八个图块应用 F(4,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="transformed">按 Winograd 坐标排列的输入变换缓冲区。</param>
    /// <param name="source">当前读取在源数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    private static unsafe void TransformEight(float[] input, float[][] transformed, int source, int width, int target)
    {
        Span<Vector256<float>> temp = stackalloc Vector256<float>[36]; Span<Vector256<float>> result = stackalloc Vector256<float>[6];
        var indices = Vector256.Create(0, 4, 8, 12, 16, 20, 24, 28);
        fixed (float* ptr = input)
            for (int c = 0; c < 6; c++)
            {
                float* start = ptr + source + c;
                var d0 = Avx2.GatherVector256(start, indices, 4); var d1 = Avx2.GatherVector256(start + width, indices, 4);
                var d2 = Avx2.GatherVector256(start + width * 2, indices, 4); var d3 = Avx2.GatherVector256(start + width * 3, indices, 4);
                var d4 = Avx2.GatherVector256(start + width * 4, indices, 4); var d5 = Avx2.GatherVector256(start + width * 5, indices, 4);
                Transform(d0, d1, d2, d3, d4, d5, result);
                for (int r = 0; r < 6; r++) temp[r * 6 + c] = result[r];
            }
        for (int r = 0; r < 6; r++)
        {
            Transform(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
            for (int c = 0; c < 6; c++) result[c].CopyTo(transformed[r * 6 + c].AsSpan(target));
        }
    }
    /// <summary>应用 F(4,3) 输入变换；一维重载写入各变换分量，面板重载处理当前空间图块。</summary>
    /// <param name="d0">一维变换的第 1 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d1">一维变换的第 2 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d2">一维变换的第 3 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d3">一维变换的第 4 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d4">一维变换的第 5 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d5">一维变换的第 6 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="output">接收一维变换结果的跨度，长度须满足变换所需分量数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Transform(Vector256<float> d0, Vector256<float> d1, Vector256<float> d2, Vector256<float> d3, Vector256<float> d4, Vector256<float> d5, Span<Vector256<float>> output)
    {
        var two = Vector256.Create(2f); var four = Vector256.Create(4f); var five = Vector256.Create(5f);
        output[0] = four * d0 - five * d2 + d4;
        output[1] = -four * (d1 + d2) + d3 + d4;
        output[2] = four * (d1 - d2) - d3 + d4;
        output[3] = -two * d1 - d2 + two * d3 + d4;
        output[4] = two * d1 - d2 - two * d3 + d4;
        output[5] = four * d1 - five * d3 + d5;
    }
    /// <summary>对八个图块应用 F(4,3) 逆变换，恢复空间输出并按签名融合偏置与激活。</summary>
    /// <param name="products">按变换坐标排列的通道乘积缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    private static unsafe void InverseEight(float[][] products, float[] output, int tile, int target, int width, float bias)
    {
        Span<Vector256<float>> temp = stackalloc Vector256<float>[24]; Span<Vector256<float>> result = stackalloc Vector256<float>[4];
        for (int c = 0; c < 6; c++)
        {
            Inverse(Vector256.LoadUnsafe(ref products[c][tile]), Vector256.LoadUnsafe(ref products[6 + c][tile]), Vector256.LoadUnsafe(ref products[12 + c][tile]), Vector256.LoadUnsafe(ref products[18 + c][tile]), Vector256.LoadUnsafe(ref products[24 + c][tile]), Vector256.LoadUnsafe(ref products[30 + c][tile]), result);
            for (int r = 0; r < 4; r++) temp[r * 6 + c] = result[r];
        }
        var bv = Vector256.Create(bias);
        fixed (float* ptr = output)
            for (int r = 0; r < 4; r++)
            {
                Inverse(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                InterleaveFour(result[0] + bv, result[1] + bv, result[2] + bv, result[3] + bv, ptr + target + r * width);
            }
    }
    /// <summary>对四个图块应用 F(4,3) 逆变换，恢复空间输出并按签名融合偏置与激活。</summary>
    /// <param name="products">按变换坐标排列的通道乘积缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    private static unsafe void InverseFour(float[][] products, float[] output, int tile, int target, int width, float bias)
    {
        Span<Vector128<float>> temp = stackalloc Vector128<float>[24]; Span<Vector128<float>> result = stackalloc Vector128<float>[4];
        for (int c = 0; c < 6; c++)
        {
            CpuWinogradPanels.Inverse4(Vector128.LoadUnsafe(ref products[c][tile]), Vector128.LoadUnsafe(ref products[6 + c][tile]),
                Vector128.LoadUnsafe(ref products[12 + c][tile]), Vector128.LoadUnsafe(ref products[18 + c][tile]),
                Vector128.LoadUnsafe(ref products[24 + c][tile]), Vector128.LoadUnsafe(ref products[30 + c][tile]), result);
            for (int r = 0; r < 4; r++) temp[r * 6 + c] = result[r];
        }
        var bv = Vector128.Create(bias);
        fixed (float* ptr = output)
            for (int r = 0; r < 4; r++)
            {
                CpuWinogradPanels.Inverse4(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                var v0 = result[0] + bv; var v1 = result[1] + bv; var v2 = result[2] + bv; var v3 = result[3] + bv;
                var lo01 = Sse.UnpackLow(v0, v1); var lo23 = Sse.UnpackLow(v2, v3);
                var hi01 = Sse.UnpackHigh(v0, v1); var hi23 = Sse.UnpackHigh(v2, v3);
                float* dst = ptr + target + r * width;
                Sse.Store(dst, Sse.Shuffle(lo01, lo23, 0x44)); Sse.Store(dst + 4, Sse.Shuffle(lo01, lo23, 0xee));
                Sse.Store(dst + 8, Sse.Shuffle(hi01, hi23, 0x44)); Sse.Store(dst + 12, Sse.Shuffle(hi01, hi23, 0xee));
            }
    }
    /// <summary>应用 F(4,3) 逆变换；一维重载还原输出分量，面板重载写回有效空间位置。</summary>
    /// <param name="d0">一维变换的第 1 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d1">一维变换的第 2 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d2">一维变换的第 3 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d3">一维变换的第 4 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d4">一维变换的第 5 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d5">一维变换的第 6 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="output">接收一维变换结果的跨度，长度须满足变换所需分量数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Inverse(Vector256<float> d0, Vector256<float> d1, Vector256<float> d2, Vector256<float> d3, Vector256<float> d4, Vector256<float> d5, Span<Vector256<float>> output)
    {
        output[0] = d0 + d1 + d2 + d3 + d4;
        output[1] = d1 - d2 + Vector256.Create(2f) * (d3 - d4);
        output[2] = d1 + d2 + Vector256.Create(4f) * (d3 + d4);
        output[3] = d1 - d2 + Vector256.Create(8f) * (d3 - d4) + d5;
    }
    /// <summary>将四个输出列的向量按图块交错写回连续的 4×4 输出块。</summary>
    /// <param name="v0">当前图块第一个输出列的 SIMD 向量。</param>
    /// <param name="v1">当前图块第二个输出列的 SIMD 向量。</param>
    /// <param name="v2">当前图块第三个输出列的 SIMD 向量。</param>
    /// <param name="v3">当前图块第四个输出列的 SIMD 向量。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void InterleaveFour(Vector256<float> v0, Vector256<float> v1, Vector256<float> v2, Vector256<float> v3, float* output)
    {
        var a = Avx.UnpackLow(v0, v1); var b = Avx.UnpackLow(v2, v3);
        var c = Avx.UnpackHigh(v0, v1); var d = Avx.UnpackHigh(v2, v3);
        var t04 = Avx.Shuffle(a, b, 0x44); var t15 = Avx.Shuffle(a, b, 0xee);
        var t26 = Avx.Shuffle(c, d, 0x44); var t37 = Avx.Shuffle(c, d, 0xee);
        Avx.Store(output, Avx.Permute2x128(t04, t15, 0x20)); Avx.Store(output + 8, Avx.Permute2x128(t26, t37, 0x20));
        Avx.Store(output + 16, Avx.Permute2x128(t04, t15, 0x31)); Avx.Store(output + 24, Avx.Permute2x128(t26, t37, 0x31));
    }
}
