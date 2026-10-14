using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>实现 F(2×2,3×3) 卷积，每个输出块以 16 个变换域通道乘积替代直接卷积的 36 个。</summary>
internal static class CpuWinograd
{
    /// <summary>按常量权重张量保存的弱引用 Winograd 变换缓存。</summary>
    private static readonly ConditionalWeakTable<Tensor, float[][]> Weights = new();
    // F(2×2,3×3) 每个输出块只需 16 个通道乘积，直接卷积需要 36 个。
    /// <summary>以 F(2,3) 变换权重和输入，缓存只读权重，经变换域通道乘法与逆变换生成 3×3 卷积结果。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="w">卷积权重，布局为 [输出通道,每组输入通道,核高,核宽]。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    public static Tensor Conv(CpuBackend backend, Tensor x, Tensor w, Tensor? bias, bool silu)
    {
        int batch = x.Shape[0], ic = x.Shape[1], h = x.Shape[2], width = x.Shape[3], oc = w.Shape[0];
        int th = (h + 1) / 2, tw = (width + 1) / 2, tiles = th * tw;
        // 用 F(2,3) 权重矩阵变换每个输入/输出通道的 3×3 核，生成 16 个坐标矩阵。
        // 参数 weight：当前深度卷积或 Winograd 变换使用的权重张量。
        // 返回值：按变换坐标排列的权重或输入数组。
        float[][] Pack(Tensor weight)
        {
            var result = Enumerable.Range(0, 16).Select(_ => new float[oc * ic]).ToArray();
            Span<float> temp = stackalloc float[12];
            for (int co = 0; co < oc; co++) for (int ci = 0; ci < ic; ci++)
            {
                int offset = (co * ic + ci) * 9;
                for (int c = 0; c < 3; c++)
                {
                    float a = weight.Data[offset + c], b = weight.Data[offset + 3 + c], d = weight.Data[offset + 6 + c];
                    temp[c] = a; temp[3 + c] = (a + b + d) * .5f; temp[6 + c] = (a - b + d) * .5f; temp[9 + c] = d;
                }
                for (int r = 0; r < 4; r++)
                {
                    float a = temp[r * 3], b = temp[r * 3 + 1], c = temp[r * 3 + 2]; int index = co * ic + ci;
                    result[r * 4][index] = a; result[r * 4 + 1][index] = (a + b + c) * .5f;
                    result[r * 4 + 2][index] = (a - b + c) * .5f; result[r * 4 + 3][index] = c;
                }
            }
            return result;
        }
        float[][] weights = w.IsConstant ? Weights.GetValue(w, Pack) : Pack(w);
        if (CpuWinograd24.ShouldUse(backend, tiles, ic, oc, w.IsConstant))
            return CpuWinograd24.Conv(backend, x, weights, bias, silu);
        if (CpuWinogradPanels.ShouldUse(backend, tiles, ic, oc))
            return CpuWinogradPanels.Conv(backend, x, weights, bias, silu, false, w.IsConstant);
        var output = Tensor.TemporaryUninitialized(batch, oc, h, width); float[] input = x.Data, target = output.Data;
        var transformed = new float[16][]; var products = new float[16][];
        for (int i = 0; i < 16; i++) { transformed[i] = ArrayPool<float>.Shared.Rent(ic * tiles); products[i] = ArrayPool<float>.Shared.Rent(oc * tiles); }
        try
        {
            for (int b = 0; b < batch; b++)
            {
                CpuScheduling.For(0, ic, backend.Parallel, ci =>
                {
                    Span<float> patch = stackalloc float[16]; Span<float> temp = stackalloc float[16];
                    for (int ty = 0; ty < th; ty++) for (int tx = 0; tx < tw; tx++)
                    {
                        if (Avx2.IsSupported && ty > 0 && ty * 2 + 2 < h && tx > 0 && tx + 8 < tw)
                        {
                            TransformEight(input, transformed, ((b * ic + ci) * h + ty * 2 - 1) * width + tx * 2 - 1, width, ci * tiles + ty * tw + tx);
                            tx += 7;
                            continue;
                        }
                        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
                        {
                            int y = ty * 2 - 1 + r, xx = tx * 2 - 1 + c;
                            patch[r * 4 + c] = (uint)y < h && (uint)xx < width ? input[((b * ic + ci) * h + y) * width + xx] : 0;
                        }
                        for (int c = 0; c < 4; c++)
                        {
                            float d0 = patch[c], d1 = patch[4 + c], d2 = patch[8 + c], d3 = patch[12 + c];
                            temp[c] = d0 - d2; temp[4 + c] = d1 + d2; temp[8 + c] = d2 - d1; temp[12 + c] = d1 - d3;
                        }
                        int index = ci * tiles + ty * tw + tx;
                        for (int r = 0; r < 4; r++)
                        {
                            float d0 = temp[r * 4], d1 = temp[r * 4 + 1], d2 = temp[r * 4 + 2], d3 = temp[r * 4 + 3];
                            transformed[r * 4][index] = d0 - d2; transformed[r * 4 + 1][index] = d1 + d2;
                            transformed[r * 4 + 2][index] = d2 - d1; transformed[r * 4 + 3][index] = d1 - d3;
                        }
                    }
                });
                CpuGemm.MultiplyBatch(weights, transformed, products, oc, tiles, ic, backend.Parallel, w.IsConstant);
                CpuScheduling.For(0, oc, backend.Parallel, co =>
                {
                    for (int ty = 0; ty < th; ty++) for (int tx = 0; tx < tw; tx++)
                    {
                        int tile = co * tiles + ty * tw + tx;
                        if (Avx.IsSupported && tx + 8 <= tw && (tx + 8) * 2 <= width && ty * 2 + 1 < h)
                        {
                            InverseEight(products, target, tile, ((b * oc + co) * h + ty * 2) * width + tx * 2, width, bias?.Data[co] ?? 0);
                            tx += 7;
                            continue;
                        }
                        float t00 = products[0][tile] + products[4][tile] + products[8][tile];
                        float t01 = products[1][tile] + products[5][tile] + products[9][tile];
                        float t02 = products[2][tile] + products[6][tile] + products[10][tile];
                        float t03 = products[3][tile] + products[7][tile] + products[11][tile];
                        float t10 = products[4][tile] - products[8][tile] - products[12][tile];
                        float t11 = products[5][tile] - products[9][tile] - products[13][tile];
                        float t12 = products[6][tile] - products[10][tile] - products[14][tile];
                        float t13 = products[7][tile] - products[11][tile] - products[15][tile];
                        float bv = bias?.Data[co] ?? 0; int y = ty * 2, xx = tx * 2, dst = ((b * oc + co) * h + y) * width + xx;
                        target[dst] = t00 + t01 + t02 + bv;
                        if (xx + 1 < width) target[dst + 1] = t01 - t02 - t03 + bv;
                        if (y + 1 < h) { target[dst + width] = t10 + t11 + t12 + bv; if (xx + 1 < width) target[dst + width + 1] = t11 - t12 - t13 + bv; }
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

    /// <summary>对八个图块应用 F(2,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="transformed">按 Winograd 坐标排列的输入变换缓冲区。</param>
    /// <param name="source">当前读取在源数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    private static unsafe void TransformEight(float[] input, float[][] transformed, int source, int width, int target)
    {
        Span<Vector256<float>> temp = stackalloc Vector256<float>[16];
        var order = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
        fixed (float* ptr = input)
        {
            for (int c = 0; c < 4; c++)
            {
                float* start = ptr + source + (c & 2);
                var d0 = GatherEvenOdd(start, c, order); var d1 = GatherEvenOdd(start + width, c, order);
                var d2 = GatherEvenOdd(start + width * 2, c, order); var d3 = GatherEvenOdd(start + width * 3, c, order);
                temp[c] = d0 - d2; temp[4 + c] = d1 + d2; temp[8 + c] = d2 - d1; temp[12 + c] = d1 - d3;
            }
        }
        for (int r = 0; r < 4; r++)
        {
            var d0 = temp[r * 4]; var d1 = temp[r * 4 + 1]; var d2 = temp[r * 4 + 2]; var d3 = temp[r * 4 + 3];
            (d0 - d2).CopyTo(transformed[r * 4].AsSpan(target));
            (d1 + d2).CopyTo(transformed[r * 4 + 1].AsSpan(target));
            (d2 - d1).CopyTo(transformed[r * 4 + 2].AsSpan(target));
            (d1 - d3).CopyTo(transformed[r * 4 + 3].AsSpan(target));
        }
    }
    /// <summary>读取相邻像素并按指定排列重组为偶数或奇数图块采样向量。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="order">重排偶数和奇数采样通道的索引向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe Vector256<float> GatherEvenOdd(float* source, int column, Vector256<int> order)
    {
        var left = Avx.LoadVector256(source); var right = Avx.LoadVector256(source + 8);
        var shuffled = (column & 1) == 0 ? Avx.Shuffle(left, right, 0x88) : Avx.Shuffle(left, right, 0xdd);
        return Avx2.PermuteVar8x32(shuffled, order);
    }

    /// <summary>对八个图块应用 F(2,3) 逆变换，恢复空间输出并按签名融合偏置与激活。</summary>
    /// <param name="products">按变换坐标排列的通道乘积缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    private static unsafe void InverseEight(float[][] products, float[] output, int tile, int target, int width, float bias)
    {
        var bv = Vector256.Create(bias);
        var t00 = Vector256.LoadUnsafe(ref products[0][tile]) + Vector256.LoadUnsafe(ref products[4][tile]) + Vector256.LoadUnsafe(ref products[8][tile]);
        var t01 = Vector256.LoadUnsafe(ref products[1][tile]) + Vector256.LoadUnsafe(ref products[5][tile]) + Vector256.LoadUnsafe(ref products[9][tile]);
        var t02 = Vector256.LoadUnsafe(ref products[2][tile]) + Vector256.LoadUnsafe(ref products[6][tile]) + Vector256.LoadUnsafe(ref products[10][tile]);
        var t03 = Vector256.LoadUnsafe(ref products[3][tile]) + Vector256.LoadUnsafe(ref products[7][tile]) + Vector256.LoadUnsafe(ref products[11][tile]);
        var t10 = Vector256.LoadUnsafe(ref products[4][tile]) - Vector256.LoadUnsafe(ref products[8][tile]) - Vector256.LoadUnsafe(ref products[12][tile]);
        var t11 = Vector256.LoadUnsafe(ref products[5][tile]) - Vector256.LoadUnsafe(ref products[9][tile]) - Vector256.LoadUnsafe(ref products[13][tile]);
        var t12 = Vector256.LoadUnsafe(ref products[6][tile]) - Vector256.LoadUnsafe(ref products[10][tile]) - Vector256.LoadUnsafe(ref products[14][tile]);
        var t13 = Vector256.LoadUnsafe(ref products[7][tile]) - Vector256.LoadUnsafe(ref products[11][tile]) - Vector256.LoadUnsafe(ref products[15][tile]);
        fixed (float* ptr = output)
        {
            Interleave(t00 + t01 + t02 + bv, t01 - t02 - t03 + bv, ptr + target);
            Interleave(t10 + t11 + t12 + bv, t11 - t12 - t13 + bv, ptr + target + width);
        }
    }
    /// <summary>将偶数和奇数位置的 SIMD 向量交错写回连续输出。</summary>
    /// <param name="even">偶数位置的 SIMD 采样或输出向量。</param>
    /// <param name="odd">奇数位置的 SIMD 采样或输出向量。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void Interleave(Vector256<float> even, Vector256<float> odd, float* output)
    {
        var low = Avx.UnpackLow(even, odd); var high = Avx.UnpackHigh(even, odd);
        Avx.Store(output, Avx.Permute2x128(low, high, 0x20));
        Avx.Store(output + 8, Avx.Permute2x128(low, high, 0x31));
    }
}
