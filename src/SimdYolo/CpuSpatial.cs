using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>提供最近邻缩放、深度卷积、池化和特定转置卷积的 CPU SIMD 实现。</summary>
internal static class CpuSpatial
{
    /// <summary>按常量权重张量保存的转置卷积矩阵缓存。</summary>
    private static readonly ConditionalWeakTable<Tensor, float[]> TransposedWeights = new();
    /// <summary>是否启用展开的 3×3 深度卷积快路径。</summary>
    private static readonly bool DepthwiseThree = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_DEPTHWISE_3X3") == "1";
    /// <summary>按最近邻映射缩放 NCHW 平面，2 倍放大时使用相邻元素重复快路径。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="w">目标空间宽度。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    public static Tensor ResizeNearest(Tensor x, int h, int w)
    {
        int ih = x.Shape[2], iw = x.Shape[3], channels = x.Shape[0] * x.Shape[1];
        var result = Tensor.TemporaryUninitialized(x.Shape[0], x.Shape[1], h, w); float[] input = x.Data, output = result.Data;
        int[] columns = Enumerable.Range(0, w).Select(xx => xx * iw / w).ToArray();
        // 处理一个批次/通道的独立空间平面，将结果写入其独占输出区。
        // 参数 c：当前批次/通道平面的索引。
        void Channel(int c)
        {
            if (h == ih * 2 && w == iw * 2)
                for (int y = 0; y < ih; y++)
                {
                    int src = (c * ih + y) * iw, dst = (c * h + y * 2) * w;
                    RepeatTwo(input, output, src, dst, iw); Array.Copy(output, dst, output, dst + w, w);
                }
            else
                for (int y = 0; y < h; y++)
                {
                    int src = (c * ih + y * ih / h) * iw, dst = (c * h + y) * w;
                    for (int xx = 0; xx < w; xx++) output[dst + xx] = input[src + columns[xx]];
                }
        }
        if (result.Length > 32768 && channels > 1) CpuScheduling.For(0, channels, CpuScheduling.Options, Channel);
        else for (int c = 0; c < channels; c++) Channel(c);
        return result;
    }
    /// <summary>将一个输入行的每个采样重复两次写入输出行。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="source">当前读取在源数组中的起始元素偏移。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="length">需要读取或重复的有效输入元素数。</param>
    private static unsafe void RepeatTwo(float[] input, float[] output, int source, int target, int length)
    {
        fixed (float* src = input, dst = output)
        {
            int j = 0;
            if (Avx.IsSupported)
                for (; j <= length - 8; j += 8)
                {
                    var value = Avx.LoadVector256(src + source + j); var low = Avx.UnpackLow(value, value); var high = Avx.UnpackHigh(value, value);
                    Avx.Store(dst + target + j * 2, Avx.Permute2x128(low, high, 0x20));
                    Avx.Store(dst + target + j * 2 + 8, Avx.Permute2x128(low, high, 0x31));
                }
            for (; j < length; j++) dst[target + j * 2] = dst[target + j * 2 + 1] = src[source + j];
        }
    }
    /// <summary>为每个批次和通道计算独立卷积，优先尝试小平面填充或专用 SIMD 路径。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="w">卷积权重，布局为 [输出通道,每组输入通道,核高,核宽]。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="strides">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    public static void Depthwise(Tensor x, Tensor w, Tensor? bias, Tensor output, int[] strides, int[] pads, int[] dilation, bool silu, ParallelOptions parallel)
    {
        if (CpuDepthwisePadded.TryRun(x, w, bias, output, strides, pads, dilation, silu, parallel)) return;
        int channels = x.Shape[0] * x.Shape[1], h = x.Shape[2], width = x.Shape[3], kh = w.Shape[2], kw = w.Shape[3], oh = output.Shape[2], ow = output.Shape[3];
        float[] input = x.Data, weights = w.Data, target = output.Data;
        CpuScheduling.For(0, channels, parallel, bc =>
        {
            int co = bc % x.Shape[1], weight = co * kh * kw, channel = bc * h * width; float bv = bias?.Data[co] ?? 0;
            for (int oy = 0; oy < oh; oy++)
            {
                int ox = 0;
                for (; ox < ow; ox++)
                {
                    int iy0 = oy * strides[0] - pads[0], ix0 = ox * strides[1] - pads[1], dst = (bc * oh + oy) * ow + ox;
                    if (DepthwiseThree && Avx2.IsSupported && Fma.IsSupported && kh == 3 && kw == 3 && dilation[0] == 1 && dilation[1] == 1 &&
                        strides[1] is 1 or 2 && iy0 >= 0 && iy0 + 2 < h && ix0 >= 0 && ox <= ow - 8 && (long)ix0 + 7 * strides[1] + 2 < width)
                    {
                        DepthwiseThreeByThree(input, weights, target, channel, weight, iy0, ix0, width, strides[1], dst, bv, silu);
                        ox += 7;
                    }
                    else if (Avx2.IsSupported && Fma.IsSupported && strides[1] == 2 && ix0 >= 0 && ox <= ow - 8 && (long)ix0 + 16 + (long)(kw - 1) * dilation[1] <= width)
                    {
                        DepthwiseStrideTwo(input, weights, target, channel, weight, iy0, ix0, h, width, kh, kw, dilation, dst, bv, silu);
                        ox += 7;
                    }
                    else if (strides[1] == 1 && ix0 >= 0 && ox <= ow - Vector<float>.Count && (long)ix0 + Vector<float>.Count + (long)(kw - 1) * dilation[1] <= width)
                    {
                        var value = new Vector<float>(bv);
                        for (int ky = 0; ky < kh; ky++)
                        {
                            int iy = iy0 + ky * dilation[0]; if ((uint)iy >= h) continue;
                            for (int kx = 0; kx < kw; kx++) value = Vector.MultiplyAddEstimate(new Vector<float>(input, channel + iy * width + ix0 + kx * dilation[1]), new Vector<float>(weights[weight + ky * kw + kx]), value);
                        }
                        if (silu) value = CpuActivation.Silu(value);
                        value.CopyTo(target, dst); ox += Vector<float>.Count - 1;
                    }
                    else
                    {
                        float value = bv;
                        for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
                        {
                            int iy = iy0 + ky * dilation[0], ix = ix0 + kx * dilation[1];
                            if ((uint)iy < h && (uint)ix < width) value += input[channel + iy * width + ix] * weights[weight + ky * kw + kx];
                        }
                        target[dst] = silu ? value * TensorOps.Sigmoid(value) : value;
                    }
                }
            }
        });
    }
    /// <summary>对无边界的 3×3 区域展开九项 FMA；步长为 2 时共享相邻输出所需的输入加载。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="channel">当前输入通道的零基索引，或该通道在输入数组中的起始偏移。</param>
    /// <param name="weight">当前通道卷积核在权重数组中的起始元素偏移。</param>
    /// <param name="iy">当前卷积输出所对应的输入起始行。</param>
    /// <param name="ix">当前卷积输出所对应的输入起始列。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="stride">卷积的水平采样步长。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    private static unsafe void DepthwiseThreeByThree(float[] input, float[] weights, float[] output, int channel, int weight,
                                                    int iy, int ix, int width, int stride, int target, float bias, bool silu)
    {
        var value = Vector256.Create(bias);
        fixed (float* src = input, kernel = weights, dst = output)
        {
            float* row = src + channel + iy * width + ix;
            if (stride == 1)
            {
                value = Fma.MultiplyAdd(Avx.LoadVector256(row), Vector256.Create(kernel[weight]), value);
                value = Fma.MultiplyAdd(Avx.LoadVector256(row + 1), Vector256.Create(kernel[weight + 1]), value);
                value = Fma.MultiplyAdd(Avx.LoadVector256(row + 2), Vector256.Create(kernel[weight + 2]), value);
                row += width;
                value = Fma.MultiplyAdd(Avx.LoadVector256(row), Vector256.Create(kernel[weight + 3]), value);
                value = Fma.MultiplyAdd(Avx.LoadVector256(row + 1), Vector256.Create(kernel[weight + 4]), value);
                value = Fma.MultiplyAdd(Avx.LoadVector256(row + 2), Vector256.Create(kernel[weight + 5]), value);
                row += width;
                value = Fma.MultiplyAdd(Avx.LoadVector256(row), Vector256.Create(kernel[weight + 6]), value);
                value = Fma.MultiplyAdd(Avx.LoadVector256(row + 1), Vector256.Create(kernel[weight + 7]), value);
                value = Fma.MultiplyAdd(Avx.LoadVector256(row + 2), Vector256.Create(kernel[weight + 8]), value);
            }
            else
            {
                var order = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
                var shift = Vector256.Create(1, 2, 3, 4, 5, 6, 7, 0);
                for (int ky = 0; ky < 3; ky++, row += width)
                {
                    var first = Avx.LoadVector256(row);
                    var second = Avx.LoadVector256(row + 8);
                    var even = Avx2.PermuteVar8x32(Avx.Shuffle(first, second, 0x88), order);
                    var odd = Avx2.PermuteVar8x32(Avx.Shuffle(first, second, 0xdd), order);
                    var next = Avx.Blend(Avx2.PermuteVar8x32(even, shift), Vector256.Create(row[16]), 0x80);
                    value = Fma.MultiplyAdd(even, Vector256.Create(kernel[weight + ky * 3]), value);
                    value = Fma.MultiplyAdd(odd, Vector256.Create(kernel[weight + ky * 3 + 1]), value);
                    value = Fma.MultiplyAdd(next, Vector256.Create(kernel[weight + ky * 3 + 2]), value);
                }
            }
            if (silu) value = CpuActivation.Silu(value);
            Avx.Store(dst + target, value);
        }
    }
    /// <summary>以偶数位置采样计算步长为 2 的八个深度卷积输出，并处理偏置和可选 SiLU。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="channel">当前输入通道的零基索引，或该通道在输入数组中的起始偏移。</param>
    /// <param name="weight">当前通道卷积核在权重数组中的起始元素偏移。</param>
    /// <param name="iy0">当前卷积输出所对应的输入起始行。</param>
    /// <param name="ix0">当前卷积输出所对应的输入起始列。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="kh">卷积核高度。</param>
    /// <param name="kw">卷积核宽度。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    private static unsafe void DepthwiseStrideTwo(float[] input, float[] weights, float[] output, int channel, int weight, int iy0, int ix0, int h, int width, int kh, int kw, int[] dilation, int target, float bias, bool silu)
    {
        var order = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7); var value = Vector256.Create(bias);
        fixed (float* src = input, dst = output)
        {
            for (int ky = 0; ky < kh; ky++)
            {
                int iy = iy0 + ky * dilation[0]; if ((uint)iy >= h) continue;
                for (int kx = 0; kx < kw; kx++)
                {
                    float* start = src + channel + iy * width + ix0 + kx * dilation[1];
                    var samples = Avx2.PermuteVar8x32(Avx.Shuffle(Avx.LoadVector256(start), Avx.LoadVector256(start + 8), 0x88), order);
                    value = Fma.MultiplyAdd(samples, Vector256.Create(weights[weight + ky * kw + kx]), value);
                }
            }
            if (silu) value = CpuActivation.Silu(value);
            Avx.Store(dst + target, value);
        }
    }
    /// <summary>在给定窗口、步长和填充下计算平均池化，按配置决定分母是否计入填充。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="kernel">池化窗口尺寸，或方形深度卷积核的边长。</param>
    /// <param name="stride">空间步长，顺序为高、宽。</param>
    /// <param name="pad">空间填充，顺序为上、左、下、右。</param>
    /// <param name="oh">输出平面高度。</param>
    /// <param name="ow">输出平面宽度。</param>
    /// <param name="countIncludePad">平均池化的分母是否包含填充位置。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    public static Tensor AveragePool(Tensor x, int[] kernel, int[] stride, int[] pad, int oh, int ow, bool countIncludePad)
    {
        if (CpuAverageTwo.ShouldUse(x, kernel, stride, pad, oh, ow)) return CpuAverageTwo.Run(x);
        int channels = x.Shape[0] * x.Shape[1], h = x.Shape[2], w = x.Shape[3], kh = kernel[0], kw = kernel[1];
        var output = Tensor.TemporaryUninitialized(x.Shape[0], x.Shape[1], oh, ow); float[] data = x.Data, target = output.Data;
        // 处理一个批次/通道的独立空间平面，将结果写入其独占输出区。
        // 参数 c：当前批次/通道平面的索引。
        void Channel(int c)
        {
            for (int y = 0; y < oh; y++) for (int xx = 0; xx < ow; xx++)
            {
                int sy0 = y * stride[0] - pad[0], sx0 = xx * stride[1] - pad[1], dst = (c * oh + y) * ow + xx;
                if (stride[1] == 1 && sx0 >= 0 && xx <= ow - Vector<float>.Count && sx0 + Vector<float>.Count + kw - 1 <= w)
                {
                    var sum = Vector<float>.Zero; int rows = 0;
                    for (int ky = 0; ky < kh; ky++)
                    {
                        int sy = sy0 + ky; if ((uint)sy >= h) continue;
                        rows++;
                        for (int kx = 0; kx < kw; kx++) sum += new Vector<float>(data, (c * h + sy) * w + sx0 + kx);
                    }
                    (sum / new Vector<float>(countIncludePad ? kh * kw : rows * kw)).CopyTo(target, dst);
                    xx += Vector<float>.Count - 1;
                }
                else
                {
                    float sum = 0; int count = 0;
                    for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
                    {
                        int sy = sy0 + ky, sx = sx0 + kx;
                        if ((uint)sy < h && (uint)sx < w) { sum += data[(c * h + sy) * w + sx]; count++; }
                    }
                    target[dst] = sum / (countIncludePad ? kh * kw : count);
                }
            }
        }
        if ((long)channels * oh * ow >= 32768 && channels > 1) CpuScheduling.For(0, channels, CpuScheduling.Options, Channel);
        else for (int c = 0; c < channels; c++) Channel(c);
        return output;
    }
    /// <summary>计算无填充、步长为 2 的 2×2 转置卷积，将矩阵乘积散布到四种输出奇偶坐标。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="w">转置卷积权重，布局为 [输入通道,每组输出通道,核高,核宽]。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    public static Tensor UpsampleConv(Tensor x, Tensor w, Tensor? bias)
    {
        int batch = x.Shape[0], ic = x.Shape[1], oc = w.Shape[1], h = x.Shape[2], width = x.Shape[3], spatial = h * width;
        // 将 2×2 转置卷积权重按四种输出奇偶坐标重排为矩阵行。
        // 参数 tensor：需要验证形状与存储长度的张量。
        // 返回值：满足当前布局或长度要求的连续 FP32 数组。
        float[] Pack(Tensor tensor)
        {
            var result = new float[tensor.Length];
            for (int co = 0; co < oc; co++) for (int phase = 0; phase < 4; phase++) for (int ci = 0; ci < ic; ci++)
                result[(co * 4 + phase) * ic + ci] = tensor.Data[(ci * oc + co) * 4 + phase];
            return result;
        }
        float[] weights = w.IsConstant ? TransposedWeights.GetValue(w, Pack) : Pack(w);
        var output = Tensor.TemporaryUninitialized(batch, oc, h * 2, width * 2); var temp = ArrayPool<float>.Shared.Rent(oc * 4 * spatial);
        try
        {
            for (int b = 0; b < batch; b++)
            {
                var input = batch == 1 ? x.Data : x.Data.AsSpan(b * ic * spatial, ic * spatial).ToArray();
                CpuGemm.Multiply(weights, input, temp, oc * 4, spatial, ic, CpuScheduling.Options);
                CpuScheduling.For(0, oc, CpuScheduling.Options, co =>
                {
                    float bv = bias?.Data[co] ?? 0; int target = (b * oc + co) * spatial * 4;
                    for (int y = 0; y < h; y++) for (int xx = 0; xx < width; xx++)
                    {
                        int a = y * width + xx, dst = target + y * width * 4 + xx * 2;
                        if (Avx.IsSupported && xx <= width - 8)
                        {
                            ScatterEight(temp, output.Data, co * 4 * spatial + a, spatial, dst, width * 2, bv);
                            xx += 7;
                            continue;
                        }
                        output.Data[dst] = temp[co * 4 * spatial + a] + bv;
                        output.Data[dst + 1] = temp[(co * 4 + 1) * spatial + a] + bv;
                        output.Data[dst + width * 2] = temp[(co * 4 + 2) * spatial + a] + bv;
                        output.Data[dst + width * 2 + 1] = temp[(co * 4 + 3) * spatial + a] + bv;
                    }
                });
            }
        }
        finally { ArrayPool<float>.Shared.Return(temp); }
        return output;
    }
    /// <summary>将八个矩阵结果按目标空间间距散布到输出，并补加通道偏置。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="target">接收当前面板结果的预分配 FP32 缓冲区。</param>
    /// <param name="index">工作线程、任务或输出元素的索引。</param>
    /// <param name="spatial">每个通道的空间元素数。</param>
    /// <param name="output">当前输出在目标数组中的起始元素偏移。</param>
    /// <param name="stride">目标输出平面的行跨度，以 FP32 元素数表示。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    private static unsafe void ScatterEight(float[] source, float[] target, int index, int spatial, int output, int stride, float bias)
    {
        fixed (float* src = source, dst = target)
        {
            var vb = Vector256.Create(bias);
            var a = Avx.LoadVector256(src + index) + vb; var b = Avx.LoadVector256(src + index + spatial) + vb;
            var low = Avx.UnpackLow(a, b); var high = Avx.UnpackHigh(a, b);
            Avx.Store(dst + output, Avx.Permute2x128(low, high, 0x20)); Avx.Store(dst + output + 8, Avx.Permute2x128(low, high, 0x31));
            a = Avx.LoadVector256(src + index + spatial * 2) + vb; b = Avx.LoadVector256(src + index + spatial * 3) + vb;
            low = Avx.UnpackLow(a, b); high = Avx.UnpackHigh(a, b);
            Avx.Store(dst + output + stride, Avx.Permute2x128(low, high, 0x20)); Avx.Store(dst + output + stride + 8, Avx.Permute2x128(low, high, 0x31));
        }
    }
    /// <summary>在给定窗口内选择最大值；窗口越界位置不参与最大值归约。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="kernel">池化窗口尺寸，或方形深度卷积核的边长。</param>
    /// <param name="stride">空间步长，顺序为高、宽。</param>
    /// <param name="pad">空间填充，顺序为上、左、下、右。</param>
    /// <param name="oh">输出平面高度。</param>
    /// <param name="ow">输出平面宽度。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    public static Tensor MaxPool(Tensor x, int[] kernel, int[] stride, int[] pad, int oh, int ow)
    {
        int channels = x.Shape[0] * x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var output = Tensor.TemporaryUninitialized(x.Shape[0], x.Shape[1], oh, ow); float[] data = x.Data, target = output.Data;
        var horizontal = ArrayPool<float>.Shared.Rent(channels * h * ow);
        try
        {
            // 处理一个批次/通道的独立空间平面，将结果写入其独占输出区。
            // 参数 c：当前批次/通道平面的索引。
            void Channel(int c)
            {
                for (int y = 0; y < h; y++)
                {
                    int xx = 0, src = (c * h + y) * w, dst = (c * h + y) * ow;
                    while (xx < ow)
                    {
                        int sx = xx * stride[1] - pad[1];
                        if (stride[1] == 1 && xx <= ow - Vector<float>.Count && sx >= 0 && sx + Vector<float>.Count + kernel[1] - 1 <= w)
                        {
                            var value = new Vector<float>(float.NegativeInfinity);
                            for (int k = 0; k < kernel[1]; k++) value = Vector.Max(value, new Vector<float>(data, src + sx + k));
                            value.CopyTo(horizontal, dst + xx); xx += Vector<float>.Count;
                        }
                        else
                        {
                            float value = float.NegativeInfinity;
                            for (int k = 0; k < kernel[1]; k++) if ((uint)(sx + k) < w) value = MathF.Max(value, data[src + sx + k]);
                            horizontal[dst + xx++] = value;
                        }
                    }
                }
                for (int y = 0; y < oh; y++)
                {
                    int sy = y * stride[0] - pad[0], xx = 0, dst = (c * oh + y) * ow;
                    for (; xx <= ow - Vector<float>.Count; xx += Vector<float>.Count)
                    {
                        var value = new Vector<float>(float.NegativeInfinity);
                        for (int k = 0; k < kernel[0]; k++) if ((uint)(sy + k) < h) value = Vector.Max(value, new Vector<float>(horizontal, (c * h + sy + k) * ow + xx));
                        value.CopyTo(target, dst + xx);
                    }
                    for (; xx < ow; xx++)
                    {
                        float value = float.NegativeInfinity;
                        for (int k = 0; k < kernel[0]; k++) if ((uint)(sy + k) < h) value = MathF.Max(value, horizontal[(c * h + sy + k) * ow + xx]);
                        target[dst + xx] = value;
                    }
                }
            }
            if ((long)channels * h * ow >= 32768) CpuScheduling.For(0, channels, CpuScheduling.Options, Channel);
            else for (int c = 0; c < channels; c++) Channel(c);
        }
        finally { ArrayPool<float>.Shared.Return(horizontal); }
        return output;
    }
}
