using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

// 狭窄的深度卷积平面中，标量边界和尾部处理占比很高。
// 每个参与线程复用一个零填充平面，让全部输出使用连续 SIMD 加载。
// 卷积核行/列的归约次序保持不变。
/// <summary>为小型深度卷积平面构造线程私有的零填充缓冲区，使边界位置也能使用连续 SIMD 加载。</summary>
internal static class CpuDepthwisePadded
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_DEPTHWISE_PADDED") != "0";

    /// <summary>检查深度卷积尺寸、步长、填充和有限权重，符合条件时使用零填充 SIMD 快路径。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="weight">当前深度卷积或 Winograd 变换使用的权重张量。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="strides">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    /// <returns>已完成快路径计算时为 true；条件不满足、输出未由本方法计算时为 false。</returns>
    internal static bool TryRun(Tensor input, Tensor weight, Tensor? bias, Tensor output, int[] strides, int[] pads,
                                int[] dilation, bool silu, ParallelOptions parallel)
    {
        int kernel = weight.Shape[2], h = input.Shape[2], width = input.Shape[3], radius = kernel / 2;
        if (!Enabled || !Avx2.IsSupported || !Fma.IsSupported || kernel is not (3 or 5 or 7) || weight.Shape[3] != kernel ||
            strides[0] != 1 || strides[1] != 1 || dilation[0] != 1 || dilation[1] != 1 ||
            pads.Length != 4 || pads.Any(pad => pad != radius) || output.Shape[2] != h || output.Shape[3] != width ||
            width is < 8 or > 80 || h < kernel || (long)(h + kernel - 1) * (width + kernel - 1) > 16384) return false;
        // 被原实现跳过的填充位置不能改成 0×NaN 或 0×Infinity。
        // 因此含非有限权重的卷积核继续走原有边界判定路径。
        foreach (float value in weight.Data) if (!float.IsFinite(value)) return false;
        Run(input.Data, weight.Data, bias?.Data, output.Data, input.Shape[0], input.Shape[1], h, width, kernel, silu, parallel);
        return true;
    }

    /// <summary>每个参与线程复用一个填充平面，对批次与通道独立地计算深度卷积。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="batch">要处理的批次数。</param>
    /// <param name="channels">要处理的通道数。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="kernel">方形深度卷积核的边长。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static void Run(float[] input, float[] weights, float[]? bias, float[] output, int batch, int channels,
                             int h, int width, int kernel, bool silu, ParallelOptions parallel)
    {
        int radius = kernel / 2, paddedWidth = checked(width + kernel - 1), paddedHeight = checked(h + kernel - 1);
        int size = checked(paddedWidth * paddedHeight);
        CpuScheduling.For(0, checked(batch * channels), parallel,
            () => ArrayPool<float>.Shared.Rent(size),
            (bc, _, padded) =>
            {
                padded.AsSpan(0, size).Clear();
                int source = bc * h * width;
                for (int y = 0; y < h; y++) Array.Copy(input, source + y * width, padded, (y + radius) * paddedWidth + radius, width);
                int c = bc % channels;
                Convolve(padded, weights, output, c * kernel * kernel, bc * h * width, h, width, paddedWidth, kernel, bias?[c] ?? 0, silu);
                return padded;
            },
            padded => ArrayPool<float>.Shared.Return(padded));
    }

    /// <summary>在零填充平面上按核行/核列顺序累积 16、8、4 或单个输出，可融合偏置与 SiLU。</summary>
    /// <param name="padded">已包含零填充的线程私有输入平面。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="weight">当前通道卷积核在权重数组中的起始元素偏移。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="paddedWidth">包含左右填充的临时平面行跨度。</param>
    /// <param name="kernel">方形深度卷积核的边长。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    private static unsafe void Convolve(float[] padded, float[] weights, float[] output, int weight, int target,
                                         int h, int width, int paddedWidth, int kernel, float bias, bool silu)
    {
        fixed (float* src = padded, wk = weights, dst = output)
            for (int y = 0; y < h; y++)
            {
                int x = 0;
                for (; x <= width - 16; x += 16)
                {
                    var a = Vector256.Create(bias);
                    var b = a;
                    for (int ky = 0; ky < kernel; ky++)
                    {
                        float* row = src + (y + ky) * paddedWidth + x;
                        for (int kx = 0; kx < kernel; kx++)
                        {
                            var coefficient = Vector256.Create(wk[weight + ky * kernel + kx]);
                            a = Fma.MultiplyAdd(Avx.LoadVector256(row + kx), coefficient, a);
                            b = Fma.MultiplyAdd(Avx.LoadVector256(row + kx + 8), coefficient, b);
                        }
                    }
                    if (silu) { a = CpuActivation.Silu(a); b = CpuActivation.Silu(b); }
                    Avx.Store(dst + target + y * width + x, a);
                    Avx.Store(dst + target + y * width + x + 8, b);
                }
                for (; x <= width - 8; x += 8)
                {
                    var value = Vector256.Create(bias);
                    for (int ky = 0; ky < kernel; ky++)
                    {
                        float* row = src + (y + ky) * paddedWidth + x;
                        for (int kx = 0; kx < kernel; kx++) value = Fma.MultiplyAdd(Avx.LoadVector256(row + kx), Vector256.Create(wk[weight + ky * kernel + kx]), value);
                    }
                    if (silu) value = CpuActivation.Silu(value);
                    Avx.Store(dst + target + y * width + x, value);
                }
                for (; x <= width - 4; x += 4)
                {
                    var value = Vector128.Create(bias);
                    for (int ky = 0; ky < kernel; ky++)
                    {
                        float* row = src + (y + ky) * paddedWidth + x;
                        for (int kx = 0; kx < kernel; kx++) value = Fma.MultiplyAdd(Sse.LoadVector128(row + kx), Vector128.Create(wk[weight + ky * kernel + kx]), value);
                    }
                    if (silu) value = CpuActivation.Silu(value);
                    Sse.Store(dst + target + y * width + x, value);
                }
                for (; x < width; x++)
                {
                    float value = bias;
                    for (int ky = 0; ky < kernel; ky++)
                        for (int kx = 0; kx < kernel; kx++) value += src[(y + ky) * paddedWidth + x + kx] * wk[weight + ky * kernel + kx];
                    dst[target + y * width + x] = silu ? value * TensorOps.Sigmoid(value) : value;
                }
            }
    }
}
