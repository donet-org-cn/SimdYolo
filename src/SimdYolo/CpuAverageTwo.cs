using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>实现无填充、步长为 1 的 2×2 平均池化，连续 SIMD 加载保持四项求和顺序。</summary>
internal static class CpuAverageTwo
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_AVG2") != "0";
    /// <summary>判断输入是否为 AVX 可用、无填充且步长为 1 的 2×2 平均池化。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="kernel">池化窗口尺寸，或方形深度卷积核的边长。</param>
    /// <param name="stride">空间步长，顺序为高、宽。</param>
    /// <param name="pad">空间填充，顺序为上、左、下、右。</param>
    /// <param name="oh">输出平面高度。</param>
    /// <param name="ow">输出平面宽度。</param>
    /// <returns>满足当前快路径的形状、模式和硬件要求时为 true；否则为 false。</returns>
    internal static bool ShouldUse(Tensor x, int[] kernel, int[] stride, int[] pad, int oh, int ow)
        => Enabled && Avx.IsSupported && kernel[0] == 2 && kernel[1] == 2 && stride[0] == 1 && stride[1] == 1 &&
           pad.All(value => value == 0) && x.Shape[2] >= 2 && x.Shape[3] >= 2 && oh == x.Shape[2] - 1 && ow == x.Shape[3] - 1;

    /// <summary>以连续 SIMD 加载计算 2×2 平均池化，按批次/通道平面并行且保留四项求和顺序。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static Tensor Run(Tensor x)
    {
        int h = x.Shape[2], w = x.Shape[3], oh = h - 1, ow = w - 1, channels = checked(x.Shape[0] * x.Shape[1]);
        var result = Tensor.TemporaryUninitialized(x.Shape[0], x.Shape[1], oh, ow);
        float[] input = x.Data, output = result.Data;
        // 以连续向量读取完成一个批次/通道平面的 2×2 平均池化。
        // 参数 channel：当前输入通道的零基索引，或该通道在输入数组中的起始偏移。
        unsafe void Channel(int channel)
        {
            fixed (float* src = input, dst = output)
                for (int y = 0; y < oh; y++)
                {
                    float* a = src + (channel * h + y) * w, b = a + w, target = dst + (channel * oh + y) * ow;
                    int column = 0;
                    for (; column <= ow - 16; column += 16)
                    {
                        var left = Vector256<float>.Zero + Avx.LoadVector256(a + column);
                        var right = Vector256<float>.Zero + Avx.LoadVector256(a + column + 8);
                        left += Avx.LoadVector256(a + column + 1); right += Avx.LoadVector256(a + column + 9);
                        left += Avx.LoadVector256(b + column); right += Avx.LoadVector256(b + column + 8);
                        left += Avx.LoadVector256(b + column + 1); right += Avx.LoadVector256(b + column + 9);
                        Avx.Store(target + column, left / Vector256.Create(4f));
                        Avx.Store(target + column + 8, right / Vector256.Create(4f));
                    }
                    for (; column <= ow - 8; column += 8)
                    {
                        var value = Vector256<float>.Zero + Avx.LoadVector256(a + column);
                        value += Avx.LoadVector256(a + column + 1); value += Avx.LoadVector256(b + column); value += Avx.LoadVector256(b + column + 1);
                        Avx.Store(target + column, value / Vector256.Create(4f));
                    }
                    for (; column < ow; column++)
                    {
                        float value = 0; value += a[column]; value += a[column + 1]; value += b[column]; value += b[column + 1];
                        target[column] = value / 4;
                    }
                }
        }
        if ((long)channels * oh * ow >= 32768 && channels > 1) CpuScheduling.For(0, channels, CpuScheduling.Options, Channel);
        else for (int channel = 0; channel < channels; channel++) Channel(channel);
        return result;
    }
}
