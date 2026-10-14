using System.Numerics;

namespace SimdYolo;

/// <summary>在矩阵乘积上按通道补加偏置，并可融合 SiLU 激活。</summary>
internal static class CpuEpilogue
{
    /// <summary>逐通道给矩阵乘积补加偏置，并在启用时对所有空间位置计算 SiLU。</summary>
    /// <param name="data">连续 FP32 元素缓冲区。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="channels">要处理的通道数。</param>
    /// <param name="spatial">每个通道的空间元素数。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    public static void Apply(float[] data, float[]? bias, int channels, int spatial, bool silu, ParallelOptions parallel)
    {
        if (bias is null && !silu) return;
        // 对一个输出通道补加偏置，并根据标志执行 SiLU。
        // 参数 c：当前批次/通道平面的索引。
        void Channel(int c)
        {
            int start = c * spatial, end = start + spatial, j = start; var vb = new Vector<float>(bias?[c] ?? 0);
            for (; j <= end - Vector<float>.Count; j += Vector<float>.Count)
            {
                var value = new Vector<float>(data, j) + vb;
                if (silu) value = CpuActivation.Silu(value);
                value.CopyTo(data, j);
            }
            for (; j < end; j++) { float value = data[j] + (bias?[c] ?? 0); data[j] = silu ? value * TensorOps.Sigmoid(value) : value; }
        }
        if ((long)channels * spatial >= 32768 && channels > 1) CpuScheduling.For(0, channels, parallel, Channel);
        else for (int c = 0; c < channels; c++) Channel(c);
    }
}
