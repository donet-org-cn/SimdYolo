using System.Numerics;

namespace SimdYolo;

/// <summary>沿候选位置分块计算 DFL 概率期望，保留各分箱的归约次序。</summary>
internal static class CpuDfl
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_DFL_TILES") != "0";

    // 沿候选位置轴分块，使批次为 1、仅四条距离通道时仍能利用全部核心。
    // 各分箱的归约次序与原内核一致。
    /// <summary>按候选位置计算四侧距离分布的稳定 Softmax 与分箱期望，将结果写入预分配距离数组。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="destination">保存解码结果的预分配缓冲区。</param>
    /// <param name="bins">每条边距离分布的分箱数量。</param>
    /// <param name="spatial">每个通道的空间元素数。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    internal static void Decode(Tensor input, float[] destination, int bins, int spatial, ParallelOptions parallel)
    {
        float[] source = input.Data;
        int sides = input.Shape[0] * 4;
        if (bins == 1) { Array.Copy(source, destination, sides * spatial); return; }
        const int tile = 256;
        int tiles = (spatial + tile - 1) / tile;
        // 对一条边的最多 256 个候选位置归约分箱概率并写入距离期望。
        // 参数 block：边框侧边与 256 位置分块组合的任务索引。
        void Block(int block)
        {
            int side = block / tiles, start = (block % tiles) * tile, end = Math.Min(start + tile, spatial);
            int offset = side * bins * spatial, output = side * spatial, anchor = start;
            for (; anchor <= end - Vector<float>.Count; anchor += Vector<float>.Count)
            {
                var maximum = new Vector<float>(float.NegativeInfinity);
                for (int bin = 0; bin < bins; bin++) maximum = Vector.Max(maximum, new Vector<float>(source, offset + bin * spatial + anchor));
                var sum = Vector<float>.Zero;
                var numerator = Vector<float>.Zero;
                for (int bin = 0; bin < bins; bin++)
                {
                    var exp = CpuActivation.Exp(new Vector<float>(source, offset + bin * spatial + anchor) - maximum);
                    sum += exp;
                    numerator += exp * new Vector<float>(bin);
                }
                (numerator / sum).CopyTo(destination, output + anchor);
            }
            for (; anchor < end; anchor++)
            {
                float maximum = float.NegativeInfinity, sum = 0, numerator = 0;
                for (int bin = 0; bin < bins; bin++) maximum = MathF.Max(maximum, source[offset + bin * spatial + anchor]);
                for (int bin = 0; bin < bins; bin++)
                {
                    float exp = MathF.Exp(source[offset + bin * spatial + anchor] - maximum);
                    sum += exp; numerator += exp * bin;
                }
                destination[output + anchor] = numerator / sum;
            }
        }
        int count = sides * tiles;
        if ((long)sides * bins * spatial >= 32768 && count > 1) CpuScheduling.For(0, count, parallel, Block);
        else for (int block = 0; block < count; block++) Block(block);
    }
}
