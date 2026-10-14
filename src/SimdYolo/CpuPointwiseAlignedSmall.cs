using System.Buffers;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

// 小平面采用 48 列任务时，可运行块数可能少于线程数。
// 改用 16 列任务仍保留物理缓存行边界，并复用原 6×16 FP32 FMA 内核。
/// <summary>对较小空间平面使用按物理缓存行对齐的 16 列任务，提高并行任务数。</summary>
internal static class CpuPointwiseAlignedSmall
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_POINTWISE_ALIGNED_SMALL") != "0";
    /// <summary>判断空间平面是否足够小，且 16 列缓存行任务较 48 列任务更能填满配置线程。</summary>
    /// <param name="rows">矩阵逻辑行数。</param>
    /// <param name="columns">矩阵逻辑列数。</param>
    /// <param name="workers">此层可参与计算的工作线程数。</param>
    /// <returns>满足当前快路径的形状、模式和硬件要求时为 true；否则为 false。</returns>
    internal static bool ShouldUse(int rows, int columns, int workers) => Enabled && rows >= 6 &&
        columns <= 48L * Math.Max(1, workers);

    /// <summary>按源张量通道顺序直接打包 1×1 卷积输入，使用互不重叠的空间面板写出偏置与可选 SiLU 结果。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="inputs">按通道拼接顺序排列的输入张量。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static unsafe void Conv(CpuBackend backend, IReadOnlyList<Tensor> inputs, Tensor weights, Tensor? bias, Tensor output, bool silu)
    {
        if (!Avx.IsSupported || !Fma.IsSupported || !weights.IsConstant) throw new ArgumentException("Small aligned pointwise requires AVX/FMA and constant weights.");
        int m = output.Shape[1], n = checked(output.Shape[2] * output.Shape[3]), k = weights.Shape[1];
        if (n % 16 != 0) throw new ArgumentException("Small aligned pointwise requires cache-line-sized channel rows.");
        if (m == 0 || n == 0) return;
        float[][] sources = inputs.Select(input => input.Data).ToArray();
        int[] channels = inputs.Select(input => input.Shape[1]).ToArray();
        var matrix = CpuPackedGemm.Prepare(weights.Data, m, k, true); float[] target = output.Data; float[]? biases = bias?.Data;
        fixed (float* targetStart = target)
        {
            int prefix = (int)((64 - ((nuint)targetStart & 63)) & 63) / sizeof(float);
            int full = (n - prefix) / 16, tail = n - prefix - full * 16;
            int edges = prefix != 0 || tail != 0 ? 1 : 0;
            // 在指定对齐偏移处打包当前 16 列片段，并写入对应输出空间位置。
            // 参数 column：当前面板的首列索引。
            // 参数 take：当前面板中实际有效的位置数。
            // 参数 scratch：当前线程私有的临时工作缓冲区。
            // 参数 offset：打包数据在缓冲区中的起始偏移。
            void Piece(int column, int take, float[] scratch, int offset)
            {
                Pack(sources, channels, scratch, n, column, take, offset);
                CpuPackedGemm.Multiply(matrix, scratch, target, n, column, take, offset);
                CpuPointwise24.Epilogue(target, biases, m, n, column, take, silu);
            }
            // 固定线程私有临时数组，完成一个缓存行面板，或独占全部行首/行尾碎片。
            // 参数 index：工作线程、任务或输出元素的索引。
            // 参数 scratch：当前线程私有的临时工作缓冲区。
            void Task(int index, float[] scratch)
            {
                fixed (float* scratchStart = scratch)
                {
                    int offset = (int)((32 - ((nuint)scratchStart & 31)) & 31) / sizeof(float);
                    if (edges != 0 && index == 0)
                    {
                        if (prefix != 0) Piece(0, prefix, scratch, offset);
                        if (tail != 0) Piece(prefix + full * 16, tail, scratch, offset);
                    }
                    else Piece(prefix + (index - edges) * 16, 16, scratch, offset);
                    _ = System.Threading.Volatile.Read(ref scratchStart[0]);
                }
            }
            int tasks = full + edges, length = checked(k * 16 + 7);
            if ((long)m * n * k > 262144 && backend.Parallel.MaxDegreeOfParallelism != 1)
                CpuScheduling.For(0, tasks, backend.Parallel,
                    () => ArrayPool<float>.Shared.Rent(length),
                    (index, _, scratch) => { Task(index, scratch); return scratch; },
                    scratch => ArrayPool<float>.Shared.Return(scratch));
            else
            {
                var scratch = ArrayPool<float>.Shared.Rent(length);
                try { for (int index = 0; index < tasks; index++) Task(index, scratch); }
                finally { ArrayPool<float>.Shared.Return(scratch); }
            }
            _ = System.Threading.Volatile.Read(ref targetStart[0]);
        }
    }

    /// <summary>按源张量与通道顺序，将当前 K×16 面板打包到指定对齐偏移处，未使用列补零。</summary>
    /// <param name="sources">按通道拼接顺序排列的源数组。</param>
    /// <param name="channels">各源张量的输入通道数量，顺序与源数组一致。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="column">当前面板的首列索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="offset">打包数据在缓冲区中的起始偏移。</param>
    private static unsafe void Pack(float[][] sources, int[] channels, float[] packed, int n, int column, int take, int offset)
    {
        fixed (float* target = packed)
        {
            int row = 0;
            for (int s = 0; s < sources.Length; s++)
                fixed (float* source = sources[s])
                    for (int ci = 0; ci < channels[s]; ci++, row++)
                    {
                        float* src = source + ci * n + column, dst = target + offset + row * 16;
                        if (take == 16)
                        {
                            Avx.Store(dst, Avx.LoadVector256(src));
                            Avx.Store(dst + 8, Avx.LoadVector256(src + 8));
                        }
                        else
                        {
                            for (int j = 0; j < take; j++) dst[j] = src[j];
                            for (int j = take; j < 16; j++) dst[j] = 0;
                        }
                    }
        }
    }
}

