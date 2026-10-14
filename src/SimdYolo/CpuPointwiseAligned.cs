using System.Buffers;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

// 保留原 4×24 算术，每个工作线程同时拥有两个相邻面板。
// 起始碎片长度按固定后的真实输出地址计算，使后续每个 48 列任务覆盖三个完整缓存行。
// N 是 16 的倍数时，全部通道行保持同一物理对齐。
// 跨相邻行可能共享缓存行的首尾碎片必须交给同一个任务。
/// <summary>按输出数组的实际物理地址分配 48 列任务，使相邻线程写入完整且互不重叠的缓存行。</summary>
internal static class CpuPointwiseAligned
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_POINTWISE_ALIGNED") != "0";
    /// <summary>是否将线程私有打包输入的起点对齐到 32 字节。</summary>
    private static readonly bool AlignInput = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_POINTWISE_ALIGNED_B") != "0";

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
        if (!Avx.IsSupported || !Fma.IsSupported || !weights.IsConstant) throw new ArgumentException("Aligned pointwise requires AVX/FMA and constant weights.");
        int m = output.Shape[1], n = checked(output.Shape[2] * output.Shape[3]), k = weights.Shape[1];
        if (n % 16 != 0) throw new ArgumentException("Aligned pointwise requires cache-line-sized channel rows.");
        if (m == 0 || n == 0) return;
        if (CpuPointwiseAlignedSmall.ShouldUse(m, n, backend.Parallel.MaxDegreeOfParallelism))
        {
            CpuPointwiseAlignedSmall.Conv(backend, inputs, weights, bias, output, silu);
            return;
        }
        float[][] sources = inputs.Select(input => input.Data).ToArray();
        int[] channels = inputs.Select(input => input.Shape[1]).ToArray();
        var matrix = CpuPointwise24.Prepare(weights.Data, m, k); float[] target = output.Data; float[]? biases = bias?.Data;
        fixed (float* targetStart = target)
        {
            int prefix = (int)((64 - ((nuint)targetStart & 63)) & 63) / sizeof(float);
            int full = (n - prefix) / 48, tail = n - prefix - full * 48;
            int edges = prefix != 0 || tail != 0 ? 1 : 0;
            // 在指定对齐偏移处打包当前 24 列片段，并写入对应输出空间位置。
            // 参数 column：当前面板的首列索引。
            // 参数 take：当前面板中实际有效的位置数。
            // 参数 scratch：当前线程私有的临时工作缓冲区。
            // 参数 offset：打包数据在缓冲区中的起始偏移。
            void Piece(int column, int take, float[] scratch, int offset)
            {
                CpuPointwise24.Pack(sources, channels, scratch, n, column, take, offset);
                CpuPointwise24.Multiply(matrix, scratch, target, n, column, take, offset);
                CpuPointwise24.Epilogue(target, biases, m, n, column, take, silu);
            }
            // 固定线程私有临时数组，完成两个相邻 24 列面板，或独占全部行首/行尾碎片。
            // 参数 index：工作线程、任务或输出元素的索引。
            // 参数 scratch：当前线程私有的临时工作缓冲区。
            void Task(int index, float[] scratch)
            {
                fixed (float* scratchStart = scratch)
                {
                    int offset = AlignInput ? (int)((32 - ((nuint)scratchStart & 31)) & 31) / sizeof(float) : 0;
                    if (edges != 0 && index == 0)
                    {
                        if (prefix != 0) Piece(0, prefix, scratch, offset);
                        int first = prefix + full * 48;
                        if (tail != 0) Piece(first, Math.Min(24, tail), scratch, offset);
                        if (tail > 24) Piece(first + 24, tail - 24, scratch, offset);
                    }
                    else
                    {
                        int column = prefix + (index - edges) * 48;
                        Piece(column, 24, scratch, offset);
                        Piece(column + 24, 24, scratch, offset);
                    }
                    // 末尾的易失读取使固定的临时指针保持存活。
                    // 必须等全部嵌套打包与矩阵乘法返回后才能解除固定。
                    _ = System.Threading.Volatile.Read(ref scratchStart[0]);
                }
            }
            int tasks = full + edges, length = checked(k * 24 + 7);
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
            // 整个并行计算区域中均保持输出地址固定。
            // GC 不能在同一层计算中途改变其物理缓存行对齐。
            _ = System.Threading.Volatile.Read(ref targetStart[0]);
        }
    }
}

