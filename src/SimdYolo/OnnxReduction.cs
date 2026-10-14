using System.Numerics;

namespace SimdYolo;

/// <summary>用纯 C# 执行 ONNX 图，并在保持输入输出约定的前提下进行常量折叠和算子融合。</summary>
internal sealed partial class OnnxModel
{
    /// <summary>按行使用 SIMD 或 Vulkan 求最后一维最大值，并保持 NaN 和正负零规则。</summary>
    /// <param name="x">参与计算的输入张量。</param>
    /// <param name="keep">是否把归约轴保留为大小为 1 的维度。</param>
    /// <param name="parallel">当前 CPU 后端的并行配置；null 使用该方法的串行回退。</param>
    /// <returns>按 keep 设置保留或移除最后一维的归约张量。</returns>
    private static Tensor ReduceMaxLast(Tensor x, bool keep, ParallelOptions? parallel)
    {
        int size = x.Shape[^1], rows = Tensor.Count(x.Shape[..^1]);
        int[] shape = keep ? x.Shape[..^1].Concat([1]).ToArray() : x.Shape[..^1];
        if (OnnxDeviceOperations.Enabled && OnnxDeviceOperations.CanReduce(x, rows, size) && DeviceScope.For(x) is VulkanBackend owner)
            return OnnxDeviceOperations.ReduceMaxLast(owner, x, shape, rows, size);
        var output = Tensor.Temporary(shape); float[] source = x.Data, destination = output.Data;
        // start 为一行首元素偏移；返回保留 NaN 与正负零语义的最后一维最大值。
        float Row(int start)
        {
            int end = start + size, position = start;
            var maximum = new Vector<float>(float.NegativeInfinity);
            for (; position <= end - Vector<float>.Count; position += Vector<float>.Count)
            {
                var value = new Vector<float>(source, position);
                // 硬件最大值指令可能忽略 NaN；发现 NaN 时直接返回，
                // 避免 SIMD 归约把标量 MathF.Max 会保留的 NaN 丢弃。
                if (!Vector.EqualsAll(value, value))
                    for (int lane = 0; lane < Vector<float>.Count; lane++) if (float.IsNaN(source[position + lane])) return source[position + lane];
                maximum = Vector.Max(maximum, value);
            }
            float result = float.NegativeInfinity;
            for (int lane = 0; lane < Vector<float>.Count; lane++) result = MathF.Max(result, maximum[lane]);
            for (; position < end; position++) result = MathF.Max(result, source[position]);
            if (result == 0)
            {
                // MathF.Max 对零值相等时优先保留正零，而向量硬件可能选择后者。
                // 零最大值时额外检查位模式，使正负零规则与标量一致。
                for (int index = start; index < end; index++) if (BitConverter.SingleToInt32Bits(source[index]) == 0) return 0;
                return BitConverter.Int32BitsToSingle(int.MinValue);
            }
            return result;
        }
        // CPU 最后一维归约每个调度块处理的行数量。
        const int blockRows = 64;
        // block 为行块编号；把此块最多 64 行的最大值写入各自输出槽，无返回值。
        void Block(int block)
        {
            int first = block * blockRows, last = Math.Min(first + blockRows, rows);
            for (int row = first; row < last; row++) destination[row] = Row(row * size);
        }
        int blocks = (rows + blockRows - 1) / blockRows;
        if (parallel is not null && x.Length >= 32768 && blocks > 1) CpuScheduling.For(0, blocks, parallel, Block);
        else for (int block = 0; block < blocks; block++) Block(block);
        return output;
    }
}
