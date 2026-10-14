using System.Numerics;

namespace SimdYolo;

/// <summary>原地解码姿态关键点坐标和可见性概率，按关键点划分互不重叠的写入任务。</summary>
internal static class CpuPose
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_POSE_VECTOR") != "0";

    // 保留 PT 姿态头的原地存储与 FP32 乘加次序。
    // 每个工作线程拥有一个关键点，通道写入互不重叠。
    /// <summary>按原有 FP32 乘加次序原地还原关键点的图像坐标；存在第三分量时将其转换为可见性概率。</summary>
    /// <param name="raw">NCHW 关键点预测；解码会原地改写其底层数组。</param>
    /// <param name="points">每个候选位置的关键点数量。</param>
    /// <param name="dimensions">每个关键点的分量数，2 为坐标，3 还包含可见性。</param>
    /// <param name="stride">当前特征层相对于输入图像的空间缩放步长。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    /// <param name="yolo26">使用 YOLO26 的原始偏移加半像素锚点公式；false 使用旧版两倍偏移。</param>
    /// <returns>形状为 [批次,关键点数×分量数,空间位置数] 的视图，与输入共享已原地解码的数组。</returns>
    internal static Tensor Decode(Tensor raw, int points, int dimensions, float stride, ParallelOptions parallel, bool yolo26 = false)
    {
        int batch = raw.Shape[0], height = raw.Shape[2], width = raw.Shape[3], spatial = height * width;
        int channels = points * dimensions;
        Tensor result = raw.Reshape(batch, channels, spatial);
        float[] values = raw.Data;
        var two = new Vector<float>(yolo26 ? 1f : 2f);
        var scale = new Vector<float>(stride);
        var step = new Vector<float>(Vector<float>.Count);
        var lanes = new Vector<float>(Enumerable.Range(0, Vector<float>.Count).Select(lane => lane + (yolo26 ? .5f : 0f)).ToArray());
        // 原地解码一个批次中的一个关键点，坐标与可见性通道互不重叠。
        // 参数 index：工作线程、任务或输出元素的索引。
        void Point(int index)
        {
            int offset = index * dimensions * spatial;
            for (int row = 0; row < height; row++)
            {
                int start = row * width, column = 0;
                var xs = lanes;
                var ys = new Vector<float>(row + (yolo26 ? .5f : 0f));
                for (; column <= width - Vector<float>.Count; column += Vector<float>.Count, xs += step)
                {
                    int position = offset + start + column;
                    ((new Vector<float>(values, position) * two + xs) * scale).CopyTo(values, position);
                    ((new Vector<float>(values, position + spatial) * two + ys) * scale).CopyTo(values, position + spatial);
                    if (dimensions == 3)
                        CpuActivation.Sigmoid(new Vector<float>(values, position + 2 * spatial)).CopyTo(values, position + 2 * spatial);
                }
                for (; column < width; column++)
                {
                    int position = offset + start + column;
                    values[position] = (yolo26 ? values[position] + (column + .5f) : values[position] * 2 + column) * stride;
                    values[position + spatial] = (yolo26 ? values[position + spatial] + (row + .5f) : values[position + spatial] * 2 + row) * stride;
                    if (dimensions == 3) values[position + 2 * spatial] = TensorOps.Sigmoid(values[position + 2 * spatial]);
                }
            }
        }
        int count = batch * points;
        if ((long)count * dimensions * spatial >= 65536 && count > 1) CpuScheduling.For(0, count, parallel, Point);
        else for (int index = 0; index < count; index++) Point(index);
        return result;
    }
}
