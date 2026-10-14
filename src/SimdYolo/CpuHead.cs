using System.Numerics;

namespace SimdYolo;

/// <summary>融合检测头的 DFL 距离期望、类别 Sigmoid 与网格坐标解码。</summary>
internal static class CpuHead
{
    /// <summary>归约四侧 DFL 分箱概率得到距离期望，以网格中心和特征步长还原边框，同时计算类别 Sigmoid。</summary>
    /// <param name="boxes">NCHW 形式的四侧距离分箱预测。</param>
    /// <param name="scores">NCHW 形式的类别 logits。</param>
    /// <param name="bins">每条边距离分布的分箱数量。</param>
    /// <param name="stride">当前特征层相对于输入图像的空间缩放步长。</param>
    /// <param name="xyxy">是否返回左右上下角坐标；关闭时返回中心坐标与宽高。</param>
    /// <param name="parallel">控制最大并行度、取消令牌和调度器的选项。</param>
    /// <returns>形状为 [批次,4+类别数,空间位置数] 的边框坐标和类别概率张量。</returns>
    public static Tensor Decode(Tensor boxes, Tensor scores, int bins, float stride, bool xyxy, ParallelOptions? parallel = null)
    {
        int batch = boxes.Shape[0], h = boxes.Shape[2], w = boxes.Shape[3], spatial = h * w, nc = scores.Shape[1];
        var output = Tensor.TemporaryUninitialized(batch, nc + 4, spatial); float[] data = boxes.Data;
        // 无 DFL 的 YOLO26 直接消费回归值，省去距离工作区和整张特征图复制。
        var distances = bins == 1 ? data : Tensor.TemporaryUninitialized(batch, 4, spatial).Data;
        float[] values = output.Data, logits = scores.Data;
        if (bins != 1 && CpuDfl.Enabled) CpuDfl.Decode(boxes, distances, bins, spatial, parallel ?? CpuScheduling.Options);
        // 为一个批次的边框侧边或类别通道写入距离期望或 Sigmoid 概率。
        // 参数 bc：批次与输出通道组合的索引。
        void Channel(int bc)
        {
            int b = bc / (nc + 4), side = bc % (nc + 4);
            if (side >= 4)
            {
                int channel = side - 4, srcScore = (b * nc + channel) * spatial, dstScore = (b * (nc + 4) + side) * spatial, s = 0;
                for (; s <= spatial - Vector<float>.Count; s += Vector<float>.Count) CpuActivation.Sigmoid(new Vector<float>(logits, srcScore + s)).CopyTo(values, dstScore + s);
                for (; s < spatial; s++) values[dstScore + s] = TensorOps.Sigmoid(logits[srcScore + s]);
                return;
            }
            if (bins == 1 || CpuDfl.Enabled) return;
            int src = (b * 4 + side) * bins * spatial, dst = (b * 4 + side) * spatial, a = 0;
            for (; a <= spatial - Vector<float>.Count; a += Vector<float>.Count)
            {
                var max = new Vector<float>(float.NegativeInfinity);
                for (int r = 0; r < bins; r++) max = Vector.Max(max, new Vector<float>(data, src + r * spatial + a));
                var sum = Vector<float>.Zero; var numerator = sum;
                for (int r = 0; r < bins; r++) { var e = CpuActivation.Exp(new Vector<float>(data, src + r * spatial + a) - max); sum += e; numerator += e * new Vector<float>(r); }
                (numerator / sum).CopyTo(distances, dst + a);
            }
            for (; a < spatial; a++)
            {
                float max = float.NegativeInfinity, sum = 0, numerator = 0;
                for (int r = 0; r < bins; r++) max = MathF.Max(max, data[src + r * spatial + a]);
                for (int r = 0; r < bins; r++) { float e = MathF.Exp(data[src + r * spatial + a] - max); sum += e; numerator += e * r; }
                distances[dst + a] = numerator / sum;
            }
        }
        if ((long)batch * (nc + 4 * bins) * spatial >= 32768) CpuScheduling.For(0, batch * (nc + 4), parallel ?? CpuScheduling.Options, Channel);
        else for (int c = 0; c < batch * (nc + 4); c++) Channel(c);
        Vector<float> scale = new(stride), half = new(.5f), step = new(Vector<float>.Count);
        var lanes = new Vector<float>(Enumerable.Range(0, Vector<float>.Count).Select(lane => lane + .5f).ToArray());
        for (int b = 0; b < batch; b++) for (int row = 0; row < h; row++)
        {
            int column = 0, start = row * w; var xs = lanes; var ys = new Vector<float>(row + .5f);
            // 超过 FP32 连续整数范围的极宽行保持原标量整数转换，避免累计网格坐标改变舍入。
            for (; w <= 16777216 && column <= w - Vector<float>.Count; column += Vector<float>.Count, xs += step)
            {
                int d = b * 4 * spatial + start + column, dst = b * (nc + 4) * spatial + start + column;
                var x1 = (xs - new Vector<float>(distances, d)) * scale; var y1 = (ys - new Vector<float>(distances, d + spatial)) * scale;
                var x2 = (xs + new Vector<float>(distances, d + 2 * spatial)) * scale; var y2 = (ys + new Vector<float>(distances, d + 3 * spatial)) * scale;
                (xyxy ? x1 : (x1 + x2) * half).CopyTo(values, dst); (xyxy ? y1 : (y1 + y2) * half).CopyTo(values, dst + spatial);
                (xyxy ? x2 : x2 - x1).CopyTo(values, dst + 2 * spatial); (xyxy ? y2 : y2 - y1).CopyTo(values, dst + 3 * spatial);
            }
            for (; column < w; column++)
            {
                int a = start + column;
                int d = b * 4 * spatial + a, dst = b * (nc + 4) * spatial + a;
                float x1 = (a % w + .5f - distances[d]) * stride, y1 = (a / w + .5f - distances[d + spatial]) * stride;
                float x2 = (a % w + .5f + distances[d + 2 * spatial]) * stride, y2 = (a / w + .5f + distances[d + 3 * spatial]) * stride;
                values[dst] = xyxy ? x1 : (x1 + x2) * .5f; values[dst + spatial] = xyxy ? y1 : (y1 + y2) * .5f;
                values[dst + 2 * spatial] = xyxy ? x2 : x2 - x1; values[dst + 3 * spatial] = xyxy ? y2 : y2 - y1;
            }
        }
        return output;
    }
}
