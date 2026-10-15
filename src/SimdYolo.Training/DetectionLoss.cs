using System.Numerics;

namespace SimdYolo.Training;

/// <summary>YOLOv8 检测损失：TAL 标签分配、BCE、CIoU 与 DFL；分配结果不参与求导。</summary>
internal static class DetectionLoss
{
    internal readonly record struct Result(float Box, float Classification, float Dfl, int Positives)
    { internal float Total => Box + Classification + Dfl; }

    /// <summary>解码 DFL，执行停止梯度的 TAL 分配并计算加权损失；需要时写入头输出的累加梯度。</summary>
    /// <param name="boxes">按尺度排列、每个位置含64通道的框分布 logits。</param><param name="classes">相同位置的未激活分类 logits。</param>
    /// <param name="strides">各尺度像素步长。</param><param name="truth">已映射至输入像素坐标的真值框。</param>
    /// <param name="nc">分类通道数量。</param><param name="backward">是否向两个输出分支写入梯度。</param>
    /// <returns>已按分配分数归一化的 CIoU、BCE、DFL 与正样本数。</returns>
    /// <exception cref="ArgumentException">框与类别通道或空间形状不一致。</exception>
    internal static Result Compute(GradientValue[] boxes, GradientValue[] classes, float[] strides, TrainingBox[] truth, int nc, bool backward = true)
    {
        const int bins = 16; int anchors = boxes.Sum(v => v.Value.Shape[2] * v.Value.Shape[3]);
        var location = new (int Level, int Pixel, int Spatial)[anchors];
        var centersX = new float[anchors]; var centersY = new float[anchors]; var strideAt = new float[anchors];
        var decoded = new TrainingBox[anchors]; var probabilities = new float[anchors * 4 * bins]; var expected = new float[anchors * 4];
        int anchor = 0;
        for (int level = 0; level < boxes.Length; level++)
        {
            var tensor = boxes[level].Value; int h = tensor.Shape[2], w = tensor.Shape[3], spatial = h * w;
            if (tensor.Shape[1] != 64 || classes[level].Value.Shape[1] != nc || classes[level].Value.Shape[2] != h || classes[level].Value.Shape[3] != w) throw new ArgumentException("检测头原始输出形状不匹配。");
            for (int pixel = 0; pixel < spatial; pixel++, anchor++)
            {
                location[anchor] = (level, pixel, spatial); centersX[anchor] = (pixel % w + .5f) * strides[level]; centersY[anchor] = (pixel / w + .5f) * strides[level]; strideAt[anchor] = strides[level];
                for (int d = 0; d < 4; d++)
                {
                    float maximum = float.NegativeInfinity;
                    for (int bin = 0; bin < bins; bin++) maximum = MathF.Max(maximum, tensor.Data[(d * bins + bin) * spatial + pixel]);
                    float sum = 0; int start = (anchor * 4 + d) * bins;
                    for (int bin = 0; bin < bins; bin++) { float value = MathF.Exp(tensor.Data[(d * bins + bin) * spatial + pixel] - maximum); probabilities[start + bin] = value; sum += value; }
                    float distance = 0;
                    for (int bin = 0; bin < bins; bin++) { float p = probabilities[start + bin] /= sum; distance += p * bin; }
                    expected[anchor * 4 + d] = distance;
                }
                decoded[anchor] = new(-1, centersX[anchor] - expected[anchor * 4] * strides[level], centersY[anchor] - expected[anchor * 4 + 1] * strides[level], centersX[anchor] + expected[anchor * 4 + 2] * strides[level], centersY[anchor] + expected[anchor * 4 + 3] * strides[level]);
            }
        }
        var selected = new bool[truth.Length * anchors]; var insideCandidates = new bool[selected.Length]; var overlap = new float[selected.Length]; var alignment = new float[selected.Length];
        for (int gt = 0; gt < truth.Length; gt++)
        {
            var box = truth[gt]; if ((uint)box.ClassId >= nc) throw new ArgumentException("类别编号越界。");
            var top = new PriorityQueue<int, (float Metric, int Tie)>();
            for (int a = 0; a < anchors; a++)
            {
                var (l, p, spatial) = location[a]; int index = gt * anchors + a;
                // 固定对照版本对小于最小步长的小框扩展候选范围，但损失仍监督原始框。
                float cx = (box.X1 + box.X2) * .5f, cy = (box.Y1 + box.Y2) * .5f;
                float candidateWidth = box.X2 - box.X1 < strides[0] ? (strides.Length > 1 ? strides[1] : strides[0] * 2) : box.X2 - box.X1;
                float candidateHeight = box.Y2 - box.Y1 < strides[0] ? (strides.Length > 1 ? strides[1] : strides[0] * 2) : box.Y2 - box.Y1;
                bool inside = centersX[a] - (cx - candidateWidth * .5f) > 1e-9f && cx + candidateWidth * .5f - centersX[a] > 1e-9f && centersY[a] - (cy - candidateHeight * .5f) > 1e-9f && cy + candidateHeight * .5f - centersY[a] > 1e-9f;
                insideCandidates[index] = inside;
                float iou = inside ? MathF.Max(0, BoxMath.Ciou(decoded[a], box).Value) : 0; overlap[index] = iou;
                float metric = inside ? MathF.Pow(HeadOperations.Sigmoid(classes[l].Value.Data[box.ClassId * spatial + p]), .5f) * MathF.Pow(iou, 6) : 0;
                alignment[index] = metric;
                top.Enqueue(a, (metric, -a)); if (top.Count > 10) top.Dequeue();
            }
            foreach (var item in top.UnorderedItems) selected[gt * anchors + item.Element] = insideCandidates[gt * anchors + item.Element];
        }
        var assigned = Enumerable.Repeat(-1, anchors).ToArray();
        for (int a = 0; a < anchors; a++)
        {
            int count = 0, chosen = -1;
            for (int gt = 0; gt < truth.Length; gt++) if (selected[gt * anchors + a]) { chosen = gt; count++; }
            if (count > 1)
            {
                float best = -1;
                for (int gt = 0; gt < truth.Length; gt++) if (overlap[gt * anchors + a] > best) { best = overlap[gt * anchors + a]; chosen = gt; }
                for (int gt = 0; gt < truth.Length; gt++) selected[gt * anchors + a] = gt == chosen;
            }
            assigned[a] = chosen;
        }
        var maxAlignment = new float[truth.Length]; var maxOverlap = new float[truth.Length];
        for (int gt = 0; gt < truth.Length; gt++) for (int a = 0; a < anchors; a++) if (selected[gt * anchors + a])
        { maxAlignment[gt] = MathF.Max(maxAlignment[gt], alignment[gt * anchors + a]); maxOverlap[gt] = MathF.Max(maxOverlap[gt], overlap[gt * anchors + a]); }
        var scores = new float[anchors]; float scoreSum = 0;
        for (int a = 0; a < anchors; a++) if (assigned[a] is int gt && gt >= 0)
        { scores[a] = alignment[gt * anchors + a] * maxOverlap[gt] / (maxAlignment[gt] + 1e-9f); scoreSum += scores[a]; }
        scoreSum = MathF.Max(1, scoreSum);
        float boxLoss = 0, classLoss = 0, dflLoss = 0; int positives = 0;
        for (int a = 0; a < anchors; a++)
        {
            var (level, pixel, spatial) = location[a]; int gt = assigned[a]; float weight = scores[a];
            for (int c = 0; c < nc; c++)
            {
                int index = c * spatial + pixel; float logit = classes[level].Value.Data[index]; float target = gt >= 0 && truth[gt].ClassId == c ? weight : 0;
                classLoss += (MathF.Max(logit, 0) - logit * target + MathF.Log(1 + MathF.Exp(-MathF.Abs(logit)))) * (.5f / scoreSum);
                if (backward) classes[level].Gradient[index] += (HeadOperations.Sigmoid(logit) - target) * (.5f / scoreSum);
            }
            if (gt < 0) continue; positives++;
            var targetBox = truth[gt]; float inverseStride = 1 / strideAt[a];
            var predictedGrid = new TrainingBox(-1, decoded[a].X1 * inverseStride, decoded[a].Y1 * inverseStride, decoded[a].X2 * inverseStride, decoded[a].Y2 * inverseStride);
            var targetGrid = new TrainingBox(gt, targetBox.X1 * inverseStride, targetBox.Y1 * inverseStride, targetBox.X2 * inverseStride, targetBox.Y2 * inverseStride);
            var ciou = BoxMath.Ciou(predictedGrid, targetGrid);
            float boxScale = 7.5f * weight / scoreSum; boxLoss += (1 - ciou.Value) * boxScale;
            Vector4 coordinateGradient = -ciou.Gradient * boxScale;
            float[] distanceGradient = [-coordinateGradient.X, -coordinateGradient.Y, coordinateGradient.Z, coordinateGradient.W];
            float[] distances = [(centersX[a] - targetBox.X1) / strideAt[a], (centersY[a] - targetBox.Y1) / strideAt[a], (targetBox.X2 - centersX[a]) / strideAt[a], (targetBox.Y2 - centersY[a]) / strideAt[a]];
            for (int d = 0; d < 4; d++)
            {
                float target = Math.Clamp(distances[d], 0, bins - 1 - .01f); int left = (int)target; float rightWeight = target - left;
                int start = (a * 4 + d) * bins; float scale = 1.5f * weight / (4 * scoreSum);
                // 以 log-sum-exp 计算交叉熵，避免概率下溢后截断损失而导数仍非零。
                float maxLogit = float.NegativeInfinity, expSum = 0;
                for (int bin = 0; bin < bins; bin++) maxLogit = MathF.Max(maxLogit, boxes[level].Value.Data[(d * bins + bin) * spatial + pixel]);
                for (int bin = 0; bin < bins; bin++) expSum += MathF.Exp(boxes[level].Value.Data[(d * bins + bin) * spatial + pixel] - maxLogit);
                float logSum = maxLogit + MathF.Log(expSum);
                dflLoss += scale * (logSum - (1 - rightWeight) * boxes[level].Value.Data[(d * bins + left) * spatial + pixel] - rightWeight * boxes[level].Value.Data[(d * bins + left + 1) * spatial + pixel]);
                if (backward) for (int bin = 0; bin < bins; bin++)
                {
                    float p = probabilities[start + bin], targetProbability = bin == left ? 1 - rightWeight : bin == left + 1 ? rightWeight : 0;
                    boxes[level].Gradient[(d * bins + bin) * spatial + pixel] += p * (bin - expected[a * 4 + d]) * distanceGradient[d] + scale * (p - targetProbability);
                }
            }
        }
        return new(boxLoss, classLoss, dflLoss, positives);
    }
}

/// <summary>四变量前向微分仅用于 CIoU 的坐标导数，避免每个正样本构建对象计算图。</summary>
/// <param name="Value">当前标量前向值。</param><param name="Gradient">相对预测框 x1、y1、x2、y2 的四个导数。</param>
internal readonly record struct Differential(float Value, Vector4 Gradient)
{
    public static implicit operator Differential(float value) => new(value, Vector4.Zero);
    public static Differential operator +(Differential a, Differential b) => new(a.Value + b.Value, a.Gradient + b.Gradient);
    public static Differential operator -(Differential a, Differential b) => new(a.Value - b.Value, a.Gradient - b.Gradient);
    public static Differential operator *(Differential a, Differential b) => new(a.Value * b.Value, a.Gradient * b.Value + b.Gradient * a.Value);
    public static Differential operator /(Differential a, Differential b) => new(a.Value / b.Value, (a.Gradient * b.Value - b.Gradient * a.Value) / (b.Value * b.Value));
    internal static Differential Min(Differential a, Differential b) => a.Value < b.Value ? a : a.Value > b.Value ? b : new(a.Value, (a.Gradient + b.Gradient) * .5f);
    internal static Differential Max(Differential a, Differential b) => a.Value > b.Value ? a : a.Value < b.Value ? b : new(a.Value, (a.Gradient + b.Gradient) * .5f);
    internal static Differential Atan(Differential a) => new(MathF.Atan(a.Value), a.Gradient / (1 + a.Value * a.Value));
}

internal static class BoxMath
{
    /// <summary>实现 xyxy CIoU，保留原版 alpha 的停止梯度语义。</summary>
    /// <param name="predicted">待求导的预测框。</param><param name="target">不参与求导的匹配真值框。</param>
    /// <returns>CIoU 前向值和预测坐标梯度。</returns>
    internal static Differential Ciou(TrainingBox predicted, TrainingBox target)
    {
        const float eps = 1e-7f;
        Differential x1 = new(predicted.X1, new(1, 0, 0, 0)), y1 = new(predicted.Y1, new(0, 1, 0, 0)), x2 = new(predicted.X2, new(0, 0, 1, 0)), y2 = new(predicted.Y2, new(0, 0, 0, 1));
        Differential w1 = x2 - x1, h1 = y2 - y1 + eps, w2 = target.X2 - target.X1, h2 = target.Y2 - target.Y1 + eps;
        Differential iw = Differential.Max(Differential.Min(x2, target.X2) - Differential.Max(x1, target.X1), 0), ih = Differential.Max(Differential.Min(y2, target.Y2) - Differential.Max(y1, target.Y1), 0);
        var inter = iw * ih; var iou = inter / (w1 * h1 + w2 * h2 - inter + eps);
        var cw = Differential.Max(x2, target.X2) - Differential.Min(x1, target.X1); var ch = Differential.Max(y2, target.Y2) - Differential.Min(y1, target.Y1);
        var dx = target.X1 + target.X2 - x1 - x2; var dy = target.Y1 + target.Y2 - y1 - y2;
        var rho = (dx * dx + dy * dy) / 4f; var v = Differential.Atan(w2 / h2) - Differential.Atan(w1 / h1); v = v * v * (4f / (MathF.PI * MathF.PI));
        float alpha = v.Value / (v.Value - iou.Value + 1 + eps);
        return iou - rho / (cw * cw + ch * ch + eps) - v * alpha;
    }
}
