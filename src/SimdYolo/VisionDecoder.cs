namespace SimdYolo;

/// <summary>表示原图坐标系中的一个姿态关键点及其置信度。</summary>
/// <param name="X">关键点在原图中的横坐标。</param>
/// <param name="Y">关键点在原图中的纵坐标。</param>
/// <param name="Confidence">关键点置信度；仅含二维坐标的输出使用 1。</param>
public readonly record struct Keypoint(float X, float Y, float Confidence);
/// <summary>表示原图坐标系中裁剪到实例区域的掩码，概率数组按行存储。</summary>
/// <param name="X">掩码裁剪区域在原图中的左侧像素坐标。</param>
/// <param name="Y">掩码裁剪区域在原图中的顶部像素坐标。</param>
/// <param name="Width">裁剪区域宽度，单位为原图像素。</param>
/// <param name="Height">裁剪区域高度，单位为原图像素。</param>
/// <param name="Probabilities">按行存储的掩码概率，共 Width × Height 个元素。</param>
public sealed record InstanceMask(int X, int Y, int Width, int Height, float[] Probabilities);
/// <summary>汇总一个检测框及其可选分割掩码和姿态关键点。</summary>
/// <param name="Box">已映射到原图坐标的检测框。</param>
/// <param name="Mask">实例掩码；未解码掩码或任务不含分割时为 null。</param>
/// <param name="Keypoints">原图坐标关键点；检测或分割任务使用空集合。</param>
public sealed record YoloInstance(Detection Box, InstanceMask? Mask, IReadOnlyList<Keypoint> Keypoints);

/// <summary>解码 YOLO 检测、实例分割和姿态输出，并映射回原图坐标。</summary>
public static class VisionDecoder
{
    /// <summary>解码检测框、关键点和可选实例掩码，按 letterbox 参数映射回原图。</summary>
    /// <param name="outputs">模型输出；首个张量为检测头，分割时第二个张量为 [1, 掩码通道, 高, 宽] 原型。</param>
    /// <param name="transform">输入预处理使用的缩放和填充参数。</param>
    /// <param name="options">任务类型、输出布局、类别数、关键点设置及检测筛选阈值。</param>
    /// <param name="decodeMasks">是否计算分割概率掩码；false 时分割结果仍保留检测框。</param>
    /// <returns>按检测结果排列的实例列表；无检测时为空列表。</returns>
    /// <exception cref="ArgumentException">模型输出缺失、布局有歧义、类别或关键点维度不匹配，或分割原型不满足单图要求。</exception>
    /// <exception cref="NotSupportedException">检测头不是三维张量，或分割、姿态输出采用不支持的端到端布局。</exception>
    /// <exception cref="OverflowException">关键点字段数或裁剪掩码面积超出 Int32 范围。</exception>
    public static IReadOnlyList<YoloInstance> Decode(Tensor[] outputs, Letterbox transform, DetectorOptions options, bool decodeMasks = true)
    {
        if (outputs.Length == 0) throw new ArgumentException("Model outputs missing.");
        YoloTask task = options.Task;
        if (task == YoloTask.Auto) task = outputs.Length == 2 && outputs[1].Rank == 4 ? YoloTask.Segmentation : YoloTask.Detection;
        Tensor head = outputs[0];
        if (head.Rank != 3) throw new NotSupportedException("YOLO head must be rank 3.");
        var layout = options.OutputLayout;
        if (layout == OutputLayout.Auto && task != YoloTask.Detection)
        {
            int expected = options.ClassCount > 0 ? 4 + options.ClassCount + (task == YoloTask.Pose ? options.KeypointCount * options.KeypointDimensions : outputs.Length == 2 ? outputs[1].Shape[1] : 0) : 0;
            if (expected > 0 && head.Shape[1] == expected && head.Shape[2] != expected) layout = OutputLayout.ChannelsFirst;
            else if (expected > 0 && head.Shape[2] == expected && head.Shape[1] != expected) layout = OutputLayout.ChannelsLast;
            else
            { if (head.Shape[1] == head.Shape[2]) throw new ArgumentException("Ambiguous output layout; set OutputLayout."); layout = head.Shape[1] < head.Shape[2] ? OutputLayout.ChannelsFirst : OutputLayout.ChannelsLast; }
        }
        bool first = layout == OutputLayout.ChannelsFirst;
        int fields = first ? head.Shape[1] : head.Shape[2], count = first ? head.Shape[2] : head.Shape[1];
        if (task != YoloTask.Detection && layout == OutputLayout.EndToEnd) throw new NotSupportedException("End-to-end segment/pose exports unsupported.");
        int extra = task == YoloTask.Segmentation ? outputs.Length == 2 && outputs[1].Rank == 4 ? outputs[1].Shape[1] : throw new ArgumentException("Segmentation requires [head, prototypes].") : task == YoloTask.Pose ? checked(options.KeypointCount * options.KeypointDimensions) : 0;
        int classes = options.ClassCount > 0 ? options.ClassCount : fields - 4 - extra;
        if (task != YoloTask.Detection && (classes <= 0 || 4 + classes + extra != fields)) throw new ArgumentException("Class/keypoint/mask dimensions mismatch.");
        var detections = DetectionDecoder.DecodeIndexed(head, transform, options with { OutputLayout = layout, ClassCount = task == YoloTask.Detection ? options.ClassCount : classes });
        // 统一通道优先与通道最后布局的字段读取，后续掩码/关键点计算复用同一锚点索引。
        float At(int a, int c) => head.Data[first ? c * count + a : a * fields + c];
        var result = new List<YoloInstance>();
        foreach (var (box, anchor) in detections)
        {
            InstanceMask? mask = null; Keypoint[] points = [];
            if (task == YoloTask.Pose)
            {
                if (options.KeypointDimensions is not (2 or 3)) throw new ArgumentException("Keypoints must have 2 or 3 dimensions.");
                points = new Keypoint[options.KeypointCount];
                for (int k = 0; k < points.Length; k++)
                {
                    int offset = 4 + classes + k * options.KeypointDimensions;
                    points[k] = new((At(anchor, offset) - transform.PadX) / transform.ScaleX, (At(anchor, offset + 1) - transform.PadY) / transform.ScaleY, options.KeypointDimensions == 3 ? At(anchor, offset + 2) : 1);
                }
            }
            if (task == YoloTask.Segmentation && decodeMasks)
            {
                var prototype = outputs[1]; if (prototype.Shape[0] != 1) throw new ArgumentException("Single-image mask decoding only.");
                int mh = prototype.Shape[2], mw = prototype.Shape[3], plane = mh * mw;
                // 先在线性空间融合原型与实例系数，再插值 logits；Sigmoid 必须放在插值之后。
                var low = new float[plane];
                for (int c = 0; c < extra; c++)
                {
                    float weight = At(anchor, 4 + classes + c); int i = 0; var vw = new System.Numerics.Vector<float>(weight);
                    for (; i <= plane - System.Numerics.Vector<float>.Count; i += System.Numerics.Vector<float>.Count)
                        (new System.Numerics.Vector<float>(low, i) + vw * new System.Numerics.Vector<float>(prototype.Data, c * plane + i)).CopyTo(low, i);
                    for (; i < plane; i++) low[i] += weight * prototype.Data[c * plane + i];
                }
                int left = (int)MathF.Floor(box.X), top = (int)MathF.Floor(box.Y), right = (int)MathF.Ceiling(box.X + box.Width), bottom = (int)MathF.Ceiling(box.Y + box.Height), width = right - left, height = bottom - top;
                // 按 letterbox 的有效图像区域取原型 ROI，避免将填充区域拉伸进原图掩码。
                int roiLeft = Math.Clamp((int)MathF.Round((float)transform.PadX * mw / transform.TargetWidth), 0, mw - 1);
                int roiTop = Math.Clamp((int)MathF.Round((float)transform.PadY * mh / transform.TargetHeight), 0, mh - 1);
                int roiRight = Math.Clamp((int)MathF.Round((transform.PadX + transform.SourceWidth * transform.ScaleX) * mw / transform.TargetWidth), roiLeft + 1, mw);
                int roiBottom = Math.Clamp((int)MathF.Round((transform.PadY + transform.SourceHeight * transform.ScaleY) * mh / transform.TargetHeight), roiTop + 1, mh);
                var values = new float[checked(width * height)];
                for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
                {
                    float px = roiLeft + (left + x + .5f) * (roiRight - roiLeft) / transform.SourceWidth - .5f;
                    float py = roiTop + (top + y + .5f) * (roiBottom - roiTop) / transform.SourceHeight - .5f;
                    px = Math.Clamp(px, roiLeft, roiRight - 1); py = Math.Clamp(py, roiTop, roiBottom - 1); int x0 = (int)px, y0 = (int)py, x1 = Math.Min(x0 + 1, roiRight - 1), y1 = Math.Min(y0 + 1, roiBottom - 1); float fx = px - x0, fy = py - y0;
                    float logit = (low[y0 * mw + x0] * (1 - fx) + low[y0 * mw + x1] * fx) * (1 - fy) + (low[y1 * mw + x0] * (1 - fx) + low[y1 * mw + x1] * fx) * fy;
                    values[y * width + x] = left + x >= box.X && top + y >= box.Y && left + x < box.X + box.Width && top + y < box.Y + box.Height ? TensorOps.Sigmoid(logit) : 0;
                }
                mask = new(left, top, width, height, values);
            }
            result.Add(new(box, mask, points));
        }
        return result;
    }
}
