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
    /// <exception cref="NotSupportedException">检测头不是三维张量。</exception>
    /// <exception cref="OverflowException">关键点字段数或裁剪掩码面积超出 Int32 范围。</exception>
    /// <exception cref="AggregateException">并行掩码解码中的工作线程发生异常。</exception>
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
        bool endToEnd = layout == OutputLayout.EndToEnd;
        int extra = task == YoloTask.Segmentation ? outputs.Length == 2 && outputs[1].Rank == 4 ? outputs[1].Shape[1] : throw new ArgumentException("Segmentation requires [head, prototypes].") : task == YoloTask.Pose ? checked(options.KeypointCount * options.KeypointDimensions) : 0;
        int classes = options.ClassCount > 0 ? options.ClassCount : fields - 4 - extra;
        if (task != YoloTask.Detection && (endToEnd ? fields != 6 + extra : classes <= 0 || 4 + classes + extra != fields)) throw new ArgumentException("Class/keypoint/mask dimensions mismatch.");
        Tensor detectionHead = head;
        if (endToEnd && extra > 0)
        {
            detectionHead = new Tensor(head.Shape[0], count, 6);
            for (int b = 0; b < head.Shape[0]; b++) for (int a = 0; a < count; a++) Array.Copy(head.Data, (b * count + a) * fields, detectionHead.Data, (b * count + a) * 6, 6);
        }
        var detections = DetectionDecoder.DecodeIndexed(detectionHead, transform, options with { OutputLayout = layout, ClassCount = endToEnd || task == YoloTask.Detection ? options.ClassCount : classes });
        int extraOffset = endToEnd ? 6 : 4 + classes;
        // 统一通道优先与通道最后布局的字段读取，后续掩码/关键点计算复用同一锚点索引。
        float At(int a, int c) => head.Data[first ? c * count + a : a * fields + c];
        var maskDecoder = task == YoloTask.Segmentation && decodeMasks && detections.Count > 0 ? new CpuMaskDecoder(outputs[1], transform) : null;
        var result = new YoloInstance[detections.Count];
        YoloInstance DecodeInstance(int index)
        {
            var (box, anchor) = detections[index];
            InstanceMask? mask = null; Keypoint[] points = [];
            if (task == YoloTask.Pose)
            {
                if (options.KeypointDimensions is not (2 or 3)) throw new ArgumentException("Keypoints must have 2 or 3 dimensions.");
                points = new Keypoint[options.KeypointCount];
                for (int k = 0; k < points.Length; k++)
                {
                    int offset = extraOffset + k * options.KeypointDimensions;
                    points[k] = new((At(anchor, offset) - transform.PadX) / transform.ScaleX, (At(anchor, offset + 1) - transform.PadY) / transform.ScaleY, options.KeypointDimensions == 3 ? At(anchor, offset + 2) : 1);
                }
            }
            if (task == YoloTask.Segmentation && decodeMasks)
            {
                var coefficients = new float[extra];
                for (int c = 0; c < extra; c++) coefficients[c] = At(anchor, extraOffset + c);
                mask = maskDecoder!.Decode(box, coefficients);
            }
            return new(box, mask, points);
        }
        int threads = options.CpuThreads > 0 ? options.CpuThreads : Environment.ProcessorCount;
        long area = maskDecoder is null ? 0 : detections.Sum(item => (long)MathF.Ceiling(item.Box.Width) * (long)MathF.Ceiling(item.Box.Height));
        if (maskDecoder is not null && result.Length > 1 && area >= 65536 && threads > 1)
            Parallel.For(0, result.Length, new ParallelOptions { MaxDegreeOfParallelism = Math.Min(threads, result.Length) }, index => result[index] = DecodeInstance(index));
        else for (int index = 0; index < result.Length; index++) result[index] = DecodeInstance(index);
        return result;
    }
}
