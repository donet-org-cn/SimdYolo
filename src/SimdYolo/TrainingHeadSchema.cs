namespace SimdYolo;

/// <summary>统一验证检测头训练结构，使训练、归档恢复与网页能力探测使用同一规则。</summary>
/// <param name="RegMax">每边回归通道数，十六为分布回归，一为直接距离。</param>
/// <param name="EndToEnd">是否同时训练一对多和一对一检测分支。</param>
/// <param name="Progressive">是否使用 YOLO26 的渐进损失与二次匹配。</param>
internal sealed record TrainingHeadSchema(int RegMax, bool EndToEnd, bool Progressive)
{
    /// <summary>归档使用的损失策略名称。</summary><value>标准分布回归、传统双头或渐进双头。</value>
    internal string Kind => Progressive ? "ProgressiveEndToEnd" : EndToEnd ? "LegacyEndToEnd" : "Dfl";
    /// <summary>必须完整保存与恢复的检测分支名称。</summary><value>双头模型包含一对一回归和分类分支。</value>
    internal string[] Prefixes => EndToEnd ? ["cv2", "cv3", "one2one_cv2", "one2one_cv3"] : ["cv2", "cv3"];

    /// <summary>检查未融合的三尺度检测头，拒绝缺少训练分支的推理专用权重及其他任务。</summary>
    /// <param name="head">PT 中的最终模块。</param><returns>回归通道及双分支损失策略。</returns>
    /// <exception cref="NotSupportedException">任务、卷积或分支布局尚不支持训练。</exception>
    internal static TrainingHeadSchema Read(PyObject head)
    {
        int bins = head.Int("reg_max", 16); bool dual = head.Has("one2one_cv2");
        if (head.Type.Name is not ("Detect" or "v10Detect") || bins is not (1 or 16) || bins == 1 && !dual || head.Type.Name == "v10Detect" && (!dual || bins != 16))
            throw new NotSupportedException("训练支持 YOLOv8/v9/YOLO11 标准 Detect、YOLOv10 双头和 YOLO26 双头检测 PT；分割、姿态及推理融合专用头不支持训练。");
        var schema = new TrainingHeadSchema(bins, dual, dual && bins == 1);
        foreach (string prefix in schema.Prefixes)
        {
            if (!head.Has(prefix) || head.Child(prefix).Children.Length != 3) throw new NotSupportedException("训练需要完整的三个尺度及所有 one2many/one2one 分支。");
            foreach (var branch in head.Child(prefix).Children)
            {
                if (branch.Type.Name != "Sequential" || branch.Children.Length != 3 || branch.Children[^1].Type.Name != "Conv2d") throw new NotSupportedException("检测分支必须以 Conv2d 输出原始 logits。");
                Validate(branch);
                int expected = prefix.EndsWith("cv2", StringComparison.Ordinal) ? 4 * bins : head.Int("nc");
                if (WeightShape(branch.Children[^1])[0] != expected) throw new NotSupportedException("检测头的输出通道与类别或 reg_max 不一致。");
            }
        }
        return schema;

        static void Validate(PyObject module)
        {
            if (module.Type.Name == "Sequential") { foreach (var child in module.Children) Validate(child); return; }
            bool normalized = module.Type.Name is "Conv" or "DWConv";
            if (!normalized && module.Type.Name != "Conv2d") throw new NotSupportedException("训练头只支持 Sequential、Conv、DWConv 和 Conv2d。");
            if (normalized && (!module.Has("bn") || module.Child("act").Type.Name != "SiLU")) throw new NotSupportedException("训练头需要未融合的 BatchNorm 和 SiLU。");
            var conv = normalized ? module.Child("conv") : module; var shape = WeightShape(conv);
            var padding = conv.Pair("padding", 0); int groups = conv.Int("groups", 1);
            if (shape.Length != 4 || shape[2] != shape[3] || shape[2] is not (1 or 3) || padding[0] != shape[2] / 2 || padding[1] != padding[0] || groups < 1 || shape[0] % groups != 0 || conv.Pair("stride").Any(v => v != 1) || conv.Pair("dilation").Any(v => v != 1))
                throw new NotSupportedException("训练头卷积要求保尺寸、步长1的普通、分组或深度卷积。");
        }
    }

    /// <summary>同时支持完整权重与不物化 storage 的元数据描述。</summary>
    /// <param name="conv">需要读取权重形状的卷积模块。</param><returns>权重维度，不读取张量数据。</returns>
    /// <exception cref="InvalidDataException">卷积没有有效的权重形状。</exception>
    private static int[] WeightShape(PyObject conv) => conv.Map("_parameters").GetValueOrDefault("weight") switch
    {
        Tensor tensor => tensor.Shape, PickleReader.TensorDescription description => description.Shape,
        _ => throw new InvalidDataException("检测头缺少卷积权重形状。")
    };
}
