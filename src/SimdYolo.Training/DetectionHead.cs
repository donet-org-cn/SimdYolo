namespace SimdYolo.Training;

/// <summary>复制五代三尺度检测头的全部未融合训练分支，包括深度卷积和双分支头。</summary>
internal sealed class DetectionHead
{
    internal readonly List<TrainingParameter> Parameters = [];
    internal readonly Dictionary<string, Tensor> Buffers = new(StringComparer.Ordinal);
    internal readonly float[] Strides;
    internal readonly int ClassCount;
    internal readonly TrainingHeadSchema Schema;
    private readonly Branch[] boxes, classes;
    private readonly Branch[]? oneBoxes, oneClasses;

    internal DetectionHead(PyObject head, string[] names, IReadOnlyDictionary<int, string> sourceNames, int seed)
    {
        Schema = TrainingHeadSchema.Read(head);
        ClassCount = names.Length; Strides = (float[])((Tensor?)head.Get("stride") ?? throw new InvalidDataException("缺少检测步长。")).Data.Clone();
        if (!Strides.SequenceEqual(new float[] { 8, 16, 32 })) throw new NotSupportedException("训练要求检测步长为8、16、32。");
        var random = new Random(seed);
        boxes = head.Child("cv2").Children.Select((branch, i) => BuildBranch(branch, $"cv2.{i}", false, i)).ToArray();
        classes = head.Child("cv3").Children.Select((branch, i) => BuildBranch(branch, $"cv3.{i}", true, i)).ToArray();
        if (Schema.EndToEnd)
        {
            oneBoxes = head.Child("one2one_cv2").Children.Select((branch, i) => BuildBranch(branch, $"one2one_cv2.{i}", false, i)).ToArray();
            oneClasses = head.Child("one2one_cv3").Children.Select((branch, i) => BuildBranch(branch, $"one2one_cv3.{i}", true, i)).ToArray();
        }

        Branch BuildBranch(PyObject branch, string prefix, bool classification, int scale)
        {
            var flattened = new List<(PyObject Module, string Path)>();
            void Flatten(PyObject module, string path)
            {
                if (module.Type.Name == "Sequential") { int n = 0; foreach (var child in module.Children) Flatten(child, path + "." + n++); }
                else flattened.Add((module, path));
            }
            Flatten(branch, prefix); var blocks = new Block[flattened.Count];
            for (int i = 0; i < flattened.Count; i++)
            {
                var (module, path) = flattened[i]; bool normalized = module.Type.Name is "Conv" or "DWConv"; var conv = normalized ? module.Child("conv") : module;
                var oldWeight = conv.Weight("weight"); int[] shape = (int[])oldWeight.Shape.Clone(); float[] weight = (float[])oldWeight.Data.Clone();
                float[]? bias = conv.Map("_parameters").GetValueOrDefault("bias") is Tensor b ? (float[])b.Data.Clone() : null;
                if (classification && i == flattened.Count - 1)
                {
                    int row = weight.Length / shape[0]; shape[0] = names.Length; weight = new float[names.Length * row]; bias = new float[names.Length];
                    for (int c = 0; c < names.Length; c++)
                    {
                        int original = sourceNames.FirstOrDefault(p => p.Value == names[c], new(-1, "")).Key;
                        if (original >= 0 && original < oldWeight.Shape[0])
                        { Array.Copy(oldWeight.Data, original * row, weight, c * row, row); bias[c] = conv.Weight("bias").Data[original]; }
                        else
                        {
                            float bound = MathF.Sqrt(1f / row);
                            for (int k = 0; k < row; k++) weight[c * row + k] = ((float)random.NextDouble() * 2 - 1) * bound;
                            bias[c] = MathF.Log(5f / names.Length / MathF.Pow(640f / Strides[scale], 2));
                        }
                    }
                }
                string convPath = normalized ? path + ".conv" : path;
                var wp = Add(convPath + ".weight", weight, shape, true);
                TrainingParameter? bp = bias is null ? null : Add(convPath + ".bias", bias, [bias.Length], false);
                var block = new Block(wp, bp, conv.Pair("padding", 0)[0], conv.Int("groups", 1));
                if (normalized)
                {
                    if (module.Child("act").Type.Name != "SiLU") throw new NotSupportedException("训练头仅支持 SiLU 激活。");
                    var bn = module.Child("bn");
                    block.Gamma = Add(path + ".bn.weight", (float[])bn.Weight("weight").Data.Clone(), [shape[0]], false);
                    block.Beta = Add(path + ".bn.bias", (float[])bn.Weight("bias").Data.Clone(), [shape[0]], false);
                    block.Mean = CloneBuffer(path + ".bn.running_mean", bn.Weight("running_mean"));
                    block.Variance = CloneBuffer(path + ".bn.running_var", bn.Weight("running_var"));
                    if (bn.Map("_buffers").GetValueOrDefault("num_batches_tracked") is Tensor tracked) block.Tracked = CloneBuffer(path + ".bn.num_batches_tracked", tracked);
                    block.Epsilon = bn.Float("eps", 1e-3f); block.Momentum = bn.Float("momentum", .03f);
                }
                blocks[i] = block;
            }
            return new(blocks);
        }
        TrainingParameter Add(string name, float[] values, int[] shape, bool decay)
        { var parameter = new TrainingParameter(name, values, shape, decay); Parameters.Add(parameter); return parameter; }
        float[] CloneBuffer(string name, Tensor tensor)
        { var copy = new Tensor((float[])tensor.Data.Clone(), tensor.Shape); Buffers.Add(name, copy); return copy.Data; }
    }

    /// <summary>执行三个尺度的两组分支，训练模式保留反向所需激活。</summary>
    /// <param name="features">按步长8、16、32排列的冻结特征。</param><param name="ops">当前设备的训练算子。</param>
    /// <param name="tape">本次前向对应的反向记录器。</param><param name="training">是否更新 BN 并记录训练梯度。</param>
    /// <param name="oneToOne">是否计算端到端头的 one2one 分支。</param>
    /// <returns>每尺度的原始框分布 logits 和类别 logits。</returns>
    /// <exception cref="ArgumentException">特征尺度数量不是三个。</exception>
    internal (GradientValue[] Boxes, GradientValue[] Classes) Forward(Tensor[] features, HeadOperations ops, GradientTape tape, bool training, bool oneToOne = false)
    {
        if (features.Length != 3) throw new ArgumentException("需要三个尺度的冻结特征。");
        var boxBranches = oneToOne ? oneBoxes ?? throw new InvalidOperationException("当前头没有 one2one 分支。") : boxes;
        var classBranches = oneToOne ? oneClasses! : classes;
        var boxValues = new GradientValue[3]; var classValues = new GradientValue[3];
        for (int i = 0; i < 3; i++)
        { var input = new GradientValue(features[i], false); boxValues[i] = boxBranches[i].Forward(input, ops, tape, training); classValues[i] = classBranches[i].Forward(input, ops, tape, training); }
        return (boxValues, classValues);
    }

    /// <summary>执行所有训练分支，并依据模型使用传统双头或渐进双头损失。</summary>
    /// <param name="features">冻结的三尺度特征。</param><param name="ops">训练算子。</param><param name="tape">梯度记录器。</param>
    /// <param name="truth">像素坐标真值。</param><param name="training">是否登记梯度及更新 BN。</param>
    /// <param name="epoch">从零开始的已完成轮次。</param><param name="scheduleEpochs">固定的渐进损失调度总轮数。</param>
    /// <returns>优化目标的加权损失。</returns>
    internal DetectionLoss.Result Loss(Tensor[] features, HeadOperations ops, GradientTape tape, TrainingBox[] truth, bool training, int epoch, int scheduleEpochs)
    {
        var many = Forward(features, ops, tape, training);
        if (!Schema.EndToEnd) return DetectionLoss.Compute(many.Boxes, many.Classes, Strides, truth, ClassCount, training, Schema.RegMax);
        var one = Forward(features, ops, tape, training, true);
        float manyWeight = Schema.Progressive ? ProgressiveWeight(epoch, scheduleEpochs) : 1;
        float oneWeight = Schema.Progressive ? 1 - manyWeight : 1;
        var a = DetectionLoss.Compute(many.Boxes, many.Classes, Strides, truth, ClassCount, training, Schema.RegMax, 10, 10, manyWeight);
        var b = DetectionLoss.Compute(one.Boxes, one.Classes, Strides, truth, ClassCount, training, Schema.RegMax, Schema.Progressive ? 7 : 1, 1, oneWeight);
        return new(a.Box * manyWeight + b.Box * oneWeight, a.Classification * manyWeight + b.Classification * oneWeight, a.Dfl * manyWeight + b.Dfl * oneWeight, a.Positives + b.Positives);
    }

    /// <summary>按官方逐轮更新公式计算 YOLO26 one2many 权重，恢复时不重置调度。</summary>
    /// <param name="epoch">从零开始的已完成轮数。</param><param name="epochs">首次创建时固定的调度总轮数。</param>
    /// <returns>一对多分支的权重，从零点八衰减至零点一。</returns>
    internal static float ProgressiveWeight(int epoch, int epochs) => MathF.Max(1 - (float)epoch / Math.Max(epochs - 1, 1), 0) * .7f + .1f;

    /// <summary>返回参数及运行统计的完整只读名称映射，供快照保存。</summary>
    /// <returns>参数和 BN 运行统计的独立深拷贝。</returns>
    internal Dictionary<string, Tensor> Snapshot()
    {
        var result = Parameters.ToDictionary(p => p.Name, p => new Tensor((float[])p.Values.Clone(), p.Shape), StringComparer.Ordinal);
        foreach (var pair in Buffers) result.Add(pair.Key, new Tensor((float[])pair.Value.Data.Clone(), pair.Value.Shape));
        return result;
    }

    /// <summary>严格恢复相同名称及形状的参数和 BN 统计，不允许静默跳过项。</summary>
    /// <param name="saved">来自归档的完整名称到张量映射。</param>
    /// <exception cref="InvalidDataException">参数清单或形状不一致。</exception>
    internal void Restore(IReadOnlyDictionary<string, Tensor> saved)
    {
        var keys = Parameters.Select(p => p.Name).Concat(Buffers.Keys).ToHashSet(StringComparer.Ordinal);
        if (!keys.SetEquals(saved.Keys)) throw new InvalidDataException("训练快照参数清单不完整。");
        foreach (var parameter in Parameters)
        { var tensor = saved[parameter.Name]; if (!tensor.Shape.SequenceEqual(parameter.Shape)) throw new InvalidDataException("参数形状不匹配。"); tensor.Data.CopyTo(parameter.Values, 0); Array.Clear(parameter.Gradient); parameter.HasGradient = false; }
        foreach (var pair in Buffers)
        { var tensor = saved[pair.Key]; if (!tensor.Shape.SequenceEqual(pair.Value.Shape)) throw new InvalidDataException("BN 统计形状不匹配。"); tensor.Data.CopyTo(pair.Value.Data, 0); }
    }

    private sealed record Branch(Block[] Blocks)
    {
        internal GradientValue Forward(GradientValue value, HeadOperations ops, GradientTape tape, bool training)
        {
            foreach (var block in Blocks)
            {
                value = ops.Conv(tape, value, block.Weight, block.Bias, block.Padding, block.Groups);
                if (block.Gamma is null) continue;
                value = ops.BatchNorm(tape, value, block.Gamma, block.Beta!, block.Mean!, block.Variance!, block.Epsilon, block.Momentum, training);
                value = HeadOperations.Silu(tape, value, training);
                if (training && block.Tracked is not null) block.Tracked[0]++;
            }
            return value;
        }
    }
    private sealed class Block(TrainingParameter weight, TrainingParameter? bias, int padding, int groups)
    {
        internal TrainingParameter Weight = weight;
        internal TrainingParameter? Bias = bias, Gamma, Beta;
        internal int Padding = padding;
        internal int Groups = groups;
        internal float[]? Mean, Variance, Tracked;
        internal float Epsilon, Momentum;
    }
}
