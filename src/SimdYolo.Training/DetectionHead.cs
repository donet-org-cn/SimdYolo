namespace SimdYolo.Training;

/// <summary>复制三尺度 YOLOv8 检测头的未融合参数；仅检测头参与更新。</summary>
internal sealed class DetectionHead
{
    internal readonly List<TrainingParameter> Parameters = [];
    internal readonly Dictionary<string, Tensor> Buffers = new(StringComparer.Ordinal);
    internal readonly float[] Strides;
    internal readonly int ClassCount;
    private readonly Branch[] boxes, classes;

    internal DetectionHead(PyObject head, string[] names, IReadOnlyDictionary<int, string> sourceNames, int seed)
    {
        if (head.Type.Name != "Detect" || head.Has("one2one_cv2") || head.Int("reg_max", 16) != 16 || head.Child("cv2").Children.Length != 3 || head.Child("cv3").Children.Length != 3)
            throw new NotSupportedException("首期仅支持 YOLOv8 三尺度、reg_max=16 的标准检测头。");
        ClassCount = names.Length; Strides = (float[])((Tensor?)head.Get("stride") ?? throw new InvalidDataException("缺少检测步长。")).Data.Clone();
        if (!Strides.SequenceEqual(new float[] { 8, 16, 32 })) throw new NotSupportedException("训练要求检测步长为8、16、32。");
        var random = new Random(seed);
        boxes = head.Child("cv2").Children.Select((branch, i) => BuildBranch(branch, $"cv2.{i}", false, i)).ToArray();
        classes = head.Child("cv3").Children.Select((branch, i) => BuildBranch(branch, $"cv3.{i}", true, i)).ToArray();

        Branch BuildBranch(PyObject branch, string prefix, bool classification, int scale)
        {
            var modules = branch.Children;
            if (modules.Length != 3 || modules[0].Type.Name != "Conv" || modules[1].Type.Name != "Conv" || modules[2].Type.Name != "Conv2d")
                throw new NotSupportedException("训练头要求 Conv-BN-SiLU、Conv-BN-SiLU、Conv2d 的 YOLOv8 布局；深度可分离或其他版本需独立适配。");
            var blocks = new Block[3];
            for (int i = 0; i < 3; i++)
            {
                string path = prefix + "." + i; var module = modules[i]; var conv = i == 2 ? module : module.Child("conv");
                if (conv.Int("groups", 1) != 1 || conv.Pair("stride").Any(v => v != 1) || conv.Pair("dilation").Any(v => v != 1)) throw new NotSupportedException("训练卷积仅支持普通步长1卷积。");
                var oldWeight = conv.Weight("weight"); int[] shape = (int[])oldWeight.Shape.Clone(); float[] weight = (float[])oldWeight.Data.Clone();
                float[]? bias = conv.Map("_parameters").GetValueOrDefault("bias") is Tensor b ? (float[])b.Data.Clone() : null;
                if (classification && i == 2)
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
                string convPath = i == 2 ? path : path + ".conv";
                var wp = Add(convPath + ".weight", weight, shape, true);
                TrainingParameter? bp = bias is null ? null : Add(convPath + ".bias", bias, [bias.Length], false);
                var block = new Block(wp, bp, conv.Pair("padding", 0)[0]);
                if (i < 2)
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
    /// <returns>每尺度的原始框分布 logits 和类别 logits。</returns>
    /// <exception cref="ArgumentException">特征尺度数量不是三个。</exception>
    internal (GradientValue[] Boxes, GradientValue[] Classes) Forward(Tensor[] features, HeadOperations ops, GradientTape tape, bool training)
    {
        if (features.Length != 3) throw new ArgumentException("需要三个尺度的冻结特征。");
        var boxValues = new GradientValue[3]; var classValues = new GradientValue[3];
        for (int i = 0; i < 3; i++)
        { var input = new GradientValue(features[i], false); boxValues[i] = boxes[i].Forward(input, ops, tape, training); classValues[i] = classes[i].Forward(input, ops, tape, training); }
        return (boxValues, classValues);
    }

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
        { var tensor = saved[parameter.Name]; if (!tensor.Shape.SequenceEqual(parameter.Shape)) throw new InvalidDataException("参数形状不匹配。"); tensor.Data.CopyTo(parameter.Values, 0); }
        foreach (var pair in Buffers)
        { var tensor = saved[pair.Key]; if (!tensor.Shape.SequenceEqual(pair.Value.Shape)) throw new InvalidDataException("BN 统计形状不匹配。"); tensor.Data.CopyTo(pair.Value.Data, 0); }
    }

    private sealed record Branch(Block[] Blocks)
    {
        internal GradientValue Forward(GradientValue value, HeadOperations ops, GradientTape tape, bool training)
        {
            foreach (var block in Blocks)
            {
                value = ops.Conv(tape, value, block.Weight, block.Bias, block.Padding);
                if (block.Gamma is null) continue;
                value = ops.BatchNorm(tape, value, block.Gamma, block.Beta!, block.Mean!, block.Variance!, block.Epsilon, block.Momentum, training);
                value = HeadOperations.Silu(tape, value, training);
                if (training && block.Tracked is not null) block.Tracked[0]++;
            }
            return value;
        }
    }
    private sealed class Block(TrainingParameter weight, TrainingParameter? bias, int padding)
    {
        internal TrainingParameter Weight = weight;
        internal TrainingParameter? Bias = bias, Gamma, Beta;
        internal int Padding = padding;
        internal float[]? Mean, Variance, Tracked;
        internal float Epsilon, Momentum;
    }
}
