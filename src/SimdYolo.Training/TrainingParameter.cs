using System.Numerics;

namespace SimdYolo.Training;

/// <summary>保存可更新的 FP32 参数、累加梯度和 SGD 动量，训练参数从不标记为推理常量。</summary>
/// <param name="name">参数在检测头内的唯一路径。</param><param name="values">独立可更新的 FP32 权重数组。</param>
/// <param name="shape">参数张量维度。</param><param name="decay">是否施加卷积权重衰减。</param>
internal sealed class TrainingParameter(string name, float[] values, int[] shape, bool decay)
{
    internal string Name { get; } = name;
    internal float[] Values { get; } = values;
    internal int[] Shape { get; } = shape;
    internal float[] Gradient { get; } = new float[values.Length];
    internal float[] Velocity { get; } = new float[values.Length];
    internal bool Decay { get; } = decay;
    internal bool HasGradient { get; set; }

    /// <summary>按有效微批次图像数量平均梯度，检查有限性后执行带动量的 SGD。</summary>
    /// <param name="options">学习率、动量和衰减配置。</param><param name="images">本次累加的实际图像数量。</param>
    /// <exception cref="ArithmeticException">更新后的权重或动量出现非有限值。</exception>
    internal void Update(TrainingOptions options, int images)
    {
        // 与 PyTorch grad=None 一致：未连接到损失的分支不更新动量，也不施加权重衰减。
        if (!HasGradient) return;
        float inverse = 1f / images, decay = Decay ? options.WeightDecay : 0;
        var inv = new Vector<float>(inverse); var wd = new Vector<float>(decay);
        var momentum = new Vector<float>(options.Momentum); var learningRate = new Vector<float>(options.LearningRate);
        int i = 0;
        for (; i <= Values.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            var weight = new Vector<float>(Values, i);
            var velocity = new Vector<float>(Velocity, i) * momentum + new Vector<float>(Gradient, i) * inv + weight * wd;
            velocity.CopyTo(Velocity, i); (weight - learningRate * velocity).CopyTo(Values, i);
        }
        for (; i < Values.Length; i++) { Velocity[i] = options.Momentum * Velocity[i] + Gradient[i] * inverse + Values[i] * decay; Values[i] -= options.LearningRate * Velocity[i]; }
        if (Values.Any(v => !float.IsFinite(v)) || Velocity.Any(v => !float.IsFinite(v))) throw new ArithmeticException("训练发生非有限参数，请降低学习率。");
        Array.Clear(Gradient);
        HasGradient = false;
    }
}

/// <summary>保存前向值和延迟分配的伴随梯度；节点只在本次训练图中存活。</summary>
/// <param name="value">前向产生的独立张量。</param><param name="requiresGradient">是否继续向该输入传播梯度；冻结特征为 false。</param>
internal sealed class GradientValue(Tensor value, bool requiresGradient = true)
{
    internal Tensor Value { get; } = value;
    internal bool RequiresGradient { get; } = requiresGradient;
    private float[]? gradient;
    internal float[] Gradient => gradient ??= new float[Value.Length];
    internal bool HasGradient => gradient is not null;
    internal float[] GradientOrZero => gradient ?? new float[Value.Length];
}

/// <summary>为检测头保存显式的反向拓扑，不增加推理张量的梯度开销。</summary>
internal sealed class GradientTape
{
    private readonly List<Action> backwards = [];
    internal void Record(Action backward) => backwards.Add(backward);
    internal void Backward()
    {
        for (int i = backwards.Count - 1; i >= 0; i--) backwards[i]();
        backwards.Clear();
    }
}
