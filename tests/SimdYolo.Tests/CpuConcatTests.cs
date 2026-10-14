using System.Reflection;
using SimdYolo;

/// <summary>提供通道拼接与点卷积融合的深度尾部、共享输入、偏置和 SiLU的回归验证。</summary>
internal static class CpuConcatTests
{
    /// <summary>向测试执行器注册通道拼接与点卷积融合的深度尾部、共享输入、偏置和 SiLU的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU ConvConcat vs scalar: channel/panel/K tails, aliases, bias and SiLU", () =>
        {
            foreach (bool scoped in new[] { false, true }) WithPool(scoped, (pool, poolType) =>
            {
                using var cpu = new CpuBackend(2); var ops = new TensorOps(cpu);
                foreach (var (counts, oc, h, w) in new[] { (new[] { 3, 5, 1 }, 17, 9, 13), (new[] { 17, 131, 125 }, 37, 5, 7) })
                {
                    var inputs = counts.Select((c, i) => Make(1, c, h, w, i * .17f)).ToArray();
                    var weight = Make(oc, counts.Sum(), 1, 1, .4f); SetConstant(weight);
                    foreach (bool silu in new[] { false, true }) foreach (bool hasBias in new[] { false, true })
                    {
                        Tensor? bias = hasBias ? Make(1, oc, 1, 1, .7f).Reshape(oc) : null;
                        var actual = ops.ConvConcat(inputs, weight, bias, silu: silu);
                        Check(actual, Scalar(inputs, weight, bias, silu));
                        if (pool is not null)
                        {
                            Type activePool = poolType!;
                            var exposed = (Tensor[])activePool.GetMethod("DetachOutputs")!.Invoke(pool, [new[] { actual }])!;
                            float[] snapshot = (float[])exposed[0].Data.Clone();
                            activePool.GetMethod("ReturnAll")!.Invoke(pool, null);
                            inputs[0].Data[0] += .1f;
                            _ = ops.ConvConcat(inputs, weight, bias, silu: silu);
                            activePool.GetMethod("ReturnAll")!.Invoke(pool, null);
                            if (!snapshot.SequenceEqual(exposed[0].Data)) throw new Exception("Public ConvConcat output was reused.");
                        }
                    }
                }
                var shared = Make(1, 3, 5, 7, .1f); var other = Make(1, 5, 5, 7, .2f);
                Tensor[] aliases = [shared, other, shared]; var aliasWeights = Make(19, 11, 1, 1, .3f); SetConstant(aliasWeights);
                Check(ops.ConvConcat(aliases, aliasWeights, silu: true), Scalar(aliases, aliasWeights, null, true));
            });
        });
        test("CPU ConvConcat fallback: batches, mutable weights and strided kernels", () =>
        {
            using var cpu = new CpuBackend(2); var ops = new TensorOps(cpu);
            Tensor[] batched = [Make(2, 3, 5, 7, .1f), Make(2, 5, 5, 7, .2f)];
            var weights = Make(17, 8, 1, 1, .3f); SetConstant(weights);
            Check(ops.ConvConcat(batched, weights, silu: true), Scalar(batched, weights, null, true));
            Tensor[] inputs = [Make(1, 3, 9, 13, .1f), Make(1, 5, 9, 13, .2f)];
            var mutable = Make(17, 8, 1, 1, .3f);
            for (int i = 0; i < 2; i++)
            {
                Check(ops.ConvConcat(inputs, mutable), Scalar(inputs, mutable, null, false));
                mutable.Data[0] += .2f; inputs[0].Data[0] += .3f;
            }
            var kernel = Make(7, 8, 3, 3, .3f); SetConstant(kernel);
            Check(ops.ConvConcat(inputs, kernel, strides: [2, 2], pads: [1, 1, 1, 1], silu: true),
                ops.Conv(TensorOps.Concat(inputs, 1), kernel, strides: [2, 2], pads: [1, 1, 1, 1], silu: true));
        });
    }

    /// <summary>生成指定尺寸和相位的确定性正弦测试数据。</summary>
    /// <param name="b">测试批次数量。</param>
    /// <param name="c">测试通道数量。</param>
    /// <param name="h">输入图像的高度。</param>
    /// <param name="w">测试图像宽度。</param>
    /// <param name="phase">确定性正弦测试数据的相位偏移。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Make(int b, int c, int h, int w, float phase)
    {
        var tensor = new Tensor(b, c, h, w);
        for (int i = 0; i < tensor.Length; i++) tensor.Data[i] = MathF.Sin(i * .031f + phase) * .1f;
        return tensor;
    }
    /// <summary>通过反射设置测试张量的固定权重标志，用于覆盖缓存路径。</summary>
    /// <param name="tensor">需要设置内部常量标志的测试张量。</param>
    private static void SetConstant(Tensor tensor)
        => typeof(Tensor).GetProperty("IsConstant", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(tensor, true);
    /// <summary>使用独立标量循环计算卷积参考结果，避免复用被测 SIMD 内核。</summary>
    /// <param name="inputs">按通道顺序参与拼接或卷积的输入张量。</param>
    /// <param name="weights">待验证的权重数据或内部打包矩阵。</param>
    /// <param name="bias">可选的逐输出通道偏置。</param>
    /// <param name="silu">是否对参考结果应用 SiLU 激活。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Scalar(Tensor[] inputs, Tensor weights, Tensor? bias, bool silu)
    {
        int batch = inputs[0].Shape[0], h = inputs[0].Shape[2], w = inputs[0].Shape[3], n = h * w, ic = inputs.Sum(x => x.Shape[1]), oc = weights.Shape[0];
        var output = new Tensor(batch, oc, h, w);
        for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int p = 0; p < n; p++)
        {
            double sum = 0; int k = 0;
            foreach (var input in inputs)
                for (int ci = 0; ci < input.Shape[1]; ci++, k++) sum += (double)input.Data[(b * input.Shape[1] + ci) * n + p] * weights.Data[co * ic + k];
            float value = (float)sum + (bias?.Data[co] ?? 0);
            output.Data[(b * oc + co) * n + p] = silu ? value / (1 + MathF.Exp(-value)) : value;
        }
        return output;
    }
    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Check(Tensor actual, Tensor expected)
    {
        if (!actual.Shape.SequenceEqual(expected.Shape)) throw new Exception("ConvConcat shape mismatch.");
        for (int i = 0; i < actual.Length; i++)
            if (!float.IsFinite(actual.Data[i]) || MathF.Abs(actual.Data[i] - expected.Data[i]) > 1e-4f * MathF.Max(1, MathF.Abs(expected.Data[i])))
                throw new Exception($"ConvConcat mismatch at {i}: {actual.Data[i]} vs {expected.Data[i]}");
    }
    /// <summary>在指定的 CPU 池化状态下执行验证，并恢复原有作用域。</summary>
    /// <param name="enabled">是否启用本次验证所需的优化或池化作用域。</param>
    /// <param name="action">在临时测试设置或设备帧作用域内执行的操作。</param>
    private static void WithPool(bool enabled, Action<object?, Type?> action)
    {
        if (!enabled) { action(null, null); return; }
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuTensorPool")!;
        FieldInfo current = type.GetField("Current", BindingFlags.Static | BindingFlags.NonPublic)!;
        object? previous = current.GetValue(null); object pool = Activator.CreateInstance(type, nonPublic: true)!;
        type.GetProperty("Parallel", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(pool, new ParallelOptions { MaxDegreeOfParallelism = 2 });
        current.SetValue(null, pool);
        try { action(pool, type); }
        finally { type.GetMethod("ReturnAll")!.Invoke(pool, null); current.SetValue(null, previous); type.GetMethod("Clear")!.Invoke(pool, null); }
    }
}
