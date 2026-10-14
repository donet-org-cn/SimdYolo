using System.Reflection;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供二十四列点卷积的拼接别名、可变输入、偏置和激活的回归验证。</summary>
internal static class CpuPointwise24Tests
{
    /// <summary>向测试执行器注册二十四列点卷积的拼接别名、可变输入、偏置和激活的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU 24-column pointwise vs double scalar: row/K/panel tails, concat aliases and SiLU", () =>
        {
            if (!Avx.IsSupported || !Fma.IsSupported) return;
            Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuPointwise24")!;
            MethodInfo method = type.GetMethod("Conv", BindingFlags.Static | BindingFlags.NonPublic)!;
            using var backend = new CpuBackend(2);
            foreach (var (m, n, counts) in new[]
            {
                (4, 24, new[] { 1, 2 }), (7, 25, new[] { 3, 5, 1 }),
                (5, 32, new[] { 7, 1 }), (9, 40, new[] { 11, 2 }),
                (16, 47, new[] { 7, 8 }), (37, 67, new[] { 17, 131, 125 }),
                (37, 64, new[] { 17, 131, 125 }),
                (37, 64, new[] { 128, 128, 257 }),
                (35, 56, new[] { 127, 130, 256 }),
                (64, 160, new[] { 32, 32 })
            })
            {
                int k = counts.Sum(); var inputs = counts.Select((c, i) => Make(c * n, .1f + i * .2f, 1, c, 1, n)).ToArray();
                var weights = Make(m * k, .4f, m, k, 1, 1);
                typeof(Tensor).GetProperty("IsConstant", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(weights, true);
                foreach (bool silu in new[] { false, true }) foreach (bool hasBias in new[] { false, true })
                {
                    Tensor? bias = hasBias ? Make(m, .7f, m) : null; var actual = new Tensor(1, m, 1, n);
                    method.Invoke(null, [backend, inputs, weights, bias, actual, silu]);
                    Check(inputs, weights, bias, actual, silu);
                    inputs[0].Data[0] += .17f;
                    method.Invoke(null, [backend, inputs, weights, bias, actual, silu]);
                    Check(inputs, weights, bias, actual, silu);
                }
            }
            var shared = Make(3 * 49, .2f, 1, 3, 1, 49);
            Tensor[] aliases = [shared, Make(5 * 49, .4f, 1, 5, 1, 49), shared];
            var aliasWeights = Make(19 * 11, .3f, 19, 11, 1, 1);
            typeof(Tensor).GetProperty("IsConstant", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(aliasWeights, true);
            var aliasOutput = new Tensor(1, 19, 1, 49);
            method.Invoke(null, [backend, aliases, aliasWeights, null, aliasOutput, true]);
            Check(aliases, aliasWeights, null, aliasOutput, true);
        });
        test("CPU pointwise batch fallback retains deep-K inputs with constant weights", () =>
        {
            using var backend = new CpuBackend(2);
            const int batch = 2, m = 37, n = 64, k = 513;
            var input = Make(batch * k * n, .1f, batch, k, 1, n);
            var weights = Make(m * k, .4f, m, k, 1, 1);
            typeof(Tensor).GetProperty("IsConstant", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(weights, true);
            for (int iteration = 0; iteration < 2; iteration++)
            {
                var actual = new TensorOps(backend).Conv(input, weights, silu: true);
                for (int b = 0; b < batch; b++) for (int row = 0; row < m; row++) for (int column = 0; column < n; column++)
                {
                    double sum = 0;
                    for (int p = 0; p < k; p++) sum += (double)weights.Data[row * k + p] * input.Data[(b * k + p) * n + column];
                    float expected = (float)sum; expected /= 1 + MathF.Exp(-expected);
                    float value = actual.Data[(b * m + row) * n + column];
                    if (!float.IsFinite(value) || MathF.Abs(value - expected) > 1e-4f * MathF.Max(1, MathF.Abs(expected)))
                        throw new Exception($"Batched pointwise [{b},{row},{column}]={value} vs {expected}");
                }
                input.Data[0] += .17f;
            }
        });
    }

    /// <summary>生成指定尺寸和相位的确定性正弦测试数据。</summary>
    /// <param name="length">需要生成的浮点元素数量。</param>
    /// <param name="phase">确定性正弦测试数据的相位偏移。</param>
    /// <param name="shape">测试张量的逻辑维度。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Make(int length, float phase, params int[] shape)
        => new(Enumerable.Range(0, length).Select(i => MathF.Sin(i * .031f + phase) * .1f).ToArray(), shape);
    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="inputs">按通道顺序参与拼接或卷积的输入张量。</param>
    /// <param name="weights">待验证的权重数据或内部打包矩阵。</param>
    /// <param name="bias">可选的逐输出通道偏置。</param>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="silu">是否对参考结果应用 SiLU 激活。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Check(Tensor[] inputs, Tensor weights, Tensor? bias, Tensor actual, bool silu)
    {
        int n = actual.Shape[3], m = actual.Shape[1], k = weights.Shape[1];
        for (int row = 0; row < m; row++) for (int column = 0; column < n; column++)
        {
            double value = 0; int p = 0;
            foreach (var input in inputs)
                for (int channel = 0; channel < input.Shape[1]; channel++, p++)
                    value += (double)weights.Data[row * k + p] * input.Data[channel * n + column];
            float expected = (float)value + (bias?.Data[row] ?? 0);
            if (silu) expected /= 1 + MathF.Exp(-expected);
            float output = actual.Data[row * n + column];
            if (!float.IsFinite(output) || MathF.Abs(expected - output) > 1e-4f * MathF.Max(1, MathF.Abs(expected)))
                throw new Exception($"Pointwise24 [{row},{column}]={output} vs {expected}");
        }
    }
}
