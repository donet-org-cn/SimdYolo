using System.Reflection;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供十六列与四十八列点卷积的物理对齐、哨兵区和公开输出生命周期的回归验证。</summary>
internal static class CpuPointwiseAlignedTests
{
    /// <summary>向测试执行器注册十六列与四十八列点卷积的物理对齐、哨兵区和公开输出生命周期的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU physically aligned 16/48-column pointwise covers row edges, changing inputs, concat aliases, deep K and activation", () =>
        {
            if (!Avx.IsSupported || !Fma.IsSupported) return;
            foreach (string core in new[] { "CpuPointwiseAligned", "CpuPointwiseAlignedSmall" })
            {
            Type type = typeof(Tensor).Assembly.GetType("SimdYolo." + core)!;
            MethodInfo conv = type.GetMethod("Conv", BindingFlags.Static | BindingFlags.NonPublic)!;
            using var backend = new CpuBackend(4);
            foreach (var (m, n, counts) in new[]
            {
                (4, 16, new[] { 1, 2 }), (7, 32, new[] { 3, 5, 1 }),
                (25, 48, new[] { 127, 130 }), (35, 64, new[] { 128, 128, 257 }),
                (64, 80, new[] { 32, 32 }), (17, 96, new[] { 64, 65 }),
                (37, 112, new[] { 257, 256 }), (80, 160, new[] { 32, 32 }), (32, 400, new[] { 64, 64 })
            })
            {
                int k = counts.Sum();
                Tensor[] inputs = counts.Select((c, i) => new Tensor(Make(c * n, .1f + i * .2f), 1, c, 1, n)).ToArray();
                var weights = new Tensor(Make(m * k, .4f), m, k, 1, 1); MarkConstant(weights);
                foreach (bool silu in new[] { false, true }) foreach (bool hasBias in new[] { false, true })
                {
                    Tensor? bias = hasBias ? new Tensor(Make(m, .7f), m) : null;
                    Tensor? retained = null; float[]? retainedValues = null;
                    float[] weightValues = (float[])weights.Data.Clone();
                    float[]? biasValues = bias is null ? null : (float[])bias.Data.Clone();
                    for (int iteration = 0; iteration < 3; iteration++)
                    {
                        var actual = GuardedOutput(m, n);
                        float[][] inputValues = inputs.Select(input => (float[])input.Data.Clone()).ToArray();
                        conv.Invoke(null, [backend, inputs, weights, bias, actual, silu]);
                        Check(inputs, weights, bias, actual, silu);
                        if (actual.Data.AsSpan(actual.Length).IndexOfAnyExcept(GuardValue) >= 0)
                            throw new Exception(core + " overwrote output suffix guard.");
                        if (retained is not null && !retained.Data.AsSpan().SequenceEqual(retainedValues))
                            throw new Exception(core + " mutated an earlier output buffer.");
                        for (int source = 0; source < inputs.Length; source++)
                            if (!inputs[source].Data.AsSpan().SequenceEqual(inputValues[source]))
                                throw new Exception(core + " mutated an input buffer.");
                        if (!weights.Data.AsSpan().SequenceEqual(weightValues) || (bias is not null && !bias.Data.AsSpan().SequenceEqual(biasValues)))
                            throw new Exception(core + " mutated constant weights or bias.");
                        retained = actual; retainedValues = (float[])actual.Data.Clone();
                        inputs[0].Data[0] += .17f;
                        if (iteration == 0) GC.Collect();
                    }
                }
            }
            var shared = new Tensor(Make(3 * 64, .2f), 1, 3, 1, 64);
            Tensor[] aliases = [shared, new Tensor(Make(5 * 64, .4f), 1, 5, 1, 64), shared];
            var aliasWeights = new Tensor(Make(19 * 11, .3f), 19, 11, 1, 1); MarkConstant(aliasWeights);
            var aliasOutput = new Tensor(1, 19, 1, 64);
            conv.Invoke(null, [backend, aliases, aliasWeights, null, aliasOutput, true]);
            Check(aliases, aliasWeights, null, aliasOutput, true);

            var odd = new Tensor(1, 19, 1, 65);
            try { conv.Invoke(null, [backend, aliases, aliasWeights, null, odd, true]); }
            catch (TargetInvocationException error) when (error.InnerException is ArgumentException) { continue; }
            throw new Exception("Aligned pointwise accepted an unaligned row stride.");
            }
        });
    }

    /// <summary>通过反射设置测试张量的固定权重标志，用于覆盖缓存路径。</summary>
    /// <param name="value">待编码、标记或比较的数值。</param>
    private static void MarkConstant(Tensor value)
        => typeof(Tensor).GetProperty("IsConstant", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(value, true);
    private const float GuardValue = 1234567f;
    /// <summary>创建逻辑尺寸不变且尾部带哨兵区的输出缓冲区，检测越界写入。</summary>
    /// <param name="channels">测试通道数量。</param>
    /// <param name="columns">空间平面或计算块中的列数。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor GuardedOutput(int channels, int columns)
    {
        var result = new Tensor(1, channels, 1, columns);
        // 为内部内核提供更大的底层数组，用尾部哨兵检测向量或面板越界，避免损坏其他 GC 对象。
        var storage = Enumerable.Repeat(GuardValue, result.Length + 31).ToArray();
        typeof(Tensor).GetField("data", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(result, storage);
        return result;
    }
    /// <summary>生成指定尺寸和相位的确定性正弦测试数据。</summary>
    /// <param name="length">需要生成的浮点元素数量。</param>
    /// <param name="phase">确定性正弦测试数据的相位偏移。</param>
    /// <returns>生成的测试数据或参考变换坐标。</returns>
    private static float[] Make(int length, float phase)
        => Enumerable.Range(0, length).Select(i => MathF.Sin(i * .031f + phase) * .2f).ToArray();
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
            double value = 0; float fp32 = 0; int p = 0;
            foreach (var input in inputs)
                for (int channel = 0; channel < input.Shape[1]; channel++, p++)
                {
                    float a = weights.Data[row * k + p], b = input.Data[channel * n + column];
                    value += (double)a * b;
                    fp32 = MathF.FusedMultiplyAdd(a, b, fp32);
                }
            float bv = bias?.Data[row] ?? 0;
            float expected = (float)value + bv;
            if (silu) expected /= 1 + MathF.Exp(-expected);
            float output = actual.Data[row * n + column];
            if (!float.IsFinite(output) || MathF.Abs(expected - output) > 1e-4f * MathF.Max(1, MathF.Abs(expected)))
                throw new Exception($"Aligned pointwise [{row},{column}]={output} vs {expected}");
            if (!silu)
            {
                fp32 += bv;
                if (BitConverter.SingleToInt32Bits(output) != BitConverter.SingleToInt32Bits(fp32))
                    throw new Exception($"Aligned pointwise changed FP32 FMA order [{row},{column}]: {output} vs {fp32}");
            }
        }
    }
}

