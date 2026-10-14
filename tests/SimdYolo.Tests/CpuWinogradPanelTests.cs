using System.Reflection;
using SimdYolo;

/// <summary>提供F2 与 F4 工作区分块的批次、行交叉和池化输出生命周期的回归验证。</summary>
internal static class CpuWinogradPanelTests
{
    /// <summary>向测试执行器注册F2 与 F4 工作区分块的批次、行交叉和池化输出生命周期的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU Winograd panels vs double scalar: F2/F4, row crossings, odd shapes, tails and batches", () =>
        {
            using var cpu = new CpuBackend(2); var ops = new TensorOps(cpu);
            // IC=8 选择 F2；在 AVX2 CPU 上，常量 IC=32 选择 F4。
            // 奇数空间尺寸覆盖面板、tile 行、补零输入与输出尾部。
            foreach (var (batch, ic, oc, h, w, constant) in new[]
            {
                (2, 8, 17, 13, 35, false), (1, 13, 19, 19, 67, true),
                (2, 8, 11, 9, 19, false), (1, 13, 19, 13, 20, true),
                (2, 8, 16, 13, 49, true), (1, 80, 80, 9, 35, true),
                (2, 32, 37, 13, 35, true), (1, 32, 41, 21, 67, true)
            })
            {
                var input = Make(batch, ic, h, w, .1f);
                var weights = Make(oc, ic, 3, 3, .3f);
                if (constant) SetConstant(weights);
                var bias = Make(1, oc, 1, 1, .7f).Reshape(oc);
                foreach (bool silu in new[] { false, true }) foreach (bool withBias in new[] { false, true })
                    Check(ops.Conv(input, weights, withBias ? bias : null, pads: [1, 1, 1, 1], silu: silu),
                          Scalar(input, weights, withBias ? bias : null, silu));
                // 工作区和常量打包不得保留旧帧数据；修改输入后，以相同形状再次执行。
                input.Data[0] += .25f;
                input.Data[input.Length / 2] -= .17f;
                if (!constant) weights.Data[0] -= .15f;
                Check(ops.Conv(input, weights, bias, pads: [1, 1, 1, 1], silu: true), Scalar(input, weights, bias, true));
            }
        });
        test("CPU Winograd panels preserve pooled public output ownership", () =>
        {
            Type poolType = typeof(Tensor).Assembly.GetType("SimdYolo.CpuTensorPool")!;
            FieldInfo current = poolType.GetField("Current", BindingFlags.Static | BindingFlags.NonPublic)!;
            object? previous = current.GetValue(null); object pool = Activator.CreateInstance(poolType, nonPublic: true)!;
            poolType.GetProperty("Parallel", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(pool, new ParallelOptions { MaxDegreeOfParallelism = 2 });
            current.SetValue(null, pool);
            try
            {
                using var cpu = new CpuBackend(2); var ops = new TensorOps(cpu);
                foreach (int ic in new[] { 8, 32 })
                {
                    var input = Make(2, ic, 17, 35, .1f); var weights = Make(37, ic, 3, 3, .3f); SetConstant(weights);
                    var output = ops.Conv(input, weights, pads: [1, 1, 1, 1], silu: true);
                    Check(output, Scalar(input, weights, null, true));
                    var detached = (Tensor[])poolType.GetMethod("DetachOutputs")!.Invoke(pool, [new[] { output }])!;
                    float[] expected = (float[])detached[0].Data.Clone();
                    poolType.GetMethod("ReturnAll")!.Invoke(pool, null);
                    input.Data[0] += .75f;
                    Check(ops.Conv(input, weights, pads: [1, 1, 1, 1], silu: true), Scalar(input, weights, null, true));
                    poolType.GetMethod("ReturnAll")!.Invoke(pool, null);
                    if (!expected.SequenceEqual(detached[0].Data)) throw new Exception("Winograd public output was reused.");
                }
            }
            finally
            {
                poolType.GetMethod("ReturnAll")!.Invoke(pool, null); current.SetValue(null, previous);
                poolType.GetMethod("Clear")!.Invoke(pool, null);
            }
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
    /// <param name="input">待验证的输入数据。</param>
    /// <param name="weights">待验证的权重数据或内部打包矩阵。</param>
    /// <param name="bias">可选的逐输出通道偏置。</param>
    /// <param name="silu">是否对参考结果应用 SiLU 激活。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Scalar(Tensor input, Tensor weights, Tensor? bias, bool silu)
    {
        int batch = input.Shape[0], ic = input.Shape[1], h = input.Shape[2], w = input.Shape[3], oc = weights.Shape[0], n = h * w;
        var output = new Tensor(batch, oc, h, w);
        for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
        {
            double sum = 0;
            for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
            {
                int yy = y + ky - 1, xx = x + kx - 1;
                if ((uint)yy < h && (uint)xx < w)
                    sum += (double)input.Data[(b * ic + ci) * n + yy * w + xx] * weights.Data[(co * ic + ci) * 9 + ky * 3 + kx];
            }
            float v = (float)sum + (bias?.Data[co] ?? 0);
            output.Data[(b * oc + co) * n + y * w + x] = silu ? v / (1 + MathF.Exp(-v)) : v;
        }
        return output;
    }
    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Check(Tensor actual, Tensor expected)
    {
        if (!actual.Shape.SequenceEqual(expected.Shape)) throw new Exception("Winograd shape mismatch.");
        for (int i = 0; i < actual.Length; i++)
            if (!float.IsFinite(actual.Data[i]) || MathF.Abs(actual.Data[i] - expected.Data[i]) > 1e-4f * MathF.Max(1, MathF.Abs(expected.Data[i])))
                throw new Exception($"Winograd mismatch at {i}: {actual.Data[i]} vs {expected.Data[i]}");
    }
}
