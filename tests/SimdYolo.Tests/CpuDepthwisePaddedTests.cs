using System.Reflection;
using SimdYolo;

/// <summary>提供窄平面深度卷积的补零工作区、非有限权重和边界回退的回归验证。</summary>
internal static class CpuDepthwisePaddedTests
{
    /// <summary>向测试执行器注册窄平面深度卷积的补零工作区、非有限权重和边界回退的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU padded depthwise SIMD preserves narrow planes, odd tails, batches and changing weights", () =>
        {
            if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported || !System.Runtime.Intrinsics.X86.Fma.IsSupported) return;
            var run = typeof(Tensor).Assembly.GetType("SimdYolo.CpuDepthwisePadded")!.GetMethod("Run", BindingFlags.NonPublic | BindingFlags.Static)!;
            const int batch = 2, channels = 5;
            foreach (int kernel in new[] { 3, 5, 7 })
                foreach (int width in new[] { 8, 11, 17, 20, 32, 40 })
                    foreach (bool silu in new[] { false, true })
                    {
                        int h = kernel + 2, radius = kernel / 2;
                        float[] input = Enumerable.Range(0, batch * channels * h * width).Select(i => MathF.Sin(i * .13f)).ToArray();
                        float[] weights = Enumerable.Range(0, channels * kernel * kernel).Select(i => MathF.Cos(i * .37f) * .09f).ToArray();
                        float[] bias = Enumerable.Range(0, channels).Select(c => c * .07f - .13f).ToArray();
                        float[] actual = new float[input.Length];
                        for (int frame = 0; frame < 2; frame++)
                        {
                            if (frame != 0) { input[0] += .7f; weights[0] -= .2f; }
                            run.Invoke(null, [input, weights, bias, actual, batch, channels, h, width, kernel, silu, new ParallelOptions { MaxDegreeOfParallelism = 3 }]);
                            for (int b = 0; b < batch; b++) for (int c = 0; c < channels; c++) for (int y = 0; y < h; y++) for (int x = 0; x < width; x++)
                            {
                                double sum = bias[c];
                                for (int ky = 0; ky < kernel; ky++) for (int kx = 0; kx < kernel; kx++)
                                {
                                    int iy = y - radius + ky, ix = x - radius + kx;
                                    if ((uint)iy < h && (uint)ix < width) sum += (double)input[((b * channels + c) * h + iy) * width + ix] * weights[(c * kernel + ky) * kernel + kx];
                                }
                                float expected = (float)(silu ? sum / (1 + Math.Exp(-sum)) : sum), value = actual[((b * channels + c) * h + y) * width + x];
                                if (!float.IsFinite(value) || MathF.Abs(value - expected) > 4e-6f * MathF.Max(1, MathF.Abs(expected))) throw new Exception($"Padded depthwise k{kernel} w{width} changed output {value} vs {expected}.");
                            }
                        }
                    }
        });
        test("CPU padded depthwise exceptional weights retain skipped-padding semantics", () =>
        {
            using var cpu = new CpuBackend(2);
            const int size = 8;
            var input = new Tensor(Enumerable.Repeat(1f, size * size).ToArray(), 1, 1, size, size);
            var weight = new Tensor(Enumerable.Repeat(1f, 9).ToArray(), 1, 1, 3, 3);
            foreach (float exceptional in new[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN })
            {
                weight.Data[0] = exceptional;
                Tensor actual = new TensorOps(cpu).Conv(input, weight, null, [1, 1], [1, 1, 1, 1], [1, 1], 1);
                if (actual.Data[0] != 4 || actual.Data[1] != 6 || actual.Data[size] != 6) throw new Exception("A skipped depthwise tap became zero times a non-finite weight.");
                if (float.IsNaN(exceptional) ? !float.IsNaN(actual.Data[size + 1]) : actual.Data[size + 1] != exceptional) throw new Exception("Depthwise non-finite interior behavior changed.");
            }
        });
    }
}
