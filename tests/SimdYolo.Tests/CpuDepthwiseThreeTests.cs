using System.Reflection;
using SimdYolo;

/// <summary>提供三乘三深度卷积的共享加载、步长、批次和可变缓冲区的回归验证。</summary>
internal static class CpuDepthwiseThreeTests
{
    /// <summary>向测试执行器注册三乘三深度卷积的共享加载、步长、批次和可变缓冲区的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU depthwise three-by-three shared loads preserve borders, tails, batches and mutable buffers", () =>
        {
            if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported || !System.Runtime.Intrinsics.X86.Fma.IsSupported) return;
            var kernel = typeof(Tensor).Assembly.GetType("SimdYolo.CpuSpatial")!.GetMethod("DepthwiseThreeByThree", BindingFlags.Static | BindingFlags.NonPublic)!;
            foreach (int stride in new[] { 1, 2 })
                foreach (bool silu in new[] { false, true })
                {
                    const int h = 5, width = 23, channels = 3;
                    var input = Enumerable.Range(0, h * width * channels).Select(i => MathF.Sin(i * .17f)).ToArray();
                    var weights = Enumerable.Range(0, channels * 9).Select(i => MathF.Cos(i * .3f) * .4f).ToArray();
                    var output = new float[8 * channels + 7];
                    for (int frame = 0; frame < 2; frame++)
                    {
                        if (frame != 0) { input[0] += .75f; weights[0] -= .2f; }
                        for (int c = 0; c < channels; c++)
                        {
                            float bias = c * .2f - .1f;
                            kernel.Invoke(null, [input, weights, output, c * h * width, c * 9, 0, 0, width, stride, c * 8 + 3, bias, silu]);
                            for (int x = 0; x < 8; x++)
                            {
                                float sum = bias;
                                for (int ky = 0; ky < 3; ky++)
                                    for (int kx = 0; kx < 3; kx++) sum = MathF.FusedMultiplyAdd(input[c * h * width + ky * width + x * stride + kx], weights[c * 9 + ky * 3 + kx], sum);
                                float expected = silu ? sum / (1 + MathF.Exp(-sum)) : sum;
                                float actual = output[c * 8 + x + 3];
                                if (silu ? MathF.Abs(actual - expected) > 4e-7f * MathF.Max(1, MathF.Abs(expected)) : BitConverter.SingleToInt32Bits(actual) != BitConverter.SingleToInt32Bits(expected))
                                    throw new Exception($"Depthwise stride {stride} shared loads changed the accumulation: {actual} vs {expected}.");
                            }
                        }
                    }
                    if (output[0] != 0 || output[1] != 0 || output[2] != 0 || output[^1] != 0) throw new Exception("Depthwise vector store crossed its output block.");
                }
            using var backend = new CpuBackend(3);
            foreach (int stride in new[] { 1, 2 })
                foreach (int width in new[] { 17, 19, 32 })
                {
                    const int batch = 2, channels = 3, h = 7;
                    var input = new Tensor(Enumerable.Range(0, batch * channels * h * width).Select(i => MathF.Sin(i * .17f)).ToArray(), batch, channels, h, width);
                    var weight = new Tensor(Enumerable.Range(0, channels * 9).Select(i => MathF.Cos(i * .3f) * .1f).ToArray(), channels, 1, 3, 3);
                    var bias = new Tensor([-.1f, .2f, .5f], channels);
                    var actual = new TensorOps(backend).Conv(input, weight, bias, [stride, stride], [1, 1, 1, 1], [1, 1], channels, true);
                    int oh = actual.Shape[2], ow = actual.Shape[3];
                    for (int b = 0; b < batch; b++) for (int c = 0; c < channels; c++) for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++)
                    {
                        double sum = bias.Data[c];
                        for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                        {
                            int iy = y * stride - 1 + ky, ix = x * stride - 1 + kx;
                            if ((uint)iy < h && (uint)ix < width) sum += (double)input.Data[((b * channels + c) * h + iy) * width + ix] * weight.Data[c * 9 + ky * 3 + kx];
                        }
                        float expected = (float)(sum / (1 + Math.Exp(-sum))), value = actual.Data[((b * channels + c) * oh + y) * ow + x];
                        if (MathF.Abs(value - expected) > 4e-6f * MathF.Max(1, MathF.Abs(expected))) throw new Exception("Depthwise border/tail changed scalar convolution.");
                    }
                }
        });
    }
}
