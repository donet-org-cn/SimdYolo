using SimdYolo;

/// <summary>提供GPU 卷积计算块的通道尾部、空间尾部、偏置和 SiLU的回归验证。</summary>
internal static class GpuConvolutionKernelTests
{
    /// <summary>向测试执行器注册GPU 卷积计算块的通道尾部、空间尾部、偏置和 SiLU的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <param name="devices">需要验证的计算后端集合。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test, IEnumerable<ComputeDevice> devices)
    {
        test("Frame convolution kernel variants: channel/spatial tails, batches, bias and SiLU", () =>
        {
            foreach (var (ic, oc, h, width, kernel, stride) in new[]
            {
                (17, 33, 13, 15, 1, 1), (32, 64, 23, 25, 1, 1),
                (17, 33, 13, 15, 3, 2), (32, 64, 40, 44, 3, 2),
                (17, 64, 13, 15, 3, 1), (32, 64, 23, 25, 3, 1),
                (32, 64, 24, 28, 3, 1), (32, 33, 24, 28, 3, 1),
                (32, 64, 41, 45, 3, 1), (17, 128, 33, 35, 3, 1), (17, 64, 13, 25, 3, 1),
                (17, 33, 41, 45, 3, 2), (17, 64, 13, 24, 1, 1),
                (17, 80, 13, 24, 1, 1), (17, 32, 13, 24, 1, 1), (17, 24, 13, 24, 1, 1), (17, 16, 13, 24, 1, 1)
            })
            {
                const int batch = 2;
                int pad = kernel == 3 ? 1 : 0, oh = (h + 2 * pad - kernel) / stride + 1, ow = (width + 2 * pad - kernel) / stride + 1;
                float[] weight = Enumerable.Range(0, oc * ic * kernel * kernel).Select(i => MathF.Cos(i * .37f) * .03f).ToArray();
                float[] bias = Enumerable.Range(0, oc).Select(i => i * .01f).ToArray();
                foreach (bool silu in new[] { false, true }) foreach (bool hasBias in new[] { false, true })
                {
                    var nodes = new List<OnnxFixtures.Node>
                    {
                        new("Conv", hasBias ? ["images", "weight", "bias"] : ["images", "weight"], [silu ? "conv" : "output"],
                            [OnnxFixtures.Integers("pads", pad, pad, pad, pad), OnnxFixtures.Integers("strides", stride, stride)])
                    };
                    if (silu) { nodes.Add(new("Sigmoid", ["conv"], ["sigmoid"])); nodes.Add(new("Mul", ["conv", "sigmoid"], ["output"])); }
                    var weights = new List<OnnxFixtures.Weight> { new("weight", weight, [oc, ic, kernel, kernel]) };
                    if (hasBias) weights.Add(new("bias", bias, [oc]));
                    string path = OnnxFixtures.Write("gpu-convolution-kernels", [batch, ic, h, width], nodes.ToArray(), weights.ToArray(), ("output", [batch, oc, oh, ow], 1));
                    foreach (var device in devices)
                    {
                        using var detector = new YoloDetector(path, new() { Device = device });
                        var input = new Tensor(batch, ic, h, width);
                        for (int iteration = 0; iteration < 2; iteration++)
                        {
                            for (int i = 0; i < input.Length; i++) input.Data[i] = MathF.Sin(i * .17f + iteration * .3f) * .2f;
                            var actual = detector.PredictRaw(input)[0];
                            for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++)
                            {
                                double sum = hasBias ? bias[co] : 0;
                                for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < kernel; ky++) for (int kx = 0; kx < kernel; kx++)
                                {
                                    int sy = y * stride - pad + ky, sx = x * stride - pad + kx;
                                    if ((uint)sy < h && (uint)sx < width)
                                        sum += (double)input.Data[((b * ic + ci) * h + sy) * width + sx] * weight[((co * ic + ci) * kernel + ky) * kernel + kx];
                                }
                                float expected = (float)sum;
                                if (silu) expected /= 1 + MathF.Exp(-expected);
                                float value = actual.Data[((b * oc + co) * oh + y) * ow + x];
                                if (!float.IsFinite(value) || MathF.Abs(value - expected) > 1e-4f * MathF.Max(1, MathF.Abs(expected)))
                                    throw new Exception($"Convolution kernel mismatch at {b}/{co}/{y}/{x}: {value} vs {expected}");
                            }
                        }
                    }
                }
            }
        });
    }
}
