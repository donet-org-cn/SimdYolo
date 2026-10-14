using SimdYolo;

/// <summary>提供GPU 深度卷积向量内核的边界、步长和通道尾部的回归验证。</summary>
internal static class GpuDepthwiseVectorTests
{
    /// <summary>向测试执行器注册GPU 深度卷积向量内核的边界、步长和通道尾部的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <param name="devices">需要验证的计算后端集合。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test, IEnumerable<ComputeDevice> devices)
    {
        test("Depthwise vector3x3: strides, channel tails, batches, borders, bias and SiLU", () =>
        {
            string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_DEPTHWISE_VECTOR");
            Environment.SetEnvironmentVariable("SIMDYOLO_GPU_DEPTHWISE_VECTOR", "1");
            try
            {
                foreach (var (channels, h, width, stride) in new[] { (3, 1, 4, 1), (17, 13, 24, 1), (17, 13, 40, 2), (32, 14, 40, 2) })
                {
                    const int batch = 2;
                    int oh = (h - 1) / stride + 1, ow = (width - 1) / stride + 1;
                    float[] weight = Enumerable.Range(0, channels * 9).Select(i => MathF.Cos(i * .37f) * .03f).ToArray();
                    float[] bias = Enumerable.Range(0, channels).Select(i => i * .01f).ToArray();
                    foreach (bool silu in new[] { false, true }) foreach (bool hasBias in new[] { false, true })
                    {
                        var nodes = new List<OnnxFixtures.Node>
                        {
                            new("Conv", hasBias ? ["images", "weight", "bias"] : ["images", "weight"], [silu ? "conv" : "output"],
                                [OnnxFixtures.Integer("group", channels), OnnxFixtures.Integers("pads", 1, 1, 1, 1), OnnxFixtures.Integers("strides", stride, stride)])
                        };
                        if (silu) { nodes.Add(new("Sigmoid", ["conv"], ["sigmoid"])); nodes.Add(new("Mul", ["conv", "sigmoid"], ["output"])); }
                        var weights = new List<OnnxFixtures.Weight> { new("weight", weight, [channels, 1, 3, 3]) };
                        if (hasBias) weights.Add(new("bias", bias, [channels]));
                        string path = OnnxFixtures.Write("gpu-depthwise-vector", [batch, channels, h, width], nodes.ToArray(), weights.ToArray(), ("output", [batch, channels, oh, ow], 1));
                        foreach (var device in devices)
                        {
                            using var detector = new YoloDetector(path, new() { Device = device });
                            var input = new Tensor(batch, channels, h, width);
                            for (int iteration = 0; iteration < 2; iteration++)
                            {
                                for (int i = 0; i < input.Length; i++) input.Data[i] = MathF.Sin(i * .17f + iteration * .3f) * .2f;
                                var actual = detector.PredictRaw(input)[0];
                                for (int b = 0; b < batch; b++) for (int channel = 0; channel < channels; channel++) for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++)
                                {
                                    double sum = hasBias ? bias[channel] : 0;
                                    for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                                    {
                                        int sy = y * stride - 1 + ky, sx = x * stride - 1 + kx;
                                        if ((uint)sy < h && (uint)sx < width)
                                            sum += (double)input.Data[((b * channels + channel) * h + sy) * width + sx] * weight[channel * 9 + ky * 3 + kx];
                                    }
                                    float expected = (float)sum;
                                    if (silu) expected /= 1 + MathF.Exp(-expected);
                                    float value = actual.Data[((b * channels + channel) * oh + y) * ow + x];
                                    if (!float.IsFinite(value) || MathF.Abs(value - expected) > 1e-4f * MathF.Max(1, MathF.Abs(expected)))
                                        throw new Exception($"Depthwise vector3x3 mismatch at {b}/{channel}/{y}/{x}: {value} vs {expected}");
                                }
                            }
                        }
                    }
                }
            }
            finally { Environment.SetEnvironmentVariable("SIMDYOLO_GPU_DEPTHWISE_VECTOR", previous); }
        });
    }
}
