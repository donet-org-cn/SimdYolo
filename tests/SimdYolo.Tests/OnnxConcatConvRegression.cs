using SimdYolo;

/// <summary>提供ONNX 拼接点卷积融合的共享消费者和动态图回退的回归验证。</summary>
internal static class OnnxConcatConvRegression
{
    /// <summary>向测试执行器注册ONNX 拼接点卷积融合的共享消费者和动态图回退的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <param name="devices">需要验证的计算后端集合。</param>
    public static void Run(Action<string, Action> test, IEnumerable<ComputeDevice> devices)
    {
        foreach (var device in devices)
            test($"ONNX concat pointwise fusion, consumers and dynamic fallback ({device})", () => Verify(device));
    }

    /// <summary>在指定设备上核对拼接点卷积融合和共享消费者的回归结果。</summary>
    /// <param name="device">本次验证使用的设备类型。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Verify(ComputeDevice device)
    {
        var variants = new[]
        {
            (Axis: 1, Shared: false, Bias: true, Silu: true),
            (Axis: -3, Shared: false, Bias: false, Silu: false),
            (Axis: 1, Shared: true, Bias: true, Silu: true)
        };
        for (int variantIndex = 0; variantIndex < variants.Length; variantIndex++)
        {
            var variant = variants[variantIndex];
            int channels = 6, outputChannels = 32, kernelChannels = channels;
            var weights = Enumerable.Range(0, outputChannels * kernelChannels).Select(index => (index * 17 % 41 - 20) / 61f).ToArray();
            var bias = Enumerable.Range(0, outputChannels).Select(index => (index - 15) / 100f).ToArray();
            var initializers = new List<OnnxFixtures.Weight> { new("w", weights, [outputChannels, kernelChannels, 1, 1]), new("split", [1, 2], [2], 7) };
            if (variant.Bias) initializers.Add(new("bias", bias, [outputChannels]));
            var nodes = new List<OnnxFixtures.Node>
            {
                new("Split", ["images", "split"], ["left", "right"], [OnnxFixtures.Integer("axis", 1)]),
                new("Concat", ["right", "left", "images"], ["joined"], [OnnxFixtures.Integer("axis", variant.Axis)]),
                new("Conv", variant.Bias ? ["joined", "w", "bias"] : ["joined", "w"], [variant.Silu ? "conv" : "output"],
                    [OnnxFixtures.Integers("strides", 1, 1), OnnxFixtures.Integers("pads", 0, 0, 0, 0), OnnxFixtures.Integer("group", 1)])
            };
            if (variant.Silu)
            {
                nodes.Add(new("Sigmoid", ["conv"], ["sigmoid"]));
                nodes.Add(new("Mul", ["conv", "sigmoid"], ["output"]));
            }
            var declaredOutputs = new List<(string, int[], int)> { ("output", [-1, outputChannels, -1, -1], 1) };
            if (variant.Shared) declaredOutputs.Add(("joined", [-1, channels, -1, -1], 1));
            string path = OnnxFixtures.Write($"concat-conv-{device}-{variantIndex}", [-1, 3, -1, -1], nodes.ToArray(), initializers.ToArray(), declaredOutputs.ToArray());
            var operations = new List<string>();
            using var detector = new YoloDetector(path, new() { Device = device, CpuThreads = 2, LayerCompleted = timing => operations.Add(timing.Operation) });
            Tensor? saved = null; float[]? snapshot = null;
            foreach (var (batch, height, width) in new[] { (1, 13, 19), (2, 11, 17), (1, 13, 19) })
            {
                var input = new Tensor(Enumerable.Range(0, batch * 3 * height * width).Select(index => (index * 13 + height * 7) % 101 / 23f - 2).ToArray(), batch, 3, height, width);
                operations.Clear(); Tensor[] outputs = detector.PredictRaw(input);
                int outputBatch = batch, outputHeight = height, outputWidth = width;
                if (!outputs[0].Shape.SequenceEqual([outputBatch, outputChannels, outputHeight, outputWidth])) throw new Exception("Concat-convolution shape mismatch.");
                int[] channelMap = [1, 2, 0, 0, 1, 2];
                for (int b = 0; b < outputBatch; b++) for (int channel = 0; channel < outputChannels; channel++) for (int y = 0; y < outputHeight; y++) for (int x = 0; x < outputWidth; x++)
                {
                    float expected = variant.Bias ? bias[channel] : 0;
                    for (int k = 0; k < kernelChannels; k++) expected += input.Data[(b * 3 + channelMap[k]) * height * width + y * width + x] * weights[channel * kernelChannels + k];
                    if (variant.Silu) expected /= 1 + MathF.Exp(-expected);
                    float actual = outputs[0].Data[((b * outputChannels + channel) * outputHeight + y) * outputWidth + x];
                    if (MathF.Abs(expected - actual) > 3e-5f + MathF.Abs(expected) * 3e-4f) throw new Exception($"Concat-convolution scalar mismatch: {actual} != {expected}.");
                }
                bool eligibleGraph = device == ComputeDevice.Cpu && !variant.Shared;
                if (operations.Contains("_ConcatConv") != eligibleGraph) throw new Exception("Concat-convolution graph fusion did not respect backend/axis/consumer/geometry.");
                if (variant.Shared)
                    for (int b = 0; b < batch; b++) for (int channel = 0; channel < channels; channel++) for (int pixel = 0; pixel < height * width; pixel++)
                        if (outputs[1].Data[(b * channels + channel) * height * width + pixel] != input.Data[(b * 3 + channelMap[channel]) * height * width + pixel]) throw new Exception("Concat public consumer was changed.");
                if (saved is not null && !saved.Data.SequenceEqual(snapshot!)) throw new Exception("Concat-convolution overwrote a previous output.");
                saved = outputs[0]; snapshot = (float[])saved.Data.Clone();
            }
        }
    }
}
