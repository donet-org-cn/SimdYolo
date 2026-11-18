using SimdYolo;

internal static class VulkanWinogradPackedTests
{
    internal static void Run(Action<string, Action> test)
    {
        test("Vulkan packed Winograd covers channel/K/spatial tails, both transforms and replayed inputs", () =>
        {
            const int batch = 2, ic = 19, oc = 20;
            float[] weights = Enumerable.Range(0, oc * ic * 9).Select(i => MathF.Cos(i * .37f) * .03f).ToArray();
            float[] bias = Enumerable.Range(0, oc).Select(i => i * .01f).ToArray();
            foreach (var (height, width) in new[] { (13, 15), (20, 20), (31, 39), (40, 40) })
            {
                string path = OnnxFixtures.Write("packed-winograd", [batch, ic, height, width],
                    [new("Conv", ["images", "weights", "bias"], ["conv"], [OnnxFixtures.Integers("pads", 1, 1, 1, 1)]),
                     new("Sigmoid", ["conv"], ["sigmoid"]), new("Mul", ["conv", "sigmoid"], ["output"])],
                    [new("weights", weights, [oc, ic, 3, 3]), new("bias", bias, [oc])],
                    ("output", [batch, oc, height, width], 1));
                using var detector = new YoloDetector(path, new() { Device = ComputeDevice.Vulkan });
                var input = new Tensor(batch, ic, height, width);
                for (int frame = 0; frame < 4; frame++)
                {
                    for (int i = 0; i < input.Length; i++) input.Data[i] = MathF.Sin(i * .17f + frame * .3f) * .2f;
                    Tensor output = detector.PredictRaw(input)[0];
                    for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++)
                        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
                        {
                            double sum = bias[co];
                            for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                            {
                                int iy = y + ky - 1, ix = x + kx - 1;
                                if ((uint)iy < height && (uint)ix < width)
                                    sum += (double)input.Data[((b * ic + ci) * height + iy) * width + ix] * weights[((co * ic + ci) * 3 + ky) * 3 + kx];
                            }
                            float expected = (float)sum;
                            expected /= 1 + MathF.Exp(-expected);
                            float actual = output.Data[((b * oc + co) * height + y) * width + x];
                            if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > 1e-4f * MathF.Max(1, MathF.Abs(expected)))
                                throw new Exception($"Packed Winograd mismatch {height}x{width}, frame {frame}, {b}/{co}/{y}/{x}: {actual} vs {expected}.");
                        }
                }
                if (Environment.GetEnvironmentVariable("SIMDYOLO_VULKAN_REPLAY") != "0" && detector.ReplayedFrames == 0)
                    throw new Exception("Packed Winograd graph did not replay.");
            }
        });
        test("Vulkan Winograd scratch reuse survives buffer growth, layout changes and replay", () =>
        {
            const int inputChannels = 16, hiddenChannels = 20, outputChannels = 32;
            var pads = OnnxFixtures.Integers("pads", 1, 1, 1, 1);
            static float[] Weights(int length) => Enumerable.Range(0, length).Select(i => MathF.Cos(i * .32f) * .023f).ToArray();
            string model = OnnxFixtures.Write("winograd-scratch", [1, inputChannels, -1, -1],
                [new("Conv", ["images", "w1"], ["first"], [pads]),
                 new("Conv", ["first", "w2"], ["second"], [pads]),
                 new("Conv", ["second", "w3"], ["third"], [pads]),
                 new("Conv", ["first", "w2"], ["branch"], [pads])],
                [new("w1", Weights(hiddenChannels * inputChannels * 9), [hiddenChannels, inputChannels, 3, 3]),
                 new("w2", Weights(outputChannels * hiddenChannels * 9), [outputChannels, hiddenChannels, 3, 3]),
                 new("w3", Weights(outputChannels * outputChannels * 9), [outputChannels, outputChannels, 3, 3])],
                ("first", [1, hiddenChannels, -1, -1], 1), ("second", [1, outputChannels, -1, -1], 1),
                ("third", [1, outputChannels, -1, -1], 1), ("branch", [1, outputChannels, -1, -1], 1));
            using var cpu = new YoloDetector(model, new() { Device = ComputeDevice.Cpu, CpuThreads = 2 });
            using var gpu = new YoloDetector(model, new() { Device = ComputeDevice.Vulkan });
            foreach (var (height, width) in new[] { (13, 15), (40, 40), (33, 33), (13, 15) })
            {
                var input = new Tensor(1, inputChannels, height, width);
                for (int frame = 0; frame < 3; frame++)
                {
                    for (int i = 0; i < input.Length; i++) input.Data[i] = MathF.Sin(i * .31f + frame) * .3f;
                    var expected = cpu.PredictRaw(input);
                    var actual = gpu.PredictRaw(input);
                    for (int slot = 0; slot < expected.Length; slot++)
                    {
                        if (!expected[slot].Shape.AsSpan().SequenceEqual(actual[slot].Shape))
                            throw new Exception("Vulkan scratch reuse changed an output shape.");
                        for (int j = 0; j < expected[slot].Length; j++)
                        {
                            float left = expected[slot].Data[j], right = actual[slot].Data[j];
                            if (!float.IsFinite(right) || MathF.Abs(left - right) > 1e-4f)
                                throw new Exception($"Winograd scratch output mismatch for {height}x{width}, frame {frame}, output {slot} at {j}.");
                        }
                    }
                }
            }
            if (Environment.GetEnvironmentVariable("SIMDYOLO_VULKAN_REPLAY") != "0" && gpu.ReplayedFrames == 0)
                throw new Exception("No frame replayed after resizing the Winograd workspace.");
        });
    }
}
