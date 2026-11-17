using SimdYolo;

internal static class MaxPoolCeilTests
{
    internal static void Run(Action<string, Action> test, IEnumerable<ComputeDevice> devices)
    {
        foreach (var device in devices)
            test($"MaxPool ceil mode includes partial windows and omits padding-only starts ({device})", () => Verify(device));
    }

    private static void Verify(ComputeDevice device)
    {
        foreach (var (height, width, kernel, stride, pads, oh, ow) in new[]
        {
            (4, 34, 3, 2, new[] { 0, 0, 0, 0 }, 2, 17),
            (5, 33, 2, 2, new[] { 1, 1, 1, 1 }, 3, 17),
            (2, 35, 3, 2, new[] { 0, 0, 0, 0 }, 1, 17),
            (4, 36, 3, 2, new[] { 0, 1, 1, 0 }, 2, 18)
        })
        {
            int[] shape = [2, 2, height, width];
            int[] outputShape = [2, 2, oh, ow];
            string path = OnnxFixtures.Write("maxpool-ceil", shape,
                [new("MaxPool", ["images"], ["pooled"],
                    [OnnxFixtures.Integers("kernel_shape", kernel, kernel),
                     OnnxFixtures.Integers("strides", stride, stride), OnnxFixtures.Integers("pads", pads),
                     OnnxFixtures.Integer("ceil_mode", 1)])], [], ("pooled", outputShape, 1));
            using var model = new YoloDetector(path, new() { Device = device, CpuThreads = 2 });
            var input = new Tensor(shape);
            for (int frame = 0; frame < 3; frame++)
            {
                for (int i = 0; i < input.Length; i++)
                    input.Data[i] = -1 - (i * 73 + frame * 29) % 257;
                var actual = model.PredictRaw(input)[0];
                if (!actual.Shape.AsSpan().SequenceEqual(outputShape))
                    throw new Exception($"Incorrect MaxPool shape for {height}x{width}: {actual}.");
                for (int plane = 0; plane < 4; plane++)
                    for (int y = 0; y < oh; y++)
                        for (int x = 0; x < ow; x++)
                        {
                            float expected = float.NegativeInfinity;
                            for (int ky = 0; ky < kernel; ky++)
                                for (int kx = 0; kx < kernel; kx++)
                                {
                                    int sy = y * stride + ky - pads[0], sx = x * stride + kx - pads[1];
                                    if ((uint)sy < height && (uint)sx < width)
                                        expected = MathF.Max(expected, input.Data[(plane * height + sy) * width + sx]);
                                }
                            if (actual.Data[(plane * oh + y) * ow + x] != expected)
                                throw new Exception($"Pooling included a padding value at {plane}/{y}/{x}.");
                        }
            }
        }
    }
}
