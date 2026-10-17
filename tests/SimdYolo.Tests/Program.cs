using SimdYolo;

int passed = 0;
// 执行命名用例；name 用于筛选和输出，test 为断言回调。成功累加计数，失败打印异常并以非零状态退出。
void Test(string name, Action test)
{
    string? filter = Environment.GetEnvironmentVariable("SIMDYOLO_TEST_FILTER");
    if (!string.IsNullOrEmpty(filter) && !name.Contains(filter, StringComparison.OrdinalIgnoreCase)) return;
    try { test(); passed++; Console.WriteLine("PASS " + name); }
    catch (Exception error) { Console.Error.WriteLine($"FAIL {name}\n{error}"); Environment.Exit(1); }
}
// 按相对/绝对容差 eps 比较 actual 与 expected；非有限实际值或超出容差时抛出断言异常。
void Near(float actual, float expected, float eps = 1e-4f) { if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > eps * MathF.Max(1, MathF.Abs(expected))) throw new Exception($"{actual} != {expected}"); }
// 在 backend 上验证矩阵乘法，覆盖非整块尺寸、深度尾部和输入更新；使用独立 double 循环作参考。
void Gemm(IComputeBackend backend)
{
    var random = new Random(123);
    foreach (var (m, n, k) in new[] { (1, 1, 1), (3, 19, 7), (17, 33, 21), (32, 64, 128), (37, 35, 273) })
    {
        var a = Enumerable.Range(0, m * k).Select(_ => (float)random.NextDouble() - .5f).ToArray(); var b = Enumerable.Range(0, k * n).Select(_ => (float)random.NextDouble() - .5f).ToArray(); var c = new float[m * n];
        backend.MatMul(a, b, c, m, n, k);
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { double expected = 0; for (int p = 0; p < k; p++) expected += (double)a[i * k + p] * b[p * n + j]; Near(c[i * n + j], (float)expected); }
        a[0] += 1; backend.MatMul(a, b, c, m, n, k); double updated = 0; for (int p = 0; p < k; p++) updated += (double)a[p] * b[p * n]; Near(c[0], (float)updated);
    }
}
using var cpu = new CpuBackend();
Test("Main library is managed IL and references only .NET framework assemblies", () =>
{
    var assembly = typeof(YoloDetector).Assembly;
    using var stream = File.OpenRead(assembly.Location); using var pe = new System.Reflection.PortableExecutable.PEReader(stream);
    if (pe.PEHeaders.CorHeader is not { } cor || (cor.Flags & System.Reflection.PortableExecutable.CorFlags.ILOnly) == 0) throw new Exception("Main library is not managed IL.");
    foreach (var reference in assembly.GetReferencedAssemblies())
        if (reference.Name is not ("netstandard" or "mscorlib" or "Microsoft.CSharp") && reference.Name?.StartsWith("System", StringComparison.Ordinal) != true) throw new Exception("External library dependency: " + reference.Name);
});
Test("SIMD GEMM, tails and changing buffers", () => Gemm(cpu));
CpuConcatTests.Run(Test);
CpuWinogradPanelTests.Run(Test);
CpuPackedGemmTests.Run(Test);
CpuPointwise24Tests.Run(Test);
CpuActivationTests.Run(Test);
MaskPipelineTests.Run(Test);
CpuExecutorTests.Run(Test);
CpuHeadOptimizationTests.Run(Test);
CpuHeadCoordinatesTests.Run(Test);
PtRoutingReuseTests.Run(Test);
CpuWeightBatchTests.Run(Test);
CpuMatMulOffsetsTests.Run(Test);
ModelReaderOptimizationTests.Run(Test);
OnnxYolo26HeadTests.Run(Test, args.Contains("--vulkan"));
ModelInfoTests.Run(Test);
Yolo26Tests.Run(Test, args.Contains("--vulkan"));
CpuGemm24Tests.Run(Test);
CpuWinograd424Tests.Run(Test);
CpuOnnxEndToEndTests.Run(Test);
CpuPtEndToEndTests.Run(Test);
CpuTransposeSimdTests.Run(Test);
CpuDepthwiseThreeTests.Run(Test);
CpuDepthwisePaddedTests.Run(Test);
CpuPointwiseAlignedTests.Run(Test);
CpuConvolutionPackTests.Run(Test);
CpuConvolutionAlignedTests.Run(Test);
CpuAverageTwoTests.Run(Test);
OnnxCpuZeroAddTests.Run(Test, FixtureDevices());
OnnxConcatConvRegression.Run(Test, FixtureDevices());
GpuConvolutionKernelTests.Run(Test, FixtureDevices());
GpuDepthwiseVectorTests.Run(Test, FixtureDevices());
OnnxDeviceHeadTests.Run(Test, FixtureDevices());
if (args.Contains("--vulkan")) { VulkanPipelineCacheTests.Run(Test); VulkanReplayTests.Run(Test); GpuTopKTests.Run(Test); OnnxEndToEndTests.Run(Test); VulkanBufferViewsTests.Run(Test); GpuPackedPointwise32Tests.Run(Test); }
Test("Grouped convolution vs scalar", () =>
{
    var ops = new TensorOps(cpu); var x = new Tensor(Enumerable.Range(0, 2 * 4 * 7 * 9).Select(i => MathF.Sin(i)).ToArray(), 2, 4, 7, 9);
    var w = new Tensor(Enumerable.Range(0, 6 * 2 * 3 * 3).Select(i => MathF.Cos(i) * .1f).ToArray(), 6, 2, 3, 3); var bias = new Tensor([1, 2, 3, 4, 5, 6], 6);
    var y = ops.Conv(x, w, bias, [2, 1], [1, 2, 1, 0], [1, 2], 2);
    for (int b = 0; b < 2; b++) for (int c = 0; c < 6; c++) for (int oy = 0; oy < y.Shape[2]; oy++) for (int ox = 0; ox < y.Shape[3]; ox++)
    {
        double v = bias.Data[c]; for (int ci = 0; ci < 2; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
        { int iy = oy * 2 - 1 + ky, ix = ox - 2 + kx * 2; if ((uint)iy < 7 && (uint)ix < 9) v += (double)x.Data[((b * 4 + c / 3 * 2 + ci) * 7 + iy) * 9 + ix] * w.Data[((c * 2 + ci) * 3 + ky) * 3 + kx]; }
        Near(y.Data[((b * 6 + c) * y.Shape[2] + oy) * y.Shape[3] + ox], (float)v);
    }
});
Test("Implicit convolution panels: strides, borders, batches and mutable weights", () =>
{
    foreach (var (batch, ic, oc, h, width, kernel, stride, pad, dilation) in new[]
    {
        (2, 3, 16, 15, 19, 3, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }),
        (2, 17, 19, 9, 11, 3, new[] { 2, 1 }, new[] { 1, 2, 0, 1 }, new[] { 1, 2 }),
        (2, 4, 7, 7, 11, 1, new[] { 1, 1 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }),
        (1, 3, 8, 19, 64, 5, new[] { 2, 2 }, new[] { 2, 3, 1, 2 }, new[] { 1, 1 })
    })
    {
        var input = new Tensor(Enumerable.Range(0, batch * ic * h * width).Select(i => MathF.Sin(i)).ToArray(), batch, ic, h, width);
        var weight = new Tensor(Enumerable.Range(0, oc * ic * kernel * kernel).Select(i => MathF.Cos(i) * .05f).ToArray(), oc, ic, kernel, kernel);
        var bias = new Tensor(Enumerable.Range(0, oc).Select(i => i * .01f).ToArray(), oc);
        for (int iteration = 0; iteration < 2; iteration++)
        {
            if (iteration != 0) { input.Data[0] += 1; weight.Data[0] += .1f; }
            var output = new TensorOps(cpu).Conv(input, weight, bias, stride, pad, dilation, silu: iteration != 0);
            for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int y = 0; y < output.Shape[2]; y++) for (int x = 0; x < output.Shape[3]; x++)
            {
                double sum = bias.Data[co];
                for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < kernel; ky++) for (int kx = 0; kx < kernel; kx++)
                { int sy = y * stride[0] - pad[0] + ky * dilation[0], sx = x * stride[1] - pad[1] + kx * dilation[1]; if ((uint)sy < h && (uint)sx < width) sum += (double)input.Data[((b * ic + ci) * h + sy) * width + sx] * weight.Data[((co * ic + ci) * kernel + ky) * kernel + kx]; }
                float expected = (float)sum; if (iteration != 0) expected /= 1 + MathF.Exp(-expected);
                Near(output.Data[((b * oc + co) * output.Shape[2] + y) * output.Shape[3] + x], expected);
            }
        }
    }
});
Test("Public shape mutation is rejected before unsafe kernels", () =>
{
    // action 必须在进入不安全内核前拒绝损坏形状；没有抛出预期异常时让该测试失败。
    void Rejected(Action action)
    {
        try { action(); } catch (ArgumentException) { return; }
        throw new Exception("Inconsistent tensor storage was accepted.");
    }
    var x = new Tensor(1, 3, 9, 11); var w = new Tensor(6, 3, 3, 3); var ops = new TensorOps(cpu);
    x.Shape[3] = 99; Rejected(() => ops.Conv(x, w)); Rejected(() => TensorOps.Pool(x, [3, 3], [1, 1], [1, 1, 1, 1])); Rejected(() => TensorOps.ResizeNearest(x, 18, 22));
    x.Shape[3] = 11; w.Shape[3] = 99; Rejected(() => ops.Conv(x, w));
    var a = new Tensor(4, 7); var b = new Tensor(7, 8); a.Shape[0] = 40; Rejected(() => ops.MatMul(a, b));
    var wide = new Tensor(Enumerable.Range(0, 3 * 32).Select(i => (float)i).ToArray(), 1, 1, 3, 32);
    var extreme = ops.Conv(wide, new Tensor([2, 9], 1, 1, 1, 2), null, [1, 2], [0, 0, 0, int.MaxValue], [1, int.MaxValue], 1);
    if (!extreme.Shape.SequenceEqual([1, 1, 3, 16])) throw new Exception("Convolution dimension arithmetic overflowed.");
    for (int row = 0; row < 3; row++) for (int column = 0; column < 16; column++) Near(extreme.Data[row * 16 + column], wide.Data[row * 32 + column * 2] * 2);
});
Test("Broadcast/transpose/softmax", () =>
{
    var x = new Tensor([1, 2, 3, 4, 5, 6], 2, 3); var y = TensorOps.Binary(x, new Tensor([10, 20, 30], 3), (a, b) => a + b);
    Near(y.Data[5], 36); var t = TensorOps.Transpose(x); Near(t.Data[1], 4); var soft = TensorOps.Softmax(new Tensor([1000, 1001, 1002], 1, 3), 1); Near(soft.Data.Sum(), 1); Near(soft.Data[2], .66524096f);
});
Test("Separable padded max pool vs scalar", () =>
{
    var x = new Tensor(Enumerable.Range(0, 2 * 3 * 9 * 13).Select(i => MathF.Sin(i)).ToArray(), 2, 3, 9, 13);
    foreach (int stride in new[] { 1, 2 })
    {
        var y = TensorOps.Pool(x, [5, 3], [stride, stride], [2, 1, 2, 1]);
        for (int c = 0; c < 6; c++) for (int oy = 0; oy < y.Shape[2]; oy++) for (int ox = 0; ox < y.Shape[3]; ox++)
        {
            float expected = float.NegativeInfinity;
            for (int ky = 0; ky < 5; ky++) for (int kx = 0; kx < 3; kx++) { int sy = oy * stride - 2 + ky, sx = ox * stride - 1 + kx; if ((uint)sy < 9 && (uint)sx < 13) expected = MathF.Max(expected, x.Data[(c * 9 + sy) * 13 + sx]); }
            Near(y.Data[(c * y.Shape[2] + oy) * y.Shape[3] + ox], expected);
        }
    }
});
Test("SIMD average pool, padding and excluded samples vs scalar", () =>
{
    var x = new Tensor(Enumerable.Range(0, 2 * 3 * 9 * 43).Select(i => MathF.Sin(i)).ToArray(), 2, 3, 9, 43);
    foreach (var (kernel, stride, pad) in new[] { (2, 1, 0), (3, 1, 1), (3, 2, 1) }) foreach (bool include in new[] { false, true })
    {
        var output = TensorOps.Pool(x, [kernel, kernel], [stride, stride], [pad, pad, pad, pad], average: true, countIncludePad: include);
        for (int channel = 0; channel < 6; channel++) for (int y = 0; y < output.Shape[2]; y++) for (int xx = 0; xx < output.Shape[3]; xx++)
        {
            double sum = 0; int count = 0;
            for (int ky = 0; ky < kernel; ky++) for (int kx = 0; kx < kernel; kx++) { int sy = y * stride - pad + ky, sx = xx * stride - pad + kx; if ((uint)sy < 9 && (uint)sx < 43) { sum += x.Data[(channel * 9 + sy) * 43 + sx]; count++; } }
            Near(output.Data[(channel * output.Shape[2] + y) * output.Shape[3] + xx], (float)(sum / (include ? kernel * kernel : count)));
        }
    }
});
Test("SIMD depthwise strided convolution vs scalar", () =>
{
    foreach (var (width, dilation, pad) in new[] { (43, 1, new[] { 1, 1, 1, 1 }), (64, 1, new[] { 1, 1, 1, 1 }), (43, 2, new[] { 2, 1, 0, 3 }) })
    {
        const int batch = 2, channels = 5, height = 17;
        var input = new Tensor(Enumerable.Range(0, batch * channels * height * width).Select(i => MathF.Sin(i)).ToArray(), batch, channels, height, width);
        var weight = new Tensor(Enumerable.Range(0, channels * 9).Select(i => MathF.Cos(i) * .1f).ToArray(), channels, 1, 3, 3);
        var bias = new Tensor(Enumerable.Range(0, channels).Select(i => i * .01f).ToArray(), channels);
        foreach (bool silu in new[] { false, true })
        {
            var output = new TensorOps(cpu).Conv(input, weight, bias, [2, 2], pad, [dilation, dilation], channels, silu);
            for (int b = 0; b < batch; b++) for (int c = 0; c < channels; c++) for (int y = 0; y < output.Shape[2]; y++) for (int xx = 0; xx < output.Shape[3]; xx++)
            {
                double sum = bias.Data[c];
                for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) { int sy = y * 2 - pad[0] + ky * dilation, sx = xx * 2 - pad[1] + kx * dilation; if ((uint)sy < height && (uint)sx < width) sum += (double)input.Data[((b * channels + c) * height + sy) * width + sx] * weight.Data[(c * 3 + ky) * 3 + kx]; }
                float expected = (float)sum; if (silu) expected /= 1 + MathF.Exp(-expected);
                Near(output.Data[((b * channels + c) * output.Shape[2] + y) * output.Shape[3] + xx], expected);
            }
        }
    }
});
Test("GEMM transposed convolution vs independent scalar scatter", () =>
{
    var x = new Tensor(Enumerable.Range(0, 2 * 5 * 3 * 7).Select(i => MathF.Sin(i)).ToArray(), 2, 5, 3, 7);
    var w = new Tensor(Enumerable.Range(0, 5 * 4 * 2 * 2).Select(i => MathF.Cos(i)).ToArray(), 5, 4, 2, 2); var bias = new Tensor([.1f, .2f, .3f, .4f], 4);
    var actual = TensorOps.ConvTranspose(x, w, bias, [2, 2], [0, 0, 0, 0], [1, 1], [0, 0], 1); var expected = new Tensor(actual.Shape);
    for (int b = 0; b < 2; b++) for (int c = 0; c < 4; c++) Array.Fill(expected.Data, bias.Data[c], (b * 4 + c) * 6 * 14, 6 * 14);
    for (int b = 0; b < 2; b++) for (int ci = 0; ci < 5; ci++) for (int co = 0; co < 4; co++) for (int y = 0; y < 3; y++) for (int xx = 0; xx < 7; xx++) for (int ky = 0; ky < 2; ky++) for (int kx = 0; kx < 2; kx++)
        expected.Data[((b * 4 + co) * 6 + y * 2 + ky) * 14 + xx * 2 + kx] += x.Data[((b * 5 + ci) * 3 + y) * 7 + xx] * w.Data[((ci * 4 + co) * 2 + ky) * 2 + kx];
    for (int i = 0; i < actual.Length; i++) Near(actual.Data[i], expected.Data[i]);
});
Test("Winograd convolution vs double scalar, odd dimensions", () =>
{
    var x = new Tensor(Enumerable.Range(0, 2 * 17 * 9 * 11).Select(i => MathF.Sin(i)).ToArray(), 2, 17, 9, 11);
    var w = new Tensor(Enumerable.Range(0, 19 * 17 * 3 * 3).Select(i => MathF.Cos(i) * .1f).ToArray(), 19, 17, 3, 3);
    var actual = new TensorOps(cpu).Conv(x, w, null, [1, 1], [1, 1, 1, 1]);
    for (int b = 0; b < 2; b++) for (int co = 0; co < 19; co++) for (int y = 0; y < 9; y++) for (int xx = 0; xx < 11; xx++)
    {
        double sum = 0;
        for (int ci = 0; ci < 17; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) { int sy = y - 1 + ky, sx = xx - 1 + kx; if ((uint)sy < 9 && (uint)sx < 11) sum += (double)x.Data[((b * 17 + ci) * 9 + sy) * 11 + sx] * w.Data[((co * 17 + ci) * 3 + ky) * 3 + kx]; }
        Near(actual.Data[((b * 19 + co) * 9 + y) * 11 + xx], (float)sum);
    }
});
Test("Letterbox, BGR, padding and mapping", () =>
{
    var (input, t) = ImagePreprocessor.Letterbox([0, 0, 255, 0, 255, 0], 2, 1, 4, 4, true);
    Near(input.Data[4], 1); Near(input.Data[0], 114f / 255); if (t.PadY != 1) throw new Exception("Padding mismatch.");
});
Test("Class-aware NMS and v10 no-NMS", () =>
{
    var output = new Tensor([10, 10, 10, 10, .9f, .1f, 10, 10, 10, 10, .8f, .1f, 10, 10, 10, 10, .1f, .95f], 1, 3, 6);
    var tx = new Letterbox(20, 20, 20, 20, 1, 1, 0, 0); var options = new DetectorOptions { OutputLayout = OutputLayout.ChannelsLast };
    if (DetectionDecoder.Decode(output, tx, options).Count != 2) throw new Exception("NMS mismatch.");
    var end = new Tensor([1, 1, 10, 10, .9f, 0, 1, 1, 10, 10, .8f, 0], 1, 2, 6);
    if (DetectionDecoder.Decode(end, tx, options with { OutputLayout = OutputLayout.EndToEnd }).Count != 2) throw new Exception("v10 NMS must be skipped.");
});
Test("Mask coefficients excluded from classification, cropped mask", () =>
{
    var head = new Tensor([2, 10, 2, 10, 4, 2, 4, 2, .9f, .1f, 20, 100, 0, 0], 1, 7, 2); var proto = new Tensor(1, 2, 4, 4); Array.Fill(proto.Data, 1);
    var instances = VisionDecoder.Decode([head, proto], new(12, 12, 12, 12, 1, 1, 0, 0), new() { Task = YoloTask.Segmentation, ClassCount = 1 });
    if (instances.Count != 1 || instances[0].Box.ClassId != 0 || instances[0].Mask is not { Width: 4, Height: 4 } mask) throw new Exception("Mask decoding mismatch.");
    Near(mask.Probabilities[0], 1);
});
Test("Pose coordinates and class score", () =>
{
    var head = new Tensor([20, 40, 20, 40, 10, 10, 10, 10, .9f, .1f, 20, 100, 30, 100, .8f, .7f], 1, 8, 2);
    var instances = VisionDecoder.Decode([head], new(50, 50, 100, 100, 2, 2, 0, 0), new() { Task = YoloTask.Pose, ClassCount = 1, KeypointCount = 1, KeypointDimensions = 3 });
    if (instances.Count != 1) throw new Exception("Pose classification mismatch."); Near(instances[0].Keypoints[0].X, 10); Near(instances[0].Keypoints[0].Y, 15); Near(instances[0].Keypoints[0].Confidence, .8f);
});
// 返回命令行选定的测试设备；默认包含 CPU，显式开关增加 Vulkan 或 Metal。
IEnumerable<ComputeDevice> FixtureDevices()
{
    yield return ComputeDevice.Cpu;
    if (args.Contains("--vulkan")) yield return ComputeDevice.Vulkan;
}
Test("ONNX slices: dynamic data, aliases, negative bounds and reverse", () =>
{
    foreach (var (starts, ends, axes, steps, outputShape) in new[]
    {
        (new[] { 1 }, new[] { 3 }, new[] { 1 }, new[] { 1 }, new[] { 1, 2, 4, 5 }),
        (new[] { -4 }, new[] { -1 }, new[] { -1 }, new[] { 1 }, new[] { 1, 3, 4, 3 }),
        (new[] { 1, 1, 0 }, new[] { 3, 4, 5 }, new[] { 1, 2, 3 }, new[] { 1, 1, 1 }, new[] { 1, 2, 3, 5 }),
        (new[] { -1 }, new[] { -100 }, new[] { -1 }, new[] { -1 }, new[] { 1, 3, 4, 5 })
    })
    {
        string path = OnnxFixtures.Write("slice", [1, 3, 4, 5], [new("Slice", ["images", "starts", "ends", "axes", "steps"], ["output"])],
            [new("starts", starts.Select(v => (float)v).ToArray(), [starts.Length], 7), new("ends", ends.Select(v => (float)v).ToArray(), [ends.Length], 7), new("axes", axes.Select(v => (float)v).ToArray(), [axes.Length], 7), new("steps", steps.Select(v => (float)v).ToArray(), [steps.Length], 7)], ("output", outputShape, 1));
        foreach (var device in FixtureDevices())
        {
            using var detector = new YoloDetector(path, new() { Device = device });
            Tensor? saved = null; float[]? snapshot = null;
            for (int iteration = 0; iteration < 2; iteration++)
            {
                var input = new Tensor(Enumerable.Range(0, 60).Select(v => v + iteration * .25f).ToArray(), 1, 3, 4, 5);
                Tensor result = detector.PredictRaw(input)[0];
                if (saved is not null && !saved.Data.SequenceEqual(snapshot!)) throw new Exception("Slice overwrote a previous output.");
                if (!result.Shape.SequenceEqual(outputShape)) throw new Exception("Slice shape mismatch.");
                int[] origin = [0, 0, 0, 0], increment = [1, 1, 1, 1];
                for (int i = 0; i < axes.Length; i++) { int axis = axes[i] < 0 ? axes[i] + 4 : axes[i]; origin[axis] = starts[i] < 0 ? input.Shape[axis] + starts[i] : starts[i]; increment[axis] = steps[i]; }
                for (int i = 0; i < result.Length; i++)
                {
                    int index = i, source = 0, stride = 1;
                    for (int axis = 3; axis >= 0; axis--) { source += (origin[axis] + index % outputShape[axis] * increment[axis]) * stride; index /= outputShape[axis]; stride *= input.Shape[axis]; }
                    Near(result.Data[i], input.Data[source]);
                }
                saved = result; snapshot = (float[])result.Data.Clone();
            }
        }
    }
});
Test("ONNX shape caches respect data-dependent reshape and public output edits", () =>
{
    string path = OnnxFixtures.Write("data-shape", [1, 1, 1, 4],
        [new("Flatten", ["images"], ["flat"]), new("Slice", ["flat", "start", "end", "axis"], ["params2d"]), new("Reshape", ["params2d", "two"], ["params1d"]), new("Cast", ["params1d"], ["target"], [OnnxFixtures.Integer("to", 7)]), new("Reshape", ["images", "target"], ["reshaped"]), new("Shape", ["reshaped"], ["output"])],
        [new("start", [0], [1], 7), new("end", [2], [1], 7), new("axis", [1], [1], 7), new("two", [2], [1], 7)], ("output", [2], 7));
    foreach (var device in FixtureDevices())
    {
        using var detector = new YoloDetector(path, new() { Device = device });
        foreach (var dimensions in new[] { new[] { 2, 2 }, new[] { 1, 4 }, new[] { 2, 2 } })
        {
            var input = new Tensor([(float)dimensions[0], dimensions[1], 7, 9], 1, 1, 1, 4);
            var output = detector.PredictRaw(input)[0]; Near(output.Data[0], dimensions[0]); Near(output.Data[1], dimensions[1]); output.Data[0] = -123;
        }
    }
    path = OnnxFixtures.Write("stable-shape", [-1, 3, -1, -1], [new("Shape", ["images"], ["output"])], [], ("output", [4], 7));
    foreach (var device in FixtureDevices())
    {
        using var detector = new YoloDetector(path, new() { Device = device });
        for (int width = 2; width <= 12; width++)
        {
            var input = new Tensor(2, 3, 4, width); var output = detector.PredictRaw(input)[0];
            for (int i = 0; i < 4; i++) Near(output.Data[i], input.Shape[i]); output.Data[0] = -9;
            var second = detector.PredictRaw(input)[0]; Near(second.Data[0], 2);
        }
    }
});
Test("ONNX Conv BN SiLU fusion and multiple consumers vs scalar", () =>
{
    float[] w = Enumerable.Range(0, 6 * 3 * 9).Select(i => MathF.Cos(i) * .1f).ToArray();
    float[] bias = [.1f, -.2f, .3f, .4f, -.5f, .6f], gamma = [1, 2, .5f, 1.5f, .7f, 2], beta = [.1f, .2f, -.1f, .3f, .4f, .5f], mean = [.2f, -.1f, .1f, 0, .4f, -.2f], variance = [.5f, 2, .7f, 1, 1.2f, .9f];
    foreach (bool share in new[] { false, true })
    {
        var outputInfos = share ? new[] { ("output", new[] { 1, 6, 4, 5 }, 1), ("conv", new[] { 1, 6, 4, 5 }, 1) } : new[] { ("output", new[] { 1, 6, 4, 5 }, 1) };
        string path = OnnxFixtures.Write("batch-norm", [1, 3, 4, 5],
            [new("Conv", ["images", "w", "bias"], ["conv"], [OnnxFixtures.Integers("pads", 1, 1, 1, 1)]), new("BatchNormalization", ["conv", "gamma", "beta", "mean", "variance"], ["bn"], [OnnxFixtures.Real("epsilon", .001f)]), new("Sigmoid", ["bn"], ["sigmoid"]), new("Mul", ["bn", "sigmoid"], ["output"])],
            [new("w", w, [6, 3, 3, 3]), new("bias", bias, [6]), new("gamma", gamma, [6]), new("beta", beta, [6]), new("mean", mean, [6]), new("variance", variance, [6])], outputInfos);
        foreach (var device in FixtureDevices())
        {
            using var detector = new YoloDetector(path, new() { Device = device });
            for (int iteration = 0; iteration < 2; iteration++)
            {
                var input = new Tensor(Enumerable.Range(0, 60).Select(i => MathF.Sin(i + iteration)).ToArray(), 1, 3, 4, 5); var outputs = detector.PredictRaw(input);
                for (int co = 0; co < 6; co++) for (int y = 0; y < 4; y++) for (int x = 0; x < 5; x++)
                {
                    double sum = bias[co];
                    for (int ci = 0; ci < 3; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                    { int sy = y - 1 + ky, sx = x - 1 + kx; if ((uint)sy < 4 && (uint)sx < 5) sum += (double)input.Data[(ci * 4 + sy) * 5 + sx] * w[((co * 3 + ci) * 3 + ky) * 3 + kx]; }
                    float bn = ((float)sum - mean[co]) * gamma[co] / MathF.Sqrt(variance[co] + .001f) + beta[co]; int index = (co * 4 + y) * 5 + x;
                    Near(outputs[0].Data[index], bn / (1 + MathF.Exp(-bn))); if (share) Near(outputs[1].Data[index], (float)sum);
                }
            }
        }
    }
});
Test("ONNX Sigmoid Concat fusion: arbitrary axes, tails and shared consumers", () =>
{
    foreach (var shape in new[] { new[] { 2, 3, 2, 5 }, new[] { 1, 3, 128, 257 }, new[] { 1, 3, 0, 5 } })
    foreach (int axis in new[] { 0, 1, 2 }) foreach (bool shared in new[] { false, true })
    {
        int count = shape.Aggregate(1, (a, b) => a * b); float[] other = Enumerable.Range(0, count).Select(i => MathF.Cos(i)).ToArray();
        var resultShape = (int[])shape.Clone(); resultShape[axis] *= 2;
        var outputInfos = shared ? new[] { ("output", resultShape, 1), ("sigmoid", shape, 1) } : new[] { ("output", resultShape, 1) };
        string path = OnnxFixtures.Write("sigmoid-concat", shape, [new("Sigmoid", ["images"], ["sigmoid"]), new("Concat", ["other", "sigmoid"], ["output"], [OnnxFixtures.Integer("axis", axis)])], [new("other", other, shape)], outputInfos);
        foreach (var device in FixtureDevices())
        {
            using var detector = new YoloDetector(path, new() { Device = device });
            var input = new Tensor(Enumerable.Range(0, count).Select(i => MathF.Sin(i)).ToArray(), shape); var outputs = detector.PredictRaw(input);
            int inner = shape.Skip(axis + 1).Aggregate(1, (a, b) => a * b), outer = shape.Take(axis).Aggregate(1, (a, b) => a * b), block = shape[axis] * inner;
            for (int row = 0; row < outer; row++) for (int i = 0; i < block; i++)
            {
                Near(outputs[0].Data[row * 2 * block + i], other[row * block + i]);
                Near(outputs[0].Data[row * 2 * block + block + i], 1 / (1 + MathF.Exp(-input.Data[row * block + i])));
            }
            if (shared) for (int i = 0; i < count; i++) Near(outputs[1].Data[i], 1 / (1 + MathF.Exp(-input.Data[i])));
        }
    }
});
Test("ONNX TopK heap preserves ties, NaNs, zero signs and arbitrary axes", () =>
{
    float[] pattern = [float.NaN, 0, BitConverter.Int32BitsToSingle(int.MinValue), float.PositiveInfinity, float.NegativeInfinity, 1, 1, 3, 3, float.NaN];
    var input = new Tensor(Enumerable.Range(0, 30).Select(i => pattern[i % pattern.Length]).ToArray(), 2, 3, 5);
    foreach (int axisValue in new[] { 0, 1, -1 }) foreach (bool largest in new[] { false, true }) foreach (int k in new[] { 0, 2 })
    {
        int axis = axisValue < 0 ? 3 + axisValue : axisValue; var shape = (int[])input.Shape.Clone(); shape[axis] = k;
        string path = OnnxFixtures.Write("top-k", input.Shape, [new("TopK", ["images", "k"], ["values", "indices"], [OnnxFixtures.Integer("axis", axisValue), OnnxFixtures.Integer("largest", largest ? 1 : 0)])], [new("k", [k], [1], 7)], ("values", shape, 1), ("indices", shape, 7));
        foreach (var device in FixtureDevices())
        {
            using var detector = new YoloDetector(path, new() { Device = device }); var result = detector.PredictRaw(input);
            int inner = input.Shape.Skip(axis + 1).Aggregate(1, (a, b) => a * b), outer = input.Shape.Take(axis).Aggregate(1, (a, b) => a * b), size = input.Shape[axis];
            for (int row = 0; row < outer; row++) for (int column = 0; column < inner; column++)
            {
                var selected = Enumerable.Range(0, size).OrderBy(i => largest ? -input.Data[(row * size + i) * inner + column] : input.Data[(row * size + i) * inner + column]).ThenBy(i => i).Take(k).ToArray();
                for (int j = 0; j < k; j++)
                {
                    int output = (row * k + j) * inner + column; float value = input.Data[(row * size + selected[j]) * inner + column]; Near(result[1].Data[output], selected[j]);
                    if (float.IsNaN(value)) { if (!float.IsNaN(result[0].Data[output])) throw new Exception("TopK NaN was lost."); }
                    else if (BitConverter.SingleToInt32Bits(result[0].Data[output]) != BitConverter.SingleToInt32Bits(value)) throw new Exception("TopK value/tie order changed.");
                }
            }
        }
    }
});
Test("ONNX ReduceMax contiguous rows matches scalar including NaN and zero", () =>
{
    const int width = 19;
    var input = new Tensor(Enumerable.Range(0, 6 * width).Select(i => MathF.Sin(i)).ToArray(), 2, 3, width);
    input.Data[5] = float.NaN; Array.Fill(input.Data, BitConverter.Int32BitsToSingle(int.MinValue), width, width); input.Data[2 * width] = 0;
    for (int i = 2 * width + 1; i < 3 * width; i++) input.Data[i] = BitConverter.Int32BitsToSingle(int.MinValue);
    string path = OnnxFixtures.Write("reduce-max", input.Shape, [new("ReduceMax", ["images"], ["output"], [OnnxFixtures.Integers("axes", -1), OnnxFixtures.Integer("keepdims", 0)])], [], ("output", [2, 3], 1));
    foreach (var device in FixtureDevices())
    {
        using var detector = new YoloDetector(path, new() { Device = device }); var result = detector.PredictRaw(input)[0];
        for (int row = 0; row < 6; row++)
        {
            float expected = float.NegativeInfinity; for (int j = 0; j < width; j++) expected = MathF.Max(expected, input.Data[row * width + j]);
            if (float.IsNaN(expected)) { if (!float.IsNaN(result.Data[row])) throw new Exception("ReduceMax NaN was lost."); }
            else if (BitConverter.SingleToInt32Bits(result.Data[row]) != BitConverter.SingleToInt32Bits(expected)) throw new Exception("ReduceMax result changed.");
        }
    }
});
Test("Public pooled output ownership and duplicate aliases", () =>
{
    string path = OnnxFixtures.Write("output-alias", [1, 3, 4, 5], [new("Add", ["images", "one"], ["output"]), new("Identity", ["output"], ["alias"])], [new("one", [1], [])], ("output", [1, 3, 4, 5], 1), ("alias", [1, 3, 4, 5], 1));
    using var detector = new YoloDetector(path);
    var input = new Tensor(Enumerable.Range(0, 60).Select(i => (float)i).ToArray(), 1, 3, 4, 5);
    var outputs = detector.PredictRaw(input); outputs[0].Data[0] = -9; Near(outputs[1].Data[0], 1);
    float[][] saved = outputs.Select(t => (float[])t.Data.Clone()).ToArray(); input.Data[0] += 10; detector.PredictRaw(input);
    for (int i = 0; i < outputs.Length; i++) if (!outputs[i].Data.SequenceEqual(saved[i])) throw new Exception("Transferred output was recycled.");
});
Test("Nested independent sessions restore CPU pools and GPU scopes", () =>
{
    string innerPath = OnnxFixtures.Write("nested-inner", [1, 3, 4, 5], [new("Mul", ["images", "two"], ["output"])], [new("two", [2], [])], ("output", [1, 3, 4, 5], 1));
    string outerPath = OnnxFixtures.Write("nested-outer", [1, 3, 4, 5], [new("Add", ["images", "one"], ["sum"]), new("Mul", ["sum", "two"], ["output"])], [new("one", [1], []), new("two", [2], [])], ("output", [1, 3, 4, 5], 1));
    foreach (var outerDevice in FixtureDevices()) foreach (var innerDevice in FixtureDevices())
    {
        using var inner = new YoloDetector(innerPath, new() { Device = innerDevice });
        var input = new Tensor(Enumerable.Range(0, 60).Select(i => i * .1f).ToArray(), 1, 3, 4, 5);
        var saved = new List<Tensor>();
        using var outer = new YoloDetector(outerPath, new() { Device = outerDevice, LayerCompleted = _ => saved.Add(inner.PredictRaw(input)[0]) });
        for (int iteration = 0; iteration < 2; iteration++)
        {
            var output = outer.PredictRaw(input)[0];
            for (int i = 0; i < output.Length; i++) Near(output.Data[i], (input.Data[i] + 1) * 2);
            foreach (var tensor in saved) for (int i = 0; i < tensor.Length; i++) Near(tensor.Data[i], input.Data[i] * 2);
        }
    }
});
Test("Frame convolution transforms: batches, tile edges and channel tails", () =>
{
    foreach (var (ic, oc, h, width) in new[] { (17, 19, 13, 15), (16, 16, 40, 44), (17, 19, 43, 45) })
    {
        const int batch = 2;
        float[] weights = Enumerable.Range(0, oc * ic * 9).Select(i => MathF.Cos(i) * .02f).ToArray();
        float[] bias = Enumerable.Range(0, oc).Select(i => i * .01f).ToArray();
        string path = OnnxFixtures.Write("frame-conv", [batch, ic, h, width], [new("Conv", ["images", "weight", "bias"], ["output"], [OnnxFixtures.Integers("pads", 1, 1, 1, 1)])], [new("weight", weights, [oc, ic, 3, 3]), new("bias", bias, [oc])], ("output", [batch, oc, h, width], 1));
        var input = new Tensor(Enumerable.Range(0, batch * ic * h * width).Select(i => MathF.Sin(i)).ToArray(), batch, ic, h, width);
        float[] expected = new float[batch * oc * h * width];
        for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int y = 0; y < h; y++) for (int x = 0; x < width; x++)
        {
            double sum = bias[co];
            for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
            { int sy = y - 1 + ky, sx = x - 1 + kx; if ((uint)sy < h && (uint)sx < width) sum += (double)input.Data[((b * ic + ci) * h + sy) * width + sx] * weights[((co * ic + ci) * 3 + ky) * 3 + kx]; }
            expected[((b * oc + co) * h + y) * width + x] = (float)sum;
        }
        foreach (var device in FixtureDevices())
        {
            using var detector = new YoloDetector(path, new() { Device = device });
            for (int iteration = 0; iteration < 2; iteration++)
            {
                var output = detector.PredictRaw(input)[0];
                for (int i = 0; i < expected.Length; i++) Near(output.Data[i], expected[i]);
            }
        }
    }
});
Test("Winograd shared inputs respect offsets, reshaped aliases and frame lifetime", () =>
{
    foreach (var (h, width) in new[] { (13, 16), (43, 48) })
    {
        const int channels = 16;
        float[] a = Enumerable.Range(0, 16 * channels * 9).Select(i => MathF.Cos(i) * .02f).ToArray();
        float[] b = Enumerable.Range(0, 19 * channels * 9).Select(i => MathF.Sin(i) * .03f).ToArray();
        float[] ba = Enumerable.Range(0, 16).Select(i => i * .01f).ToArray(), bb = Enumerable.Range(0, 19).Select(i => i * -.02f).ToArray();
        string path = OnnxFixtures.Write("shared-frame-conv", [1, channels * 2, h, width],
            [new("Add", ["images", "zero"], ["shared"]),
             new("Slice", ["shared", "start0", "end16", "axis"], ["left"]),
             new("Identity", ["left"], ["alias"]),
             new("Conv", ["left", "wa", "ba"], ["first"], [OnnxFixtures.Integers("pads", 1, 1, 1, 1)]),
             new("Conv", ["alias", "wb", "bb"], ["second"], [OnnxFixtures.Integers("pads", 1, 1, 1, 1)]),
             new("Slice", ["shared", "start16", "end32", "axis"], ["right"]),
             new("Conv", ["right", "wa", "ba"], ["third"], [OnnxFixtures.Integers("pads", 1, 1, 1, 1)]),
             new("Reshape", ["left", "shape"], ["reshape"]),
             new("Conv", ["reshape", "wa", "ba"], ["fourth"], [OnnxFixtures.Integers("pads", 1, 1, 1, 1)])],
            [new("zero", [0], []), new("start0", [0], [1], 7), new("end16", [16], [1], 7), new("start16", [16], [1], 7), new("end32", [32], [1], 7), new("axis", [1], [1], 7),
             new("shape", [1, channels, width, h], [4], 7), new("wa", a, [16, channels, 3, 3]), new("wb", b, [19, channels, 3, 3]), new("ba", ba, [16]), new("bb", bb, [19])],
            ("first", [1, 16, h, width], 1), ("second", [1, 19, h, width], 1), ("third", [1, 16, h, width], 1), ("fourth", [1, 16, width, h], 1));
        foreach (var device in FixtureDevices())
        {
            // 逐层回调会强制执行中间 Flush；当前帧的缓存仍须保持有效。
            using var detector = new YoloDetector(path, new() { Device = device, LayerCompleted = _ => { } });
            var input = new Tensor(1, channels * 2, h, width);
            for (int iteration = 0; iteration < 2; iteration++)
            {
                for (int i = 0; i < input.Length; i++) input.Data[i] = MathF.Sin(i) + iteration * (i % 17) * .03f;
                var output = detector.PredictRaw(input);
                for (int branch = 0; branch < output.Length; branch++)
                {
                    float[] weights = branch == 1 ? b : a, bias = branch == 1 ? bb : ba;
                    int oh = branch == 3 ? width : h, ow = branch == 3 ? h : width, channelBase = branch == 2 ? channels : 0;
                    for (int co = 0; co < bias.Length; co++) for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++)
                    {
                        double sum = bias[co];
                        for (int ci = 0; ci < channels; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                        {
                            int sy = y - 1 + ky, sx = x - 1 + kx;
                            if ((uint)sy < oh && (uint)sx < ow) sum += (double)input.Data[(channelBase + ci) * h * width + sy * ow + sx] * weights[(co * channels + ci) * 9 + ky * 3 + kx];
                        }
                        Near(output[branch].Data[(co * oh + y) * ow + x], (float)sum);
                    }
                }
            }
        }
    }
});
Test("Frame tiled MatMul batched and broadcast operands vs double scalar", () =>
{
    const int batch = 2, m = 37, k = 23, n = 41;
    foreach (bool broadcast in new[] { false, true })
    {
        int bBatches = broadcast ? 1 : batch;
        float[] weights = Enumerable.Range(0, bBatches * k * n).Select(i => MathF.Cos(i) * .1f).ToArray();
        string path = OnnxFixtures.Write("frame-matmul", [batch, m, k], [new("MatMul", ["images", "w"], ["output"])], [new("w", weights, broadcast ? new[] { k, n } : new[] { batch, k, n })], ("output", [batch, m, n], 1));
        var input = new Tensor(Enumerable.Range(0, batch * m * k).Select(i => MathF.Sin(i)).ToArray(), batch, m, k);
        foreach (var device in FixtureDevices())
        {
            using var detector = new YoloDetector(path, new() { Device = device }); var output = detector.PredictRaw(input)[0];
            for (int b = 0; b < batch; b++) for (int row = 0; row < m; row++) for (int col = 0; col < n; col++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += (double)input.Data[(b * m + row) * k + p] * weights[((broadcast ? 0 : b) * k + p) * n + col];
                Near(output.Data[(b * m + row) * n + col], (float)sum);
            }
        }
    }
});
Test("ONNX DFL fusion and nonstandard-layout fallback vs scalar", () =>
{
    const int bins = 3;
    foreach (bool standard in new[] { true, false })
    {
        int rows = standard ? 4 : 5, positions = standard ? 15 : 12;
        string path = OnnxFixtures.Write("dfl", [1, 4 * bins, 15],
            [new("Reshape", ["images", "shape1"], ["reshape1"]), new("Transpose", ["reshape1"], ["transpose"], [OnnxFixtures.Integers("perm", 0, 2, 1, 3)]), new("Softmax", ["transpose"], ["softmax"], [OnnxFixtures.Integer("axis", 1)]), new("Conv", ["softmax", "w"], ["conv"]), new("Reshape", ["conv", "shape2"], ["output"])],
            [new("shape1", [1, rows, bins, positions], [4], 7), new("shape2", [1, 4, 15], [3], 7), new("w", [0, 1, 2], [1, bins, 1, 1])], ("output", [1, 4, 15], 1));
        foreach (var device in FixtureDevices())
        {
            using var detector = new YoloDetector(path, new() { Device = device });
            var input = new Tensor(Enumerable.Range(0, 180).Select(i => MathF.Sin(i)).ToArray(), 1, 12, 15); var output = detector.PredictRaw(input)[0];
            for (int row = 0; row < rows; row++) for (int position = 0; position < positions; position++)
            {
                double sum = 0, weighted = 0;
                for (int bin = 0; bin < bins; bin++) { double value = Math.Exp(input.Data[(row * bins + bin) * positions + position]); sum += value; weighted += value * bin; }
                Near(output.Data[row * positions + position], (float)(weighted / sum));
            }
        }
    }
});
// 在 gpu 上验证分组、批次、膨胀和 SiLU 卷积，逐元素对照独立 CPU 路径；不返回张量。
void ConvGpu(IComputeBackend gpu)
{
    var random = new Random(13);
    foreach (int groups in new[] { 1, 2 })
    {
        var x = new Tensor(Enumerable.Range(0, 2 * 4 * 9 * 11).Select(_ => (float)random.NextDouble()).ToArray(), 2, 4, 9, 11);
        var w = new Tensor(Enumerable.Range(0, 6 * (4 / groups) * 3 * 3).Select(_ => (float)random.NextDouble() - .5f).ToArray(), 6, 4 / groups, 3, 3); var bias = new Tensor([1, 2, 3, 4, 5, 6], 6);
        foreach (bool silu in new[] { false, true })
        {
            var reference = new TensorOps(cpu).Conv(x, w, bias, [2, 1], [1, 2, 1, 0], [1, 2], groups, silu);
            var actual = new TensorOps(gpu).Conv(x, w, bias, [2, 1], [1, 2, 1, 0], [1, 2], groups, silu);
            for (int i = 0; i < reference.Length; i++) Near(actual.Data[i], reference.Data[i]);
        }
    }
}
if (args.Contains("--vulkan")) { using var gpu = new VulkanBackend(); Console.WriteLine(gpu.Name); Test("Vulkan GEMM vs double reference", () => Gemm(gpu)); Test("Vulkan direct Conv, grouped/batched/dilated/SiLU", () => ConvGpu(gpu)); }
if (args.Contains("--metal")) { using var gpu = new MetalBackend(); Console.WriteLine(gpu.Name); Test("Metal GEMM vs double reference", () => Gemm(gpu)); Test("Metal direct Conv, grouped/batched/dilated/SiLU", () => ConvGpu(gpu)); }
Console.WriteLine($"{passed} tests passed.");
int modelIndex = Array.IndexOf(args, "--models");
if (modelIndex >= 0)
{
    string cache = args[modelIndex + 1];
    foreach (string name in new[] { "yolov8n.pt", "yolo11n.pt", "yolov8n-seg.pt", "yolo11n-pose.pt" })
    {
        using var baseline = new YoloDetector(Path.Combine(cache, name)); using var accelerated = new YoloDetector(Path.Combine(cache, name), new() { Device = ComputeDevice.Vulkan });
        Test($"Persistent weights and changing shape: {name}", () =>
        {
            var random = new Random(13);
            Tensor[]? saved = null; float[][]? savedData = null;
            foreach (int width in new[] { 128, 160, 128 })
            {
                var x = new Tensor(Enumerable.Range(0, 3 * 128 * width).Select(_ => (float)random.NextDouble()).ToArray(), 1, 3, 128, width);
                var expected = baseline.PredictRaw(x); var actual = accelerated.PredictRaw(x);
                if (saved is not null) for (int i = 0; i < saved.Length; i++) if (!saved[i].Data.SequenceEqual(savedData![i])) throw new Exception("Inference reused a public output buffer.");
                saved = [.. expected, .. actual]; savedData = saved.Select(t => (float[])t.Data.Clone()).ToArray();
                for (int i = 0; i < expected.Length; i++)
                { if (!expected[i].Shape.SequenceEqual(actual[i].Shape)) throw new Exception("Output shapes differ."); for (int j = 0; j < expected[i].Length; j++) Near(actual[i].Data[j], expected[i].Data[j], .003f); }
            }
            var batched = new Tensor(Enumerable.Range(0, 2 * 3 * 64 * 96).Select(_ => (float)random.NextDouble()).ToArray(), 2, 3, 64, 96);
            var batchReference = baseline.PredictRaw(batched); var batchActual = accelerated.PredictRaw(batched);
            for (int i = 0; i < batchReference.Length; i++) { if (!batchReference[i].Shape.SequenceEqual(batchActual[i].Shape)) throw new Exception("Batch shapes differ."); for (int j = 0; j < batchReference[i].Length; j++) Near(batchActual[i].Data[j], batchReference[i].Data[j], .003f); }
        });
    }
    Console.WriteLine($"{passed} total tests passed.");
}
