using SimdYolo;

/// <summary>提供ONNX 设备归约与索引收集的动态形状、负索引和越界约束的回归验证。</summary>
internal static class OnnxDeviceHeadTests
{
    /// <summary>向测试执行器注册ONNX 设备归约与索引收集的动态形状、负索引和越界约束的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <param name="devices">需要验证的计算后端集合。</param>
    internal static void Run(Action<string, Action> test, IEnumerable<ComputeDevice> devices)
    {
        foreach (var device in devices)
        {
            test($"ONNX device ReduceMax: dynamic rows, NaNs, infinities and signed zero ({device})", () => Enabled(() => Reduce(device)));
            test($"ONNX device GatherElements: axes, negative indices, tails and bounds ({device})", () => Enabled(() => Gather(device)));
        }
    }

    /// <summary>暂时启用专项优化开关，执行验证后恢复原有环境设置。</summary>
    /// <param name="action">在临时测试设置或设备帧作用域内执行的操作。</param>
    private static void Enabled(Action action)
    {
        string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_ONNX_HEAD");
        Environment.SetEnvironmentVariable("SIMDYOLO_GPU_ONNX_HEAD", "1");
        try { action(); }
        finally { Environment.SetEnvironmentVariable("SIMDYOLO_GPU_ONNX_HEAD", previous); }
    }

    /// <summary>比较设备归约与独立参考，覆盖 NaN、无穷和符号零。</summary>
    /// <param name="device">本次验证使用的设备类型。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Reduce(ComputeDevice device)
    {
        foreach (bool keep in new[] { false, true })
        {
            string path = OnnxFixtures.Write($"device-max-{device}-{keep}", [-1, -1, -1],
                [new("ReduceMax", ["images"], ["output"], [OnnxFixtures.Integers("axes", -1), OnnxFixtures.Integer("keepdims", keep ? 1 : 0)])], [],
                ("output", keep ? [-1, -1, 1] : [-1, -1], 1));
            using var detector = new YoloDetector(path, new() { Device = device });
            foreach (int size in new[] { 80, 257, 0, 80 })
            {
                var input = new Tensor(2, 3, size);
                for (int row = 0; row < 6; row++) for (int column = 0; column < size; column++)
                {
                    input.Data[row * size + column] = row switch
                    {
                        0 => BitConverter.Int32BitsToSingle(int.MinValue),
                        1 => column % 2 == 0 ? 0 : BitConverter.Int32BitsToSingle(int.MinValue),
                        2 => column == size - 1 ? float.PositiveInfinity : -column,
                        3 => column == 17 ? BitConverter.Int32BitsToSingle(unchecked((int)0x7fc12345)) : -column,
                        4 => float.NegativeInfinity,
                        _ => MathF.Sin(column * .17f) - 3
                    };
                }
                Tensor actual = detector.PredictRaw(input)[0];
                int[] expectedShape = keep ? [2, 3, 1] : [2, 3];
                if (!actual.Shape.SequenceEqual(expectedShape)) throw new Exception("Device ReduceMax shape mismatch.");
                for (int row = 0; row < 6; row++)
                {
                    float expected = float.NegativeInfinity;
                    for (int column = 0; column < size; column++) expected = MathF.Max(expected, input.Data[row * size + column]);
                    Equal(actual.Data[row], expected, "ReduceMax");
                }
            }
        }
    }

    /// <summary>验证设备索引收集的轴、负索引、尾部和越界拒绝。</summary>
    /// <param name="device">本次验证使用的设备类型。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Gather(ComputeDevice device)
    {
        var variants = new[]
        {
            (Source: new[] { 32, 19 }, Indices: new[] { 5, 19 }, Axis: 0),
            (Source: new[] { 2, 32, 9 }, Indices: new[] { 2, 7, 9 }, Axis: 1),
            (Source: new[] { 2, 3, 7, 17 }, Indices: new[] { 2, 3, 7, 11 }, Axis: -1),
            (Source: new[] { 2, 2, 3, 4, 7 }, Indices: new[] { 2, 2, 2, 4, 7 }, Axis: 2)
        };
        for (int variant = 0; variant < variants.Length; variant++)
        {
            var (shape, indexShape, declaredAxis) = variants[variant]; int axis = declaredAxis < 0 ? declaredAxis + shape.Length : declaredAxis;
            int size = shape[axis]; var input = new Tensor(shape);
            for (int i = 0; i < input.Length; i++) input.Data[i] = i * .03125f - 2;
            input.Data[0] = float.NaN; input.Data[1] = BitConverter.Int32BitsToSingle(int.MinValue);
            float[] indices = Enumerable.Range(0, Tensor.Count(indexShape)).Select(i => (i % 6) switch
            {
                0 => (float)-size, 1 => -1, 2 => 0, 3 => size - 1, 4 => -.75f, _ => size - 1 + .75f
            }).ToArray();
            string path = WriteGather($"device-gather-{device}-{variant}", shape, indexShape, declaredAxis, indices);
            using var detector = new YoloDetector(path, new() { Device = device });
            Tensor output = detector.PredictRaw(input)[0];
            if (!output.Shape.SequenceEqual(indexShape)) throw new Exception("Device GatherElements shape mismatch.");
            int[] strides = new int[shape.Length]; int stride = 1;
            for (int d = shape.Length - 1; d >= 0; d--) { strides[d] = stride; stride *= shape[d]; }
            for (int i = 0; i < indices.Length; i++)
            {
                int source = 0, remainder = i;
                for (int d = shape.Length - 1; d >= 0; d--)
                {
                    int coordinate = remainder % indexShape[d]; remainder /= indexShape[d];
                    if (d == axis) { coordinate = checked((int)indices[i]); if (coordinate < 0) coordinate += shape[d]; }
                    source += coordinate * strides[d];
                }
                Equal(output.Data[i], input.Data[source], "GatherElements");
            }
        }
        foreach (float invalid in new[] { -33f, 32f, float.NaN, float.PositiveInfinity })
        {
            string path = WriteGather($"device-gather-bounds-{device}-{BitConverter.SingleToInt32Bits(invalid)}", [32, 19], [1, 19], 0, Enumerable.Repeat(invalid, 19).ToArray());
            using var detector = new YoloDetector(path, new() { Device = device });
            try { detector.PredictRaw(new Tensor(32, 19)); throw new Exception("GatherElements accepted an invalid index."); }
            catch (InvalidDataException exception) when (exception.InnerException is ArgumentException or OverflowException) { }
        }
        foreach (bool empty in new[] { false, true })
        {
            int[] indexShape = empty ? [0, 20] : [1, 20];
            string path = WriteGather($"device-gather-dimensions-{device}-{empty}", [32, 19], indexShape, 0, new float[Tensor.Count(indexShape)]);
            using var detector = new YoloDetector(path, new() { Device = device });
            if (empty)
            {
                Tensor output = detector.PredictRaw(new Tensor(32, 19))[0];
                if (!output.Shape.SequenceEqual(indexShape) || output.Length != 0) throw new Exception("Empty GatherElements changed its shape.");
            }
            else
            {
                try { detector.PredictRaw(new Tensor(32, 19)); throw new Exception("GatherElements accepted an oversized non-axis dimension."); }
                catch (InvalidDataException exception) when (exception.InnerException is ArgumentException) { }
            }
        }
    }

    /// <summary>写入包含指定输入形状、索引形状和轴属性的收集测试图。</summary>
    /// <param name="name">测试文件、张量或 ONNX 属性的名称。</param>
    /// <param name="inputShape">测试图输入张量的逻辑维度。</param>
    /// <param name="indexShape">索引张量的逻辑维度。</param>
    /// <param name="axis">执行收集操作的维度索引。</param>
    /// <param name="indices">包含负索引或边界值的测试索引数据。</param>
    /// <returns>写入的 ONNX 测试文件路径。</returns>
    private static string WriteGather(string name, int[] inputShape, int[] indexShape, int axis, float[] indices)
        => OnnxFixtures.Write(name, inputShape,
            [new("GatherElements", ["images", "indices"], ["output"], [OnnxFixtures.Integer("axis", axis)])],
            [new("indices", indices, indexShape)], ("output", indexShape, 1));

    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <param name="operation">失败消息中标识的被测操作。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Equal(float actual, float expected, string operation)
    {
        if (float.IsNaN(expected) ? !float.IsNaN(actual) : BitConverter.SingleToInt32Bits(actual) != BitConverter.SingleToInt32Bits(expected))
            throw new Exception($"{operation} differs from its scalar oracle: {actual} != {expected}.");
    }
}
