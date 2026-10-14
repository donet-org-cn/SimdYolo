using System.Reflection;
using SimdYolo;

/// <summary>提供GPU TopK 的 NaN、符号零、同分排序和两级检测头选择的回归验证。</summary>
internal static class GpuTopKTests
{
    /// <summary>向测试执行器注册GPU TopK 的 NaN、符号零、同分排序和两级检测头选择的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("GPU last-axis TopK preserves NaNs, signed zero, ties, offsets, tails and batches", () => WithGpu(backend =>
        {
            MethodInfo topK = typeof(VulkanBackend).GetMethod("TryTopKLastAxis", BindingFlags.Instance | BindingFlags.NonPublic)!;
            foreach (var (rows, size, count) in new[] { (2, 257, 33), (1, 513, 301), (1, 19, 19), (1, 263, 0) })
                foreach (bool largest in new[] { false, true })
                {
                    float[] source = Enumerable.Range(0, rows * size).Select(i => (i % 19 - 9) * .03f).ToArray();
                    SpecialValues(source);
                    WithFrame(backend, () =>
                    {
                        Tensor input = DeviceTensor(backend, source, [rows, size], prefix: 3);
                        var output = (Tensor[])topK.Invoke(backend, [input, count, largest])!;
                        for (int row = 0; row < rows; row++)
                        {
                            int[] expected = Enumerable.Range(0, size)
                                .OrderBy(index => largest ? -source[row * size + index] : source[row * size + index]).ThenBy(index => index).Take(count).ToArray();
                            for (int i = 0; i < expected.Length; i++)
                            {
                                int destination = row * count + i;
                                Exact(output[0].Data[destination], source[row * size + expected[i]]);
                                Exact(output[1].Data[destination], expected[i]);
                            }
                        }
                    });
                }
        }));
        test("GPU end-to-end head TopK matches existing CPU contract across both selection stages", () => WithGpu(backend =>
        {
            MethodInfo cpuTopK = typeof(Tensor).Assembly.GetType("SimdYolo.PtModel")!.GetMethod("EndToEndTopK", BindingFlags.Static | BindingFlags.NonPublic)!;
            MethodInfo gpuTopK = typeof(VulkanBackend).GetMethod("TryEndToEndTopK", BindingFlags.Instance | BindingFlags.NonPublic)!;
            foreach (var (batch, anchors, classes, maximum) in new[] { (2, 263, 5, 17), (1, 521, 7, 77), (2, 37, 3, 300), (1, 263, 1, 263) })
            {
                var cpuInput = new Tensor(batch, classes + 4, anchors);
                for (int b = 0; b < batch; b++)
                {
                    int offset = b * (classes + 4) * anchors;
                    for (int c = 0; c < 4; c++) for (int a = 0; a < anchors; a++) cpuInput.Data[offset + c * anchors + a] = b * 10000 + c * 1000 + a;
                    for (int c = 0; c < classes; c++)
                    {
                        float[] values = Enumerable.Range(0, anchors).Select(a => (a % 17 + c % 3) * .03f).ToArray();
                        SpecialValues(values);
                        values.CopyTo(cpuInput.Data, offset + (c + 4) * anchors);
                    }
                }
                Tensor expected = (Tensor)cpuTopK.Invoke(null, [cpuInput, maximum, classes])!;
                WithFrame(backend, () =>
                {
                    Tensor input = DeviceTensor(backend, cpuInput.Data, cpuInput.Shape, prefix: 3);
                    Tensor actual = (Tensor)gpuTopK.Invoke(backend, [input, maximum, classes])!;
                    if (!actual.Shape.SequenceEqual(expected.Shape)) throw new Exception("GPU head TopK shape mismatch.");
                    for (int i = 0; i < actual.Length; i++) Exact(actual.Data[i], expected.Data[i]);
                });
            }
        }));
        test("GPU ONNX end-to-end TopK keeps NaNs first and selection-rank ties", () => WithGpu(backend =>
        {
            MethodInfo gpuTopK = typeof(VulkanBackend).GetMethod("TryEndToEndTopKOnnx", BindingFlags.Instance | BindingFlags.NonPublic)!;
            foreach (var (batch, anchors, classes, maximum) in new[] { (2, 263, 5, 17), (1, 513, 7, 77), (2, 37, 3, 37), (1, 263, 1, 263) })
            {
                var cpuInput = new Tensor(batch, classes + 4, anchors);
                int k = Math.Min(maximum, anchors);
                for (int iteration = 0; iteration < 2; iteration++)
                {
                    var expected = new Tensor(batch, k, 6);
                    for (int b = 0; b < batch; b++)
                    {
                        int offset = b * (classes + 4) * anchors;
                        for (int c = 0; c < 4; c++) for (int a = 0; a < anchors; a++) cpuInput.Data[offset + c * anchors + a] = b * 10000 + c * 1000 + a + iteration * .25f;
                        for (int c = 0; c < classes; c++)
                        {
                            float[] values = Enumerable.Range(0, anchors).Select(a => (a % (17 + iteration) + c % 3) * .03f).ToArray();
                            SpecialValues(values);
                            values.CopyTo(cpuInput.Data, offset + (c + 4) * anchors);
                        }
                        float[] maxima = Enumerable.Range(0, anchors).Select(a =>
                        {
                            float best = float.NegativeInfinity;
                            for (int c = 0; c < classes; c++) best = MathF.Max(best, cpuInput.Data[offset + (c + 4) * anchors + a]);
                            return best;
                        }).ToArray();
                        int[] selected = Enumerable.Range(0, anchors).OrderBy(a => -maxima[a]).ThenBy(a => a).Take(k).ToArray();
                        int[] candidates = Enumerable.Range(0, k * classes)
                            .OrderBy(index => -cpuInput.Data[offset + (index % classes + 4) * anchors + selected[index / classes]])
                            .ThenBy(index => index).Take(k).ToArray();
                        for (int i = 0; i < k; i++)
                        {
                            int candidate = candidates[i], anchor = selected[candidate / classes], category = candidate % classes, destination = (b * k + i) * 6;
                            for (int c = 0; c < 4; c++) expected.Data[destination + c] = cpuInput.Data[offset + c * anchors + anchor];
                            expected.Data[destination + 4] = cpuInput.Data[offset + (category + 4) * anchors + anchor];
                            expected.Data[destination + 5] = category;
                        }
                    }
                    WithFrame(backend, () =>
                    {
                        Tensor input = DeviceTensor(backend, cpuInput.Data, cpuInput.Shape, prefix: 3);
                        Tensor actual = (Tensor)gpuTopK.Invoke(backend, [input, maximum, classes])!;
                        if (!actual.Shape.SequenceEqual(expected.Shape)) throw new Exception("GPU ONNX head TopK shape mismatch.");
                        for (int i = 0; i < actual.Length; i++) Exact(actual.Data[i], expected.Data[i]);
                    });
                }
            }
        }));
    }

    /// <summary>向测试数组写入 NaN、无穷、符号零和同分值，覆盖排序边界。</summary>
    /// <param name="values">测试浮点值或编码属性值序列。</param>
    private static void SpecialValues(float[] values)
    {
        values[0] = BitConverter.Int32BitsToSingle(0x7fc00001); values[1] = float.PositiveInfinity; values[2] = float.NegativeInfinity;
        values[3] = 0; values[4] = BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
        values[5] = 0; values[6] = BitConverter.Int32BitsToSingle(0x7fc00003);
        values[7] = float.Epsilon; values[8] = -float.Epsilon; values[9] = float.Epsilon * 2;
    }
    /// <summary>核对两个浮点值的精确表示，包括 NaN 类别和符号零约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Exact(float actual, float expected)
    {
        if (BitConverter.SingleToInt32Bits(actual) != BitConverter.SingleToInt32Bits(expected))
            throw new Exception($"GPU TopK changed value bits: {actual} vs {expected}.");
    }
    /// <summary>创建 Vulkan 后端执行验证，并在结束时释放后端资源。</summary>
    /// <param name="action">在临时测试设置或设备帧作用域内执行的操作。</param>
    private static void WithGpu(Action<VulkanBackend> action)
    {
        string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_TOPK");
        Environment.SetEnvironmentVariable("SIMDYOLO_GPU_TOPK", "1");
        try { using var backend = new VulkanBackend(); action(backend); }
        finally { Environment.SetEnvironmentVariable("SIMDYOLO_GPU_TOPK", previous); }
    }
    /// <summary>在完整 GPU 帧作用域内执行验证，并保证帧资源结束处理。</summary>
    /// <param name="backend">本次验证使用的计算后端。</param>
    /// <param name="action">在临时测试设置或设备帧作用域内执行的操作。</param>
    private static void WithFrame(VulkanBackend backend, Action action)
    {
        Type device = typeof(Tensor).Assembly.GetType("SimdYolo.IDeviceBackend")!;
        device.GetMethod("BeginFrame")!.Invoke(backend, null);
        try { action(); }
        finally { device.GetMethod("EndFrame")!.Invoke(backend, null); }
    }
    /// <summary>构造带前置哨兵区和非零偏移的设备张量视图。</summary>
    /// <param name="backend">本次验证使用的计算后端。</param>
    /// <param name="values">测试浮点值或编码属性值序列。</param>
    /// <param name="shape">测试张量的逻辑维度。</param>
    /// <param name="prefix">设备视图之前保留的哨兵区元素数量。</param>
    /// <returns>带 prefix 浮点元素偏移的借用设备视图，仅在当前 GPU 帧内有效。</returns>
    private static Tensor DeviceTensor(VulkanBackend backend, float[] values, int[] shape, int prefix)
    {
        // 用未对齐的逻辑视图验证描述符地址对齐后的元素偏移补偿。
        var uploaded = new Tensor(Enumerable.Repeat(99f, prefix).Concat(values).ToArray(), values.Length + prefix);
        object storage = typeof(VulkanBackend).GetMethod("Upload", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(backend, [uploaded])!;
        Type storageType = storage.GetType();
        object view = Activator.CreateInstance(storageType,
            [backend, storageType.GetProperty("Buffer")!.GetValue(storage), (ulong)storageType.GetProperty("Offset")!.GetValue(storage)! + (ulong)prefix * 4,
             values.Length, storageType.GetProperty("FrameId")!.GetValue(storage)])!;
        return (Tensor)Activator.CreateInstance(typeof(Tensor), BindingFlags.Instance | BindingFlags.NonPublic, null, [view, shape], null)!;
    }
}
