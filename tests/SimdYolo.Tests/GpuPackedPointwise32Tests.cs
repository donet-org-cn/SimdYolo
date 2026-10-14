using System.Reflection;
using SimdYolo;

/// <summary>提供GPU 打包点卷积的偏移视图、通道尾部和多帧输入的回归验证。</summary>
internal static class GpuPackedPointwise32Tests
{
    /// <summary>向测试执行器注册GPU 打包点卷积的偏移视图、通道尾部和多帧输入的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("Packed pointwise32 preserves input and bias views across channel tails and frames", () =>
        {
            string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_POINTWISE_PACKED32");
            Environment.SetEnvironmentVariable("SIMDYOLO_GPU_POINTWISE_PACKED32", "1");
            try
            {
                using var gpu = new VulkanBackend();
                Type device = typeof(Tensor).Assembly.GetType("SimdYolo.IDeviceBackend")!;
                var begin = device.GetMethod("BeginFrame")!; var end = device.GetMethod("EndFrame")!;
                var conv = device.GetMethod("DeviceConv")!;
                const int batch = 2, ic = 17, h = 13, width = 24, spatial = h * width;
                foreach (int oc in new[] { 24, 80 }) foreach (bool silu in new[] { false, true })
                {
                    var weights = new Tensor(Enumerable.Range(0, oc * ic).Select(i => MathF.Cos(i * .37f) * .03f).ToArray(), oc, ic, 1, 1);
                    typeof(Tensor).GetProperty("IsConstant", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.SetValue(weights, true);
                    float[] biases = Enumerable.Range(0, oc).Select(i => i * .01f).ToArray();
                    for (int iteration = 0; iteration < 2; iteration++)
                    {
                        float[] values = Enumerable.Range(0, batch * ic * spatial).Select(i => MathF.Sin(i * .17f + iteration * .3f) * .2f).ToArray();
                        begin.Invoke(gpu, null);
                        try
                        {
                            Tensor input = DeviceView(gpu, values, [batch, ic, h, width], 4);
                            Tensor bias = DeviceView(gpu, biases, [oc], 3);
                            var actual = (Tensor)conv.Invoke(gpu, [input, weights, bias, new[] { batch, oc, h, width }, new[] { 1, 1 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, 1, silu])!;
                            float[] output = actual.Data;
                            for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int column = 0; column < spatial; column++)
                            {
                                double sum = biases[co];
                                for (int ci = 0; ci < ic; ci++) sum += (double)values[(b * ic + ci) * spatial + column] * weights.Data[co * ic + ci];
                                float expected = (float)sum;
                                if (silu) expected /= 1 + MathF.Exp(-expected);
                                float value = output[(b * oc + co) * spatial + column];
                                if (!float.IsFinite(value) || MathF.Abs(value - expected) > 1e-4f * MathF.Max(1, MathF.Abs(expected)))
                                    throw new Exception($"Packed32 view convolution mismatch: {value} vs {expected}");
                            }
                        }
                        finally { end.Invoke(gpu, null); }
                    }
                }
            }
            finally { Environment.SetEnvironmentVariable("SIMDYOLO_GPU_POINTWISE_PACKED32", previous); }
        });
    }

    /// <summary>构造带前置哨兵区和非零偏移的设备张量视图。</summary>
    /// <param name="gpu">本次验证使用的 Vulkan 后端。</param>
    /// <param name="values">测试浮点值或编码属性值序列。</param>
    /// <param name="shape">测试张量的逻辑维度。</param>
    /// <param name="prefix">设备视图之前保留的哨兵区元素数量。</param>
    /// <returns>带 prefix 浮点元素偏移的借用设备视图，仅在当前 GPU 帧内有效。</returns>
    private static Tensor DeviceView(VulkanBackend gpu, float[] values, int[] shape, int prefix)
    {
        var upload = new Tensor(Enumerable.Repeat(99f, prefix).Concat(values).ToArray(), values.Length + prefix);
        object source = typeof(VulkanBackend).GetMethod("Upload", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(gpu, [upload])!;
        Type type = source.GetType();
        object view = Activator.CreateInstance(type, [gpu, type.GetProperty("Buffer")!.GetValue(source),
            (ulong)type.GetProperty("Offset")!.GetValue(source)! + (ulong)prefix * 4, values.Length, type.GetProperty("FrameId")!.GetValue(source)])!;
        return (Tensor)Activator.CreateInstance(typeof(Tensor), BindingFlags.Instance | BindingFlags.NonPublic, null, [view, shape], null)!;
    }
}
