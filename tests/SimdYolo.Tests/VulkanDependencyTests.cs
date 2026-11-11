using System.Reflection;
using SimdYolo;

/// <summary>提供Vulkan 缓冲区视图的分支、别名、传输和刷新的回归验证。</summary>
internal static class VulkanBufferViewsTests
{
    /// <summary>向测试执行器注册Vulkan 缓冲区视图的分支、别名、传输和刷新的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        var type = typeof(VulkanBackend).GetInterface("SimdYolo.IDeviceBackend")!;
        var begin = type.GetMethod("BeginFrame")!;
        var end = type.GetMethod("EndFrame")!;
        var materialize = type.GetMethod("MaterializeOutputs")!;
        test("Vulkan buffer views concatenate batched host and device tensors with empty inputs", () =>
        {
            using var gpu = new VulkanBackend();
            foreach (int batch in new[] { 1, 2, 4, 5 })
            {
                float[] left = Enumerable.Range(0, batch * 16).Select(i => (float)i).ToArray();
                float[] right = Enumerable.Range(0, batch * 8).Select(i => -(float)i).ToArray();
                var first = new Tensor(left, batch, 2, 2, 4);
                var second = new Tensor(right, batch, 1, 2, 4);
                var empty = new Tensor(batch, 0, 2, 4);
                begin.Invoke(gpu, null);
                try
                {
                    var onDevice = TensorOps.Arithmetic(first, new Tensor([2f], []), 12);
                    var result = TensorOps.Concat([empty, onDevice, second, onDevice], 1);
                    materialize.Invoke(gpu, [new[] { result }]);
                    if (!result.Shape.SequenceEqual([batch, 5, 2, 4]))
                        throw new Exception("Incorrect batched concatenation shape.");
                    for (int b = 0; b < batch; b++)
                    {
                        var row = result.Data.AsSpan(b * 40, 40);
                        for (int i = 0; i < 16; i++)
                            if (row[i] != left[b * 16 + i] * 2 || row[24 + i] != row[i])
                                throw new Exception($"Incorrect device slice for batch {b}.");
                        if (!row.Slice(16, 8).SequenceEqual(right.AsSpan(b * 8, 8)))
                            throw new Exception($"Incorrect host slice for batch {b}.");
                    }
                }
                finally { end.Invoke(gpu, null); }
            }
        });
        test("Vulkan buffer views preserve independent branches, transfers, aliases and flushes", () =>
        {
                using var gpu = new VulkanBackend();
                var offset = Constant(.25f); var scale = Constant(-.375f); var last = Constant(.125f);
                for (int frame = 0; frame < 4; frame++)
                {
                    float[] values = Enumerable.Range(0, 256).Select(i => MathF.Sin(i * .17f + frame)).ToArray();
                    var input = new Tensor(values, 1, 4, 8, 8);
                    begin.Invoke(gpu, null);
                    try
                    {
                        var left = TensorOps.Arithmetic(input, offset, 10); var right = TensorOps.Arithmetic(input, scale, 12);
                        var views = TensorOps.Split(left, 1, [2, 2]);
                        var viewResult = TensorOps.Arithmetic(views[1], scale, 12);
                        var concatenated = TensorOps.Concat([left, right], 1);
                        var joined = TensorOps.Arithmetic(concatenated, last, 10);
                        // 该读取强制提交命令；随后设备操作必须能看到这次提交产生的写入。
                        float[] host = joined.Data;
                        var afterFlush = TensorOps.Arithmetic(joined, offset, 10);
                        Tensor[] outputs = [left, right, concatenated, afterFlush, viewResult];
                        materialize.Invoke(gpu, [outputs]);
                        for (int i = 0; i < values.Length; i++)
                        {
                            Near(left.Data[i], values[i] + .25f);
                            Near(right.Data[i], values[i] * -.375f);
                            Near(concatenated.Data[i], values[i] + .25f);
                            Near(concatenated.Data[i + values.Length], values[i] * -.375f);
                            Near(host[i], values[i] + .25f + .125f);
                            Near(afterFlush.Data[i], values[i] + .25f + .125f + .25f);
                            Near(afterFlush.Data[i + values.Length], values[i] * -.375f + .125f + .25f);
                            if (i < 128) Near(viewResult.Data[i], (values[i + 128] + .25f) * -.375f);
                        }
                    }
                    finally { end.Invoke(gpu, null); }
                }
            static Tensor Constant(float value)
            {
                var tensor = new Tensor([value], []);
                typeof(Tensor).GetProperty("IsConstant", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)!.SetValue(tensor, true);
                return tensor;
            }
            static void Near(float actual, float expected)
            {
                if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > 1e-6f) throw new Exception($"Dependency result {actual} != {expected}");
            }
        });
    }
}
