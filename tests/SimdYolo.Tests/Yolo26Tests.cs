using System.Reflection;
using SimdYolo;

/// <summary>验证 YOLO26 姿态公式、实例附加字段、端到端解码及 GPU 单桶回归。</summary>
internal static class Yolo26Tests
{
    /// <summary>注册 CPU 和可选 Vulkan 回归。</summary>
    /// <param name="test">执行命名断言的测试回调。</param>
    /// <param name="vulkan">是否验证 Vulkan 解码和附件收集。</param>
    internal static void Run(Action<string, Action> test, bool vulkan)
    {
        test("YOLO26 pose SIMD matches independent half-pixel decoding across batches and tails", () =>
        {
            var method = typeof(Tensor).Assembly.GetType("SimdYolo.CpuPose")!.GetMethod("Decode", BindingFlags.NonPublic | BindingFlags.Static)!;
            foreach (int dimensions in new[] { 2, 3 }) foreach (bool yolo26 in new[] { false, true })
            {
                var input = new Tensor(Enumerable.Range(0, 2 * 5 * dimensions * 3 * 19).Select(i => MathF.Sin(i)).ToArray(), 2, 5 * dimensions, 3, 19);
                float[] original = (float[])input.Data.Clone(); var output = (Tensor)method.Invoke(null, [input, 5, dimensions, 8f, new ParallelOptions(), yolo26])!;
                for (int b = 0; b < 2; b++) for (int k = 0; k < 5; k++) for (int d = 0; d < dimensions; d++) for (int a = 0; a < 57; a++)
                {
                    int index = ((b * 5 + k) * dimensions + d) * 57 + a;
                    float expected = d == 2 ? 1f / (1f + MathF.Exp(-original[index])) : (original[index] * (yolo26 ? 1 : 2) + (d == 0 ? a % 19 : a / 19) + (yolo26 ? .5f : 0)) * 8;
                    Near(output.Data[index], expected);
                }
            }
        });
        test("YOLO26 end-to-end pose/mask payloads retain the selected detection row and skip NMS", () =>
        {
            var transform = new Letterbox(32, 32, 32, 32, 1, 1, 0, 0);
            var pose = new Tensor([1, 2, 20, 24, .9f, 0, 3, 4, .8f, 1, 2, 20, 24, .7f, 0, 13, 14, .6f], 1, 2, 9);
            var options = new DetectorOptions { Task = YoloTask.Pose, OutputLayout = OutputLayout.EndToEnd, KeypointCount = 1, ClassCount = 1 };
            var points = VisionDecoder.Decode([pose], transform, options);
            if (points.Count != 2 || points[0].Keypoints[0].X != 3 || points[1].Keypoints[0].X != 13) throw new Exception("End-to-end pose payload changed or NMS removed a row.");
            var masks = new Tensor([1, 2, 20, 24, .9f, 0, 3, 1, 2, 20, 24, .7f, 0, -3], 1, 2, 7);
            var prototype = new Tensor(Enumerable.Repeat(1f, 64).ToArray(), 1, 1, 8, 8);
            var instances = VisionDecoder.Decode([masks, prototype], transform, options with { Task = YoloTask.Segmentation });
            if (instances.Count != 2 || instances[0].Mask!.Probabilities.Max() < .9f || instances[1].Mask!.Probabilities.Max() > .1f) throw new Exception("Mask coefficients were not attached to selected rows.");
        });
        test("YOLO26 CPU TopK preserves extra channels, batches, classes and equal-score anchors", () => CheckTopK(null));
        if (!vulkan) return;
        test("YOLO26 Vulkan TopK keeps extra channels on device with offset views and batches", () => GpuTopKTests.WithGpu(backend => GpuTopKTests.WithFrame(backend, () => CheckTopK(backend))));
        test("YOLO26 Vulkan single-bin regression passes raw distances and pose uses new anchors", () => GpuTopKTests.WithGpu(backend => GpuTopKTests.WithFrame(backend, () =>
        {
            var type = typeof(Tensor).Assembly.GetType("SimdYolo.DeviceOps")!;
            var boxes = GpuTopKTests.DeviceTensor(backend, [1, 2, 3, 4], [1, 4, 1, 1], 3);
            var scores = GpuTopKTests.DeviceTensor(backend, [0], [1, 1, 1, 1], 3);
            var result = (Tensor)type.GetMethod("DecodeHead")!.Invoke(null, [boxes, scores, 1, 8f, true])!;
            float[] expected = [-4, -12, 28, 36, .5f]; for (int i = 0; i < expected.Length; i++) Near(result.Data[i], expected[i]);
            var raw = GpuTopKTests.DeviceTensor(backend, [1, 2, 0], [1, 3, 1, 1], 3);
            var points = (Tensor)type.GetMethod("DecodePose")!.Invoke(null, [raw, 1, 3, 8f, true])!;
            Near(points.Data[0], 12); Near(points.Data[1], 20); Near(points.Data[2], .5f);
        })));
    }

    /// <summary>以独立候选排序检查 CPU 或 GPU 的框与附加字段收集。</summary>
    /// <param name="backend">GPU 后端；null 验证 CPU。</param>
    private static void CheckTopK(VulkanBackend? backend)
    {
        const int batch = 2, anchors = 37, classes = 3, extra = 7, maximum = 13, fields = 6 + extra;
        var source = new Tensor(batch, 4 + classes + extra, anchors);
        for (int b = 0; b < batch; b++) for (int c = 0; c < source.Shape[1]; c++) for (int a = 0; a < anchors; a++)
            source.Data[(b * source.Shape[1] + c) * anchors + a] = c is >= 4 and < 7 ? (a % 11) * .01f + c * .1f : b * 10000 + c * 100 + a;
        MethodInfo select = typeof(Tensor).Assembly.GetType("SimdYolo.PtModel")!.GetMethod("EndToEndTopK", BindingFlags.NonPublic | BindingFlags.Static)!;
        Tensor output = backend is null ? (Tensor)select.Invoke(null, [source, maximum, classes])! :
            (Tensor)typeof(VulkanBackend).GetMethod("TryEndToEndTopK", BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(backend, [GpuTopKTests.DeviceTensor(backend, source.Data, source.Shape, 3), maximum, classes])!;
        if (!output.Shape.SequenceEqual([batch, maximum, fields])) throw new Exception("TopK lost instance fields.");
        for (int b = 0; b < batch; b++)
        {
            int offset = b * source.Shape[1] * anchors;
            var selected = Enumerable.Range(0, anchors).OrderByDescending(a => Enumerable.Range(0, classes).Max(c => source.Data[offset + (c + 4) * anchors + a])).ThenBy(a => a).Take(maximum).ToArray();
            var best = selected.SelectMany(a => Enumerable.Range(0, classes).Select(c => (Anchor: a, Class: c, Score: source.Data[offset + (c + 4) * anchors + a]))).OrderByDescending(item => item.Score).ThenBy(item => item.Anchor).ThenBy(item => item.Class).Take(maximum).ToArray();
            for (int i = 0; i < maximum; i++)
            {
                int target = (b * maximum + i) * fields; var item = best[i];
                for (int c = 0; c < 4; c++) Near(output.Data[target + c], source.Data[offset + c * anchors + item.Anchor]);
                Near(output.Data[target + 4], item.Score); Near(output.Data[target + 5], item.Class);
                for (int c = 0; c < extra; c++) Near(output.Data[target + 6 + c], source.Data[offset + (classes + 4 + c) * anchors + item.Anchor]);
            }
        }
    }

    /// <summary>检查 FP32 值在绝对容差内接近独立参考值。</summary>
    /// <param name="actual">实际结果。</param>
    /// <param name="expected">独立期望结果。</param>
    /// <exception cref="Exception">结果非有限或偏差超过容差。</exception>
    private static void Near(float actual, float expected) { if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > .0001f) throw new Exception($"{actual} != {expected}"); }
}
