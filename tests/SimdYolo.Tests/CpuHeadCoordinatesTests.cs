using System.Reflection;
using SimdYolo;

/// <summary>验证检测头坐标 SIMD 的行边界、批次和单桶零复制，不依赖模型权重。</summary>
internal static class CpuHeadCoordinatesTests
{
    /// <summary>注册独立标量坐标参考和输入存储保护测试。</summary>
    /// <param name="test">执行命名断言的测试回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU head coordinates: bins1/16, batches, row tails and XYXY/XYWH match scalar", () =>
        {
            var decode = typeof(Tensor).Assembly.GetType("SimdYolo.CpuHead")!.GetMethod("Decode", BindingFlags.Public | BindingFlags.Static)!;
            foreach (int width in new[] { 1, 3, 19, 80 }) foreach (int bins in new[] { 1, 16 }) foreach (bool xyxy in new[] { false, true })
            {
                const int batch = 2, height = 3, classes = 3; int spatial = width * height;
                var boxes = new Tensor(batch, 4 * bins, height, width); var scores = new Tensor(batch, classes, height, width);
                for (int b = 0; b < batch; b++) for (int side = 0; side < 4; side++) for (int r = 0; r < bins; r++) for (int a = 0; a < spatial; a++)
                    boxes.Data[((b * 4 + side) * bins + r) * spatial + a] = bins == 1 ? MathF.Sin(a + side * .3f + b) * 4 : .2f;
                float[] saved = (float[])boxes.Data.Clone();
                var output = (Tensor)decode.Invoke(null, [boxes, scores, bins, 8f, xyxy, new ParallelOptions { MaxDegreeOfParallelism = 1 }])!;
                for (int b = 0; b < batch; b++) for (int a = 0; a < spatial; a++)
                {
                    float Distance(int side) => bins == 1 ? saved[(b * 4 + side) * spatial + a] : 7.5f;
                    float x1 = (a % width + .5f - Distance(0)) * 8, y1 = (a / width + .5f - Distance(1)) * 8;
                    float x2 = (a % width + .5f + Distance(2)) * 8, y2 = (a / width + .5f + Distance(3)) * 8;
                    float[] expected = [xyxy ? x1 : (x1 + x2) * .5f, xyxy ? y1 : (y1 + y2) * .5f, xyxy ? x2 : x2 - x1, xyxy ? y2 : y2 - y1, .5f, .5f, .5f];
                    for (int c = 0; c < expected.Length; c++)
                    {
                        float actual = output.Data[(b * (classes + 4) + c) * spatial + a];
                        // 单桶没有概率归约，坐标要求位一致；多桶的现有指数回退可产生 FP32 舍入差异。
                        bool equal = bins == 1 ? BitConverter.SingleToInt32Bits(actual) == BitConverter.SingleToInt32Bits(expected[c]) : float.IsFinite(actual) && MathF.Abs(actual - expected[c]) <= 2e-5f * MathF.Max(1, MathF.Abs(expected[c]));
                        if (!equal) throw new Exception($"SIMD changed coordinate operation order: width={width}, bins={bins}, xyxy={xyxy}, batch={b}, anchor={a}, channel={c}, actual={actual:R}, expected={expected[c]:R}.");
                    }
                }
                if (!boxes.Data.SequenceEqual(saved)) throw new Exception("Head modified regression storage.");
            }
        });
        test("CPU head coordinates: NaN payloads, infinities, signed zeros and arbitrary strides", () =>
        {
            var decode = typeof(Tensor).Assembly.GetType("SimdYolo.CpuHead")!.GetMethod("Decode", BindingFlags.Public | BindingFlags.Static)!;
            float[] special = [BitConverter.Int32BitsToSingle(0x7fc12345), BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321)), float.PositiveInfinity, float.NegativeInfinity, 0, -0f, .1234567f, -2.345678f];
            foreach (float stride in new[] { 1.25f, -3.75f, 0f }) foreach (bool xyxy in new[] { false, true })
            {
                const int width = 19, height = 2, spatial = width * height;
                var boxes = new Tensor(Enumerable.Range(0, 4 * spatial).Select(i => special[i % special.Length]).ToArray(), 1, 4, height, width);
                var output = (Tensor)decode.Invoke(null, [boxes, new Tensor(1, 1, height, width), 1, stride, xyxy, new ParallelOptions { MaxDegreeOfParallelism = 1 }])!;
                for (int a = 0; a < spatial; a++)
                {
                    float x1 = (a % width + .5f - boxes.Data[a]) * stride, y1 = (a / width + .5f - boxes.Data[spatial + a]) * stride;
                    float x2 = (a % width + .5f + boxes.Data[2 * spatial + a]) * stride, y2 = (a / width + .5f + boxes.Data[3 * spatial + a]) * stride;
                    float[] expected = [xyxy ? x1 : (x1 + x2) * .5f, xyxy ? y1 : (y1 + y2) * .5f, xyxy ? x2 : x2 - x1, xyxy ? y2 : y2 - y1];
                    for (int c = 0; c < 4; c++) if (BitConverter.SingleToInt32Bits(output.Data[c * spatial + a]) != BitConverter.SingleToInt32Bits(expected[c])) throw new Exception("Special coordinate bits changed.");
                }
            }
        });
    }
}
