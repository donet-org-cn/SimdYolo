using System.Reflection;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供二乘二平均池化的 FP32 加法顺序、奇数尾部、多批次和输出所有权的回归验证。</summary>
internal static class CpuAverageTwoTests
{
    /// <summary>向测试执行器注册二乘二平均池化的 FP32 加法顺序、奇数尾部、多批次和输出所有权的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU two-by-two average pooling preserves FP32 order, odd tails, batches and input ownership", () =>
        {
            if (!Avx.IsSupported) return;
            MethodInfo run = typeof(Tensor).Assembly.GetType("SimdYolo.CpuAverageTwo")!.GetMethod("Run", BindingFlags.NonPublic | BindingFlags.Static)!;
            foreach (int width in new[] { 2, 7, 8, 9, 16, 17, 20, 31, 32, 33, 40, 65, 320 })
            {
                int height = width == 320 ? 81 : 7, channels = 6;
                var input = new Tensor(Enumerable.Range(0, channels * height * width).Select(i => MathF.Sin(i * .13f)).ToArray(), 2, 3, height, width);
                Tensor? saved = null; float[]? snapshot = null;
                for (int frame = 0; frame < 3; frame++)
                {
                    float[] original = (float[])input.Data.Clone();
                    Tensor output = (Tensor)run.Invoke(null, [input])!;
                    if (!output.Shape.SequenceEqual([2, 3, height - 1, width - 1])) throw new Exception("Average pooling shape changed.");
                    for (int c = 0; c < channels; c++) for (int y = 0; y < height - 1; y++) for (int x = 0; x < width - 1; x++)
                    {
                        int start = (c * height + y) * width + x;
                        float expected = 0; expected += input.Data[start]; expected += input.Data[start + 1]; expected += input.Data[start + width]; expected += input.Data[start + width + 1]; expected /= 4;
                        float actual = output.Data[(c * (height - 1) + y) * (width - 1) + x];
                        if (float.IsNaN(expected) ? !float.IsNaN(actual) : BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual)) throw new Exception("Average pooling changed FP32 order.");
                    }
                    if (!original.AsSpan().SequenceEqual(input.Data)) throw new Exception("Average pooling modified input.");
                    if (saved is not null && !snapshot!.AsSpan().SequenceEqual(saved.Data)) throw new Exception("Average pooling recycled published output.");
                    saved = output; snapshot = (float[])output.Data.Clone();
                    input.Data[0] += .17f;
                    if (frame == 1)
                    {
                        input.Data[3] = BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
                        input.Data[11] = float.NaN; input.Data[17] = float.PositiveInfinity; input.Data[23] = float.NegativeInfinity;
                        input.Data[29] = float.Epsilon; input.Data[31] = -float.Epsilon;
                    }
                }
            }
        });
    }
}
