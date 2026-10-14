using System.Reflection;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供检测头指数近似、分布式框回归和姿态向量解码的回归验证。</summary>
internal static class CpuHeadOptimizationTests
{
    /// <summary>向测试执行器注册检测头指数近似、分布式框回归和姿态向量解码的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU head FP32 exponential: dense double oracle and exact special-value fallback", Exp);
        test("CPU tiled DFL: bins, batches, tile tails, extreme logits and changing inputs", Dfl);
        test("CPU vector pose: odd widths, batches, coordinate operation order and storage alias", Pose);
    }

    /// <summary>以稠密双精度参考和特殊值测试 CPU 指数近似。</summary>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Exp()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuActivation")!;
        var exp = type.GetMethod("FastExp", BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<Func<Vector256<float>, Vector256<float>>>();
        Span<float> lanes = stackalloc float[8];
        for (int start = 0; start < 174000; start += 8)
        {
            for (int lane = 0; lane < 8; lane++) lanes[lane] = -87f + (start + lane) * .001f;
            Vector256<float> input = Vector256.Create(lanes), output = exp(input);
            for (int lane = 0; lane < 8; lane++)
            {
                double expected = Math.Exp(input[lane]);
                if (Math.Abs(output[lane] / expected - 1) > 4e-7)
                    throw new Exception($"FP32 exponential {input[lane]}: {output[lane]} vs {expected}.");
            }
        }
        foreach (var input in new[]
        {
            Vector256.Create(BitConverter.Int32BitsToSingle(unchecked((int)0x7fc12345)), float.PositiveInfinity, float.NegativeInfinity, 0f, -0f, float.Epsilon, -float.Epsilon, 88f),
            Vector256.Create(-100f, 100f, -200f, 200f, float.MaxValue, float.MinValue, -87f, 87f)
        })
        {
            var expected = Vector256.Exp(input); var actual = exp(input);
            for (int lane = 0; lane < 8; lane++)
                if (BitConverter.SingleToInt32Bits(actual[lane]) != BitConverter.SingleToInt32Bits(expected[lane]))
                    throw new Exception($"Exponential special-value fallback changed lane {lane}.");
        }
    }

    /// <summary>核对分布式框回归的分块、批次、极端 logits 和可变输入。</summary>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Dfl()
    {
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuDfl")!;
        var decode = type.GetMethod("Decode", BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<Action<Tensor, float[], int, int, ParallelOptions>>();
        var parallel = new ParallelOptions { MaxDegreeOfParallelism = 4 };
        foreach (var (batch, bins, spatial) in new[] { (1, 1, 17), (2, 7, 259), (1, 16, 8400), (2, 16, 1) })
        {
            var source = new Tensor(batch, 4 * bins, spatial);
            for (int frame = 0; frame < 2; frame++)
            {
                for (int i = 0; i < source.Length; i++) source.Data[i] = MathF.Sin(i * .117f + frame) * 15 + (i % 5) * 13;
                if (bins == 1)
                {
                    source.Data[0] = BitConverter.Int32BitsToSingle(unchecked((int)0x7fc12345));
                    source.Data[1] = -0f; source.Data[2] = float.PositiveInfinity;
                }
                else
                {
                    source.Data[0] = float.NaN;
                    source.Data[spatial] = float.PositiveInfinity;
                    if (spatial > 1) for (int bin = 0; bin < bins; bin++) source.Data[bin * spatial + 1] = float.NegativeInfinity;
                }
                var output = new float[batch * 4 * spatial]; Array.Fill(output, 12345);
                decode(source, output, bins, spatial, parallel);
                for (int side = 0; side < batch * 4; side++) for (int anchor = 0; anchor < spatial; anchor++)
                {
                    int offset = side * bins * spatial + anchor, target = side * spatial + anchor;
                    if (bins == 1)
                    {
                        if (BitConverter.SingleToInt32Bits(output[target]) != BitConverter.SingleToInt32Bits(source.Data[offset])) throw new Exception("Single-bin DFL did not preserve storage bits.");
                        continue;
                    }
                    double maximum = double.NegativeInfinity;
                    for (int bin = 0; bin < bins; bin++) maximum = Math.Max(maximum, source.Data[offset + bin * spatial]);
                    double numerator = 0, sum = 0;
                    for (int bin = 0; bin < bins; bin++)
                    {
                        double value = Math.Exp(source.Data[offset + bin * spatial] - maximum);
                        sum += value; numerator += value * bin;
                    }
                    float expected = (float)(numerator / sum), actual = output[target];
                    Near(actual, expected, 2e-6f, "DFL");
                }
            }
        }
    }

    /// <summary>核对姿态向量解码的坐标运算顺序、尾部和输入别名。</summary>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Pose()
    {
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuPose")!;
        var decode = type.GetMethod("Decode", BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<Func<Tensor, int, int, float, ParallelOptions, bool, Tensor>>();
        var parallel = new ParallelOptions { MaxDegreeOfParallelism = 4 };
        foreach (var (batch, points, dimensions, height, width, stride) in new[]
        {
            (2, 17, 3, 80, 80, 8f), (2, 5, 3, 3, 19, 16f), (1, 17, 2, 7, 7, 32f), (2, 1, 3, 1, 1, 3f)
        })
        {
            int channels = points * dimensions, spatial = height * width;
            for (int frame = 0; frame < 2; frame++)
            {
                var raw = new Tensor(batch, channels, height, width);
                for (int i = 0; i < raw.Length; i++) raw.Data[i] = MathF.Sin(i * .179f + frame) * 14;
                if (dimensions == 3 && spatial > 7)
                {
                    float[] special = [float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, -0f, 100f, -100f, float.Epsilon];
                    Array.Copy(special, 0, raw.Data, spatial * 2, special.Length);
                }
                float[] source = (float[])raw.Data.Clone(), originalStorage = raw.Data;
                Tensor result = decode(raw, points, dimensions, stride, parallel, false);
                if (!ReferenceEquals(originalStorage, result.Data) || !result.Shape.SequenceEqual([batch, channels, spatial])) throw new Exception("Pose changed its in-place storage contract.");
                for (int index = 0; index < batch * points; index++) for (int anchor = 0; anchor < spatial; anchor++)
                {
                    int offset = index * dimensions * spatial + anchor;
                    float x = (source[offset] * 2 + anchor % width) * stride;
                    float y = (source[offset + spatial] * 2 + anchor / width) * stride;
                    if (BitConverter.SingleToInt32Bits(result.Data[offset]) != BitConverter.SingleToInt32Bits(x) ||
                        BitConverter.SingleToInt32Bits(result.Data[offset + spatial]) != BitConverter.SingleToInt32Bits(y)) throw new Exception("Pose changed the coordinate multiply/add operation order.");
                    if (dimensions == 3) Near(result.Data[offset + 2 * spatial], 1f / (1f + MathF.Exp(-source[offset + 2 * spatial])), 4e-7f, "Pose sigmoid");
                }
            }
        }
    }

    /// <summary>按指定容差比较两个浮点值，并在失败时标明被测操作。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <param name="tolerance">允许的相对或绝对误差系数。</param>
    /// <param name="operation">失败消息中标识的被测操作。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Near(float actual, float expected, float tolerance, string operation)
    {
        if (float.IsNaN(expected)) { if (!float.IsNaN(actual)) throw new Exception(operation + " did not preserve NaN."); return; }
        if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > tolerance * MathF.Max(1, MathF.Abs(expected)))
            throw new Exception($"{operation}: {actual} vs {expected}.");
    }
}
