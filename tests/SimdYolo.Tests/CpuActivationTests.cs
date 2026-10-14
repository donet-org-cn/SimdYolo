using System.Reflection;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供CPU 向量激活的有限值误差、NaN、无穷和标量回退的回归验证。</summary>
internal static class CpuActivationTests
{
    /// <summary>向测试执行器注册CPU 向量激活的有限值误差、NaN、无穷和标量回退的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU four-lane activation vs dense double oracle and exact special fallback", () =>
        {
            var type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuActivation")!;
            var method = type.GetMethod("Silu", BindingFlags.NonPublic | BindingFlags.Static, [typeof(Vector128<float>)])!;
            var silu = method.CreateDelegate<Func<Vector128<float>, Vector128<float>>>();
            for (int start = 0; start < 174000; start += 4)
            {
                var input = Vector128.Create(-87f + start * .001f, -87f + (start + 1) * .001f, -87f + (start + 2) * .001f, -87f + (start + 3) * .001f);
                var result = silu(input);
                for (int lane = 0; lane < 4; lane++)
                {
                    float x = input[lane], expected = (float)(x / (1 + Math.Exp(-(double)x)));
                    if (MathF.Abs(result[lane] - expected) > 4e-7f * MathF.Max(1, MathF.Abs(expected))) throw new Exception("Four-lane activation differs from double oracle.");
                }
            }
            foreach (var input in new[] { Vector128.Create(float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0f), Vector128.Create(88f, -88f, 200f, -200f) })
            {
                var expected = input / (Vector128<float>.One + Vector128.Exp(-input)); var actual = silu(input);
                for (int lane = 0; lane < 4; lane++) if (BitConverter.SingleToInt32Bits(expected[lane]) != BitConverter.SingleToInt32Bits(actual[lane])) throw new Exception("Four-lane special fallback changed bits.");
            }
        });
        test("CPU FP32 SIMD activation vs double: dense normal range and special values", () =>
        {
            if (!Avx2.IsSupported || !Fma.IsSupported) return;
            var type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuActivation")!;
            var method = type.GetMethod("FastSilu", BindingFlags.NonPublic | BindingFlags.Static)!;
            var silu = method.CreateDelegate<Func<Vector256<float>, Vector256<float>>>();
            Span<float> inputs = stackalloc float[8];
            for (int start = 0; start < 174000; start += 8)
            {
                for (int lane = 0; lane < 8; lane++) inputs[lane] = -87f + (start + lane) * .001f;
                Check(Vector256.Create(inputs), silu);
            }
            Check(Vector256.Create(float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, -0f, 88f, -88f, float.Epsilon), silu);
            Check(Vector256.Create(-200f, 200f, -87f, 87f, -1f, 1f, -1e-20f, 1e-20f), silu);
            static void Check(Vector256<float> input, Func<Vector256<float>, Vector256<float>> silu)
            {
                var result = silu(input);
                for (int lane = 0; lane < 8; lane++)
                {
                    float x = input[lane];
                    float expected = (float)(x / (1 + Math.Exp(-(double)x))), actual = result[lane];
                    if (float.IsNaN(expected)) { if (!float.IsNaN(actual)) throw new Exception("Activation NaN mismatch."); continue; }
                    if (float.IsInfinity(expected)) { if (actual != expected) throw new Exception("Activation infinity mismatch."); continue; }
                    // 标量 float 指数比 double 更早溢出；超出正常区间时，
                    // 使用原有 float 运算作参考，保留其溢出语义。
                    if (MathF.Abs(x) > 87) expected = x / (1 + MathF.Exp(-x));
                    if (MathF.Abs(actual - expected) > 4e-7f * MathF.Max(1, MathF.Abs(expected)))
                        throw new Exception($"Activation {x}: {actual} vs {expected}");
                    if (x == 0 && BitConverter.SingleToInt32Bits(actual) != BitConverter.SingleToInt32Bits(x)) throw new Exception("Activation lost signed zero.");
                }
            }
        });
    }
}
