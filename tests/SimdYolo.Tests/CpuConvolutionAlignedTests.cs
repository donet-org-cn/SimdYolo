using System.Reflection;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供普通卷积的物理缓存行对齐、可变权重和 SIMD 尾部的回归验证。</summary>
internal static class CpuConvolutionAlignedTests
{
    /// <summary>向测试执行器注册普通卷积的物理缓存行对齐、可变权重和 SIMD 尾部的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU physically aligned implicit convolution: cache-line fragments, mutable weights and SiLU", () =>
        {
            if (!Avx2.IsSupported || !Fma.IsSupported) return;
            Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuConvolution")!;
            MethodInfo aligned = type.GetMethod("ConvAligned", BindingFlags.Static | BindingFlags.NonPublic)!;
            Type packedType = typeof(Tensor).Assembly.GetType("SimdYolo.CpuPackedGemm")!;
            MethodInfo prepare = packedType.GetMethod("Prepare", BindingFlags.Static | BindingFlags.NonPublic)!;
            using var backend = new CpuBackend(4);
            foreach (var (ic, oc, size, kernel, stride) in new[] { (3, 7, 16, 3, 2), (8, 19, 40, 3, 2), (43, 32, 40, 3, 2), (5, 17, 8, 5, 1) })
                foreach (bool cached in new[] { false, true })
                {
                    Tensor input = Make(.1f, 1, ic, size, size), weights = Make(.4f, oc, ic, kernel, kernel), bias = Make(.7f, oc);
                    object? matrix = cached ? prepare.Invoke(null, [weights.Data, oc, ic * kernel * kernel, true]) : null;
                    int pad = kernel / 2, width = size / stride;
                    for (int frame = 0; frame < 4; frame++)
                    {
                        bool silu = frame % 2 != 0;
                        var output = new Tensor(1, oc, width, width);
                        aligned.Invoke(null, [backend, input, weights, bias, output, new[] { stride, stride }, new[] { pad, pad, pad, pad }, new[] { 1, 1 }, silu, matrix]);
                        for (int co = 0; co < oc; co++) for (int y = 0; y < width; y++) for (int x = 0; x < width; x++)
                        {
                            double sum = 0; float fp32 = 0;
                            for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < kernel; ky++) for (int kx = 0; kx < kernel; kx++)
                            {
                                int sy = y * stride - pad + ky, sx = x * stride - pad + kx;
                                float value = (uint)sy < size && (uint)sx < size ? input.Data[(ci * size + sy) * size + sx] : 0;
                                float weight = weights.Data[((co * ic + ci) * kernel + ky) * kernel + kx];
                                sum += (double)value * weight; fp32 = MathF.FusedMultiplyAdd(weight, value, fp32);
                            }
                            fp32 += bias.Data[co]; float expected = (float)sum + bias.Data[co];
                            if (silu) expected /= 1 + MathF.Exp(-expected);
                            float actual = output.Data[(co * width + y) * width + x];
                            if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > 1e-4f * MathF.Max(1, MathF.Abs(expected))) throw new Exception($"Aligned implicit convolution [{co},{y},{x}]={actual} vs {expected}");
                            if (!silu && BitConverter.SingleToInt32Bits(actual) != BitConverter.SingleToInt32Bits(fp32)) throw new Exception("Aligned implicit convolution changed FP32 FMA order.");
                        }
                        input.Data[0] += .17f;
                        if (!cached) weights.Data[0] -= .07f;
                    }
                }
        });
    }
    /// <summary>生成指定尺寸和相位的确定性正弦测试数据。</summary>
    /// <param name="phase">确定性正弦测试数据的相位偏移。</param>
    /// <param name="shape">测试张量的逻辑维度。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Make(float phase, params int[] shape)
        => new(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => MathF.Sin(i * .031f + phase) * .1f).ToArray(), shape);
}
