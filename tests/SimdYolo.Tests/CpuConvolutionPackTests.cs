using System.Reflection;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供步长二卷积的共享输入加载、边界和逐位打包一致性的回归验证。</summary>
internal static class CpuConvolutionPackTests
{
    /// <summary>向测试执行器注册步长二卷积的共享输入加载、边界和逐位打包一致性的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU stride-two shared input pack is bit-exact: row crossings, boundaries, batches, tails and specials", () =>
        {
            if (!Avx2.IsSupported) return;
            Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuConvolutionPack")!;
            MethodInfo pack = type.GetMethod("Pack", BindingFlags.Static | BindingFlags.NonPublic)!;
            const int channels = 3, height = 9, batch = 2, prefix = 23, suffix = 31;
            int[] bits = [0, unchecked((int)0x80000000), 1, unchecked((int)0x80000001),
                0x7f800000, unchecked((int)0xff800000), 0x7fc12345, unchecked((int)0xffc54321),
                0x3f800000, unchecked((int)0xbf800000), 0x3dcccccd];
            foreach (int width in new[] { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 35, 39, 40, 41, 64, 65, 80, 81, 160, 640 })
                foreach (int padY in new[] { 0, 1 }) foreach (int padX in new[] { 0, 1 })
                {
                    int oh = (height + 2 * padY - 3) / 2 + 1, ow = (width + 2 * padX - 3) / 2 + 1;
                    if (width + 2 * padX < 3 || ow <= 0) continue;
                    float[] input = new float[prefix + batch * channels * height * width + suffix];
                    for (int i = 0; i < input.Length; i++) input[i] = BitConverter.Int32BitsToSingle(bits[i % bits.Length]);
                    for (int b = 0; b < batch; b++)
                        for (int start = 0; start < oh * ow; start += 16)
                        {
                            int take = Math.Min(16, oh * ow - start), batchStart = prefix + b * channels * height * width;
                            float[] packed = Enumerable.Repeat(42f, channels * 9 * 16 + suffix).ToArray();
                            pack.Invoke(null, [input, packed, batchStart, channels, height, width, ow, start, take, padY, padX]);
                            for (int ci = 0; ci < channels; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                                for (int j = 0; j < 16; j++)
                                {
                                    float expected = 0;
                                    if (j < take)
                                    {
                                        int sy = (start + j) / ow * 2 - padY + ky, sx = (start + j) % ow * 2 - padX + kx;
                                        if ((uint)sy < height && (uint)sx < width)
                                            expected = input[batchStart + (ci * height + sy) * width + sx];
                                    }
                                    int index = (ci * 9 + ky * 3 + kx) * 16 + j;
                                    int actualBits = BitConverter.SingleToInt32Bits(packed[index]), expectedBits = BitConverter.SingleToInt32Bits(expected);
                                    if (actualBits != expectedBits)
                                        throw new Exception($"Shared input pack width={width} pad={padY}/{padX} batch={b} start={start} [{ci},{ky},{kx},{j}]: {actualBits:X8} vs {expectedBits:X8}");
                                }
                            for (int i = channels * 9 * 16; i < packed.Length; i++)
                                if (packed[i] != 42f) throw new Exception("Shared input pack overwrote suffix.");
                        }
                }
        });
        test("CPU stride-two shared pack retains convolution, bias and SiLU across input changes", () =>
        {
            using var backend = new CpuBackend(2);
            foreach (var (ic, oc, height, width, pad) in new[]
            {
                (3, 16, 17, 40, 1), (8, 17, 11, 80, 1),
                (17, 19, 7, 35, 1), (3, 7, 9, 65, 0)
            })
            {
                const int batch = 2;
                var input = Make(.1f, batch, ic, height, width);
                var weights = Make(.4f, oc, ic, 3, 3);
                var bias = Make(.7f, oc);
                typeof(Tensor).GetProperty("IsConstant", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(weights, true);
                foreach (bool silu in new[] { false, true }) foreach (bool hasBias in new[] { false, true })
                {
                    var output = new TensorOps(backend).Conv(input, weights, hasBias ? bias : null, [2, 2], [pad, pad, pad, pad], [1, 1], silu: silu);
                    for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int y = 0; y < output.Shape[2]; y++) for (int x = 0; x < output.Shape[3]; x++)
                    {
                        double sum = 0;
                        for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                        {
                            int sy = y * 2 - pad + ky, sx = x * 2 - pad + kx;
                            if ((uint)sy < height && (uint)sx < width)
                                sum += (double)input.Data[((b * ic + ci) * height + sy) * width + sx] * weights.Data[((co * ic + ci) * 3 + ky) * 3 + kx];
                        }
                        float expected = (float)sum + (hasBias ? bias.Data[co] : 0);
                        if (silu) expected /= 1 + MathF.Exp(-expected);
                        float value = output.Data[((b * oc + co) * output.Shape[2] + y) * output.Shape[3] + x];
                        if (!float.IsFinite(value) || MathF.Abs(value - expected) > 1e-4f * MathF.Max(1, MathF.Abs(expected)))
                            throw new Exception($"Shared input convolution [{b},{co},{y},{x}]={value} vs {expected}");
                    }
                    input.Data[0] += .17f;
                    input.Data[input.Length / 2] -= .31f;
                }
            }
        });
    }

    /// <summary>生成指定尺寸和相位的确定性正弦测试数据。</summary>
    /// <param name="phase">确定性正弦测试数据的相位偏移。</param>
    /// <param name="shape">测试张量的逻辑维度。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Make(float phase, params int[] shape)
    {
        var tensor = new Tensor(shape);
        for (int i = 0; i < tensor.Length; i++) tensor.Data[i] = MathF.Sin(i * .031f + phase) * .1f;
        return tensor;
    }
}
