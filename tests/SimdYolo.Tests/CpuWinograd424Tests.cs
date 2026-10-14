using System.Reflection;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供F4 卷积的二十四列变换块、裁剪边界和独立标量参考的回归验证。</summary>
internal static class CpuWinograd424Tests
{
    /// <summary>向测试执行器注册F4 卷积的二十四列变换块、裁剪边界和独立标量参考的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU F4 24-column panels vs double scalar: SIMD rows, row crossings, odd borders, panel tails, bias and batches", () =>
        {
            if (!Avx2.IsSupported || !Fma.IsSupported) return;
            using var cpu = new CpuBackend(2);
            foreach (var (batch, ic, oc, h, width) in new[]
            {
                (2, 5, 7, 21, 39), (1, 33, 17, 17, 67),
                (2, 8, 16, 9, 19), (1, 32, 32, 21, 81)
            })
            {
                var x = Make([batch, ic, h, width], .1f); var w = Make([oc, ic, 3, 3], .3f);
                var bias = Make([oc], .7f); float[][] transformed = TransformWeights(w);
                foreach (bool silu in new[] { false, true }) foreach (bool withBias in new[] { false, true })
                    Check(Conv(cpu, x, transformed, withBias ? bias : null, silu), Scalar(x, w, withBias ? bias : null, silu));
                x.Data[0] += .4f; x.Data[x.Length / 2] -= .17f;
                Check(Conv(cpu, x, transformed, bias, true), Scalar(x, w, bias, true));
            }
        });
        test("CPU small F4 global coordinate path vs double scalar: padded border SIMD, odd dimensions and batches", () =>
        {
            if (!Avx2.IsSupported || !Fma.IsSupported) return;
            using var cpu = new CpuBackend(12);
            MethodInfo method = typeof(Tensor).Assembly.GetType("SimdYolo.CpuWinograd4")!.GetMethod("Conv", BindingFlags.Static | BindingFlags.Public)!;
            foreach (var (batch, ic, oc, h, width) in new[] { (1, 80, 80, 20, 20), (2, 33, 17, 21, 23), (1, 32, 16, 39, 37) })
            {
                var input = Make([batch, ic, h, width], .1f); var weights = Make([oc, ic, 3, 3], .3f); var bias = Make([oc], .7f);
                typeof(Tensor).GetProperty("IsConstant", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(weights, true);
                foreach (bool silu in new[] { false, true })
                    Check((Tensor)method.Invoke(null, [cpu, input, weights, bias, silu])!, Scalar(input, weights, bias, silu));
                input.Data[0] += .4f; input.Data[input.Length - 1] -= .17f;
                Check((Tensor)method.Invoke(null, [cpu, input, weights, bias, true])!, Scalar(input, weights, bias, true));
            }
        });
        test("CPU F4 border SIMD input transform preserves scalar FP32 bits, clipped corners, pooled padding and row crossings", () =>
        {
            if (!Avx2.IsSupported || !Fma.IsSupported) return;
            MethodInfo method = typeof(Tensor).Assembly.GetType("SimdYolo.CpuWinogradPanels")!
                .GetMethods(BindingFlags.Static | BindingFlags.NonPublic)
                .Single(candidate => candidate.Name == "Transform4" && candidate.GetParameters()[0].ParameterType == typeof(float[]));
            foreach (var (h, width, tile, take) in new[] { (21,39,0,24), (21,39,2,8), (21,39,42,8), (9,19,0,15), (1,39,0,10), (5,1,0,2) })
            {
                const int batch = 2, ic = 3, columns = 24;
                int tw = (width + 3) / 4, stride = ic * columns + 16;
                var input = Make([batch, ic, h, width], .13f);
                input.Data[0] = -0f; input.Data[input.Length / 2] = 0;
                // 特殊值保留既有 Winograd NaN 语义；本优化不要求某个固定的 NaN 位模式。
                input.Data[Math.Min(2, input.Length - 1)] = float.PositiveInfinity;
                input.Data[input.Length - 1] = float.NaN;
                float[] original = (float[])input.Data.Clone();
                for (int b = 0; b < batch; b++)
                {
                    float[] actual = Enumerable.Repeat(42f, 36 * stride + 13).ToArray();
                    float[] expected = (float[])actual.Clone();
                    method.Invoke(null, [input.Data, actual, b, ic, h, width, tw, tile, take, stride, columns]);
                    ReferenceTransform4(input.Data, expected, b, ic, h, width, tw, tile, take, stride, columns);
                    for (int i = 0; i < actual.Length; i++)
                        if (float.IsNaN(expected[i]) ? !float.IsNaN(actual[i]) : BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(expected[i]))
                            throw new Exception($"F4 border transform bits ({h},{width},{tile},{take}) batch {b} at {i}: {actual[i]} vs {expected[i]}");
                }
                for (int i = 0; i < original.Length; i++)
                    if (BitConverter.SingleToInt32Bits(original[i]) != BitConverter.SingleToInt32Bits(input.Data[i])) throw new Exception("F4 border transform mutated input.");
            }
        });
    }

    private static readonly MethodInfo ConvMethod = typeof(Tensor).Assembly.GetType("SimdYolo.CpuWinograd424")!
        .GetMethod("Conv", BindingFlags.Static | BindingFlags.NonPublic)!;
    /// <summary>通过反射运行指定 Winograd 分块内核，取得被测卷积结果。</summary>
    /// <param name="backend">本次验证使用的计算后端。</param>
    /// <param name="x">待计算的输入张量。</param>
    /// <param name="weights">待验证的权重数据或内部打包矩阵。</param>
    /// <param name="bias">可选的逐输出通道偏置。</param>
    /// <param name="silu">是否对参考结果应用 SiLU 激活。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Conv(CpuBackend backend, Tensor x, float[][] weights, Tensor? bias, bool silu)
        => (Tensor)ConvMethod.Invoke(null, [backend, x, weights, bias, silu])!;
    /// <summary>生成指定尺寸和相位的确定性正弦测试数据。</summary>
    /// <param name="shape">测试张量的逻辑维度。</param>
    /// <param name="phase">确定性正弦测试数据的相位偏移。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Make(int[] shape, float phase)
    {
        var tensor = new Tensor(shape);
        for (int i = 0; i < tensor.Length; i++) tensor.Data[i] = MathF.Sin(i * .031f + phase) * .1f;
        return tensor;
    }
    /// <summary>以独立标量矩阵变换生成 F4 卷积的权重参考坐标。</summary>
    /// <param name="w">按输出通道、输入通道、核行、核列保存的卷积权重。</param>
    /// <returns>生成的测试数据或参考变换坐标。</returns>
    private static float[][] TransformWeights(Tensor w)
    {
        float[] g = [.25f, 0, 0, -1f / 6, -1f / 6, -1f / 6, -1f / 6, 1f / 6, -1f / 6,
                     1f / 24, 1f / 12, 1f / 6, 1f / 24, -1f / 12, 1f / 6, 0, 0, 1];
        int oc = w.Shape[0], ic = w.Shape[1];
        var weights = Enumerable.Range(0, 36).Select(_ => new float[oc * ic]).ToArray();
        Span<float> temp = stackalloc float[18];
        for (int co = 0; co < oc; co++) for (int ci = 0; ci < ic; ci++)
        {
            int start = (co * ic + ci) * 9, dst = co * ic + ci;
            for (int r = 0; r < 6; r++) for (int c = 0; c < 3; c++)
                temp[r * 3 + c] = g[r * 3] * w.Data[start + c] + g[r * 3 + 1] * w.Data[start + 3 + c] + g[r * 3 + 2] * w.Data[start + 6 + c];
            for (int r = 0; r < 6; r++) for (int c = 0; c < 6; c++)
                weights[r * 6 + c][dst] = temp[r * 3] * g[c * 3] + temp[r * 3 + 1] * g[c * 3 + 1] + temp[r * 3 + 2] * g[c * 3 + 2];
        }
        return weights;
    }
    /// <summary>以标量顺序生成 F4 输入变换参考，覆盖批次、裁剪边界和工作区步长。</summary>
    /// <param name="input">待验证的输入数据。</param>
    /// <param name="output">用于接收计算结果的预分配缓冲区。</param>
    /// <param name="batch">测试批次数量或当前批次索引。</param>
    /// <param name="channels">测试通道数量。</param>
    /// <param name="h">输入图像的高度。</param>
    /// <param name="width">输入图像的宽度。</param>
    /// <param name="tw">每行 Winograd tile 的数量。</param>
    /// <param name="tile">当前变换块的起始 tile 索引。</param>
    /// <param name="take">当前计算块实际处理的列或 tile 数。</param>
    /// <param name="stride">变换坐标在工作区中的步长。</param>
    /// <param name="columns">空间平面或计算块中的列数。</param>
    private static void ReferenceTransform4(float[] input, float[] output, int batch, int channels, int h, int width, int tw, int tile, int take, int stride, int columns)
    {
        Span<float> patch = stackalloc float[36], temp = stackalloc float[36], values = stackalloc float[6];
        for (int ci = 0; ci < channels; ci++)
        {
            int channel = (batch * channels + ci) * h * width, row = ci * columns;
            for (int j = 0; j < take; j++)
            {
                int ty = (tile + j) / tw, tx = (tile + j) % tw;
                for (int r = 0; r < 6; r++) for (int c = 0; c < 6; c++)
                {
                    int yy = ty * 4 - 1 + r, xx = tx * 4 - 1 + c;
                    patch[r * 6 + c] = (uint)yy < h && (uint)xx < width ? input[channel + yy * width + xx] : 0;
                }
                for (int c = 0; c < 6; c++)
                {
                    ReferenceTransform(patch[c], patch[6 + c], patch[12 + c], patch[18 + c], patch[24 + c], patch[30 + c], values);
                    for (int r = 0; r < 6; r++) temp[r * 6 + c] = values[r];
                }
                for (int r = 0; r < 6; r++)
                {
                    ReferenceTransform(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], values);
                    for (int c = 0; c < 6; c++) output[(r * 6 + c) * stride + row + j] = values[c];
                }
            }
            for (int coordinate = 0; coordinate < 36; coordinate++) output.AsSpan(coordinate * stride + row + take, columns - take).Clear();
        }
    }
    /// <summary>按 F4 的一维变换顺序计算六个输入值的参考坐标。</summary>
    /// <param name="a">一维 F4 变换的第 1 个输入值。</param>
    /// <param name="b">一维 F4 变换的第 2 个输入值。</param>
    /// <param name="c">一维 F4 变换的第 3 个输入值。</param>
    /// <param name="d">一维 F4 变换的第 4 个输入值。</param>
    /// <param name="e">一维 F4 变换的第 5 个输入值。</param>
    /// <param name="f">一维 F4 变换的第 6 个输入值。</param>
    /// <param name="output">用于接收计算结果的预分配缓冲区。</param>
    private static void ReferenceTransform(float a, float b, float c, float d, float e, float f, Span<float> output)
    {
        output[0] = 4 * a - 5 * c + e; output[1] = -4 * (b + c) + d + e;
        output[2] = 4 * (b - c) - d + e; output[3] = -2 * b - c + 2 * d + e;
        output[4] = 2 * b - c - 2 * d + e; output[5] = 4 * b - 5 * d + f;
    }
    /// <summary>使用独立标量循环计算卷积参考结果，避免复用被测 SIMD 内核。</summary>
    /// <param name="x">待计算的输入张量。</param>
    /// <param name="w">按输出通道、输入通道、核行、核列保存的卷积权重。</param>
    /// <param name="bias">可选的逐输出通道偏置。</param>
    /// <param name="silu">是否对参考结果应用 SiLU 激活。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Scalar(Tensor x, Tensor w, Tensor? bias, bool silu)
    {
        int batch = x.Shape[0], ic = x.Shape[1], h = x.Shape[2], width = x.Shape[3], oc = w.Shape[0], n = h * width;
        var output = new Tensor(batch, oc, h, width);
        for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int y = 0; y < h; y++) for (int column = 0; column < width; column++)
        {
            double sum = 0;
            for (int ci = 0; ci < ic; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
            {
                int yy = y + ky - 1, xx = column + kx - 1;
                if ((uint)yy < h && (uint)xx < width)
                    sum += (double)x.Data[(b * ic + ci) * n + yy * width + xx] * w.Data[(co * ic + ci) * 9 + ky * 3 + kx];
            }
            float value = (float)sum + (bias?.Data[co] ?? 0);
            output.Data[(b * oc + co) * n + y * width + column] = silu ? value / (1 + MathF.Exp(-value)) : value;
        }
        return output;
    }
    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Check(Tensor actual, Tensor expected)
    {
        if (!actual.Shape.SequenceEqual(expected.Shape)) throw new Exception("F4 24-panel shape mismatch.");
        for (int i = 0; i < actual.Length; i++)
            if (!float.IsFinite(actual.Data[i]) || MathF.Abs(actual.Data[i] - expected.Data[i]) > 1e-4f * MathF.Max(1, MathF.Abs(expected.Data[i])))
                throw new Exception($"F4 24-panel mismatch at {i}: {actual.Data[i]} vs {expected.Data[i]}");
    }
}
