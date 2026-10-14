using System.Reflection;
using SimdYolo;

/// <summary>验证批次权重缓存的形状与源替换失效，以及取消预清零后卷积和池化的完整写入。</summary>
internal static class CpuWeightBatchTests
{
    /// <summary>注册轻量 CPU 批次缓存和脏缓冲区覆盖回归用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">输出不符合独立参考结果或缓存保留旧源数据时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU constant weight batch tracks source replacements, shape changes and zero K", Batch);
        test("CPU convolution and max pool overwrite every element of a dirty pooled output", DirtyOutputs);
        test("CPU broadcast arithmetic, activation, scaling and concat overwrite dirty outputs and preserve empty shapes", DirtyElementwise);
    }

    /// <summary>替换同一权重批次内的源数组并改变 M/K，独立检查结果；同时覆盖零 K 输出清零语义。</summary>
    /// <exception cref="Exception">矩阵结果不满足 FP32 FMA 参考或零 K 输出未清零时。</exception>
    private static void Batch()
    {
        var gemm = typeof(Tensor).Assembly.GetType("SimdYolo.CpuGemm")!.GetMethod("MultiplyBatch")!;
        float[][] sources = [Make(91, .13f), Make(91, -.21f)];
        foreach (var (m, k) in new[] { (7, 13), (7, 13), (13, 7), (7, 13) })
        {
            int n = 19;
            float[][] inputs = [Make(k * n, .37f), Make(k * n, -.71f)];
            float[][] outputs = [Enumerable.Repeat(float.NaN, m * n).ToArray(), Enumerable.Repeat(float.NaN, m * n).ToArray()];
            gemm.Invoke(null, [sources, inputs, outputs, m, n, k, new ParallelOptions { MaxDegreeOfParallelism = 1 }, true]);
            for (int batch = 0; batch < 2; batch++) for (int row = 0; row < m; row++) for (int column = 0; column < n; column++)
            {
                float expected = 0;
                for (int index = 0; index < k; index++) expected = MathF.FusedMultiplyAdd(sources[batch][row * k + index], inputs[batch][index * n + column], expected);
                Near(outputs[batch][row * n + column], expected);
            }
            // 批次数组身份保持不变；内部源矩阵的替换必须使整个批次描述符失效。
            sources[1] = Make(91, sources[1][0] + .59f);
        }
        float[][] zeroOutput = [Enumerable.Repeat(7f, 21).ToArray(), Enumerable.Repeat(9f, 21).ToArray()];
        gemm.Invoke(null, [new[] { Array.Empty<float>(), Array.Empty<float>() }, new[] { Array.Empty<float>(), Array.Empty<float>() },
            zeroOutput, 7, 3, 0, new ParallelOptions { MaxDegreeOfParallelism = 1 }, true]);
        if (zeroOutput.Any(output => output.Any(value => BitConverter.SingleToInt32Bits(value) != 0))) throw new Exception("Zero K GEMM did not clear output.");
    }

    /// <summary>把池中待复用的目标数组写为 NaN，再验证各卷积分支和最大池化完整覆盖有效输出。</summary>
    /// <exception cref="Exception">旧池内容泄漏到输出，或输出偏离独立标量参考时。</exception>
    private static void DirtyOutputs()
    {
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuTensorPool")!;
        FieldInfo current = type.GetField("Current", BindingFlags.Static | BindingFlags.NonPublic)!;
        object? previous = current.GetValue(null), pool = Activator.CreateInstance(type, nonPublic: true)!;
        MethodInfo rent = type.GetMethod("Rent")!, release = type.GetMethod("ReturnAll")!;
        current.SetValue(null, pool);
        try
        {
            using var cpu = new CpuBackend(1); var ops = new TensorOps(cpu);
            foreach (var (batch, ic, oc, kernel, stride, groups, height, width) in new[]
            {
                (1, 3, 7, 1, 1, 1, 9, 19), (2, 3, 7, 3, 2, 1, 17, 19),
                (2, 5, 5, 3, 1, 5, 17, 19), (1, 4, 6, 3, 1, 2, 9, 11)
            })
            {
                var input = new Tensor(Make(batch * ic * height * width, .19f), batch, ic, height, width);
                var weights = new Tensor(Make(oc * (ic / groups) * kernel * kernel, -.29f), oc, ic / groups, kernel, kernel);
                var bias = new Tensor(Make(oc, .43f), oc);
                int pad = kernel / 2, oh = (height + 2 * pad - kernel) / stride + 1, ow = (width + 2 * pad - kernel) / stride + 1;
                var dirty = (float[])rent.Invoke(pool, [batch * oc * oh * ow, false])!;
                Array.Fill(dirty, float.NaN); release.Invoke(pool, null);
                var output = ops.Conv(input, weights, bias, [stride, stride], [pad, pad, pad, pad], groups: groups);
                if (!ReferenceEquals(output.Data, dirty)) throw new Exception("Convolution test did not reuse the poisoned output array.");
                int icg = ic / groups, ocg = oc / groups;
                for (int b = 0; b < batch; b++) for (int co = 0; co < oc; co++) for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++)
                {
                    double expected = bias.Data[co];
                    for (int ci = 0; ci < icg; ci++) for (int ky = 0; ky < kernel; ky++) for (int kx = 0; kx < kernel; kx++)
                    {
                        int iy = y * stride - pad + ky, ix = x * stride - pad + kx;
                        if ((uint)iy < height && (uint)ix < width)
                            expected += (double)input.Data[((b * ic + co / ocg * icg + ci) * height + iy) * width + ix] * weights.Data[((co * icg + ci) * kernel + ky) * kernel + kx];
                    }
                    Near(output.Data[((b * oc + co) * oh + y) * ow + x], (float)expected);
                }
                release.Invoke(pool, null);
            }
            var poolInput = new Tensor(Make(2 * 3 * 9 * 13, -.53f), 2, 3, 9, 13);
            var poolDirty = (float[])rent.Invoke(pool, [poolInput.Length, false])!;
            Array.Fill(poolDirty, float.NaN); release.Invoke(pool, null);
            var pooled = TensorOps.Pool(poolInput, [5, 3], [1, 1], [2, 1, 2, 1]);
            if (!ReferenceEquals(pooled.Data, poolDirty)) throw new Exception("Max-pool test did not reuse the poisoned output array.");
            for (int c = 0; c < 6; c++) for (int y = 0; y < 9; y++) for (int x = 0; x < 13; x++)
            {
                float expected = float.NegativeInfinity;
                for (int ky = 0; ky < 5; ky++) for (int kx = 0; kx < 3; kx++)
                {
                    int iy = y - 2 + ky, ix = x - 1 + kx;
                    if ((uint)iy < 9 && (uint)ix < 13) expected = MathF.Max(expected, poolInput.Data[(c * 9 + iy) * 13 + ix]);
                }
                Near(pooled.Data[(c * 9 + y) * 13 + x], expected);
            }
            release.Invoke(pool, null);
            var reused = (float[])rent.Invoke(pool, [poolInput.Length, true])!;
            if (reused.Any(value => BitConverter.SingleToInt32Bits(value) != 0)) throw new Exception("Reused zero-initialized pool buffer contained old data.");
        }
        finally
        {
            release.Invoke(pool, null);
            type.GetMethod("Clear")!.Invoke(pool, null);
            current.SetValue(null, previous);
        }
    }

    /// <summary>以脏池检查广播、尾部、零维标量、空张量以及不同拼接轴的逐元素覆盖。</summary>
    /// <exception cref="Exception">输出保留旧池内容、偏离参考结果或改变空张量形状时。</exception>
    private static void DirtyElementwise()
    {
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuTensorPool")!;
        FieldInfo current = type.GetField("Current", BindingFlags.Static | BindingFlags.NonPublic)!;
        object? previous = current.GetValue(null), pool = Activator.CreateInstance(type, nonPublic: true)!;
        MethodInfo rent = type.GetMethod("Rent")!, release = type.GetMethod("ReturnAll")!;
        current.SetValue(null, pool);
        try
        {
            // 按 expected 的长度租用并污染缓冲区；operation 必须覆盖全部元素并保持 shape。
            void Check(Func<Tensor> operation, float[] expected, int[] shape)
            {
                var dirty = (float[])rent.Invoke(pool, [expected.Length, false])!;
                Array.Fill(dirty, float.NaN); release.Invoke(pool, null);
                var output = operation();
                if (!output.Shape.SequenceEqual(shape) || !ReferenceEquals(output.Data, dirty)) throw new Exception("Elementwise test did not use the expected dirty output or shape.");
                for (int index = 0; index < expected.Length; index++) Near(output.Data[index], expected[index]);
                release.Invoke(pool, null);
            }
            foreach (var (leftShape, rightShape) in new (int[], int[])[]
            {
                ([2, 3, 5], [1, 3, 1]), ([1, 3, 1], [2, 3, 5]), ([2, 3, 5], [2, 3, 5]),
                ([], [2, 3, 5]), ([2, 3, 5], []), ([2, 1, 5], [1, 3, 1])
            })
            {
                var left = new Tensor(Make(leftShape.Aggregate(1, (a, b) => a * b), .9f), leftShape);
                var right = new Tensor(Make(rightShape.Aggregate(1, (a, b) => a * b), 1.1f), rightShape);
                // 独立映射三维广播坐标；零维 value 始终返回其唯一标量。
                float At(Tensor value, int batch, int channel, int x)
                {
                    if (value.Rank == 0) return value.Data[0];
                    int b = value.Shape[0] == 1 ? 0 : batch, c = value.Shape[1] == 1 ? 0 : channel, column = value.Shape[2] == 1 ? 0 : x;
                    return value.Data[(b * value.Shape[1] + c) * value.Shape[2] + column];
                }
                foreach (int operation in new[] { 10, 11, 12, 13 })
                {
                    var expected = new float[30];
                    for (int b = 0; b < 2; b++) for (int c = 0; c < 3; c++) for (int x = 0; x < 5; x++)
                    {
                        float a = At(left, b, c, x), r = At(right, b, c, x);
                        expected[(b * 3 + c) * 5 + x] = operation switch { 10 => a + r, 11 => a - r, 12 => a * r, _ => a / r };
                    }
                    Check(() => TensorOps.Arithmetic(left, right, operation), expected, [2, 3, 5]);
                }
            }
            foreach (int[] shape in new[] { Array.Empty<int>(), new[] { 1 }, new[] { 19 }, new[] { 2, 3, 5 }, new[] { 2, 0, 5 } })
            {
                var input = new Tensor(Make(shape.Aggregate(1, (a, b) => a * b), -.13f), shape);
                Check(() => TensorOps.Scale(input, -.37f), input.Data.Select(value => value * -.37f).ToArray(), shape);
                foreach (bool silu in new[] { false, true })
                    Check(() => TensorOps.Activation(input, silu), input.Data.Select(value => (silu ? value : 1f) / (1f + MathF.Exp(-value))).ToArray(), shape);
            }
            foreach (int axis in new[] { 0, 1, 2 })
            {
                int[] aShape = [2, 3, 5], bShape = [2, 3, 5]; bShape[axis] = 1;
                var left = new Tensor(Make(aShape.Aggregate(1, (a, b) => a * b), .13f), aShape);
                var right = new Tensor(Make(bShape.Aggregate(1, (a, b) => a * b), .61f), bShape);
                int[] shape = (int[])aShape.Clone(); shape[axis] += bShape[axis];
                var expected = new float[shape.Aggregate(1, (a, b) => a * b)];
                for (int b = 0; b < shape[0]; b++) for (int c = 0; c < shape[1]; c++) for (int x = 0; x < shape[2]; x++)
                {
                    int[] coordinates = [b, c, x]; Tensor source = coordinates[axis] < aShape[axis] ? left : right;
                    if (ReferenceEquals(source, right)) coordinates[axis] -= aShape[axis];
                    expected[(b * shape[1] + c) * shape[2] + x] = source.Data[(coordinates[0] * source.Shape[1] + coordinates[1]) * source.Shape[2] + coordinates[2]];
                }
                Check(() => TensorOps.Concat([left, right], axis), expected, shape);
            }
            var empty = new Tensor(2, 0, 5);
            var full = new Tensor(Make(30, .29f), 2, 3, 5);
            Check(() => TensorOps.Concat([empty, full], 1), (float[])full.Data.Clone(), full.Shape);
            Check(() => TensorOps.Concat([empty, empty], 1), [], empty.Shape);
            Check(() => TensorOps.Arithmetic(empty, new Tensor([1f], []), 10), [], empty.Shape);
        }
        finally
        {
            release.Invoke(pool, null);
            type.GetMethod("Clear")!.Invoke(pool, null);
            current.SetValue(null, previous);
        }
    }

    /// <summary>生成有限的确定性正弦数据。</summary>
    /// <param name="length">目标数组的元素数量。</param>
    /// <param name="phase">正弦函数的相位。</param>
    /// <returns>指定长度的测试数组。</returns>
    private static float[] Make(int length, float phase)
        => Enumerable.Range(0, length).Select(index => MathF.Sin(index * .031f + phase) * .17f).ToArray();

    /// <summary>按 FP32 运算允许的舍入误差比较结果，并拒绝非有限输出。</summary>
    /// <param name="actual">待验证的实际结果。</param>
    /// <param name="expected">独立参考结果。</param>
    /// <exception cref="Exception">实际输出非有限或超出容差时。</exception>
    private static void Near(float actual, float expected)
    {
        if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > 2e-5f * MathF.Max(1, MathF.Abs(expected)))
            throw new Exception($"CPU buffer/cache result {actual} vs {expected}.");
    }
}
