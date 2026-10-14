using System.Reflection;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>验证 CPU 批次矩阵乘法的广播与共享数组偏移、独立结果所有权及原有 FP32 归约次序。</summary>
internal static class CpuMatMulOffsetsTests
{
    /// <summary>注册偏移批次 GEMM 的数值、范围、公开生命周期和自定义后端契约用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU offset batched MatMul preserves broadcast, tails, zero dimensions, aliases and changing inputs", Broadcast);
        test("CPU offset GEMM validates ranges and preserves guarded packed panel writes", Ranges);
        test("CPU offset MatMul preserves pooled public ownership and custom backend contracts", Ownership);
    }

    /// <summary>通过独立标量与逐批旧后端调用检查广播、非整块、零维度、输入别名和数据更新。</summary>
    /// <exception cref="Exception">输出形状、FP32 结果或独立参考结果不符合预期时。</exception>
    private static void Broadcast()
    {
        using var backend = new CpuBackend(3); var ops = new TensorOps(backend);
        foreach (var (aShape, bShape) in new (int[], int[])[]
        {
            ([3, 5], [5, 19]), ([2, 3, 5], [1, 5, 17]), ([2, 1, 3, 7], [1, 4, 7, 19]),
            ([1, 3, 6, 13], [2, 1, 13, 33]), ([2, 2, 32, 257], [1, 2, 257, 35]),
            ([2, 1, 7, 19], [1, 3, 19, 5]), ([2, 3, 0], [1, 0, 19]),
            ([2, 0, 7], [1, 7, 19]), ([2, 3, 7], [1, 7, 0]), ([0, 3, 7], [1, 7, 19])
        })
        {
            var a = new Tensor(Make(Count(aShape), .19f), aShape);
            var b = new Tensor(Make(Count(bShape), -.31f), bShape);
            for (int pass = 0; pass < 2; pass++)
            {
                var actual = ops.MatMul(a, b); Check(backend, a, b, actual);
                if (a.Length > 0) a.Data[0] += .027f;
                if (b.Length > 0) b.Data[b.Length - 1] -= .019f;
            }
        }
        float[] shared = Make(18, .57f);
        var aliasA = new Tensor(shared, 2, 3, 3); var aliasB = new Tensor(shared, 1, 2, 3, 3);
        Check(backend, aliasA, aliasB, ops.MatMul(aliasA, aliasB));
        if (!ReferenceEquals(aliasA.Data, aliasB.Data)) throw new Exception("Input alias test did not share its array.");
    }

    /// <summary>检查低层批次偏移的边界、重复源引用、零 K，以及 AVX 左/右/输出偏移的保护区。</summary>
    /// <exception cref="Exception">范围错误未拒绝、保护区被覆盖或矩阵结果错误时。</exception>
    private static void Ranges()
    {
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuGemm")!;
        MethodInfo batch = type.GetMethod("MultiplyOffsetsBatch", BindingFlags.NonPublic | BindingFlags.Static)!;
        const int m = 7, n = 19, k = 13;
        float[] a = Make(2 * m * k + 27, .11f), b = Make(2 * k * n + 29, -.17f);
        int[] ao = [5, m * k + 12, 5], bo = [7, k * n + 18, k * n + 18];
        float[] output = Enumerable.Repeat(12345f, ao.Length * m * n + 13).ToArray();
        batch.Invoke(null, [a, b, output, ao, bo, m, n, k, new ParallelOptions { MaxDegreeOfParallelism = 2 }]);
        for (int t = 0; t < ao.Length; t++) for (int row = 0; row < m; row++) for (int column = 0; column < n; column++)
        {
            double expected = 0;
            for (int p = 0; p < k; p++) expected += (double)a[ao[t] + row * k + p] * b[bo[t] + p * n + column];
            Near(output[(t * m + row) * n + column], (float)expected);
        }
        if (output.AsSpan(ao.Length * m * n).ContainsAnyExcept(12345f)) throw new Exception("Offset batch overwrote trailing output guard.");
        foreach (object[] arguments in new[]
        {
            new object[] { a, b, output, new[] { -1 }, new[] { 0 }, m, n, k, new ParallelOptions() },
            [a, b, output, new[] { a.Length }, new[] { 0 }, m, n, k, new ParallelOptions()],
            [a, b, output, new[] { 0 }, Array.Empty<int>(), m, n, k, new ParallelOptions()],
            [a, b, new float[1], new[] { 0 }, new[] { 0 }, m, n, k, new ParallelOptions()],
            [a, a, a, new[] { 0 }, new[] { 0 }, 3, 3, 3, new ParallelOptions()]
        })
            Throws<ArgumentException>(() => batch.Invoke(null, arguments));
        Throws<OverflowException>(() => batch.Invoke(null, [a, b, output, new[] { 0 }, new[] { 0 }, int.MaxValue, n, 2, new ParallelOptions()]));
        Array.Fill(output, 12345f);
        batch.Invoke(null, [Array.Empty<float>(), Array.Empty<float>(), output, new[] { 0, 0 }, new[] { 0, 0 }, 3, 7, 0, new ParallelOptions()]);
        for (int index = 0; index < 42; index++) if (BitConverter.SingleToInt32Bits(output[index]) != 0) throw new Exception("Zero K batch did not write positive zero.");
        if (output.AsSpan(42).ContainsAnyExcept(12345f)) throw new Exception("Zero K batch overwrote output guard.");
        if (!Avx.IsSupported || !Fma.IsSupported) return;
        MethodInfo avx = type.GetMethod("MultiplyAvx", BindingFlags.NonPublic | BindingFlags.Static)!;
        foreach (var (rows, depth, column, take) in new[] { (32, 257, 2, 16), (7, 19, 3, 7), (3, 0, 1, 4) })
        {
            const int width = 21, aOffset = 9, bOffset = 7, cOffset = 11;
            float[] left = Make(aOffset + rows * depth + 13, -.23f), packed = Make(bOffset + depth * 16 + 13, .41f);
            float[] target = Enumerable.Repeat(12345f, cOffset + rows * width + 13).ToArray();
            avx.Invoke(null, [left, packed, target, rows, width, depth, column, take, bOffset, cOffset, aOffset]);
            for (int row = 0; row < rows; row++) for (int x = 0; x < width; x++)
            {
                int index = cOffset + row * width + x;
                if (x < column || x >= column + take)
                {
                    if (target[index] != 12345f) throw new Exception("AVX panel overwrote a neighboring column.");
                    continue;
                }
                float expected = 0;
                for (int p = 0; p < depth; p++) expected = MathF.FusedMultiplyAdd(left[aOffset + row * depth + p], packed[bOffset + p * 16 + x - column], expected);
                if (BitConverter.SingleToInt32Bits(target[index]) != BitConverter.SingleToInt32Bits(expected)) throw new Exception("Offset AVX panel changed FP32 FMA order.");
            }
            if (target.AsSpan(0, cOffset).ContainsAnyExcept(12345f) || target.AsSpan(cOffset + rows * width).ContainsAnyExcept(12345f)) throw new Exception("AVX panel overwrote prefix/suffix guards.");
        }
    }

    /// <summary>验证结果脱离临时池后仍稳定，并确认自定义后端仍收到精确长度的逐批独立矩阵。</summary>
    /// <exception cref="Exception">公开结果被下一次调用覆盖，或自定义后端输入合同改变时。</exception>
    private static void Ownership()
    {
        Type poolType = typeof(Tensor).Assembly.GetType("SimdYolo.CpuTensorPool")!;
        FieldInfo current = poolType.GetField("Current", BindingFlags.Static | BindingFlags.NonPublic)!;
        object? previous = current.GetValue(null); object pool = Activator.CreateInstance(poolType, nonPublic: true)!;
        current.SetValue(null, pool);
        try
        {
            using var cpu = new CpuBackend(2); var ops = new TensorOps(cpu);
            var a = new Tensor(Make(2 * 3 * 5, .31f), 2, 3, 5);
            var b = new Tensor(Make(5 * 7, .61f), 1, 5, 7);
            var first = ops.MatMul(a, b);
            first = ((Tensor[])poolType.GetMethod("DetachOutputs")!.Invoke(pool, [new[] { first }])!)[0];
            float[] snapshot = (float[])first.Data.Clone(); poolType.GetMethod("ReturnAll")!.Invoke(pool, null);
            a.Data[0] += .3f;
            var second = ops.MatMul(a, b); Check(cpu, a, b, second);
            if (!first.Data.SequenceEqual(snapshot) || ReferenceEquals(first.Data, second.Data)) throw new Exception("Offset MatMul reused a published output array.");
            using var custom = new ExactBufferBackend(); var customOutput = new TensorOps(custom).MatMul(a, b);
            if (custom.Calls != 2) throw new Exception("Custom backend no longer receives one call per logical batch.");
            Check(cpu, a, b, customOutput, exact: false);
        }
        finally
        {
            poolType.GetMethod("ReturnAll")!.Invoke(pool, null); poolType.GetMethod("Clear")!.Invoke(pool, null); current.SetValue(null, previous);
        }
    }

    /// <summary>以独立坐标映射和逐批旧 MatMul 合同计算参考，同时检查 double 标量误差。</summary>
    /// <param name="cpu">用于逐批旧矩阵接口的 CPU 后端。</param>
    /// <param name="a">左输入张量。</param>
    /// <param name="b">右输入张量。</param>
    /// <param name="actual">待验证的输出。</param>
    /// <param name="exact">是否要求与原有 CPU 微内核逐元素位相同。</param>
    /// <exception cref="Exception">形状或结果不满足参考时。</exception>
    private static void Check(CpuBackend cpu, Tensor a, Tensor b, Tensor actual, bool exact = true)
    {
        int m = a.Shape[^2], k = a.Shape[^1], n = b.Shape[^1];
        int[] batch = TensorOps.BroadcastShape(a.Shape[..^2], b.Shape[..^2]), shape = [.. batch, m, n];
        if (!actual.Shape.SequenceEqual(shape)) throw new Exception("Offset MatMul output shape mismatch.");
        for (int t = 0; t < Count(batch); t++)
        {
            int ai = SourceOffset(t, batch, a.Shape[..^2], m * k), bi = SourceOffset(t, batch, b.Shape[..^2], k * n);
            float[] left = a.Data.AsSpan(ai, m * k).ToArray(), right = b.Data.AsSpan(bi, k * n).ToArray(), reference = new float[m * n];
            cpu.MatMul(left, right, reference, m, n, k);
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++)
            {
                int index = row * n + column; float value = actual.Data[t * m * n + index];
                double expected = 0;
                for (int p = 0; p < k; p++) expected += (double)left[row * k + p] * right[p * n + column];
                Near(value, (float)expected);
                if (exact && BitConverter.SingleToInt32Bits(value) != BitConverter.SingleToInt32Bits(reference[index])) throw new Exception("Offset MatMul changed the original FP32 accumulation.");
            }
        }
    }

    /// <summary>从输出批次坐标独立构造源矩阵编号，大小为 1 的轴映射到零。</summary>
    /// <param name="flat">输出批次的扁平编号。</param>
    /// <param name="batch">输出批次形状。</param>
    /// <param name="source">源批次形状。</param>
    /// <param name="matrixSize">单个源矩阵的元素数。</param>
    /// <returns>对应源矩阵的元素起始偏移。</returns>
    private static int SourceOffset(int flat, int[] batch, int[] source, int matrixSize)
    {
        var coordinates = new int[batch.Length];
        for (int axis = batch.Length - 1; axis >= 0; axis--) { coordinates[axis] = flat % batch[axis]; flat /= batch[axis]; }
        int index = 0;
        for (int axis = 0; axis < source.Length; axis++) index = index * source[axis] + (source[axis] == 1 ? 0 : coordinates[batch.Length - source.Length + axis]);
        return index * matrixSize;
    }

    /// <summary>计算测试形状的元素数，空形状代表一个标量。</summary>
    /// <param name="shape">合法且较小的测试形状。</param>
    /// <returns>各维度乘积。</returns>
    private static int Count(int[] shape) => shape.Aggregate(1, (left, right) => left * right);

    /// <summary>生成小幅度有限的确定性测试数据。</summary>
    /// <param name="length">数组元素数量。</param>
    /// <param name="phase">正弦相位。</param>
    /// <returns>指定长度的独立数组。</returns>
    private static float[] Make(int length, float phase) => Enumerable.Range(0, length).Select(index => MathF.Sin(index * .031f + phase) * .07f).ToArray();

    /// <summary>比较独立标量参考，拒绝非有限输出。</summary>
    /// <param name="actual">实际结果。</param>
    /// <param name="expected">参考结果。</param>
    /// <exception cref="Exception">结果超出 FP32 合理舍入范围时。</exception>
    private static void Near(float actual, float expected)
    {
        if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > 2e-5f * MathF.Max(1, MathF.Abs(expected))) throw new Exception($"Offset MatMul {actual} vs {expected}.");
    }

    /// <summary>核对反射调用中传播的实际异常类型。</summary>
    /// <typeparam name="T">应当出现的异常类型。</typeparam>
    /// <param name="action">应当失败的调用。</param>
    /// <exception cref="Exception">调用成功或异常类型错误时。</exception>
    private static void Throws<T>(Action action) where T : Exception
    {
        try { action(); }
        catch (TargetInvocationException error) when (error.InnerException is T) { return; }
        throw new Exception("Offset GEMM failed to reject an invalid argument: " + typeof(T).Name);
    }

    /// <summary>检查公开自定义后端仍收到精确矩阵数组，不要求其支持共享数组偏移。</summary>
    private sealed class ExactBufferBackend : IComputeBackend
    {
        /// <summary>逐批调用次数。</summary>
        /// <value>已接收的逻辑批次数量。</value>
        internal int Calls { get; private set; }
        /// <summary>测试后端名称。</summary>
        /// <value>用于识别精确缓冲区合同测试的固定名称。</value>
        public string Name => "exact-buffer fixture";
        /// <summary>检查缓冲区尺寸并使用 double 标量计算参考结果。</summary>
        /// <param name="a">精确长度的左矩阵。</param>
        /// <param name="b">精确长度的右矩阵。</param>
        /// <param name="c">精确长度的目标矩阵。</param>
        /// <param name="m">矩阵行数。</param>
        /// <param name="n">矩阵列数。</param>
        /// <param name="k">归约深度。</param>
        /// <exception cref="Exception">自定义后端的精确数组合同改变时。</exception>
        public void MatMul(float[] a, float[] b, float[] c, int m, int n, int k)
        {
            if (a.Length != m * k || b.Length != k * n || c.Length != m * n) throw new Exception("Custom backend array contract changed.");
            Calls++;
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++)
            {
                double value = 0;
                for (int p = 0; p < k; p++) value += (double)a[row * k + p] * b[p * n + column];
                c[row * n + column] = (float)value;
            }
        }
        /// <summary>测试后端没有待释放资源。</summary>
        public void Dispose() { }
    }
}
