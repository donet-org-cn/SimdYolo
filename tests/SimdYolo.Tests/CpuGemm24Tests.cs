using System.Reflection;
using System.Runtime.Intrinsics.X86;
using SimdYolo;

/// <summary>提供四行二十四列矩阵微内核的偏移、行尾、深度分块和缓存布局的回归验证。</summary>
internal static class CpuGemm24Tests
{
    /// <summary>向测试执行器注册四行二十四列矩阵微内核的偏移、行尾、深度分块和缓存布局的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU four-row/24-column GEMM vs double scalar: offsets, tails, deep K and cache layout", () =>
        {
            if (!Avx.IsSupported || !Fma.IsSupported) return;
            foreach (var (m, n, k) in new[]
            {
                (1, 1, 1), (4, 19, 7), (17, 25, 21), (32, 64, 128),
                (37, 73, 513), (35, 56, 385)
            })
            {
                float[] a = Make(m * k, .1f), b = Make(k * n, .4f);
                Check(a, b, m, n, k);
                b[0] -= .15f;
                Check(a, b, m, n, k);
            }
            float[] shared = Make(37 * 273, .3f);
            object first = Prepare(shared, 37, 273);
            if (!ReferenceEquals(first, Prepare(shared, 37, 273))) throw new Exception("Four-row weight pack was not cached.");
            Check(shared, Make(273 * 25, .4f), 37, 25, 273);
            Check(shared, Make(37 * 25, .4f), 273, 25, 37);
            Check(shared, Make(273 * 25, .4f), 37, 25, 273);
        });
        test("CPU four-row/24-column GEMM preserves zero K and rejects invalid panel bounds", () =>
        {
            if (!Avx.IsSupported || !Fma.IsSupported) return;
            object zero = Prepare([], 3, 0);
            float[] output = Enumerable.Repeat(42f, 23 + 3 * 7 + 29).ToArray();
            Multiply(zero, [], output, 7, 2, 4, 0, 23);
            for (int i = 0; i < output.Length; i++)
            {
                int logical = i - 23;
                bool cleared = logical >= 0 && logical < 21 && logical % 7 is >= 2 and < 6;
                if (output[i] != (cleared ? 0 : 42)) throw new Exception("Four-row GEMM zero-K output mismatch.");
            }
            object weights = Prepare(Make(12, .3f), 3, 4);
            foreach (var (n, col, take, bo, co) in new[]
            {
                (7, 0, 8, 0, 0), (7, -1, 1, 0, 0), (25, 0, 25, 0, 0),
                (7, 0, 7, 1, 0), (7, 0, 7, 0, 1), (7, 0, 7, -1, 0)
            })
            {
                try { Multiply(weights, new float[96], new float[21], n, col, take, bo, co); }
                catch (TargetInvocationException error) when (error.InnerException is ArgumentException) { continue; }
                throw new Exception("Four-row GEMM accepted invalid panel bounds.");
            }
        });
    }

    private static readonly Type Type = typeof(Tensor).Assembly.GetType("SimdYolo.CpuPointwise24")!;
    private static readonly MethodInfo PrepareMethod = Type.GetMethod("Prepare", BindingFlags.Static | BindingFlags.NonPublic)!;
    private static readonly MethodInfo MultiplyMethod = Type.GetMethod("Multiply", BindingFlags.Static | BindingFlags.NonPublic)!;
    /// <summary>通过反射调用内部权重打包器，准备可复用的矩阵测试对象。</summary>
    /// <param name="a">按行保存的权重矩阵 A。</param>
    /// <param name="m">矩阵输出行数。</param>
    /// <param name="k">矩阵归约深度。</param>
    /// <returns>准备完成的内部权重打包矩阵。</returns>
    private static object Prepare(float[] a, int m, int k) => PrepareMethod.Invoke(null, [a, m, k])!;
    /// <summary>通过反射运行矩阵计算块，覆盖输入与输出偏移以及列尾部。</summary>
    /// <param name="weights">待验证的权重数据或内部打包矩阵。</param>
    /// <param name="input">待验证的输入数据。</param>
    /// <param name="output">用于接收计算结果的预分配缓冲区。</param>
    /// <param name="n">矩阵输出列数。</param>
    /// <param name="column">当前计算块在输出矩阵中的起始列。</param>
    /// <param name="take">当前计算块实际处理的列或 tile 数。</param>
    /// <param name="bOffset">打包输入起始位置，以浮点元素计。</param>
    /// <param name="cOffset">输出起始位置，以浮点元素计。</param>
    private static void Multiply(object weights, float[] input, float[] output, int n, int column, int take, int bOffset, int cOffset)
        => MultiplyMethod.Invoke(null, [weights, input, output, n, column, take, bOffset, cOffset]);
    /// <summary>生成指定尺寸和相位的确定性正弦测试数据。</summary>
    /// <param name="length">需要生成的浮点元素数量。</param>
    /// <param name="phase">确定性正弦测试数据的相位偏移。</param>
    /// <returns>生成的测试数据或参考变换坐标。</returns>
    private static float[] Make(int length, float phase)
        => Enumerable.Range(0, length).Select(i => MathF.Sin(i * .031f + phase) * .2f).ToArray();
    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="a">按行保存的权重矩阵 A。</param>
    /// <param name="b">按行保存的矩阵输入 B。</param>
    /// <param name="m">矩阵输出行数。</param>
    /// <param name="n">矩阵输出列数。</param>
    /// <param name="k">矩阵归约深度。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Check(float[] a, float[] b, int m, int n, int k)
    {
        const int bOffset = 17, cOffset = 23;
        object weights = Prepare(a, m, k);
        float[] output = Enumerable.Repeat(42f, cOffset + m * n + 29).ToArray();
        float[] input = Enumerable.Repeat(123f, bOffset + k * 24 + 31).ToArray();
        for (int column = 0; column < n; column += 24)
        {
            int take = Math.Min(24, n - column);
            for (int p = 0; p < k; p++) for (int j = 0; j < 24; j++)
                input[bOffset + p * 24 + j] = j < take ? b[p * n + column + j] : 0;
            Multiply(weights, input, output, n, column, take, bOffset, cOffset);
        }
        for (int r = 0; r < m; r++) for (int c = 0; c < n; c++)
        {
            double expected = 0;
            for (int p = 0; p < k; p++) expected += (double)a[r * k + p] * b[p * n + c];
            float actual = output[cOffset + r * n + c];
            if (!float.IsFinite(actual) || MathF.Abs(actual - (float)expected) > 1e-4f * MathF.Max(1, MathF.Abs((float)expected)))
                throw new Exception($"Four-row GEMM mismatch ({m},{n},{k})[{r},{c}]: {actual} vs {expected}");
        }
        for (int i = 0; i < cOffset; i++) if (output[i] != 42) throw new Exception("Four-row GEMM overwrote prefix.");
        for (int i = cOffset + m * n; i < output.Length; i++) if (output[i] != 42) throw new Exception("Four-row GEMM overwrote suffix.");
    }
}
