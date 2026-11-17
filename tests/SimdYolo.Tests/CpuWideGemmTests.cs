using System.Numerics;
using System.Reflection;
using System.Runtime.Intrinsics;
using SimdYolo;

internal static unsafe class CpuWideGemmTests
{
    private delegate void WidePanel(float* a, float* b, float* c, int rows, int stride, int depth, bool packedRows);
    private delegate void PortablePanel(float[] a, float[] b, float[] c, int rows, int stride, int depth,
                                        int column, int take, int outputOffset);
    private delegate long Int8Dot(sbyte[] a, int aOffset, sbyte[] b, int bOffset, int length);

    internal static void Run(Action<string, Action> test)
    {
        test("CPU AVX-512 GEMM requires explicit opt-in", VerifyOptIn);
        test("Vector512 GEMM preserves FMA order, packed weights, depth blocks and output offsets", VerifyWide);
        test("Portable 16-column convolution panels preserve subpanel strides and tails", VerifyPortable);
        test("INT8 dot product preserves signed products, vector tails and block sums", VerifyInt8);
    }

    private static void VerifyOptIn()
    {
        using var backend = new CpuBackend(1);
        bool optedIn = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_AVX512") == "1";
        bool usesWideGemm = backend.Name == "CPU SIMD (AVX-512 GEMM)";
        if (usesWideGemm != (optedIn && Vector512.IsHardwareAccelerated))
            throw new Exception($"Unexpected SIMD dispatch for AVX-512 opt-in {optedIn}: {backend.Name}");
    }

    private static void VerifyWide()
    {
        var kernel = Load<WidePanel>("SimdYolo.CpuGemm512", "Multiply");
        foreach (var (rows, depth) in new[] { (1, 9), (5, 33), (6, 65), (37, 273) })
        {
            const int stride = 35, column = 7, offset = 13;
            float[] a = Values(rows * depth, 0.17f), b = Values(16 * depth, 0.37f);
            var packed = new float[(rows + 5) / 6 * depth * 6];
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < depth; k++)
                    packed[(r / 6 * depth + k) * 6 + r % 6] = a[r * depth + k];
            foreach (bool interleaved in new[] { false, true })
            {
                var output = Enumerable.Repeat(12345f, offset + rows * stride + 9).ToArray();
                fixed (float* left = interleaved ? packed : a, right = b, target = output)
                    kernel(left, right, target + offset + column, rows, stride, depth, interleaved);
                Check(a, b, output, rows, stride, depth, column, 16, offset, exactFma: true);
            }
        }
    }

    private static void VerifyPortable()
    {
        // The Arm64 entry point uses a 128-bit Vector<float>. A 256-bit test process
        // also exercises this API with one subpanel; 512-bit Vector<T> is outside its contract.
        if (Vector<float>.Count > 8) return;
        var kernel = Load<PortablePanel>("SimdYolo.CpuGemmPortable", "MultiplyPanel16");
        const int rows = 7, stride = 29, depth = 67, column = 3, offset = 5;
        float[] a = Values(rows * depth, 0.19f), b = Values(16 * depth, 0.39f);
        foreach (int take in new[] { 1, 4, 8, 11, 16 })
        {
            var output = Enumerable.Repeat(12345f, offset + rows * stride + 9).ToArray();
            kernel(a, b, output, rows, stride, depth, column, take, offset);
            Check(a, b, output, rows, stride, depth, column, take, offset, exactFma: false);
        }
    }

    private static float[] Values(int count, float frequency) =>
        Enumerable.Range(0, count).Select(i => MathF.Sin(i * frequency + 0.3f)).ToArray();

    private static void VerifyInt8()
    {
        var dot = Load<Int8Dot>("SimdYolo.PrecisionCpuBackend", "Dot");
        const int leftOffset = 3, rightOffset = 7;
        var a = new sbyte[200_032];
        var b = new sbyte[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = unchecked((sbyte)(i * 59));
            b[i] = unchecked((sbyte)(i * 73 + 127));
        }
        foreach (int count in new[] { 0, 1, 15, 16, 31, 32, 63, 64, 65, 4095, 4096, 4097 })
        {
            long expected = 0;
            for (int i = 0; i < count; i++) expected += a[i + leftOffset] * b[i + rightOffset];
            if (dot(a, leftOffset, b, rightOffset, count) != expected)
                throw new Exception($"Incorrect signed INT8 dot product for {count} elements.");
        }
        Array.Fill(a, sbyte.MinValue);
        Array.Fill(b, sbyte.MinValue);
        const int large = 200_013;
        if (dot(a, leftOffset, b, rightOffset, large) != (long)large * 128 * 128)
            throw new Exception("INT8 dot product overflowed before widening the accumulated result.");
    }

    private static T Load<T>(string type, string method) where T : Delegate =>
        typeof(Tensor).Assembly.GetType(type)!.GetMethod(method, BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<T>();

    private static void Check(float[] a, float[] b, float[] actual, int rows, int stride, int depth, int column,
                              int take, int offset, bool exactFma)
    {
        for (int index = 0; index < actual.Length; index++)
        {
            int position = index - offset;
            int row = position / stride, lane = position % stride - column;
            if (position < 0 || row >= rows || (uint)lane >= take)
            {
                if (actual[index] != 12345f) throw new Exception($"GEMM overwrote index {index} outside its panel.");
                continue;
            }
            float expected = 0;
            for (int k = 0; k < depth; k++)
                expected = MathF.FusedMultiplyAdd(a[row * depth + k], b[k * 16 + lane], expected);
            float tolerance = exactFma ? 0 : 1e-5f * MathF.Max(1, MathF.Abs(expected));
            if (!float.IsFinite(actual[index]) || MathF.Abs(actual[index] - expected) > tolerance)
                throw new Exception($"GEMM mismatch at {row}/{lane}: {actual[index]} vs {expected}.");
        }
    }
}
