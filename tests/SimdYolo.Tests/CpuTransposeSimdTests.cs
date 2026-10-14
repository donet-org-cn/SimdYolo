using System.Reflection;
using SimdYolo;

/// <summary>提供八乘八 SIMD 转置的逐位复制、分块边界和奇数尾部的回归验证。</summary>
internal static class CpuTransposeSimdTests
{
    /// <summary>向测试执行器注册八乘八 SIMD 转置的逐位复制、分块边界和奇数尾部的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU AVX transpose preserves every FP32 bit, panel borders, odd tails and prefix planes", () =>
        {
            if (!System.Runtime.Intrinsics.X86.Avx.IsSupported) return;
            var transpose = typeof(Tensor).Assembly.GetType("SimdYolo.CpuTranspose")!.GetMethod("PlaneSimd", BindingFlags.Static | BindingFlags.NonPublic)!;
            var random = new Random(17321);
            foreach (var (rows, columns) in new[] { (1, 1), (3, 9), (8, 8), (9, 17), (17, 7), (20, 20), (31, 33), (32, 32), (33, 65), (16, 400), (400, 400) })
            {
                const int planes = 2, prefix = 3, suffix = 7;
                int count = rows * columns * planes;
                var source = new float[prefix + count + suffix];
                var target = Enumerable.Repeat(BitConverter.Int32BitsToSingle(0x7fc54321), source.Length).ToArray();
                for (int i = 0; i < source.Length; i++)
                {
                    int bits = random.Next(int.MinValue, int.MaxValue);
                    bits = (i % 13) switch { 0 => 0, 1 => int.MinValue, 2 => 1, 3 => unchecked((int)0x80000001), 4 => 0x7f800000, 5 => unchecked((int)0xff800000), 6 => 0x7fc00001 + i, 7 => 0x7f800001 + i, _ => bits };
                    source[i] = BitConverter.Int32BitsToSingle(bits);
                }
                for (int frame = 0; frame < 2; frame++)
                {
                    if (frame != 0) source[prefix] = BitConverter.Int32BitsToSingle(unchecked((int)0xffc76543));
                    for (int p = 0; p < planes; p++) transpose.Invoke(null, [source, target, prefix + p * rows * columns, rows, columns]);
                    for (int p = 0; p < planes; p++) for (int y = 0; y < rows; y++) for (int x = 0; x < columns; x++)
                    {
                        int src = prefix + p * rows * columns + y * columns + x, dst = prefix + p * rows * columns + x * rows + y;
                        if (BitConverter.SingleToInt32Bits(source[src]) != BitConverter.SingleToInt32Bits(target[dst])) throw new Exception($"AVX transpose {rows}x{columns} changed bits at ({y},{x}).");
                    }
                    for (int i = 0; i < prefix; i++) if (BitConverter.SingleToInt32Bits(target[i]) != 0x7fc54321) throw new Exception("Transpose wrote before its plane.");
                    for (int i = target.Length - suffix; i < target.Length; i++) if (BitConverter.SingleToInt32Bits(target[i]) != 0x7fc54321) throw new Exception("Transpose wrote after its plane.");
                }
            }
        });
        test("CPU last-axis transpose dispatch preserves prefix dimensions and input mutation", () =>
        {
            foreach (int planes in new[] { 2, 6 })
            {
                const int rows = 17, columns = 33;
                var source = new Tensor(Enumerable.Range(0, planes * rows * columns).Select(i => (float)i).ToArray(), 1, planes, rows, columns);
                for (int frame = 0; frame < 2; frame++)
                {
                    if (frame != 0) source.Data[0] = -1234;
                    Tensor target = TensorOps.Transpose(source, [0, 1, 3, 2]);
                    if (!target.Shape.SequenceEqual([1, planes, columns, rows])) throw new Exception("Transpose prefix shape changed.");
                    for (int p = 0; p < planes; p++) for (int y = 0; y < rows; y++) for (int x = 0; x < columns; x++)
                        if (target.Data[(p * columns + x) * rows + y] != source.Data[(p * rows + y) * columns + x]) throw new Exception("Transpose prefix offset or mutable-input behavior changed.");
                }
            }
        });
    }
}
