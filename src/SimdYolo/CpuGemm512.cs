using System.Runtime.Intrinsics;

namespace SimdYolo;

internal static unsafe class CpuGemm512
{
    private const int Columns = 16;
    private const int Rows = 6;
    private static readonly bool enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_AVX512") == "1";

    internal static bool Enabled => Vector512.IsHardwareAccelerated && enabled;

    // The existing packed panel is exactly one 512-bit vector per reduction step.
    // The same kernel handles row-major A and the six-row interleaved constant weights.
    internal static void Multiply(float* a, float* b, float* c, int m, int n, int k, bool packedRows)
    {
        int rowStride = packedRows ? 1 : k;
        int reductionStride = packedRows ? Rows : 1;
        int depth = k >= 256 && m >= 32 ? 128 : k;
        for (int begin = 0; begin < k; begin += depth)
        {
            int end = Math.Min(k, begin + depth);
            int row = 0;
            for (; row <= m - Rows; row += Rows)
            {
                float* output = c + row * n;
                var s0 = begin == 0 ? Vector512<float>.Zero : Vector512.Load(output);
                var s1 = begin == 0 ? Vector512<float>.Zero : Vector512.Load(output + n);
                var s2 = begin == 0 ? Vector512<float>.Zero : Vector512.Load(output + 2 * n);
                var s3 = begin == 0 ? Vector512<float>.Zero : Vector512.Load(output + 3 * n);
                var s4 = begin == 0 ? Vector512<float>.Zero : Vector512.Load(output + 4 * n);
                var s5 = begin == 0 ? Vector512<float>.Zero : Vector512.Load(output + 5 * n);
                float* left = a + row * k + begin * reductionStride;
                float* right = b + begin * Columns;
                for (int p = begin; p < end; p++, left += reductionStride, right += Columns)
                {
                    var rhs = Vector512.Load(right);
                    s0 = Vector512.FusedMultiplyAdd(Vector512.Create(left[0]), rhs, s0);
                    s1 = Vector512.FusedMultiplyAdd(Vector512.Create(left[rowStride]), rhs, s1);
                    s2 = Vector512.FusedMultiplyAdd(Vector512.Create(left[2 * rowStride]), rhs, s2);
                    s3 = Vector512.FusedMultiplyAdd(Vector512.Create(left[3 * rowStride]), rhs, s3);
                    s4 = Vector512.FusedMultiplyAdd(Vector512.Create(left[4 * rowStride]), rhs, s4);
                    s5 = Vector512.FusedMultiplyAdd(Vector512.Create(left[5 * rowStride]), rhs, s5);
                }
                s0.Store(output);
                s1.Store(output + n);
                s2.Store(output + 2 * n);
                s3.Store(output + 3 * n);
                s4.Store(output + 4 * n);
                s5.Store(output + 5 * n);
            }
            float* lastBlock = a + row * k;
            for (int remaining = 0; row + remaining < m; remaining++)
            {
                float* output = c + (row + remaining) * n;
                var sum = begin == 0 ? Vector512<float>.Zero : Vector512.Load(output);
                float* left = lastBlock + remaining * rowStride + begin * reductionStride;
                float* right = b + begin * Columns;
                for (int p = begin; p < end; p++, left += reductionStride, right += Columns)
                    sum = Vector512.FusedMultiplyAdd(Vector512.Create(*left), Vector512.Load(right), sum);
                sum.Store(output);
            }
        }
    }
}
