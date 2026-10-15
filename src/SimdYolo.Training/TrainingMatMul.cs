using System.Numerics;
using System.Runtime.CompilerServices;

namespace SimdYolo.Training;

/// <summary>直接读取两侧连续行计算 dW，避免物化并转置整个 im2col 矩阵。</summary>
internal static class TrainingMatMul
{
    /// <summary>计算 C[M,N]=A[M,K]×B[N,K]ᵀ，使用 2×4 SIMD 点积微内核。</summary>
    /// <param name="a">左矩阵的行主序数据。</param><param name="b">尚未转置的右矩阵行数据。</param>
    /// <param name="c">覆盖写入的输出矩阵。</param><param name="m">左矩阵行数。</param>
    /// <param name="n">右矩阵行数。</param><param name="k">归约长度。</param><param name="parallel">会话线程预算。</param>
    /// <remarks>训练卷积已校验形状；每块独占两行输出，不存在跨线程梯度累加。</remarks>
    internal static unsafe void RightTransposed(float[] a, float[] b, float[] c, int m, int n, int k, ParallelOptions parallel)
    {
        void Rows(int block)
        {
            int row = block * 2;
            fixed (float* ap = a, bp = b, cp = c)
            {
                int column = 0;
                if (row + 1 < m) for (; column + 4 <= n; column += 4)
                {
                    var s00 = Vector<float>.Zero; var s01 = s00; var s02 = s00; var s03 = s00;
                    var s10 = s00; var s11 = s00; var s12 = s00; var s13 = s00;
                    int p = 0;
                    for (; p <= k - Vector<float>.Count; p += Vector<float>.Count)
                    {
                        var a0 = Unsafe.ReadUnaligned<Vector<float>>(ap + row * k + p); var a1 = Unsafe.ReadUnaligned<Vector<float>>(ap + (row + 1) * k + p);
                        var b0 = Unsafe.ReadUnaligned<Vector<float>>(bp + column * k + p); var b1 = Unsafe.ReadUnaligned<Vector<float>>(bp + (column + 1) * k + p);
                        var b2 = Unsafe.ReadUnaligned<Vector<float>>(bp + (column + 2) * k + p); var b3 = Unsafe.ReadUnaligned<Vector<float>>(bp + (column + 3) * k + p);
                        s00 += a0 * b0; s01 += a0 * b1; s02 += a0 * b2; s03 += a0 * b3;
                        s10 += a1 * b0; s11 += a1 * b1; s12 += a1 * b2; s13 += a1 * b3;
                    }
                    cp[row * n + column] = Vector.Sum(s00); cp[row * n + column + 1] = Vector.Sum(s01); cp[row * n + column + 2] = Vector.Sum(s02); cp[row * n + column + 3] = Vector.Sum(s03);
                    cp[(row + 1) * n + column] = Vector.Sum(s10); cp[(row + 1) * n + column + 1] = Vector.Sum(s11); cp[(row + 1) * n + column + 2] = Vector.Sum(s12); cp[(row + 1) * n + column + 3] = Vector.Sum(s13);
                    for (; p < k; p++) for (int j = 0; j < 4; j++) { cp[row * n + column + j] += ap[row * k + p] * bp[(column + j) * k + p]; cp[(row + 1) * n + column + j] += ap[(row + 1) * k + p] * bp[(column + j) * k + p]; }
                }
                for (int rr = row; rr < Math.Min(row + 2, m); rr++) for (int col = column; col < n; col++)
                {
                    int p = 0; var sum = Vector<float>.Zero;
                    for (; p <= k - Vector<float>.Count; p += Vector<float>.Count) sum += Unsafe.ReadUnaligned<Vector<float>>(ap + rr * k + p) * Unsafe.ReadUnaligned<Vector<float>>(bp + col * k + p);
                    float value = Vector.Sum(sum); for (; p < k; p++) value += ap[rr * k + p] * bp[col * k + p]; cp[rr * n + col] = value;
                }
            }
        }
        if ((long)m * n * k >= 262144 && parallel.MaxDegreeOfParallelism != 1) CpuScheduling.For(0, (m + 1) / 2, parallel, Rows);
        else for (int block = 0; block < (m + 1) / 2; block++) Rows(block);
    }
}
