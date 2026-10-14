using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>通过分块与 SIMD 位重排交换张量最后两个轴，保留 NaN 载荷和有符号零的位模式。</summary>
internal static class CpuTranspose
{
    /// <summary>是否启用 AVX 矩阵转置。</summary>
    private static readonly bool Simd = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_TRANSPOSE_SIMD") != "0" && Avx.IsSupported;
    /// <summary>交换最后两个轴，按批次平面分块转置，符合条件时使用 SIMD 位重排。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="shape">转置后的目标形状。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    public static Tensor LastTwo(Tensor x, int[] shape)
    {
        if (x.Length == 0) return Tensor.TemporaryUninitialized(shape);
        int rows = x.Shape[^2], columns = x.Shape[^1], planes = x.Length / (rows * columns);
        var output = Tensor.TemporaryUninitialized(shape); float[] source = x.Data, target = output.Data;
        // 分块限制目标工作集大小，适用于注意力层的大型通道/空间矩阵。
        // 块内访问同时避免逐元素整数除法。
        // 转置一个批次矩阵平面；选用 SIMD 或有界工作集的标量分块拷贝。
        // 参数 p：当前批次矩阵平面的索引。
        void Plane(int p)
        {
            int start = p * rows * columns;
            if (Simd)
            {
                PlaneSimd(source, target, start, rows, columns);
                return;
            }
            for (int y = 0; y < rows; y += 32) for (int xx = 0; xx < columns; xx += 32)
                for (int r = y; r < Math.Min(y + 32, rows); r++)
                    for (int c = xx; c < Math.Min(xx + 32, columns); c++) target[start + c * rows + r] = source[start + r * columns + c];
        }
        if (planes >= 4 && x.Length > 65536) CpuScheduling.For(0, planes, CpuScheduling.Options, Plane);
        else for (int p = 0; p < planes; p++) Plane(p);
        return output;
    }

    /// <summary>以 8×8 AVX 子块转置一个矩阵平面，剩余行列逐元素拷贝且不改变浮点位模式。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="target">接收当前面板结果的预分配 FP32 缓冲区。</param>
    /// <param name="start">当前批次平面在源和目标数组中的起始元素偏移。</param>
    /// <param name="rows">矩阵逻辑行数。</param>
    /// <param name="columns">矩阵逻辑列数。</param>
    private static unsafe void PlaneSimd(float[] source, float[] target, int start, int rows, int columns)
    {
        fixed (float* input = source, output = target)
        {
            // 保留原有 32 列工作集边界。
            // unpack/shuffle 仅搬运位模式，保留 NaN 载荷与有符号零，不引入浮点运算。
            for (int by = 0; by < rows; by += 32)
                for (int bx = 0; bx < columns; bx += 32)
                {
                    int endY = Math.Min(by + 32, rows), endX = Math.Min(bx + 32, columns), y = by;
                    for (; y <= endY - 8; y += 8)
                    {
                        int x = bx;
                        for (; x <= endX - 8; x += 8)
                        {
                            float* src = input + start + y * columns + x;
                            var r0 = Avx.LoadVector256(src);
                            var r1 = Avx.LoadVector256(src + columns);
                            var r2 = Avx.LoadVector256(src + columns * 2);
                            var r3 = Avx.LoadVector256(src + columns * 3);
                            var r4 = Avx.LoadVector256(src + columns * 4);
                            var r5 = Avx.LoadVector256(src + columns * 5);
                            var r6 = Avx.LoadVector256(src + columns * 6);
                            var r7 = Avx.LoadVector256(src + columns * 7);
                            var t0 = Avx.UnpackLow(r0, r1);
                            var t1 = Avx.UnpackHigh(r0, r1);
                            var t2 = Avx.UnpackLow(r2, r3);
                            var t3 = Avx.UnpackHigh(r2, r3);
                            var t4 = Avx.UnpackLow(r4, r5);
                            var t5 = Avx.UnpackHigh(r4, r5);
                            var t6 = Avx.UnpackLow(r6, r7);
                            var t7 = Avx.UnpackHigh(r6, r7);
                            var s0 = Avx.Shuffle(t0, t2, 0x44);
                            var s1 = Avx.Shuffle(t0, t2, 0xee);
                            var s2 = Avx.Shuffle(t1, t3, 0x44);
                            var s3 = Avx.Shuffle(t1, t3, 0xee);
                            var s4 = Avx.Shuffle(t4, t6, 0x44);
                            var s5 = Avx.Shuffle(t4, t6, 0xee);
                            var s6 = Avx.Shuffle(t5, t7, 0x44);
                            var s7 = Avx.Shuffle(t5, t7, 0xee);
                            float* dst = output + start + x * rows + y;
                            Avx.Store(dst, Avx.Permute2x128(s0, s4, 0x20));
                            Avx.Store(dst + rows, Avx.Permute2x128(s1, s5, 0x20));
                            Avx.Store(dst + rows * 2, Avx.Permute2x128(s2, s6, 0x20));
                            Avx.Store(dst + rows * 3, Avx.Permute2x128(s3, s7, 0x20));
                            Avx.Store(dst + rows * 4, Avx.Permute2x128(s0, s4, 0x31));
                            Avx.Store(dst + rows * 5, Avx.Permute2x128(s1, s5, 0x31));
                            Avx.Store(dst + rows * 6, Avx.Permute2x128(s2, s6, 0x31));
                            Avx.Store(dst + rows * 7, Avx.Permute2x128(s3, s7, 0x31));
                        }
                        for (; x < endX; x++)
                            for (int r = y; r < y + 8; r++) output[start + x * rows + r] = input[start + r * columns + x];
                    }
                    for (; y < endY; y++)
                        for (int x = bx; x < endX; x++) output[start + x * rows + y] = input[start + y * columns + x];
                }
        }
    }
}
