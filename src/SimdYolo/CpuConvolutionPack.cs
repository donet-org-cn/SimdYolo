using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

// 沿用 16 列的隐式 GEMM 布局，相邻核列复用同一输入向量的偶数和奇数采样。
// 本步骤只搬运数据，不引入新的变换、乘法或通道归约次序。
/// <summary>打包 3×3、步长为 2 的卷积输入，通过复用相邻核列的偶数和奇数采样降低读取成本。</summary>
internal static class CpuConvolutionPack
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_STRIDE2_PACK") != "0";

    /// <summary>检查是否启用 3×3、步长为 2、膨胀为 1 的专用打包路径。</summary>
    /// <param name="kh">卷积核高度。</param>
    /// <param name="kw">卷积核宽度。</param>
    /// <param name="stride">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    /// <returns>满足当前快路径的形状、模式和硬件要求时为 true；否则为 false。</returns>
    internal static bool ShouldUse(int kh, int kw, int[] stride, int[] pads, int[] dilation)
        => Enabled && Avx2.IsSupported && kh == 3 && kw == 3 &&
           stride[0] == 2 && stride[1] == 2 && dilation[0] == 1 && dilation[1] == 1 &&
           pads[0] is >= 0 and <= 1 && pads[1] is >= 0 and <= 1 &&
           pads[2] is >= 0 and <= 1 && pads[3] is >= 0 and <= 1;

    /// <summary>按卷积空间面板组织输入采样，复用相邻核列的偶数/奇数向量，并清零无效位置。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="batchStart">当前批次在输入数组中的起始偏移。</param>
    /// <param name="ic">输入通道数。</param>
    /// <param name="ih">输入平面高度。</param>
    /// <param name="iw">输入平面宽度。</param>
    /// <param name="ow">输出平面宽度。</param>
    /// <param name="start">当前空间面板的起始位置；out 参数返回领取区间的起点。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="padY">输入顶部填充的行数。</param>
    /// <param name="padX">输入左侧填充的列数。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static unsafe void Pack(float[] input, float[] packed, int batchStart, int ic, int ih, int iw,
                                     int ow, int start, int take, int padY, int padX)
    {
        if (batchStart < 0 || ic <= 0 || ih <= 0 || iw <= 0 || ow <= 0 || ow > (iw + 1L) / 2 || start < 0 || take < 0 || take > 16 ||
            padY is < 0 or > 1 || padX is < 0 or > 1 ||
            input.Length < checked(batchStart + checked(ic * checked(ih * iw))) || packed.Length < checked(ic * 9 * 16))
            throw new ArgumentException("Invalid stride-two convolution pack.");
        // 一个 16 位置面板最多跨越 16 个输出行。
        // 每个面板只计算一次行划分，避免按输入通道重复进行整数除法。
        Span<Segment> segments = stackalloc Segment[16];
        int count = 0;
        for (int offset = 0; offset < take;)
        {
            int position = start + offset, ox = position % ow;
            int length = Math.Min(take - offset, ow - ox);
            segments[count++] = new(position / ow * 2 - padY, ox * 2 - padX, offset, length);
            offset += length;
        }
        fixed (float* src = input, dst = packed)
            for (int ci = 0; ci < ic; ci++)
            {
                for (int ky = 0; ky < 3; ky++)
                {
                    float* p0 = dst + (ci * 9 + ky * 3) * 16, p1 = p0 + 16, p2 = p1 + 16;
                    for (int segmentIndex = 0; segmentIndex < count; segmentIndex++)
                    {
                        var segment = segments[segmentIndex]; int sy = segment.Y + ky;
                        if ((uint)sy >= ih)
                        {
                            new Span<float>(p0 + segment.Offset, segment.Length).Clear();
                            new Span<float>(p1 + segment.Offset, segment.Length).Clear();
                            new Span<float>(p2 + segment.Offset, segment.Length).Clear();
                            continue;
                        }
                        float* row = src + batchStart + (ci * ih + sy) * iw;
                        int j = 0;
                        while (j < segment.Length)
                        {
                            int sx = segment.X + j * 2, offset = segment.Offset + j, remaining = segment.Length - j;
                            if (remaining >= 16 && sx >= 0 && (long)sx + 31 < iw)
                            {
                                float last = (long)sx + 32 < iw ? row[sx + 32] : 0;
                                PackSixteen(row + sx, p0 + offset, p1 + offset, p2 + offset, last);
                                j += 16;
                            }
                            else if (remaining >= 16 && sx == -1 && iw >= 32)
                            {
                                PackSixteenLeft(row, p0 + offset, p1 + offset, p2 + offset);
                                j += 16;
                            }
                            else if (remaining >= 8 && sx >= 0 && (long)sx + 15 < iw)
                            {
                                float last = (long)sx + 16 < iw ? row[sx + 16] : 0;
                                PackEight(row + sx, p0 + offset, p1 + offset, p2 + offset, last);
                                j += 8;
                            }
                            else if (remaining >= 8 && sx == -1 && iw >= 16)
                            {
                                PackEightLeft(row, p0 + offset, p1 + offset, p2 + offset);
                                j += 8;
                            }
                            else if (remaining >= 4 && sx >= 0 && (long)sx + 7 < iw)
                            {
                                float last = (long)sx + 8 < iw ? row[sx + 8] : 0;
                                PackFour(row + sx, p0 + offset, p1 + offset, p2 + offset, last);
                                j += 4;
                            }
                            else if (remaining >= 4 && sx == -1 && iw >= 8)
                            {
                                PackFourLeft(row, p0 + offset, p1 + offset, p2 + offset);
                                j += 4;
                            }
                            else
                            {
                                p0[offset] = (uint)sx < iw ? row[sx] : 0;
                                p1[offset] = (uint)(sx + 1) < iw ? row[sx + 1] : 0;
                                p2[offset] = (uint)(sx + 2) < iw ? row[sx + 2] : 0;
                                j++;
                            }
                        }
                    }
                }
                // GEMM 会读取的全部通道均须覆写。
                // 每个批次最后一个面板的池化缓冲区尾部也要补零，不能复用旧数据。
                if (take != 16)
                    for (int q = 0; q < 9; q++) new Span<float>(dst + (ci * 9 + q) * 16 + take, 16 - take).Clear();
            }
    }

    /// <summary>保存空间面板与一个输入行相交的连续片段。</summary>
    /// <param name="Y">片段所在的输入行坐标。</param>
    /// <param name="X">片段首个输入列坐标。</param>
    /// <param name="Offset">片段在空间面板内的起始位置。</param>
    /// <param name="Length">片段包含的有效位置数。</param>
    /// <remarks>各构造参数对应同名自动生成属性，描述本记录保存的数据。</remarks>
    private readonly record struct Segment(int Y, int X, int Offset, int Length);

    /// <summary>从连续输入读取八组偶数和奇数采样，供相邻核列共享。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="even">偶数位置的 SIMD 采样或输出向量。</param>
    /// <param name="odd">奇数位置的 SIMD 采样或输出向量。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SamplesEight(float* source, out Vector256<float> even, out Vector256<float> odd)
    {
        var a = Avx.LoadVector256(source); var b = Avx.LoadVector256(source + 8);
        var order = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
        even = Avx2.PermuteVar8x32(Avx.Shuffle(a, b, 0x88), order);
        odd = Avx2.PermuteVar8x32(Avx.Shuffle(a, b, 0xdd), order);
    }

    /// <summary>将同一行的 16 个输出位置所需的三列核采样写入三个打包行。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="p0">第一列卷积核采样的打包写入地址。</param>
    /// <param name="p1">第二列卷积核采样的打包写入地址。</param>
    /// <param name="p2">第三列卷积核采样的打包写入地址。</param>
    /// <param name="last">补入末位的采样值。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackSixteen(float* source, float* p0, float* p1, float* p2, float last)
    {
        SamplesEight(source, out var even0, out var odd0);
        SamplesEight(source + 16, out var even1, out var odd1);
        var crossing = Avx.Permute2x128(even0, even1, 0x21);
        var shifted0 = Avx2.AlignRight(crossing.AsByte(), even0.AsByte(), 4).AsSingle();
        var shifted1 = ShiftOne(even1, last);
        Avx.Store(p0, even0); Avx.Store(p0 + 8, even1);
        Avx.Store(p1, odd0); Avx.Store(p1 + 8, odd1);
        Avx.Store(p2, shifted0); Avx.Store(p2 + 8, shifted1);
    }

    /// <summary>将向量各通道向前移动一个位置，并在末尾补入指定采样值。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <param name="last">补入末位的采样值。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ShiftOne(Vector256<float> value, float last)
        => Avx.Blend(Avx2.PermuteVar8x32(value, Vector256.Create(1, 2, 3, 4, 5, 6, 7, 0)), Vector256.Create(last), 0x80);

    /// <summary>将向量各通道向后移动一个位置，并在首位补入指定采样值。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <param name="first">补入首位的采样值。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ShiftBack(Vector256<float> value, float first)
        => Avx.Blend(Avx2.PermuteVar8x32(value, Vector256.Create(0, 0, 1, 2, 3, 4, 5, 6)), Vector256.Create(first), 0x01);

    /// <summary>为含左侧零填充的 16 个位置打包三列核采样。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="p0">第一列卷积核采样的打包写入地址。</param>
    /// <param name="p1">第二列卷积核采样的打包写入地址。</param>
    /// <param name="p2">第三列卷积核采样的打包写入地址。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackSixteenLeft(float* source, float* p0, float* p1, float* p2)
    {
        SamplesEight(source, out var even0, out var odd0);
        SamplesEight(source + 16, out var even1, out var odd1);
        Avx.Store(p0, ShiftBack(odd0, 0)); Avx.Store(p0 + 8, ShiftBack(odd1, source[15]));
        Avx.Store(p1, even0); Avx.Store(p1 + 8, even1);
        Avx.Store(p2, odd0); Avx.Store(p2 + 8, odd1);
    }

    /// <summary>将同一行的八个输出位置所需的三列核采样写入打包行。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="p0">第一列卷积核采样的打包写入地址。</param>
    /// <param name="p1">第二列卷积核采样的打包写入地址。</param>
    /// <param name="p2">第三列卷积核采样的打包写入地址。</param>
    /// <param name="last">补入末位的采样值。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackEight(float* source, float* p0, float* p1, float* p2, float last)
    {
        SamplesEight(source, out var even, out var odd);
        Avx.Store(p0, even); Avx.Store(p1, odd); Avx.Store(p2, ShiftOne(even, last));
    }

    /// <summary>为含左侧零填充的八个位置打包三列核采样。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="p0">第一列卷积核采样的打包写入地址。</param>
    /// <param name="p1">第二列卷积核采样的打包写入地址。</param>
    /// <param name="p2">第三列卷积核采样的打包写入地址。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackEightLeft(float* source, float* p0, float* p1, float* p2)
    {
        SamplesEight(source, out var even, out var odd);
        Avx.Store(p0, ShiftBack(odd, 0)); Avx.Store(p1, even); Avx.Store(p2, odd);
    }

    /// <summary>将同一行的四个输出位置所需的三列核采样写入打包行。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="p0">第一列卷积核采样的打包写入地址。</param>
    /// <param name="p1">第二列卷积核采样的打包写入地址。</param>
    /// <param name="p2">第三列卷积核采样的打包写入地址。</param>
    /// <param name="last">补入末位的采样值。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackFour(float* source, float* p0, float* p1, float* p2, float last)
    {
        var a = Sse.LoadVector128(source); var b = Sse.LoadVector128(source + 4);
        var even = Sse.Shuffle(a, b, 0x88); var odd = Sse.Shuffle(a, b, 0xdd);
        var next = Sse41.Blend(Sse.Shuffle(even, even, 0x39), Vector128.Create(last), 0x08);
        Sse.Store(p0, even); Sse.Store(p1, odd); Sse.Store(p2, next);
    }

    /// <summary>为含左侧零填充的四个位置打包三列核采样。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="p0">第一列卷积核采样的打包写入地址。</param>
    /// <param name="p1">第二列卷积核采样的打包写入地址。</param>
    /// <param name="p2">第三列卷积核采样的打包写入地址。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackFourLeft(float* source, float* p0, float* p1, float* p2)
    {
        var a = Sse.LoadVector128(source); var b = Sse.LoadVector128(source + 4);
        var even = Sse.Shuffle(a, b, 0x88); var odd = Sse.Shuffle(a, b, 0xdd);
        var previous = Sse41.Blend(Sse.Shuffle(odd, odd, 0x90), Vector128<float>.Zero, 0x01);
        Sse.Store(p0, previous); Sse.Store(p1, even); Sse.Store(p2, odd);
    }
}
