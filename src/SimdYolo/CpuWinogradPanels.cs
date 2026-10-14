using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

// 同一个工作线程的临时面板容纳输入变换、通道归约和逆变换。
// 变换图像与乘积图像不跨线程发布，GEMM 直接消费变换结果而无需再次打包。
/// <summary>将输入变换、通道归约和逆变换限制在同一工作线程的面板缓冲区内，避免跨线程物化变换图像。</summary>
internal static class CpuWinogradPanels
{
    /// <summary>Winograd 面板模式：0 关闭、1 强制、auto 按并行度与尺寸选择。</summary>
    private static readonly string Mode = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_WINOGRAD_PANELS") ?? "auto";
    /// <summary>是否使用连续加载与位重排代替逐图块离散读取。</summary>
    private static readonly bool ContiguousLoads = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_WINOGRAD_CONTIGUOUS_LOADS") != "0";
    /// <summary>是否一次计算并复用面板内的图块坐标。</summary>
    private static readonly bool TileCoordinates = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_TILE_COORDINATES") != "0";
    /// <summary>是否用四通道 SIMD 处理 F(4,3) 剩余图块。</summary>
    private static readonly bool FourTileF4 = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_F4_24") != "0";
    /// <summary>是否将 F(4,3) 边界块复制到临时平面后使用 SIMD 变换。</summary>
    private static readonly bool BorderSimdF4 = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_F4_BORDER_SIMD") != "0";

    /// <summary>根据 AVX2/FMA、面板数量和通道工作集选择融合路径；环境变量为 1 时跳过尺寸筛选，为 0 时关闭。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="tiles">图块总数。</param>
    /// <param name="ic">输入通道数。</param>
    /// <param name="oc">输出通道数。</param>
    /// <returns>满足当前快路径的形状、模式和硬件要求时为 true；否则为 false。</returns>
    internal static bool ShouldUse(CpuBackend backend, int tiles, int ic, int oc)
    {
        if (Mode == "0" || !Avx2.IsSupported || !Fma.IsSupported) return false;
        if (Mode == "1") return true;
        int degree = backend.Parallel.MaxDegreeOfParallelism;
        int workers = degree > 0 ? Math.Min(degree, Environment.ProcessorCount) : Environment.ProcessorCount;
        // 较小的深层特征图可以按变换坐标独立归约，提供更多并行任务。
        // 只有面板数量足以填满 CPU 时，才改用面板内融合路径。
        return (tiles + 15) / 16 >= workers && (long)ic + oc <= 512;
    }

    /// <summary>在同一线程的 16 块面板内串联 F(2,3) 或 F(4,3) 输入变换、通道归约与逆变换。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="weights">按 Winograd 变换坐标排列的 O×I 权重矩阵。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <param name="four">是否使用每块输出 4×4 的 F(4,3) 变换；否则使用 F(2,3)。</param>
    /// <param name="constant">权重是否只读，从而允许缓存其打包结果。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static Tensor Conv(CpuBackend backend, Tensor x, float[][] weights, Tensor? bias, bool silu, bool four, bool constant)
    {
        int batch = x.Shape[0], ic = x.Shape[1], h = x.Shape[2], width = x.Shape[3];
        int oc = weights[0].Length / ic, size = four ? 4 : 2, coordinates = four ? 36 : 16;
        int tw = (width + size - 1) / size, tiles = checked(((h + size - 1) / size) * tw);
        // 增加一个缓存行的间隔，避免 2 的幂次通道数使全部变换坐标集中映射到少数 L1 缓存组。
        int panels = (tiles + 15) / 16, inputStride = checked(ic * 16 + 16), productStride = checked(oc * 16 + 16);
        var packedWeights = CpuPackedGemm.Use(oc) && constant ? weights.Select(weight => CpuPackedGemm.Prepare(weight, oc, ic, true)).ToArray() : null;
        float[] input = x.Data; float[]? biases = bias?.Data;
        var output = Tensor.TemporaryUninitialized(batch, oc, h, width); float[] target = output.Data;
        int degree = backend.Parallel.MaxDegreeOfParallelism;
        int workers = degree > 0 ? Math.Min(degree, Environment.ProcessorCount) : Environment.ProcessorCount;
        int perTask = panels >= workers * 8 ? 4 : 1, blocks = (panels + perTask - 1) / perTask;

        // 为当前参与线程租用独占的输入变换与乘积工作数组。
        // 返回值：由当前工作线程独享、使用后必须归还的两个数组池缓冲区。
        Scratch Rent() => new(ArrayPool<float>.Shared.Rent(checked(coordinates * inputStride)),
                              ArrayPool<float>.Shared.Rent(checked(coordinates * productStride)));
        // 归还当前参与线程租用的输入变换与乘积工作数组。
        // 参数 scratch：当前线程私有的临时工作缓冲区。
        void Return(Scratch scratch)
        {
            ArrayPool<float>.Shared.Return(scratch.Input);
            ArrayPool<float>.Shared.Return(scratch.Products);
        }
        // 处理当前批次分配的连续空间面板，在同一线程内串联变换、通道归约与逆变换。
        // 参数 work：批次与面板块组合的任务索引。
        // 参数 scratch：当前线程私有的临时工作缓冲区。
        void Block(int work, Scratch scratch)
        {
            int b = work / blocks, first = work % blocks * perTask, end = Math.Min(panels, first + perTask);
            for (int panel = first; panel < end; panel++)
            {
                int tile = panel * 16, take = Math.Min(16, tiles - tile);
                if (four) Transform4(input, scratch.Input, b, ic, h, width, tw, tile, take, inputStride);
                else Transform2(input, scratch.Input, b, ic, h, width, tw, tile, take, inputStride);
                for (int coordinate = 0; coordinate < coordinates; coordinate++)
                    if (packedWeights is not null)
                        CpuPackedGemm.Multiply(packedWeights[coordinate], scratch.Input, scratch.Products, 16, 0, 16,
                                               coordinate * inputStride, coordinate * productStride);
                    else CpuGemm.MultiplyAvx(weights[coordinate], scratch.Input, scratch.Products, oc, 16, ic, 0, 16,
                                            coordinate * inputStride, coordinate * productStride);
                if (four) Inverse4(scratch.Products, target, biases, b, oc, h, width, tw, tile, take, productStride, silu);
                else Inverse2(scratch.Products, target, biases, b, oc, h, width, tw, tile, take, productStride, silu);
            }
        }
        if ((long)coordinates * ic * oc * tiles * batch > 262144 && degree != 1)
            CpuScheduling.For(0, checked(batch * blocks), backend.Parallel, Rent,
                (work, _, scratch) => { Block(work, scratch); return scratch; }, Return);
        else
        {
            var scratch = Rent();
            try { for (int work = 0; work < checked(batch * blocks); work++) Block(work, scratch); }
            finally { Return(scratch); }
        }
        return output;
    }

    /// <summary>保存单个工作线程租用的输入变换与乘积缓冲区。</summary>
    /// <param name="Input">按变换坐标与通道排列的线程私有输入缓冲区。</param>
    /// <param name="Products">按变换坐标与输出通道排列的线程私有乘积缓冲区。</param>
    /// <remarks>各构造参数对应同名自动生成属性，描述本记录保存的数据。</remarks>
    private sealed record Scratch(float[] Input, float[] Products);

    // 一个面板的图块位置对所有输入通道都相同。
    // 首个位置只计算一次除法与余数，后续通过行边界比较递增，避免按通道重复计算。
    /// <summary>一次计算面板中各图块的行列坐标，以行边界递增代替重复整数除法。</summary>
    /// <param name="ys">保存图块行坐标的临时跨度。</param>
    /// <param name="xs">保存图块列坐标的临时跨度。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="tw">每行图块的数量。</param>
    private static void Coordinates(Span<int> ys, Span<int> xs, int tile, int take, int tw)
    {
        int y = tile / tw, x = tile % tw;
        for (int j = 0; j < take; j++)
        {
            ys[j] = y; xs[j] = x;
            if (++x == tw) { x = 0; y++; }
        }
    }

    /// <summary>应用 F(2,3) 输入变换；一维重载写入各变换分量，面板重载处理当前空间图块。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="batch">当前批次的零基索引。</param>
    /// <param name="channels">要处理的通道数。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="tw">每行图块的数量。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="columns">面板容量，取 16 或 24。</param>
    internal static void Transform2(float[] input, float[] packed, int batch, int channels, int h, int width,
                                    int tw, int tile, int take, int stride, int columns = 16)
    {
        Span<float> patch = stackalloc float[16]; Span<float> temp = stackalloc float[16];
        Span<int> ys = stackalloc int[24]; Span<int> xs = stackalloc int[24];
        if (TileCoordinates) Coordinates(ys, xs, tile, take, tw);
        for (int ci = 0; ci < channels; ci++)
        {
            int channel = (batch * channels + ci) * h * width, row = ci * columns;
            for (int j = 0; j < take; j++)
            {
                int position = tile + j, ty = TileCoordinates ? ys[j] : position / tw, tx = TileCoordinates ? xs[j] : position % tw;
                if (j + 8 <= take && ty > 0 && ty * 2 + 2 < h && tx > 0 && tx + 8 < tw)
                {
                    TransformEight2(input, packed, channel + (ty * 2 - 1) * width + tx * 2 - 1,
                                    width, row + j, stride);
                    j += 7;
                    continue;
                }
                for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
                {
                    int yy = ty * 2 - 1 + r, xx = tx * 2 - 1 + c;
                    patch[r * 4 + c] = (uint)yy < h && (uint)xx < width ? input[channel + yy * width + xx] : 0;
                }
                for (int c = 0; c < 4; c++)
                {
                    float d0 = patch[c], d1 = patch[4 + c], d2 = patch[8 + c], d3 = patch[12 + c];
                    temp[c] = d0 - d2; temp[4 + c] = d1 + d2; temp[8 + c] = d2 - d1; temp[12 + c] = d1 - d3;
                }
                for (int r = 0; r < 4; r++)
                {
                    float d0 = temp[r * 4], d1 = temp[r * 4 + 1], d2 = temp[r * 4 + 2], d3 = temp[r * 4 + 3];
                    int dst = r * 4 * stride + row + j;
                    packed[dst] = d0 - d2; packed[dst + stride] = d1 + d2;
                    packed[dst + stride * 2] = d2 - d1; packed[dst + stride * 3] = d1 - d3;
                }
            }
            // 临时缓冲区来自池且未初始化。
            // GEMM 读取前必须覆写所有补齐的 SIMD 通道，包括每个批次最后一个面板。
            if (take != columns)
                for (int coordinate = 0; coordinate < 16; coordinate++)
                    packed.AsSpan(coordinate * stride + row + take, columns - take).Clear();
        }
    }

    /// <summary>对八个图块应用 F(2,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="source">当前读取在源数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    private static unsafe void TransformEight2(float[] input, float[] packed, int source, int width, int target, int stride)
    {
        Span<Vector256<float>> temp = stackalloc Vector256<float>[16];
        var order = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
        fixed (float* ptr = input)
            for (int c = 0; c < 4; c++)
            {
                float* start = ptr + source + (c & 2);
                var d0 = CpuWinograd.GatherEvenOdd(start, c, order); var d1 = CpuWinograd.GatherEvenOdd(start + width, c, order);
                var d2 = CpuWinograd.GatherEvenOdd(start + width * 2, c, order); var d3 = CpuWinograd.GatherEvenOdd(start + width * 3, c, order);
                temp[c] = d0 - d2; temp[4 + c] = d1 + d2; temp[8 + c] = d2 - d1; temp[12 + c] = d1 - d3;
            }
        for (int r = 0; r < 4; r++)
        {
            var d0 = temp[r * 4]; var d1 = temp[r * 4 + 1]; var d2 = temp[r * 4 + 2]; var d3 = temp[r * 4 + 3];
            (d0 - d2).CopyTo(packed.AsSpan(r * 4 * stride + target));
            (d1 + d2).CopyTo(packed.AsSpan((r * 4 + 1) * stride + target));
            (d2 - d1).CopyTo(packed.AsSpan((r * 4 + 2) * stride + target));
            (d1 - d3).CopyTo(packed.AsSpan((r * 4 + 3) * stride + target));
        }
    }

    /// <summary>应用 F(4,3) 输入变换；一维重载写入各变换分量，面板重载处理当前空间图块。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="batch">当前批次的零基索引。</param>
    /// <param name="channels">要处理的通道数。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="tw">每行图块的数量。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="columns">面板容量，取 16 或 24。</param>
    internal static unsafe void Transform4(float[] input, float[] packed, int batch, int channels, int h, int width,
                                   int tw, int tile, int take, int stride, int columns = 16)
    {
        Span<float> patch = stackalloc float[36]; Span<float> temp = stackalloc float[36]; Span<float> result = stackalloc float[6];
        Span<float> border = stackalloc float[BorderSimdF4 ? 204 : 0];
        Span<Vector128<float>> borderValues = stackalloc Vector128<float>[BorderSimdF4 ? 36 : 0];
        Span<int> ys = stackalloc int[24]; Span<int> xs = stackalloc int[24];
        if (TileCoordinates) Coordinates(ys, xs, tile, take, tw);
        for (int ci = 0; ci < channels; ci++)
        {
            int channel = (batch * channels + ci) * h * width, row = ci * columns;
            for (int j = 0; j < take; j++)
            {
                int position = tile + j, ty = TileCoordinates ? ys[j] : position / tw, tx = TileCoordinates ? xs[j] : position % tw;
                if (j + 8 <= take && ty > 0 && ty * 4 + 4 < h && tx > 0 && tx + 8 < tw)
                {
                    TransformEight4(input, packed, channel + (ty * 4 - 1) * width + tx * 4 - 1,
                                    width, row + j, stride);
                    j += 7;
                    continue;
                }
                if (BorderSimdF4 && j + 8 <= take && tx + 8 <= tw)
                {
                    CopyBorder(input, border, channel, h, width, ty * 4 - 1, tx * 4 - 1, 34);
                    fixed (float* ptr = border) TransformEight4(ptr, packed, 34, row + j, stride);
                    j += 7; continue;
                }
                if (FourTileF4 && j + 4 <= take && ty > 0 && ty * 4 + 4 < h && tx > 0 && tx + 4 < tw)
                {
                    TransformFour4(input, packed, channel + (ty * 4 - 1) * width + tx * 4 - 1,
                                   width, row + j, stride);
                    j += 3;
                    continue;
                }
                if (BorderSimdF4 && j + 4 <= take && tx + 4 <= tw)
                {
                    CopyBorder(input, border, channel, h, width, ty * 4 - 1, tx * 4 - 1, 18);
                    fixed (float* ptr = border) TransformFour4(ptr, 18, borderValues);
                    for (int coordinate = 0; coordinate < 36; coordinate++) borderValues[coordinate].CopyTo(packed.AsSpan(coordinate * stride + row + j));
                    j += 3; continue;
                }
                for (int r = 0; r < 6; r++) for (int c = 0; c < 6; c++)
                {
                    int yy = ty * 4 - 1 + r, xx = tx * 4 - 1 + c;
                    patch[r * 6 + c] = (uint)yy < h && (uint)xx < width ? input[channel + yy * width + xx] : 0;
                }
                for (int c = 0; c < 6; c++)
                {
                    CpuWinograd4.Transform(patch[c], patch[6 + c], patch[12 + c], patch[18 + c], patch[24 + c], patch[30 + c], result);
                    for (int r = 0; r < 6; r++) temp[r * 6 + c] = result[r];
                }
                for (int r = 0; r < 6; r++)
                {
                    CpuWinograd4.Transform(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                    for (int c = 0; c < 6; c++) packed[(r * 6 + c) * stride + row + j] = result[c];
                }
            }
            if (take != columns)
                for (int coordinate = 0; coordinate < 36; coordinate++)
                    packed.AsSpan(coordinate * stride + row + take, columns - take).Clear();
        }
    }

    /// <summary>将含越界区域的输入块复制到临时平面，越界元素写零。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="scratch">当前线程私有的临时工作缓冲区。</param>
    /// <param name="channel">当前输入通道的零基索引，或该通道在输入数组中的起始偏移。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="y">当前块的输入行坐标。</param>
    /// <param name="x">当前输入块的列坐标。</param>
    /// <param name="patchWidth">待复制输入块的边长。</param>
    private static void CopyBorder(float[] input, Span<float> scratch, int channel, int h, int width, int y, int x, int patchWidth)
    {
        int left = Math.Max(0, -x), sourceX = Math.Max(0, x), count = Math.Max(0, Math.Min(patchWidth - left, width - sourceX));
        for (int r = 0; r < 6; r++)
        {
            Span<float> row = scratch.Slice(r * patchWidth, patchWidth);
            int yy = y + r;
            if ((uint)yy >= h) { row.Clear(); continue; }
            row.Slice(0, left).Clear();
            input.AsSpan(channel + yy * width + sourceX, count).CopyTo(row.Slice(left, count));
            row.Slice(left + count).Clear();
        }
    }

    /// <summary>对八个图块应用 F(4,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="source">当前读取在源数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void TransformEight4(float[] input, float[] packed, int source, int width, int target, int stride)
    {
        fixed (float* ptr = input) TransformEight4(ptr + source, packed, width, target, stride);
    }

    /// <summary>对八个图块应用 F(4,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    private static unsafe void TransformEight4(float* source, float[] packed, int width, int target, int stride)
    {
        Span<Vector256<float>> patch = stackalloc Vector256<float>[36];
        Span<Vector256<float>> temp = stackalloc Vector256<float>[36]; Span<Vector256<float>> result = stackalloc Vector256<float>[6];
        var indices = Vector256.Create(0, 4, 8, 12, 16, 20, 24, 28);
        if (ContiguousLoads)
        {
            for (int r = 0; r < 6; r++) LoadSixColumns(source + r * width, patch.Slice(r * 6, 6));
            for (int c = 0; c < 6; c++)
            {
                CpuWinograd4.Transform(patch[c], patch[6 + c], patch[12 + c], patch[18 + c], patch[24 + c], patch[30 + c], result);
                for (int r = 0; r < 6; r++) temp[r * 6 + c] = result[r];
            }
        }
        else
        {
            for (int c = 0; c < 6; c++)
            {
                float* start = source + c;
                CpuWinograd4.Transform(Avx2.GatherVector256(start, indices, 4), Avx2.GatherVector256(start + width, indices, 4),
                    Avx2.GatherVector256(start + width * 2, indices, 4), Avx2.GatherVector256(start + width * 3, indices, 4),
                    Avx2.GatherVector256(start + width * 4, indices, 4), Avx2.GatherVector256(start + width * 5, indices, 4), result);
                for (int r = 0; r < 6; r++) temp[r * 6 + c] = result[r];
            }
        }
        for (int r = 0; r < 6; r++)
        {
            CpuWinograd4.Transform(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
            for (int c = 0; c < 6; c++) result[c].CopyTo(packed.AsSpan((r * 6 + c) * stride + target));
        }
    }

    /// <summary>对四个图块应用 F(4,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="packed">当前工作线程使用的连续打包缓冲区。</param>
    /// <param name="source">当前读取在源数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    private static unsafe void TransformFour4(float[] input, float[] packed, int source, int width, int target, int stride)
    {
        Span<Vector128<float>> transformed = stackalloc Vector128<float>[36];
        TransformFour4(input, source, width, transformed);
        for (int coordinate = 0; coordinate < 36; coordinate++)
            transformed[coordinate].CopyTo(packed.AsSpan(coordinate * stride + target));
    }

    /// <summary>对四个图块应用 F(4,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="transformed">按 Winograd 坐标排列的输入变换缓冲区。</param>
    /// <param name="source">当前读取在源数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    internal static void TransformFour4Global(float[] input, float[][] transformed, int source, int width, int target)
    {
        Span<Vector128<float>> values = stackalloc Vector128<float>[36];
        TransformFour4(input, source, width, values);
        for (int coordinate = 0; coordinate < 36; coordinate++) values[coordinate].CopyTo(transformed[coordinate].AsSpan(target));
    }

    /// <summary>对四个图块应用 F(4,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="source">当前读取在源数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="transformed">按 Winograd 坐标排列的输入变换缓冲区。</param>
    private static unsafe void TransformFour4(float[] input, int source, int width, Span<Vector128<float>> transformed)
    {
        fixed (float* ptr = input) TransformFour4(ptr + source, width, transformed);
    }

    /// <summary>对四个图块应用 F(4,3) 输入变换，将结果按变换坐标写入缓冲区。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="transformed">按 Winograd 坐标排列的输入变换缓冲区。</param>
    private static unsafe void TransformFour4(float* source, int width, Span<Vector128<float>> transformed)
    {
        Span<Vector128<float>> patch = stackalloc Vector128<float>[36];
        Span<Vector128<float>> temp = stackalloc Vector128<float>[36]; Span<Vector128<float>> result = stackalloc Vector128<float>[6];
        for (int r = 0; r < 6; r++) LoadSixColumns4(source + r * width, patch.Slice(r * 6, 6));
        for (int c = 0; c < 6; c++)
        {
            Transform4(patch[c], patch[6 + c], patch[12 + c], patch[18 + c], patch[24 + c], patch[30 + c], result);
            for (int r = 0; r < 6; r++) temp[r * 6 + c] = result[r];
        }
        for (int r = 0; r < 6; r++)
        {
            Transform4(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
            for (int c = 0; c < 6; c++) transformed[r * 6 + c] = result[c];
        }
    }

    /// <summary>通过连续读取与位重排获取四个图块各自的六列输入。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="columns">接收六列 SIMD 数据的临时跨度。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void LoadSixColumns4(float* source, Span<Vector128<float>> columns)
    {
        var a = Sse.LoadVector128(source); var b = Sse.LoadVector128(source + 4);
        var c = Sse.LoadVector128(source + 8); var d = Sse.LoadVector128(source + 12);
        var loAB = Sse.Shuffle(a, b, 0x44); var loCD = Sse.Shuffle(c, d, 0x44);
        var hiAB = Sse.Shuffle(a, b, 0xee); var hiCD = Sse.Shuffle(c, d, 0xee);
        columns[0] = Sse.Shuffle(loAB, loCD, 0x88); columns[1] = Sse.Shuffle(loAB, loCD, 0xdd);
        columns[2] = Sse.Shuffle(hiAB, hiCD, 0x88); columns[3] = Sse.Shuffle(hiAB, hiCD, 0xdd);
        columns[4] = Sse41.Blend(Sse.Shuffle(columns[0], columns[0], 0x39), Vector128.Create(source[16]), 0x08);
        columns[5] = Sse41.Blend(Sse.Shuffle(columns[1], columns[1], 0x39), Vector128.Create(source[17]), 0x08);
    }

    /// <summary>应用 F(4,3) 输入变换；一维重载写入各变换分量，面板重载处理当前空间图块。</summary>
    /// <param name="d0">一维变换的第 1 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d1">一维变换的第 2 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d2">一维变换的第 3 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d3">一维变换的第 4 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d4">一维变换的第 5 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d5">一维变换的第 6 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="output">接收一维变换结果的跨度，长度须满足变换所需分量数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transform4(Vector128<float> d0, Vector128<float> d1, Vector128<float> d2, Vector128<float> d3,
                                   Vector128<float> d4, Vector128<float> d5, Span<Vector128<float>> output)
    {
        var two = Vector128.Create(2f); var four = Vector128.Create(4f); var five = Vector128.Create(5f);
        output[0] = four * d0 - five * d2 + d4;
        output[1] = -four * (d1 + d2) + d3 + d4;
        output[2] = four * (d1 - d2) - d3 + d4;
        output[3] = -two * d1 - d2 + two * d3 + d4;
        output[4] = two * d1 - d2 - two * d3 + d4;
        output[5] = four * d1 - five * d3 + d5;
    }

    /// <summary>通过连续读取与位重排获取八个图块各自的六列输入。</summary>
    /// <param name="source">连续的源 FP32 缓冲区或其固定指针。</param>
    /// <param name="columns">接收六列 SIMD 数据的临时跨度。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void LoadSixColumns(float* source, Span<Vector256<float>> columns)
    {
        var a = Avx.LoadVector256(source); var b = Avx.LoadVector256(source + 8);
        var c = Avx.LoadVector256(source + 16); var d = Avx.LoadVector256(source + 24);
        var evenAB = Avx.Shuffle(a, b, 0x88); var evenCD = Avx.Shuffle(c, d, 0x88);
        var oddAB = Avx.Shuffle(a, b, 0xdd); var oddCD = Avx.Shuffle(c, d, 0xdd);
        var order = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        columns[0] = Avx2.PermuteVar8x32(Avx.Shuffle(evenAB, evenCD, 0x88), order);
        columns[1] = Avx2.PermuteVar8x32(Avx.Shuffle(oddAB, oddCD, 0x88), order);
        columns[2] = Avx2.PermuteVar8x32(Avx.Shuffle(evenAB, evenCD, 0xdd), order);
        columns[3] = Avx2.PermuteVar8x32(Avx.Shuffle(oddAB, oddCD, 0xdd), order);
        var next = Vector256.Create(1, 2, 3, 4, 5, 6, 7, 0);
        columns[4] = Avx.Blend(Avx2.PermuteVar8x32(columns[0], next), Avx.BroadcastScalarToVector256(source + 32), 0x80);
        columns[5] = Avx.Blend(Avx2.PermuteVar8x32(columns[1], next), Avx.BroadcastScalarToVector256(source + 33), 0x80);
    }

    /// <summary>应用 F(2,3) 逆变换；一维重载还原输出分量，面板重载写回有效空间位置。</summary>
    /// <param name="products">按变换坐标排列的通道乘积缓冲区。</param>
    /// <param name="target">接收当前面板结果的预分配 FP32 缓冲区。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="batch">当前批次的零基索引。</param>
    /// <param name="oc">输出通道数。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="tw">每行图块的数量。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <param name="columns">面板容量，取 16 或 24。</param>
    internal static void Inverse2(float[] products, float[] target, float[]? bias, int batch, int oc, int h, int width,
                                  int tw, int tile, int take, int stride, bool silu, int columns = 16)
    {
        Span<int> ys = stackalloc int[24]; Span<int> xs = stackalloc int[24];
        if (TileCoordinates) Coordinates(ys, xs, tile, take, tw);
        for (int co = 0; co < oc; co++)
        {
            float bv = bias?[co] ?? 0; int channel = (batch * oc + co) * h * width;
            for (int j = 0; j < take; j++)
            {
                int position = tile + j, ty = TileCoordinates ? ys[j] : position / tw, tx = TileCoordinates ? xs[j] : position % tw, src = co * columns + j;
                int yy = ty * 2, xx = tx * 2, dst = channel + yy * width + xx;
                if (j + 8 <= take && tx + 8 <= tw && (tx + 8) * 2 <= width && yy + 1 < h)
                {
                    InverseEight2(products, target, src, dst, width, stride, bv, silu);
                    j += 7;
                    continue;
                }
                float t00 = products[src] + products[4 * stride + src] + products[8 * stride + src];
                float t01 = products[stride + src] + products[5 * stride + src] + products[9 * stride + src];
                float t02 = products[2 * stride + src] + products[6 * stride + src] + products[10 * stride + src];
                float t03 = products[3 * stride + src] + products[7 * stride + src] + products[11 * stride + src];
                float t10 = products[4 * stride + src] - products[8 * stride + src] - products[12 * stride + src];
                float t11 = products[5 * stride + src] - products[9 * stride + src] - products[13 * stride + src];
                float t12 = products[6 * stride + src] - products[10 * stride + src] - products[14 * stride + src];
                float t13 = products[7 * stride + src] - products[11 * stride + src] - products[15 * stride + src];
                float v00 = t00 + t01 + t02 + bv, v01 = t01 - t02 - t03 + bv;
                float v10 = t10 + t11 + t12 + bv, v11 = t11 - t12 - t13 + bv;
                if (silu)
                {
                    var activated = Activate(Vector128.Create(v00, v01, v10, v11), true);
                    v00 = activated.GetElement(0); v01 = activated.GetElement(1); v10 = activated.GetElement(2); v11 = activated.GetElement(3);
                }
                target[dst] = v00;
                if (xx + 1 < width) target[dst + 1] = v01;
                if (yy + 1 < h)
                {
                    target[dst + width] = v10;
                    if (xx + 1 < width) target[dst + width + 1] = v11;
                }
            }
        }
    }

    /// <summary>对八个图块应用 F(2,3) 逆变换，恢复空间输出并按签名融合偏置与激活。</summary>
    /// <param name="products">按变换坐标排列的通道乘积缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    private static unsafe void InverseEight2(float[] products, float[] output, int tile, int target, int width, int stride, float bias, bool silu)
    {
        var bv = Vector256.Create(bias);
        var t00 = Load(products, tile, stride, 0) + Load(products, tile, stride, 4) + Load(products, tile, stride, 8);
        var t01 = Load(products, tile, stride, 1) + Load(products, tile, stride, 5) + Load(products, tile, stride, 9);
        var t02 = Load(products, tile, stride, 2) + Load(products, tile, stride, 6) + Load(products, tile, stride, 10);
        var t03 = Load(products, tile, stride, 3) + Load(products, tile, stride, 7) + Load(products, tile, stride, 11);
        var t10 = Load(products, tile, stride, 4) - Load(products, tile, stride, 8) - Load(products, tile, stride, 12);
        var t11 = Load(products, tile, stride, 5) - Load(products, tile, stride, 9) - Load(products, tile, stride, 13);
        var t12 = Load(products, tile, stride, 6) - Load(products, tile, stride, 10) - Load(products, tile, stride, 14);
        var t13 = Load(products, tile, stride, 7) - Load(products, tile, stride, 11) - Load(products, tile, stride, 15);
        fixed (float* ptr = output)
        {
            CpuWinograd.Interleave(Activate(t00 + t01 + t02 + bv, silu), Activate(t01 - t02 - t03 + bv, silu), ptr + target);
            CpuWinograd.Interleave(Activate(t10 + t11 + t12 + bv, silu), Activate(t11 - t12 - t13 + bv, silu), ptr + target + width);
        }
    }

    /// <summary>应用 F(4,3) 逆变换；一维重载还原输出分量，面板重载写回有效空间位置。</summary>
    /// <param name="products">按变换坐标排列的通道乘积缓冲区。</param>
    /// <param name="target">接收当前面板结果的预分配 FP32 缓冲区。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="batch">当前批次的零基索引。</param>
    /// <param name="oc">输出通道数。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="tw">每行图块的数量。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="take">当前面板中实际有效的位置数。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <param name="columns">面板容量，取 16 或 24。</param>
    internal static void Inverse4(float[] products, float[] target, float[]? bias, int batch, int oc, int h, int width,
                                 int tw, int tile, int take, int stride, bool silu, int columns = 16)
    {
        Span<float> temp = stackalloc float[24]; Span<float> result = stackalloc float[4];
        Span<int> ys = stackalloc int[24]; Span<int> xs = stackalloc int[24];
        if (TileCoordinates) Coordinates(ys, xs, tile, take, tw);
        for (int co = 0; co < oc; co++)
        {
            float bv = bias?[co] ?? 0; int channel = (batch * oc + co) * h * width;
            for (int j = 0; j < take; j++)
            {
                int position = tile + j, ty = TileCoordinates ? ys[j] : position / tw, tx = TileCoordinates ? xs[j] : position % tw, src = co * columns + j;
                int yy = ty * 4, xx = tx * 4;
                if (j + 8 <= take && tx + 8 <= tw && (tx + 8) * 4 <= width && yy + 3 < h)
                {
                    InverseEight4(products, target, src, channel + yy * width + xx, width, stride, bv, silu);
                    j += 7;
                    continue;
                }
                if (FourTileF4 && j + 4 <= take && tx + 4 <= tw && (tx + 4) * 4 <= width && yy + 3 < h)
                {
                    InverseFour4(products, target, src, channel + yy * width + xx, width, stride, bv, silu);
                    j += 3;
                    continue;
                }
                for (int c = 0; c < 6; c++)
                {
                    CpuWinograd4.Inverse(products[c * stride + src], products[(6 + c) * stride + src], products[(12 + c) * stride + src],
                                        products[(18 + c) * stride + src], products[(24 + c) * stride + src], products[(30 + c) * stride + src], result);
                    for (int r = 0; r < 4; r++) temp[r * 6 + c] = result[r];
                }
                for (int r = 0; r < 4 && yy + r < h; r++)
                {
                    CpuWinograd4.Inverse(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                    int dst = channel + (yy + r) * width + xx;
                    if (silu)
                    {
                        var activated = Activate(Vector128.LoadUnsafe(ref result[0]) + Vector128.Create(bv), true);
                        if (xx + 4 <= width) activated.CopyTo(target.AsSpan(dst, 4));
                        else for (int c = 0; xx + c < width; c++) target[dst + c] = activated.GetElement(c);
                    }
                    else for (int c = 0; c < 4 && xx + c < width; c++) target[dst + c] = result[c] + bv;
                }
            }
        }
    }

    /// <summary>对八个图块应用 F(4,3) 逆变换，恢复空间输出并按签名融合偏置与激活。</summary>
    /// <param name="products">按变换坐标排列的通道乘积缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    private static unsafe void InverseEight4(float[] products, float[] output, int tile, int target, int width, int stride, float bias, bool silu)
    {
        Span<Vector256<float>> temp = stackalloc Vector256<float>[24]; Span<Vector256<float>> result = stackalloc Vector256<float>[4];
        for (int c = 0; c < 6; c++)
        {
            CpuWinograd4.Inverse(Load(products, tile, stride, c), Load(products, tile, stride, 6 + c), Load(products, tile, stride, 12 + c),
                                Load(products, tile, stride, 18 + c), Load(products, tile, stride, 24 + c), Load(products, tile, stride, 30 + c), result);
            for (int r = 0; r < 4; r++) temp[r * 6 + c] = result[r];
        }
        var bv = Vector256.Create(bias);
        fixed (float* ptr = output)
            for (int r = 0; r < 4; r++)
            {
                CpuWinograd4.Inverse(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                CpuWinograd4.InterleaveFour(Activate(result[0] + bv, silu), Activate(result[1] + bv, silu),
                                           Activate(result[2] + bv, silu), Activate(result[3] + bv, silu), ptr + target + r * width);
            }
    }

    /// <summary>对四个图块应用 F(4,3) 逆变换，恢复空间输出并按签名融合偏置与激活。</summary>
    /// <param name="products">按变换坐标排列的通道乘积缓冲区。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="target">当前写入在目标数组中的起始元素偏移。</param>
    /// <param name="width">空间平面的宽度。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="bias">当前输出通道的单精度偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    private static unsafe void InverseFour4(float[] products, float[] output, int tile, int target, int width, int stride, float bias, bool silu)
    {
        Span<Vector128<float>> temp = stackalloc Vector128<float>[24]; Span<Vector128<float>> result = stackalloc Vector128<float>[4];
        for (int c = 0; c < 6; c++)
        {
            Inverse4(Load128(products, tile, stride, c), Load128(products, tile, stride, 6 + c), Load128(products, tile, stride, 12 + c),
                     Load128(products, tile, stride, 18 + c), Load128(products, tile, stride, 24 + c), Load128(products, tile, stride, 30 + c), result);
            for (int r = 0; r < 4; r++) temp[r * 6 + c] = result[r];
        }
        var bv = Vector128.Create(bias);
        fixed (float* ptr = output)
            for (int r = 0; r < 4; r++)
            {
                Inverse4(temp[r * 6], temp[r * 6 + 1], temp[r * 6 + 2], temp[r * 6 + 3], temp[r * 6 + 4], temp[r * 6 + 5], result);
                var v0 = Activate(result[0] + bv, silu); var v1 = Activate(result[1] + bv, silu);
                var v2 = Activate(result[2] + bv, silu); var v3 = Activate(result[3] + bv, silu);
                var lo01 = Sse.UnpackLow(v0, v1); var lo23 = Sse.UnpackLow(v2, v3);
                var hi01 = Sse.UnpackHigh(v0, v1); var hi23 = Sse.UnpackHigh(v2, v3);
                float* dst = ptr + target + r * width;
                Sse.Store(dst, Sse.Shuffle(lo01, lo23, 0x44)); Sse.Store(dst + 4, Sse.Shuffle(lo01, lo23, 0xee));
                Sse.Store(dst + 8, Sse.Shuffle(hi01, hi23, 0x44)); Sse.Store(dst + 12, Sse.Shuffle(hi01, hi23, 0xee));
            }
    }

    /// <summary>应用 F(4,3) 逆变换；一维重载还原输出分量，面板重载写回有效空间位置。</summary>
    /// <param name="d0">一维变换的第 1 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d1">一维变换的第 2 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d2">一维变换的第 3 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d3">一维变换的第 4 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d4">一维变换的第 5 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="d5">一维变换的第 6 个输入分量；SIMD 版本逐通道独立计算。</param>
    /// <param name="output">接收一维变换结果的跨度，长度须满足变换所需分量数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Inverse4(Vector128<float> d0, Vector128<float> d1, Vector128<float> d2, Vector128<float> d3,
                                 Vector128<float> d4, Vector128<float> d5, Span<Vector128<float>> output)
    {
        output[0] = d0 + d1 + d2 + d3 + d4;
        output[1] = d1 - d2 + Vector128.Create(2f) * (d3 - d4);
        output[2] = d1 + d2 + Vector128.Create(4f) * (d3 + d4);
        output[3] = d1 - d2 + Vector128.Create(8f) * (d3 - d4) + d5;
    }

    /// <summary>从坐标优先的变换缓冲区读取四个图块的当前坐标。</summary>
    /// <param name="data">连续 FP32 元素缓冲区。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="coordinate">当前变换坐标的索引。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Load128(float[] data, int tile, int stride, int coordinate)
        => Vector128.LoadUnsafe(ref data[coordinate * stride + tile]);

    /// <summary>从坐标优先的变换缓冲区读取八个图块的当前坐标。</summary>
    /// <param name="data">连续 FP32 元素缓冲区。</param>
    /// <param name="tile">当前图块在平铺图块集合中的起始索引。</param>
    /// <param name="stride">相邻变换坐标平面在缓冲区中的元素跨度。</param>
    /// <param name="coordinate">当前变换坐标的索引。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Load(float[] data, int tile, int stride, int coordinate)
        => Vector256.LoadUnsafe(ref data[coordinate * stride + tile]);
    /// <summary>按标志对 SIMD 结果应用 SiLU，关闭时保持原值。</summary>
    /// <param name="v">待激活的 SIMD 向量。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Activate(Vector256<float> v, bool silu)
        => silu ? CpuActivation.Silu(v) : v;
    /// <summary>按标志对 SIMD 结果应用 SiLU，关闭时保持原值。</summary>
    /// <param name="v">待激活的 SIMD 向量。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Activate(Vector128<float> v, bool silu)
        => silu ? CpuActivation.Silu(v) : v;
}
