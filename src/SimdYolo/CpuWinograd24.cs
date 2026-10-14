using System.Buffers;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

// F(2,3) 保留原有 FP32 变换与递增输入通道的 FMA 次序。
// 24 块面板让四行 GEMM 将每次权重广播复用于三个向量，并减少空间任务数。
/// <summary>将 F(2,3) 的变换和逆变换组合到 24 块面板中，以四行 AVX/FMA 内核完成通道归约。</summary>
internal static class CpuWinograd24
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_WINOGRAD_24") != "0";

    /// <summary>判断常量权重、AVX2/FMA、通道数和图块数量是否适合线程私有的 24 块面板。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="tiles">图块总数。</param>
    /// <param name="ic">输入通道数。</param>
    /// <param name="oc">输出通道数。</param>
    /// <param name="constant">权重是否只读，从而允许缓存其打包结果。</param>
    /// <returns>满足当前快路径的形状、模式和硬件要求时为 true；否则为 false。</returns>
    internal static bool ShouldUse(CpuBackend backend, int tiles, int ic, int oc, bool constant)
    {
        if (!Enabled || !constant || !Avx2.IsSupported || !Fma.IsSupported || ic < 8 || oc < 16 || oc % 4 != 0 ||
            (long)ic + oc > 512) return false;
        int degree = backend.Parallel.MaxDegreeOfParallelism;
        int workers = degree > 0 ? Math.Min(degree, Environment.ProcessorCount) : Environment.ProcessorCount;
        // 小图像若无法用 24 块面板填满配置的线程数，则保留 16 列路径。
        // F(4,3) 不进入此分支。
        return (tiles + 23) / 24 >= workers;
    }

    /// <summary>将 F(2,3) 的 16 个变换坐标限制在 24 块面板内，经 4×24 内核完成通道归约与逆变换。</summary>
    /// <param name="backend">提供并行线程预算和 CPU 执行器的后端。</param>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="weights">按 Winograd 变换坐标排列的 O×I 权重矩阵。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    internal static Tensor Conv(CpuBackend backend, Tensor x, float[][] weights, Tensor? bias, bool silu)
    {
        int batch = x.Shape[0], ic = x.Shape[1], h = x.Shape[2], width = x.Shape[3];
        int oc = weights[0].Length / ic, tw = (width + 1) / 2;
        int tiles = checked(((h + 1) / 2) * tw), panels = (tiles + 23) / 24;
        // 坐标优先布局中保留一个缓存行的间隔。
        // 避免各变换平面反复映射到相同的少数 L1 缓存组。
        int inputStride = checked(ic * 24 + 16), productStride = checked(oc * 24 + 16);
        var packedWeights = weights.Select(weight => CpuPointwise24.Prepare(weight, oc, ic)).ToArray();
        float[] input = x.Data; float[]? biases = bias?.Data;
        var output = Tensor.TemporaryUninitialized(batch, oc, h, width); float[] target = output.Data;
        int degree = backend.Parallel.MaxDegreeOfParallelism;
        int workers = degree > 0 ? Math.Min(degree, Environment.ProcessorCount) : Environment.ProcessorCount;
        int perTask = panels >= workers * 8 ? 4 : 1, blocks = (panels + perTask - 1) / perTask;

        // 为当前参与线程租用独占的输入变换与乘积工作数组。
        // 返回值：由当前工作线程独享、使用后必须归还的两个数组池缓冲区。
        Scratch Rent() => new(ArrayPool<float>.Shared.Rent(checked(16 * inputStride)),
                              ArrayPool<float>.Shared.Rent(checked(16 * productStride)));
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
                int tile = panel * 24, take = Math.Min(24, tiles - tile);
                CpuWinogradPanels.Transform2(input, scratch.Input, b, ic, h, width, tw, tile, take, inputStride, 24);
                for (int coordinate = 0; coordinate < 16; coordinate++)
                    CpuPointwise24.Multiply(packedWeights[coordinate], scratch.Input, scratch.Products, 24, 0, 24,
                                           coordinate * inputStride, coordinate * productStride);
                CpuWinogradPanels.Inverse2(scratch.Products, target, biases, b, oc, h, width, tw, tile, take, productStride, silu, 24);
            }
        }
        if ((long)16 * ic * oc * tiles * batch > 262144 && degree != 1)
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
}
