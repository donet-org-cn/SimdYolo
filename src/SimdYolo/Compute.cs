using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>定义计算后端的名称、行主序矩阵乘法和资源释放契约。</summary>
public interface IComputeBackend : IDisposable
{
    /// <summary>可读的后端名称及当前 SIMD 位宽。</summary>
    /// <value>可读的后端名称及当前 SIMD 位宽。</value>
    string Name { get; }
    /// <summary>计算行主序矩阵乘法 C[M,N]=A[M,K]×B[K,N]，将结果写入预分配的 C。</summary>
    /// <param name="a">行主序左矩阵 A[M,K]。</param>
    /// <param name="b">行主序右矩阵 B[K,N]；微内核方法中为已打包的 K×面板列数布局。</param>
    /// <param name="c">预分配的行主序输出矩阵 C[M,N]。</param>
    /// <param name="m">左矩阵与输出矩阵的行数 M。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="k">矩阵乘法的归约长度 K。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    void MatMul(float[] a, float[] b, float[] c, int m, int n, int k);
}
/// <summary>定义直接向预分配张量写入 NCHW 卷积结果的后端契约。</summary>
public interface IConvolutionBackend
{
    /// <summary>计算 NCHW/OIHW 卷积，可融合偏置与 SiLU；按形状选择设备、Winograd、点卷积或通用 CPU 路径。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="output">预分配的目标张量、数组或固定写入指针。</param>
    /// <param name="strides">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="groups">将输入和输出通道划分的卷积分组数。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    void Conv(Tensor input, Tensor weights, Tensor? bias, Tensor output, int[] strides, int[] pads, int[] dilation, int groups, bool silu);
}

/// <summary>以纯托管 FP32 和 CPU SIMD 提供矩阵计算，并拥有受线程预算约束的专用工作线程。</summary>
public sealed class CpuBackend : IComputeBackend
{
    /// <summary>CPU 后端固定的并行线程预算。</summary>
    private readonly ParallelOptions parallel;
    /// <summary>本后端拥有的专用 CPU 执行器；关闭专用调度时为 null。</summary>
    private readonly CpuExecutor? executor;
    /// <summary>创建 CPU 后端，并在启用专用执行器时注册其线程预算。</summary>
    /// <param name="threads">线程预算；大于零时使用指定值，否则使用逻辑处理器数。</param>
    public CpuBackend(int threads = 0)
    {
        parallel = new() { MaxDegreeOfParallelism = threads > 0 ? threads : Environment.ProcessorCount };
        if (Environment.GetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR") != "0")
        {
            executor = new CpuExecutor(parallel);
            CpuScheduling.Register(parallel, executor);
        }
    }
    /// <summary>可读的后端名称及当前 SIMD 位宽。</summary>
    /// <value>可读的后端名称及当前 SIMD 位宽。</value>
    public string Name => $"CPU SIMD ({Vector<float>.Count * 32} bits)";
    /// <summary>等待 CPU 内核结束后释放专用工作线程，并解除调度注册；可重复调用。</summary>
    /// <exception cref="InvalidOperationException">试图在本后端自己的内核回调中释放它，会导致线程自等待。</exception>
    public void Dispose()
    {
        executor?.Dispose();
        CpuScheduling.Unregister(parallel);
    }
    /// <summary>当前会话或后端的 CPU 并行配置。</summary>
    /// <value>当前会话或后端的 CPU 并行配置。</value>
    internal ParallelOptions Parallel => parallel;
    /// <summary>计算行主序矩阵乘法 C[M,N]=A[M,K]×B[K,N]，将结果写入预分配的 C。</summary>
    /// <param name="a">行主序左矩阵 A[M,K]。</param>
    /// <param name="b">行主序右矩阵 B[K,N]；微内核方法中为已打包的 K×面板列数布局。</param>
    /// <param name="c">预分配的行主序输出矩阵 C[M,N]。</param>
    /// <param name="m">左矩阵与输出矩阵的行数 M。</param>
    /// <param name="n">右矩阵与输出矩阵的列数 N，亦为输出行跨度。</param>
    /// <param name="k">矩阵乘法的归约长度 K。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    public void MatMul(float[] a, float[] b, float[] c, int m, int n, int k)
        => CpuGemm.Multiply(a, b, c, m, n, k, parallel);
}

/// <summary>提供卷积、广播算术和张量布局等算子；在活动设备帧内优先使用设备后端，否则使用 CPU 实现。</summary>
/// <param name="backend">此算子集合使用的计算后端。</param>
public sealed class TensorOps(IComputeBackend backend)
{
    /// <summary>此算子集合使用的计算后端。</summary>
    /// <value>此算子集合使用的计算后端。</value>
    public IComputeBackend Backend { get; } = backend;
    /// <summary>检查张量形状的元素数是否仍与底层缓冲区长度相符。</summary>
    /// <param name="tensor">需要验证形状与存储长度的张量。</param>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    private static void ValidateShape(Tensor tensor)
    {
        try
        {
            if (Tensor.Count(tensor.Shape) == tensor.Length) return;
        }
        catch (OverflowException error)
        {
            throw new ArgumentException("Tensor shape exceeds the supported buffer length.", nameof(tensor), error);
        }
        throw new ArgumentException("Tensor shape no longer matches its buffer length.", nameof(tensor));
    }
    /// <summary>对 NCHW 张量进行分组转置卷积，并根据步长、填充和输出填充计算输出尺寸。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="w">转置卷积权重，布局为 [输入通道,每组输出通道,核高,核宽]。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="strides">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilation">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="outputPadding">转置卷积输出的额外尺寸，顺序为高、宽。</param>
    /// <param name="groups">将输入和输出通道划分的卷积分组数。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    public static Tensor ConvTranspose(Tensor x, Tensor w, Tensor? bias, int[] strides, int[] pads, int[] dilation, int[] outputPadding, int groups)
    {
        ValidateShape(x); ValidateShape(w); if (bias is not null) ValidateShape(bias);
        if (x.Rank != 4 || w.Rank != 4 || groups <= 0 || x.Shape[1] != w.Shape[0] || x.Shape[1] % groups != 0) throw new ArgumentException("Invalid ConvTranspose shape.");
        if (strides.Length != 2 || pads.Length != 4 || dilation.Length != 2 || outputPadding.Length != 2 || strides.Any(s => s <= 0) || pads.Any(p => p < 0) || dilation.Any(d => d <= 0) || outputPadding.Any(p => p < 0) || x.Shape.Any(d => d <= 0) || w.Shape.Any(d => d <= 0)) throw new ArgumentException("Invalid ConvTranspose parameters.");
        for (int i = 0; i < 2; i++) if (outputPadding[i] >= Math.Max(strides[i], dilation[i])) throw new ArgumentException("ConvTranspose output padding must be smaller than stride or dilation.");
        int batch = x.Shape[0], ic = x.Shape[1], ocg = w.Shape[1], oc = ocg * groups, ih = x.Shape[2], iw = x.Shape[3], kh = w.Shape[2], kw = w.Shape[3];
        if (bias is not null && bias.Length != oc) throw new ArgumentException("Invalid ConvTranspose bias.");
        int oh = checked((int)((long)(ih - 1) * strides[0] - pads[0] - pads[2] + (long)dilation[0] * (kh - 1) + outputPadding[0] + 1));
        int ow = checked((int)((long)(iw - 1) * strides[1] - pads[1] - pads[3] + (long)dilation[1] * (kw - 1) + outputPadding[1] + 1));
        if (oh <= 0 || ow <= 0) throw new ArgumentException("Invalid ConvTranspose output dimensions.");
        if (DeviceScope.For(x) is { } device)
        {
            var p = DeviceOps.Parameters((uint)(batch * oc * oh * ow), (uint)ic, (uint)ih, (uint)iw, (uint)oc, (uint)oh, (uint)ow, (uint)kh, (uint)kw, (uint)strides[0], (uint)strides[1], (uint)pads[0], (uint)pads[1], (uint)dilation[0], (uint)dilation[1], (uint)groups);
            return device.Compute(33, bias is null ? [x, w] : [x, w, bias], [batch, oc, oh, ow], p, batch * oc * oh * ow);
        }
        if (groups == 1 && kh == 2 && kw == 2 && strides.SequenceEqual([2, 2]) && pads.All(p => p == 0) && dilation.SequenceEqual([1, 1]) && outputPadding.All(p => p == 0)) return CpuSpatial.UpsampleConv(x, w, bias);
        var y = Tensor.Temporary(batch, oc, oh, ow); int icg = ic / groups;
        CpuScheduling.For(0, batch * oc, CpuScheduling.Options, bc =>
        {
            int b = bc / oc, co = bc % oc, group = co / ocg;
            Array.Fill(y.Data, bias?.Data[co] ?? 0, bc * oh * ow, oh * ow);
            for (int ci = group * icg; ci < (group + 1) * icg; ci++) for (int iy = 0; iy < ih; iy++) for (int ix = 0; ix < iw; ix++)
            {
                float value = x.Data[((b * ic + ci) * ih + iy) * iw + ix];
                for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
                {
                    int oy = iy * strides[0] - pads[0] + ky * dilation[0], ox = ix * strides[1] - pads[1] + kx * dilation[1];
                    if ((uint)oy < oh && (uint)ox < ow) y.Data[(bc * oh + oy) * ow + ox] += value * w.Data[((ci * ocg + co % ocg) * kh + ky) * kw + kx];
                }
            }
        });
        return y;
    }
    /// <summary>计算 NCHW/OIHW 卷积，可融合偏置与 SiLU；按形状选择设备、Winograd、点卷积或通用 CPU 路径。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="w">卷积权重，布局为 [输出通道,每组输入通道,核高,核宽]。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="strides">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilations">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="groups">将输入和输出通道划分的卷积分组数。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="NotSupportedException">输入张量的秩或布局不属于此算子支持的形式。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    public Tensor Conv(Tensor x, Tensor w, Tensor? bias = null, int[]? strides = null, int[]? pads = null, int[]? dilations = null, int groups = 1, bool silu = false)
    {
        ValidateShape(x); ValidateShape(w); if (bias is not null) ValidateShape(bias);
        if (x.Rank != 4 || w.Rank != 4) throw new NotSupportedException("Conv requires NCHW/OIHW.");
        strides ??= [1, 1]; pads ??= [0, 0, 0, 0]; dilations ??= [1, 1];
        if (strides.Length != 2 || pads.Length != 4 || dilations.Length != 2 || strides.Any(s => s <= 0) || pads.Any(p => p < 0) || dilations.Any(d => d <= 0) || x.Shape.Any(d => d <= 0) || w.Shape.Any(d => d <= 0)) throw new ArgumentException("Invalid convolution parameters.");
        int batches = x.Shape[0], ic = x.Shape[1], h = x.Shape[2], width = x.Shape[3], oc = w.Shape[0], kh = w.Shape[2], kw = w.Shape[3];
        if (groups <= 0 || ic % groups != 0 || oc % groups != 0 || w.Shape[1] != ic / groups || bias is not null && bias.Length != oc) throw new ArgumentException("Invalid convolution channels.");
        long heightNumerator = (long)h + pads[0] + pads[2] - (long)dilations[0] * (kh - 1) - 1;
        long widthNumerator = (long)width + pads[1] + pads[3] - (long)dilations[1] * (kw - 1) - 1;
        if (heightNumerator < 0 || widthNumerator < 0) throw new ArgumentException("Convolution kernel exceeds the padded input dimensions.");
        int oh = checked((int)(heightNumerator / strides[0] + 1)), ow = checked((int)(widthNumerator / strides[1] + 1));
        if (Backend is PrecisionCpuBackend low) return low.Convolve(x, w, bias, [batches, oc, oh, ow], strides, pads, dilations, groups, silu);
        if (Backend is IDeviceBackend { FrameActive: true } device) return device.DeviceConv(x, w, bias, [batches, oc, oh, ow], strides, pads, dilations, groups, silu);
        if (Backend is CpuBackend cpuWinograd && groups == 1 && kh == 3 && kw == 3 && strides.SequenceEqual([1, 1]) && pads.SequenceEqual([1, 1, 1, 1]) && dilations.SequenceEqual([1, 1]) && ic >= 8 && oc >= 8 && oh * ow >= 64)
        {
            return Avx2.IsSupported && w.IsConstant && ((ic is >= 32 and <= 64 && oc is >= 32 and <= 128 && oh <= 160 && ow <= 160) || CpuWinograd4.AdditionalHotShape(ic, oc, oh, ow)) ? CpuWinograd4.Conv(cpuWinograd, x, w, bias, silu) : CpuWinograd.Conv(cpuWinograd, x, w, bias, silu);
        }
        // 已验证的 CPU 卷积路径均完整写入输出；其他扩展后端保持原有清零约定。
        var y = Backend is CpuBackend ? Tensor.TemporaryUninitialized(batches, oc, oh, ow) : Tensor.Temporary(batches, oc, oh, ow);
        int spatial = oh * ow, cig = ic / groups, cog = oc / groups, k = cig * kh * kw;
        if (Backend is CpuBackend cpuPointwise && groups == 1 && batches == 1 && kh == 1 && kw == 1 && strides.SequenceEqual([1, 1]) && pads.All(p => p == 0))
        {
            if (CpuPointwise.FusionEnabled && CpuPointwise.Supported)
            {
                CpuPointwise.Conv(cpuPointwise, [x], w, bias, y, silu);
                return y;
            }
            cpuPointwise.MatMul(w.Data, x.Data, y.Data, oc, spatial, ic);
            CpuEpilogue.Apply(y.Data, bias?.Data, oc, spatial, silu, cpuPointwise.Parallel);
            return y;
        }
        if (groups == ic && oc == ic)
        {
            if (Backend is CpuBackend cpuDepthwise && (strides[1] == 1 || strides[1] == 2 && Avx2.IsSupported && Fma.IsSupported))
            {
                CpuSpatial.Depthwise(x, w, bias, y, strides, pads, dilations, silu, cpuDepthwise.Parallel);
                return y;
            }
            CpuScheduling.For(0, batches * ic, Backend is CpuBackend cpu ? cpu.Parallel : CpuScheduling.Options, bc =>
            {
                int channel = bc % ic;
                for (int oy = 0; oy < oh; oy++) for (int ox = 0; ox < ow; ox++)
                {
                    float value = bias?.Data[channel] ?? 0;
                    for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
                    {
                        int iy = oy * strides[0] - pads[0] + ky * dilations[0], ix = ox * strides[1] - pads[1] + kx * dilations[1];
                        if ((uint)iy < h && (uint)ix < width) value += x.Data[(bc * h + iy) * width + ix] * w.Data[(channel * kh + ky) * kw + kx];
                    }
                    y.Data[(bc * oh + oy) * ow + ox] = value;
                }
            });
            return silu ? Activation(y, true) : y;
        }
        if (Backend is IConvolutionBackend convolution)
        {
            convolution.Conv(x, w, bias, y, strides, pads, dilations, groups, silu); return y;
        }
        if (Backend is CpuBackend cpuImplicit && groups == 1 && Avx2.IsSupported && Fma.IsSupported)
        {
            CpuConvolution.Conv(cpuImplicit, x, w, bias, y, strides, pads, dilations, silu);
            return y;
        }
        float[] col = ArrayPool<float>.Shared.Rent(k * spatial), dst = ArrayPool<float>.Shared.Rent(cog * spatial);
        float[] weights = groups == 1 ? w.Data : new float[cog * k];
        try
        {
            for (int b = 0; b < batches; b++) for (int g = 0; g < groups; g++)
            {
                if (groups != 1) Array.Copy(w.Data, g * cog * k, weights, 0, weights.Length);
                if (kh == 1 && kw == 1 && strides.SequenceEqual([1, 1]) && pads.All(p => p == 0))
                    Array.Copy(x.Data, (b * ic + g * cig) * h * width, col, 0, k * spatial);
                else
                {
                    // 按输入通道和卷积核坐标展开一个 im2col 行，越界位置写零。
                    // 参数 r：展开后的 K 轴行索引。
                    void ExpandRow(int r)
                    {
                        int ci = r / (kh * kw), ky = r / kw % kh, kx = r % kw, row = r * spatial;
                        int ix0 = kx * dilations[1] - pads[1];
                        int first = Math.Clamp((-ix0 + strides[1] - 1) / strides[1], 0, ow);
                        int last = Math.Clamp((width - 1 - ix0) / strides[1] + 1, first, ow);
                        for (int oy = 0; oy < oh; oy++)
                        {
                            int iy = oy * strides[0] - pads[0] + ky * dilations[0], target = row + oy * ow;
                            if ((uint)iy >= h) { Array.Clear(col, target, ow); continue; }
                            Array.Clear(col, target, first); Array.Clear(col, target + last, ow - last);
                            int source = ((b * ic + g * cig + ci) * h + iy) * width + ix0 + first * strides[1];
                            if (strides[1] == 1) Array.Copy(x.Data, source, col, target + first, last - first);
                            else for (int ox = first; ox < last; ox++, source += strides[1]) col[target + ox] = x.Data[source];
                        }
                    }
                    if ((long)k * spatial >= 262144) CpuScheduling.For(0, k, Backend is CpuBackend cpu ? cpu.Parallel : CpuScheduling.Options, ExpandRow);
                    else for (int r = 0; r < k; r++) ExpandRow(r);
                }
                Backend.MatMul(weights, col, dst, cog, spatial, k);
                for (int co = 0; co < cog; co++)
                {
                    int target = (b * oc + g * cog + co) * spatial;
                    float bv = bias?.Data[g * cog + co] ?? 0;
                    int j = 0; var vb = new Vector<float>(bv);
                    for (; j <= spatial - Vector<float>.Count; j += Vector<float>.Count)
                    {
                        var value = new Vector<float>(dst, co * spatial + j) + vb;
                        if (silu) value = CpuActivation.Silu(value);
                        value.CopyTo(y.Data, target + j);
                    }
                    for (; j < spatial; j++) { float value = dst[co * spatial + j] + bv; y.Data[target + j] = silu ? value * Sigmoid(value) : value; }
                }
            }
        }
        finally { ArrayPool<float>.Shared.Return(col); ArrayPool<float>.Shared.Return(dst); }
        return y;
    }
    /// <summary>沿通道轴拼接输入后进行卷积；符合条件的常量 1×1 卷积直接打包各输入，省去中间拼接张量。</summary>
    /// <param name="inputs">按通道拼接顺序排列的输入张量。</param>
    /// <param name="weights">卷积或矩阵乘法权重；需要缓存的路径要求权重保持只读。</param>
    /// <param name="bias">按输出通道排列的偏置；为 null 时使用零偏置。</param>
    /// <param name="strides">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="dilations">卷积膨胀系数，顺序为高、宽。</param>
    /// <param name="groups">将输入和输出通道划分的卷积分组数。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    public Tensor ConvConcat(IReadOnlyList<Tensor> inputs, Tensor weights, Tensor? bias = null, int[]? strides = null, int[]? pads = null, int[]? dilations = null, int groups = 1, bool silu = false)
    {
        if (inputs.Count == 0) throw new ArgumentException("ConvConcat requires at least one input.", nameof(inputs));
        foreach (var input in inputs) ValidateShape(input);
        ValidateShape(weights); if (bias is not null) ValidateShape(bias);
        strides ??= [1, 1]; pads ??= [0, 0, 0, 0]; dilations ??= [1, 1];
        bool eligible = CpuPointwise.ConcatEnabled && CpuPointwise.Supported && Backend is CpuBackend && inputs[0].Rank == 4 && inputs[0].Shape[0] == 1 && inputs[0].Shape.All(d => d > 0)
            && weights.IsConstant && weights.Rank == 4 && weights.Shape.All(d => d > 0) && weights.Shape[2] == 1 && weights.Shape[3] == 1 && groups == 1
            && strides.SequenceEqual([1, 1]) && pads.SequenceEqual([0, 0, 0, 0]) && dilations.SequenceEqual([1, 1]);
        int channels = 0;
        if (eligible)
        {
            foreach (var input in inputs)
            {
                if (input.Rank != 4 || input.Shape[0] != 1 || input.Shape[1] <= 0 || input.Shape[2] != inputs[0].Shape[2] || input.Shape[3] != inputs[0].Shape[3]) { eligible = false; break; }
                channels = checked(channels + input.Shape[1]);
            }
            eligible &= channels == weights.Shape[1] && (bias is null || bias.Length == weights.Shape[0]);
        }
        if (!eligible) return Conv(Concat(inputs, 1), weights, bias, strides, pads, dilations, groups, silu);
        var output = Tensor.TemporaryUninitialized(1, weights.Shape[0], inputs[0].Shape[2], inputs[0].Shape[3]);
        CpuPointwise.Conv((CpuBackend)Backend, inputs, weights, bias, output, silu);
        return output;
    }
    /// <summary>计算张量矩阵乘法，前导批次维度按广播规则对齐。</summary>
    /// <param name="a">左侧张量操作数。</param>
    /// <param name="b">右侧张量操作数。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="NotSupportedException">输入张量的秩或布局不属于此算子支持的形式。</exception>
    public Tensor MatMul(Tensor a, Tensor b)
    {
        ValidateShape(a); ValidateShape(b);
        if (a.Rank < 2 || b.Rank < 2) throw new NotSupportedException("MatMul requires rank >= 2.");
        int m = a.Shape[^2], k = a.Shape[^1], n = b.Shape[^1];
        if (b.Shape[^2] != k) throw new ArgumentException("MatMul dimensions mismatch.");
        int[] aBatch = a.Shape[..^2], bBatch = b.Shape[..^2];
        int[] batch = BroadcastShape(aBatch, bBatch);
        if (Backend is IDeviceBackend { FrameActive: true } device) return device.DeviceMatMul(a, b, [.. batch, m, n]);
        if (Backend is CpuBackend cpu)
        {
            var output = Tensor.TemporaryUninitialized([.. batch, m, n]);
            int batchCount = Tensor.Count(batch);
            if (batchCount == 0 || m == 0 || n == 0) return output;
            if (k == 0) { Array.Clear(output.Data); return output; }
            int mk = checked(m * k), kn = checked(k * n);
            var aOffsets = new int[batchCount]; var bOffsets = new int[batchCount];
            for (int t = 0; t < batchCount; t++)
            {
                aOffsets[t] = checked(BroadcastIndex(t, batch, aBatch) * mk);
                bOffsets[t] = checked(BroadcastIndex(t, batch, bBatch) * kn);
            }
            CpuGemm.MultiplyOffsetsBatch(a.Data, b.Data, output.Data, aOffsets, bOffsets, m, n, k, cpu.Parallel);
            return output;
        }
        var result = Tensor.Temporary([.. batch, m, n]); int count = Tensor.Count(batch);
        if (m == 0 || n == 0 || k == 0) { Array.Clear(result.Data); return result; }
        for (int t = 0; t < count; t++)
        {
            int ai = BroadcastIndex(t, batch, aBatch) * m * k, bi = BroadcastIndex(t, batch, bBatch) * k * n;
            var aa = ai == 0 && a.Length == m * k ? a.Data : a.Data.AsSpan(ai, m * k).ToArray();
            var bb = bi == 0 && b.Length == k * n ? b.Data : b.Data.AsSpan(bi, k * n).ToArray();
            var cc = count == 1 ? result.Data : new float[m * n]; Backend.MatMul(aa, bb, cc, m, n, k);
            if (count != 1) cc.CopyTo(result.Data, t * m * n);
        }
        return result;
    }
    /// <summary>从最右侧维度开始合并两个形状，按维度相等或其中一方为 1 的规则确定广播结果。</summary>
    /// <param name="a">参与广播的第一个形状。</param>
    /// <param name="b">参与广播的第二个形状。</param>
    /// <returns>符合广播规则的新形状数组。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    public static int[] BroadcastShape(int[] a, int[] b)
    {
        int[] shape = new int[Math.Max(a.Length, b.Length)];
        for (int i = 1; i <= shape.Length; i++)
        {
            int x = i <= a.Length ? a[^i] : 1, y = i <= b.Length ? b[^i] : 1;
            if (x != y && x != 1 && y != 1) throw new ArgumentException("Incompatible broadcast.");
            shape[^i] = x == 1 ? y : x;
        }
        return shape;
    }
    /// <summary>将广播结果中的扁平索引映射到源张量，忽略源形状中为 1 的广播维度。</summary>
    /// <param name="flat">广播结果中的行主序扁平索引。</param>
    /// <param name="target">广播结果的目标形状。</param>
    /// <param name="source">广播映射的源形状。</param>
    /// <returns>源张量中的行主序扁平索引。</returns>
    private static int BroadcastIndex(int flat, int[] target, int[] source)
    {
        int index = 0, stride = 1;
        for (int t = target.Length - 1, s = source.Length - 1; t >= 0; t--, s--)
        {
            int c = flat % target[t]; flat /= target[t];
            if (s >= 0) { if (source[s] != 1) index += c * stride; stride *= source[s]; }
        }
        return index;
    }
    /// <summary>对两个张量按广播规则逐元素调用二元函数。</summary>
    /// <param name="a">左侧张量操作数。</param>
    /// <param name="b">右侧张量操作数。</param>
    /// <param name="f">逐元素调用的函数。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    public static Tensor Binary(Tensor a, Tensor b, Func<float, float, float> f)
    {
        ValidateShape(a); ValidateShape(b);
        var shape = BroadcastShape(a.Shape, b.Shape); var y = Tensor.Temporary(shape);
        if (a.Shape.SequenceEqual(b.Shape)) for (int i = 0; i < y.Length; i++) y.Data[i] = f(a.Data[i], b.Data[i]);
        else if (b.Length == 1) for (int i = 0; i < y.Length; i++) y.Data[i] = f(a.Data[i], b.Data[0]);
        else if (a.Length == 1) for (int i = 0; i < y.Length; i++) y.Data[i] = f(a.Data[0], b.Data[i]);
        else for (int i = 0; i < y.Length; i++) y.Data[i] = f(a.Data[BroadcastIndex(i, shape, a.Shape)], b.Data[BroadcastIndex(i, shape, b.Shape)]);
        return y;
    }
    /// <summary>对输入张量逐元素调用一元函数，并保持输入形状。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="f">逐元素调用的函数。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    public static Tensor Unary(Tensor x, Func<float, float> f)
    {
        ValidateShape(x);
        var y = Tensor.Temporary(x.Shape); for (int i = 0; i < y.Length; i++) y.Data[i] = f(x.Data[i]); return y;
    }
    /// <summary>按广播规则执行加、减、乘或除，连续的最内层元素使用 SIMD 处理。</summary>
    /// <param name="a">左侧张量操作数。</param>
    /// <param name="b">右侧张量操作数。</param>
    /// <param name="operation">运算编号：10 加、11 减、12 乘、13 除。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="ArgumentOutOfRangeException">操作编号或操作轴超出允许的范围。</exception>
    public static Tensor Arithmetic(Tensor a, Tensor b, int operation)
    {
        ValidateShape(a); ValidateShape(b);
        if (DeviceScope.For(a, b) is { } owner && Math.Max(a.Rank, b.Rank) <= 4) return DeviceOps.Binary(owner, a, b, operation);
        if (operation is < 10 or > 13) throw new ArgumentOutOfRangeException(nameof(operation));
        int[] shape = BroadcastShape(a.Shape, b.Shape); var result = Tensor.TemporaryUninitialized(shape);
        if (result.Length == 0) return result;
        float[] ad = a.Data, bd = b.Data, output = result.Data;
        int inner = shape.Length == 0 ? 1 : shape[^1], outer = result.Length / inner;
        bool scalarA = a.Rank == 0 || a.Shape[^1] == 1, scalarB = b.Rank == 0 || b.Shape[^1] == 1;
        if (a.Shape.SequenceEqual(shape) && b.Shape.SequenceEqual(shape)) { inner = result.Length; outer = 1; scalarA = scalarB = false; }
        else if (a.Length == 1 && b.Shape.SequenceEqual(shape)) { inner = result.Length; outer = 1; scalarA = true; scalarB = false; }
        else if (b.Length == 1 && a.Shape.SequenceEqual(shape)) { inner = result.Length; outer = 1; scalarA = false; scalarB = true; }
        for (int o = 0; o < outer; o++)
        {
            int dst = o * inner, ai = BroadcastIndex(dst, shape, a.Shape), bi = BroadcastIndex(dst, shape, b.Shape), j = 0;
            var va = new Vector<float>(ad[ai]); var vb = new Vector<float>(bd[bi]);
            for (; j <= inner - Vector<float>.Count; j += Vector<float>.Count)
            {
                var x = scalarA ? va : new Vector<float>(ad, ai + j); var y = scalarB ? vb : new Vector<float>(bd, bi + j);
                (operation switch { 10 => x + y, 11 => x - y, 12 => x * y, _ => x / y }).CopyTo(output, dst + j);
            }
            for (; j < inner; j++)
            {
                float x = ad[ai + (scalarA ? 0 : j)], y = bd[bi + (scalarB ? 0 : j)];
                output[dst + j] = operation switch { 10 => x + y, 11 => x - y, 12 => x * y, _ => x / y };
            }
        }
        return result;
    }
    /// <summary>将张量每个元素乘以同一个 FP32 系数。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="scale">所有元素共用的单精度乘数。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    public static Tensor Scale(Tensor x, float scale)
    {
        ValidateShape(x);
        if (DeviceScope.For(x) is { } owner) return DeviceOps.Unary(owner, x, 4, scale);
        var result = Tensor.TemporaryUninitialized(x.Shape); float[] input = x.Data, output = result.Data; var factor = new Vector<float>(scale); int i = 0;
        for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count) (new Vector<float>(input, i) * factor).CopyTo(output, i);
        for (; i < x.Length; i++) output[i] = input[i] * scale;
        return result;
    }
    /// <summary>逐元素计算 Sigmoid，或在启用时计算 SiLU。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="silu">是否在输出上融合 SiLU 激活。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    public static Tensor Activation(Tensor x, bool silu)
    {
        ValidateShape(x);
        if (DeviceScope.For(x) is { } owner) return DeviceOps.Unary(owner, x, silu ? 1 : 2);
        var y = Tensor.TemporaryUninitialized(x.Shape); int i = 0;
        for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            var v = new Vector<float>(x.Data, i); var result = CpuActivation.Sigmoid(v);
            if (silu) result *= v; result.CopyTo(y.Data, i);
        }
        for (; i < x.Length; i++) y.Data[i] = silu ? x.Data[i] * Sigmoid(x.Data[i]) : Sigmoid(x.Data[i]);
        return y;
    }
    /// <summary>计算单精度 Sigmoid：1/(1+exp(-x))。</summary>
    /// <param name="x">待计算 Sigmoid 的单精度值。</param>
    /// <returns>计算后的单精度标量结果。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
    /// <summary>沿指定轴依次拼接张量，要求其余维度相同。</summary>
    /// <param name="tensors">需要返回或拼接的张量序列。</param>
    /// <param name="axis">操作轴，允许使用相对末尾的负轴索引。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="ArgumentOutOfRangeException">操作编号或操作轴超出允许的范围。</exception>
    public static Tensor Concat(IReadOnlyList<Tensor> tensors, int axis)
    {
        foreach (var tensor in tensors) ValidateShape(tensor);
        axis = Tensor.Axis(axis, tensors[0].Rank); int[] shape = (int[])tensors[0].Shape.Clone(); shape[axis] = 0;
        foreach (var x in tensors)
        {
            if (x.Rank != shape.Length || x.Shape.Where((d, i) => i != axis && d != shape[i]).Any()) throw new ArgumentException("Concat shape mismatch.");
            shape[axis] += x.Shape[axis];
        }
        if (DeviceScope.For(tensors.ToArray()) is { } owner) return owner.DeviceConcat(tensors, axis, shape);
        var y = Tensor.TemporaryUninitialized(shape); int inner = Tensor.Count(shape[(axis + 1)..]), outer = Tensor.Count(shape[..axis]);
        for (int o = 0; o < outer; o++)
        {
            int dst = o * shape[axis] * inner;
            foreach (var x in tensors) { int size = x.Shape[axis] * inner; Array.Copy(x.Data, o * size, y.Data, dst, size); dst += size; }
        }
        return y;
    }
    /// <summary>按给定长度沿指定轴拆分张量，保持各段在原张量中的顺序。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="axis">操作轴，允许使用相对末尾的负轴索引。</param>
    /// <param name="sizes">沿拆分轴的各段长度，总和必须等于该轴长度。</param>
    /// <returns>按拆分顺序排列的各段张量。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="ArgumentOutOfRangeException">操作编号或操作轴超出允许的范围。</exception>
    public static Tensor[] Split(Tensor x, int axis, int[] sizes)
    {
        ValidateShape(x);
        axis = Tensor.Axis(axis, x.Rank); if (sizes.Sum() != x.Shape[axis]) throw new ArgumentException("Split sizes mismatch.");
        int inner = Tensor.Count(x.Shape[(axis + 1)..]), outer = Tensor.Count(x.Shape[..axis]), start = 0;
        var outputs = new List<Tensor>();
        foreach (int size in sizes)
        {
            var shape = (int[])x.Shape.Clone(); shape[axis] = size;
            if (DeviceScope.For(x) is { } owner) { outputs.Add(owner.DeviceSlice(x, shape, axis, start)); start += size; continue; }
            var y = Tensor.Temporary(shape);
            for (int o = 0; o < outer; o++) Array.Copy(x.Data, (o * x.Shape[axis] + start) * inner, y.Data, o * size * inner, size * inner);
            outputs.Add(y); start += size;
        }
        return outputs.ToArray();
    }
    /// <summary>按轴排列转置张量；默认反转全部轴，最后两个轴交换可走 CPU SIMD 快路径。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="perm">输出各轴对应的输入轴排列；为 null 时反转全部轴。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    public static Tensor Transpose(Tensor x, int[]? perm = null)
    {
        ValidateShape(x);
        perm ??= Enumerable.Range(0, x.Rank).Reverse().ToArray();
        if (perm.Length != x.Rank || perm.Distinct().Count() != x.Rank || perm.Any(p => p < 0 || p >= x.Rank)) throw new ArgumentException("Invalid permutation.");
        var shape = perm.Select(i => x.Shape[i]).ToArray();
        if (perm.Where(p => x.Shape[p] != 1).SequenceEqual(Enumerable.Range(0, x.Rank).Where(p => x.Shape[p] != 1))) return x.Reshape(shape);
        if (DeviceScope.For(x) is { } owner && x.Rank <= 4) return DeviceOps.Transpose(owner, x, perm, shape);
        if (perm.SequenceEqual(Enumerable.Range(0, x.Rank - 2).Concat([x.Rank - 1, x.Rank - 2])))
            return CpuTranspose.LastTwo(x, shape);
        var y = Tensor.Temporary(shape); var xs = Tensor.Strides(x.Shape);
        for (int i = 0; i < y.Length; i++)
        {
            int src = 0, index = i;
            for (int d = y.Rank - 1; d >= 0; d--) { src += index % shape[d] * xs[perm[d]]; index /= shape[d]; }
            y.Data[i] = x.Data[src];
        }
        return y;
    }
    /// <summary>沿指定轴先减去最大值，再计算指数和归一化概率，以提高数值稳定性。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="axis">操作轴，允许使用相对末尾的负轴索引。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="ArgumentOutOfRangeException">操作编号或操作轴超出允许的范围。</exception>
    public static Tensor Softmax(Tensor x, int axis)
    {
        ValidateShape(x);
        axis = Tensor.Axis(axis, x.Rank);
        if (DeviceScope.For(x) is { } owner) return DeviceOps.Softmax(owner, x, axis);
        var y = Tensor.Temporary(x.Shape);
        int inner = Tensor.Count(x.Shape[(axis + 1)..]), size = x.Shape[axis], outer = Tensor.Count(x.Shape[..axis]);
        float[] data = x.Data, target = y.Data;
        if (inner == 1 && size >= Vector<float>.Count)
        {
            // 对一个连续 Softmax 行计算最大值、指数总和与归一化概率。
            // 参数 o：当前连续 Softmax 行的索引。
            void Row(int o)
            {
                int start = o * size, end = start + size, c = start; var maximum = new Vector<float>(float.NegativeInfinity);
                for (; c <= end - Vector<float>.Count; c += Vector<float>.Count) maximum = Vector.Max(maximum, new Vector<float>(data, c));
                float max = float.NegativeInfinity;
                for (int lane = 0; lane < Vector<float>.Count; lane++) max = MathF.Max(max, maximum[lane]);
                for (; c < end; c++) max = MathF.Max(max, data[c]);
                var vmax = new Vector<float>(max); var sumVector = Vector<float>.Zero; c = start;
                for (; c <= end - Vector<float>.Count; c += Vector<float>.Count) { var value = CpuActivation.Exp(new Vector<float>(data, c) - vmax); value.CopyTo(target, c); sumVector += value; }
                float sum = Vector.Sum(sumVector);
                for (; c < end; c++) { float value = MathF.Exp(data[c] - max); target[c] = value; sum += value; }
                var denominator = new Vector<float>(sum); c = start;
                for (; c <= end - Vector<float>.Count; c += Vector<float>.Count) (new Vector<float>(target, c) / denominator).CopyTo(target, c);
                for (; c < end; c++) target[c] /= sum;
            }
            if ((long)outer * size >= 32768 && outer > 1) CpuScheduling.For(0, outer, CpuScheduling.Options, Row);
            else for (int o = 0; o < outer; o++) Row(o);
            return y;
        }
        for (int o = 0; o < outer; o++)
        {
            int j = 0;
            for (; j <= inner - Vector<float>.Count; j += Vector<float>.Count)
            {
                int offset = o * size * inner + j; var maximum = new Vector<float>(float.NegativeInfinity); var sum = Vector<float>.Zero;
                for (int c = 0; c < size; c++) maximum = Vector.Max(maximum, new Vector<float>(data, offset + c * inner));
                for (int c = 0; c < size; c++) { var e = CpuActivation.Exp(new Vector<float>(data, offset + c * inner) - maximum); e.CopyTo(target, offset + c * inner); sum += e; }
                for (int c = 0; c < size; c++) (new Vector<float>(target, offset + c * inner) / sum).CopyTo(target, offset + c * inner);
            }
            for (; j < inner; j++)
            {
                int offset = o * size * inner + j; float max = float.NegativeInfinity, sum = 0;
                for (int c = 0; c < size; c++) max = MathF.Max(max, x.Data[offset + c * inner]);
                for (int c = 0; c < size; c++) { float v = MathF.Exp(x.Data[offset + c * inner] - max); y.Data[offset + c * inner] = v; sum += v; }
                for (int c = 0; c < size; c++) y.Data[offset + c * inner] /= sum;
            }
        }
        return y;
    }
    /// <summary>在 NCHW 张量上计算最大池化或平均池化，可选择是否将填充计入平均值分母。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="kernel">池化窗口尺寸，或方形深度卷积核的边长。</param>
    /// <param name="strides">空间步长，顺序为高、宽。</param>
    /// <param name="pads">空间填充，顺序为上、左、下、右。</param>
    /// <param name="average">是否进行平均池化；否则进行最大池化。</param>
    /// <param name="countIncludePad">平均池化的分母是否包含填充位置。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    /// <exception cref="NotSupportedException">输入张量的秩或布局不属于此算子支持的形式。</exception>
    /// <exception cref="OverflowException">形状或缓冲区元素数量的受检查整数运算超出支持范围。</exception>
    public static Tensor Pool(Tensor x, int[] kernel, int[] strides, int[] pads, bool average = false, bool countIncludePad = true)
    {
        ValidateShape(x);
        if (x.Rank != 4) throw new NotSupportedException("Pool requires NCHW.");
        if (kernel.Length != 2 || strides.Length != 2 || pads.Length != 4 || kernel.Any(k => k <= 0) || strides.Any(s => s <= 0) || pads.Any(p => p < 0) || x.Shape[2] <= 0 || x.Shape[3] <= 0) throw new ArgumentException("Invalid pooling parameters.");
        int h = x.Shape[2], w = x.Shape[3]; long hn = (long)h + pads[0] + pads[2] - kernel[0], wn = (long)w + pads[1] + pads[3] - kernel[1];
        if (hn < 0 || wn < 0) throw new ArgumentException("Pooling kernel exceeds the padded input dimensions.");
        int oh = checked((int)(hn / strides[0] + 1)), ow = checked((int)(wn / strides[1] + 1));
        if (DeviceScope.For(x) is { } owner)
            return owner.Compute(average ? 22 : 21, [x], [x.Shape[0], x.Shape[1], oh, ow], DeviceOps.Parameters((uint)(x.Shape[0] * x.Shape[1] * oh * ow), (uint)h, (uint)w, (uint)oh, (uint)ow, (uint)kernel[0], (uint)kernel[1], (uint)strides[0], (uint)strides[1], (uint)pads[0], (uint)pads[1], countIncludePad ? 1u : 0u), x.Shape[0] * x.Shape[1] * oh * ow);
        if (!average) return CpuSpatial.MaxPool(x, kernel, strides, pads, oh, ow);
        return CpuSpatial.AveragePool(x, kernel, strides, pads, oh, ow, countIncludePad);
    }
    /// <summary>以最近邻采样将 NCHW 张量缩放到指定空间尺寸。</summary>
    /// <param name="x">输入张量；空间算子使用 NCHW 布局。</param>
    /// <param name="h">空间平面的高度。</param>
    /// <param name="w">目标空间宽度。</param>
    /// <returns>按本算子输出形状生成的临时张量；发布为公开输出前由调用方完成池所有权转移。</returns>
    /// <exception cref="ArgumentException">输入形状、缓冲区长度或当前快路径要求的参数不匹配。</exception>
    public static Tensor ResizeNearest(Tensor x, int h, int w)
    {
        ValidateShape(x);
        if (x.Rank != 4 || h <= 0 || w <= 0 || x.Shape[2] <= 0 || x.Shape[3] <= 0) throw new ArgumentException("Resize requires NCHW and positive size.");
        if (DeviceScope.For(x) is { } owner)
            return owner.Compute(20, [x], [x.Shape[0], x.Shape[1], h, w], DeviceOps.Parameters((uint)(x.Shape[0] * x.Shape[1] * h * w), (uint)x.Shape[2], (uint)x.Shape[3], (uint)h, (uint)w), x.Shape[0] * x.Shape[1] * h * w);
        return CpuSpatial.ResizeNearest(x, h, w);
    }
}
