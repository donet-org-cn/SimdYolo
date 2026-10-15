namespace SimdYolo.Training;

/// <summary>实现检测头所需的卷积、训练 BatchNorm 和 SiLU 的前向与反向。</summary>
/// <param name="backend">CPU SIMD 或 Vulkan 矩阵乘法后端，由外部管理生命周期。</param>
internal sealed class HeadOperations(IComputeBackend backend)
{
    /// <summary>以 im2col 和可变权重 GEMM 计算卷积，并登记 dW、db 和 dX；不使用只读权重缓存。</summary>
    /// <param name="tape">保存本次反向拓扑的记录器。</param><param name="input">单图 NCHW 激活或冻结特征。</param>
    /// <param name="weight">可更新卷积权重。</param><param name="bias">可选偏置。</param><param name="padding">每边零填充宽度。</param>
    /// <param name="groups">卷积分组数；支持深度卷积。</param>
    /// <returns>卷积结果及其梯度节点。</returns>
    /// <exception cref="NotSupportedException">批量、卷积核或输出尺寸不满足训练头限制。</exception>
    internal GradientValue Conv(GradientTape tape, GradientValue input, TrainingParameter weight, TrainingParameter? bias, int padding, int groups = 1)
    {
        if (groups != 1) return GroupedConv(tape, input, weight, bias, padding, groups);
        var x = input.Value; int batch = x.Shape[0], ic = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        int oc = weight.Shape[0], kh = weight.Shape[2], kw = weight.Shape[3], oh = h + 2 * padding - kh + 1, ow = w + 2 * padding - kw + 1;
        if (batch != 1 || weight.Shape[1] != ic || kh != kw || oh != h || ow != w) throw new NotSupportedException("训练头卷积要求 batch=1、方形卷积核、步长1和保尺寸填充。");
        int spatial = h * w, k = ic * kh * kw;
        float[] columns = new float[k * spatial];
        for (int c = 0; c < ic; c++) for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
        {
            int row = ((c * kh + ky) * kw + kx) * spatial;
            for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++)
            { int iy = y + ky - padding, ix = xx + kx - padding; if ((uint)iy < h && (uint)ix < w) columns[row + y * w + xx] = x.Data[(c * h + iy) * w + ix]; }
        }
        var output = new GradientValue(new Tensor(1, oc, h, w));
        backend.MatMul(weight.Values, columns, output.Value.Data, oc, spatial, k);
        if (bias is not null) for (int c = 0; c < oc; c++) for (int p = 0; p < spatial; p++) output.Value.Data[c * spatial + p] += bias.Values[c];
        tape.Record(() =>
        {
            if (!output.HasGradient) return;
            weight.HasGradient = true; if (bias is not null) bias.HasGradient = true;
            float[] dy = output.Gradient;
            float[] columnsT = Transpose(columns, k, spatial), dw = new float[weight.Values.Length];
            backend.MatMul(dy, columnsT, dw, oc, k, spatial);
            Add(weight.Gradient, dw);
            if (bias is not null) for (int c = 0; c < oc; c++) { float sum = 0; for (int p = 0; p < spatial; p++) sum += dy[c * spatial + p]; bias.Gradient[c] += sum; }
            if (!input.RequiresGradient) return;
            float[] wt = Transpose(weight.Values, oc, k), dcolumns = new float[columns.Length];
            backend.MatMul(wt, dy, dcolumns, k, spatial, oc);
            float[] dx = input.Gradient;
            for (int c = 0; c < ic; c++) for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
            {
                int row = ((c * kh + ky) * kw + kx) * spatial;
                for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++)
                { int iy = y + ky - padding, ix = xx + kx - padding; if ((uint)iy < h && (uint)ix < w) dx[(c * h + iy) * w + ix] += dcolumns[row + y * w + xx]; }
            }
        });
        return output;
    }

    /// <summary>直接计算分组卷积及其 dW/db/dX；深度卷积避免大量微型 GEMM 和设备往返。</summary>
    /// <param name="tape">反向记录器。</param><param name="input">单图 NCHW 输入。</param><param name="weight">分组权重。</param>
    /// <param name="bias">可选偏置。</param><param name="padding">对称填充。</param><param name="groups">分组数量。</param>
    /// <returns>保尺寸的卷积节点。</returns><exception cref="NotSupportedException">形状或分组不合法。</exception>
    private static GradientValue GroupedConv(GradientTape tape, GradientValue input, TrainingParameter weight, TrainingParameter? bias, int padding, int groups)
    {
        var x = input.Value; int ic = x.Shape[1], h = x.Shape[2], w = x.Shape[3], oc = weight.Shape[0], ig = weight.Shape[1], kh = weight.Shape[2], kw = weight.Shape[3];
        if (x.Shape[0] != 1 || groups < 1 || ic != ig * groups || oc % groups != 0 || kh != kw || 2 * padding != kh - 1) throw new NotSupportedException("训练分组卷积形状不合法。");
        int spatial = h * w, og = oc / groups; var output = new GradientValue(new Tensor(1, oc, h, w));
        for (int o = 0; o < oc; o++)
        {
            if (bias is not null) Array.Fill(output.Value.Data, bias.Values[o], o * spatial, spatial);
            int first = o / og * ig;
            for (int c = 0; c < ig; c++) for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
            {
                float factor = weight.Values[((o * ig + c) * kh + ky) * kw + kx];
                for (int y = Math.Max(0, padding - ky); y < Math.Min(h, h + padding - ky); y++)
                {
                    int left = Math.Max(0, padding - kx), right = Math.Min(w, w + padding - kx);
                    Accumulate(output.Value.Data, o * spatial + y * w + left, x.Data, (first + c) * spatial + (y + ky - padding) * w + left + kx - padding, right - left, factor);
                }
            }
        }
        tape.Record(() =>
        {
            if (!output.HasGradient) return;
            weight.HasGradient = true; if (bias is not null) bias.HasGradient = true;
            for (int o = 0; o < oc; o++)
            {
                if (bias is not null) { float sum = 0; for (int p = 0; p < spatial; p++) sum += output.Gradient[o * spatial + p]; bias.Gradient[o] += sum; }
                int first = o / og * ig;
                for (int c = 0; c < ig; c++) for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
                {
                    int wi = ((o * ig + c) * kh + ky) * kw + kx; float sum = 0;
                    for (int y = Math.Max(0, padding - ky); y < Math.Min(h, h + padding - ky); y++)
                    {
                        int left = Math.Max(0, padding - kx), right = Math.Min(w, w + padding - kx);
                        int xi = (first + c) * spatial + (y + ky - padding) * w + left + kx - padding, yi = o * spatial + y * w + left;
                        for (int xx = 0; xx < right - left; xx++) sum += output.Gradient[yi + xx] * x.Data[xi + xx];
                        if (input.RequiresGradient) Accumulate(input.Gradient, xi, output.Gradient, yi, right - left, weight.Values[wi]);
                    }
                    weight.Gradient[wi] += sum;
                }
            }
        });
        return output;
    }

    /// <summary>对连续行执行 SIMD 缩放累加，尾部按标量处理。</summary>
    /// <param name="target">原地累加的目标数组。</param><param name="ti">目标起始偏移。</param>
    /// <param name="source">连续源数组。</param><param name="si">源起始偏移。</param>
    /// <param name="count">待处理的元素数量。</param><param name="scale">源元素的缩放系数。</param>
    private static void Accumulate(float[] target, int ti, float[] source, int si, int count, float scale)
    {
        int p = 0, width = System.Numerics.Vector<float>.Count; var factor = new System.Numerics.Vector<float>(scale);
        for (; p <= count - width; p += width) (new System.Numerics.Vector<float>(target, ti + p) + new System.Numerics.Vector<float>(source, si + p) * factor).CopyTo(target, ti + p);
        for (; p < count; p++) target[ti + p] += source[si + p] * scale;
    }

    /// <summary>使用当前图像空间维度统计 BatchNorm，并更新供推理使用的无偏运行方差。</summary>
    /// <param name="tape">训练时登记反向闭包的记录器。</param><param name="input">单图卷积激活。</param>
    /// <param name="gamma">可训练的通道缩放。</param><param name="beta">可训练的通道偏移。</param>
    /// <param name="runningMean">可原地更新的通道运行均值。</param><param name="runningVariance">可原地更新的无偏运行方差。</param>
    /// <param name="epsilon">方差稳定项。</param><param name="momentum">运行统计的新观测权重。</param>
    /// <param name="training">为 true 时使用当前空间统计并记录反向，否则使用运行统计。</param>
    /// <returns>归一化、缩放和平移后的梯度节点。</returns>
    internal GradientValue BatchNorm(GradientTape tape, GradientValue input, TrainingParameter gamma, TrainingParameter beta, float[] runningMean, float[] runningVariance, float epsilon, float momentum, bool training)
    {
        int channels = input.Value.Shape[1], spatial = input.Value.Shape[2] * input.Value.Shape[3];
        var output = new GradientValue(new Tensor(input.Value.Shape)); float[] normalized = new float[input.Value.Length], inverse = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            int start = c * spatial; float mean = runningMean[c], variance = runningVariance[c];
            if (training)
            {
                mean = 0; for (int p = 0; p < spatial; p++) mean += input.Value.Data[start + p]; mean /= spatial;
                variance = 0; for (int p = 0; p < spatial; p++) { float delta = input.Value.Data[start + p] - mean; variance += delta * delta; } variance /= spatial;
                runningMean[c] = (1 - momentum) * runningMean[c] + momentum * mean;
                runningVariance[c] = (1 - momentum) * runningVariance[c] + momentum * variance * (spatial > 1 ? (float)spatial / (spatial - 1) : 1);
            }
            inverse[c] = 1f / MathF.Sqrt(variance + epsilon);
            for (int p = 0; p < spatial; p++) { float value = (input.Value.Data[start + p] - mean) * inverse[c]; normalized[start + p] = value; output.Value.Data[start + p] = value * gamma.Values[c] + beta.Values[c]; }
        }
        if (training) tape.Record(() =>
        {
            if (!output.HasGradient) return;
            gamma.HasGradient = beta.HasGradient = true;
            for (int c = 0; c < channels; c++)
            {
                int start = c * spatial; float sum = 0, weighted = 0;
                for (int p = 0; p < spatial; p++) { float dy = output.Gradient[start + p]; sum += dy; weighted += dy * normalized[start + p]; }
                gamma.Gradient[c] += weighted; beta.Gradient[c] += sum;
                float scale = gamma.Values[c] * inverse[c] / spatial;
                for (int p = 0; p < spatial; p++) input.Gradient[start + p] += scale * (spatial * output.Gradient[start + p] - sum - normalized[start + p] * weighted);
            }
        });
        return output;
    }

    /// <summary>登记 SiLU 的精确 sigmoid 导数，反向不复用已被修改的输出。</summary>
    /// <param name="tape">训练时保存导数闭包的记录器。</param><param name="input">激活前节点。</param><param name="training">是否登记反向计算。</param>
    /// <returns>SiLU 前向值和延迟梯度。</returns>
    internal static GradientValue Silu(GradientTape tape, GradientValue input, bool training)
    {
        var output = new GradientValue(new Tensor(input.Value.Shape)); float[] sigmoid = new float[input.Value.Length];
        for (int i = 0; i < sigmoid.Length; i++) { float x = input.Value.Data[i]; float s = Sigmoid(x); sigmoid[i] = s; output.Value.Data[i] = x * s; }
        if (training) tape.Record(() => { if (!output.HasGradient) return; for (int i = 0; i < sigmoid.Length; i++) { float s = sigmoid[i]; input.Gradient[i] += output.Gradient[i] * (s + input.Value.Data[i] * s * (1 - s)); } });
        return output;
    }
    internal static float Sigmoid(float value) => value >= 0 ? 1f / (1 + MathF.Exp(-value)) : MathF.Exp(value) / (1 + MathF.Exp(value));
    internal static float[] Transpose(float[] input, int rows, int columns)
    { var output = new float[input.Length]; for (int r = 0; r < rows; r++) for (int c = 0; c < columns; c++) output[c * rows + r] = input[r * columns + c]; return output; }
    internal static void Add(float[] destination, float[] source)
    { int i = 0; for (; i <= source.Length - System.Numerics.Vector<float>.Count; i += System.Numerics.Vector<float>.Count) (new System.Numerics.Vector<float>(destination, i) + new System.Numerics.Vector<float>(source, i)).CopyTo(destination, i); for (; i < source.Length; i++) destination[i] += source[i]; }
}
