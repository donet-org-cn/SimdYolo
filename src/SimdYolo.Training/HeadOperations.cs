using System.Numerics;

namespace SimdYolo.Training;

/// <summary>实现检测头所需的卷积、训练 BatchNorm 和 SiLU 的前向与反向。</summary>
/// <param name="backend">CPU SIMD 或 Vulkan 矩阵乘法后端，由外部管理生命周期。</param>
/// <param name="cpuThreads">Vulkan 混合路径的 CPU 线程预算，零表示使用逻辑处理器数量。</param>
internal sealed class HeadOperations(IComputeBackend backend, int cpuThreads = 0)
{
    /// <summary>CPU 调度预算；Vulkan 的小矩阵使用同样受限的托管 SIMD 路径。</summary>
    private readonly ParallelOptions parallel = backend is CpuBackend cpu ? cpu.Parallel : new() { MaxDegreeOfParallelism = cpuThreads > 0 ? cpuThreads : Environment.ProcessorCount };
    /// <summary>每个训练会话独立拥有的有界临时矩阵池，不缓存可变权重的计算结果。</summary>
    private readonly TrainingScratch scratch = new();

    /// <summary>按计算量分派训练 GEMM，避免小矩阵的 Vulkan 提交和回读超过计算本身。</summary>
    /// <param name="a">行主序左矩阵。</param><param name="b">行主序右矩阵。</param><param name="c">覆盖写入的目标矩阵。</param>
    /// <param name="m">输出行数。</param><param name="n">输出列数。</param><param name="k">归约长度。</param>
    private void MatMul(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        if (backend is VulkanBackend && (long)m * n * k <= 4_000_000) CpuGemm.Multiply(a, b, c, m, n, k, parallel);
        else backend.MatMul(a, b, c, m, n, k);
    }
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
        // 1×1 的 NCHW 空间平面已经是 GEMM 所需布局，直接借用输入，避免复制。
        var columnLease = kh == 1 ? null : scratch.Rent(k * spatial, clear: true);
        float[] columns = columnLease?.Data ?? x.Data;
        if (kh != 1) for (int c = 0; c < ic; c++) for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
        {
            int row = ((c * kh + ky) * kw + kx) * spatial;
            int left = Math.Max(0, padding - kx), right = Math.Min(w, w + padding - kx);
            for (int y = Math.Max(0, padding - ky); y < Math.Min(h, h + padding - ky); y++)
                Array.Copy(x.Data, (c * h + y + ky - padding) * w + left + kx - padding, columns, row + y * w + left, right - left);
        }
        var output = new GradientValue(new Tensor(1, oc, h, w));
        MatMul(weight.Values, columns, output.Value.Data, oc, spatial, k);
        if (bias is not null) for (int c = 0; c < oc; c++) for (int p = 0; p < spatial; p++) output.Value.Data[c * spatial + p] += bias.Values[c];
        tape.Record(() =>
        {
            using var columnsLifetime = columnLease;
            if (!output.HasGradient) return;
            bool accumulateWeight = weight.HasGradient;
            weight.HasGradient = true; if (bias is not null) bias.HasGradient = true;
            float[] dy = output.Gradient;
            using var dwLease = accumulateWeight ? scratch.Rent(weight.Values.Length) : null;
            float[] dw = dwLease?.Data ?? weight.Gradient;
            if ((backend is CpuBackend || backend is VulkanBackend && (long)oc * k * spatial <= 4_000_000) && spatial >= 64)
                TrainingMatMul.RightTransposed(dy, columns, dw, oc, k, spatial, parallel);
            else
            {
                using var transposeLease = scratch.Rent(columns.Length);
                MatMul(dy, Transpose(columns, k, spatial, transposeLease.Data), dw, oc, k, spatial);
            }
            if (accumulateWeight) Add(weight.Gradient, dw);
            if (bias is not null) for (int c = 0; c < oc; c++) { float sum = 0; for (int p = 0; p < spatial; p++) sum += dy[c * spatial + p]; bias.Gradient[c] += sum; }
            if (!input.RequiresGradient) return;
            using var wtLease = scratch.Rent(weight.Values.Length); using var dxLease = scratch.Rent(columns.Length);
            float[] wt = Transpose(weight.Values, oc, k, wtLease.Data), dcolumns = dxLease.Data;
            MatMul(wt, dy, dcolumns, k, spatial, oc);
            float[] dx = input.Gradient;
            if (kh == 1) { Add(dx, dcolumns); return; }
            for (int c = 0; c < ic; c++) for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
            {
                int row = ((c * kh + ky) * kw + kx) * spatial;
                int left = Math.Max(0, padding - kx), right = Math.Min(w, w + padding - kx);
                for (int y = Math.Max(0, padding - ky); y < Math.Min(h, h + padding - ky); y++)
                    Accumulate(dx, (c * h + y + ky - padding) * w + left + kx - padding, dcolumns, row + y * w + left, right - left, 1);
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
                        int xx = 0; var dot = Vector<float>.Zero;
                        for (; xx <= right - left - Vector<float>.Count; xx += Vector<float>.Count) dot += new Vector<float>(output.Gradient, yi + xx) * new Vector<float>(x.Data, xi + xx);
                        sum += Vector.Sum(dot);
                        for (; xx < right - left; xx++) sum += output.Gradient[yi + xx] * x.Data[xi + xx];
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
                mean = 0; for (int q = 0; q < spatial; q++) mean += input.Value.Data[start + q]; mean /= spatial;
                variance = 0; for (int q = 0; q < spatial; q++) { float delta = input.Value.Data[start + q] - mean; variance += delta * delta; } variance /= spatial;
                runningMean[c] = (1 - momentum) * runningMean[c] + momentum * mean;
                runningVariance[c] = (1 - momentum) * runningVariance[c] + momentum * variance * (spatial > 1 ? (float)spatial / (spatial - 1) : 1);
            }
            inverse[c] = 1f / MathF.Sqrt(variance + epsilon);
            int p = 0; var meanVector = new Vector<float>(mean); var invVector = new Vector<float>(inverse[c]); var gammaVector = new Vector<float>(gamma.Values[c]); var betaVector = new Vector<float>(beta.Values[c]);
            for (; p <= spatial - Vector<float>.Count; p += Vector<float>.Count)
            {
                var value = (new Vector<float>(input.Value.Data, start + p) - meanVector) * invVector;
                value.CopyTo(normalized, start + p); (value * gammaVector + betaVector).CopyTo(output.Value.Data, start + p);
            }
            for (; p < spatial; p++) { float value = (input.Value.Data[start + p] - mean) * inverse[c]; normalized[start + p] = value; output.Value.Data[start + p] = value * gamma.Values[c] + beta.Values[c]; }
        }
        if (training) tape.Record(() =>
        {
            if (!output.HasGradient) return;
            gamma.HasGradient = beta.HasGradient = true;
            for (int c = 0; c < channels; c++)
            {
                int start = c * spatial; float sum = 0, weighted = 0;
                for (int q = 0; q < spatial; q++) { float dy = output.Gradient[start + q]; sum += dy; weighted += dy * normalized[start + q]; }
                gamma.Gradient[c] += weighted; beta.Gradient[c] += sum;
                float scale = gamma.Values[c] * inverse[c] / spatial;
                int p = 0; var scaleVector = new Vector<float>(scale); var spatialVector = new Vector<float>(spatial); var sumVector = new Vector<float>(sum); var weightedVector = new Vector<float>(weighted); float[] dx = input.Gradient;
                for (; p <= spatial - Vector<float>.Count; p += Vector<float>.Count)
                    (new Vector<float>(dx, start + p) + scaleVector * (spatialVector * new Vector<float>(output.Gradient, start + p) - sumVector - new Vector<float>(normalized, start + p) * weightedVector)).CopyTo(dx, start + p);
                for (; p < spatial; p++) dx[start + p] += scale * (spatial * output.Gradient[start + p] - sum - normalized[start + p] * weighted);
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
        int i = 0;
        for (; i <= sigmoid.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            var x = new Vector<float>(input.Value.Data, i); var exp = CpuActivation.Exp(-Vector.Abs(x)); var denominator = Vector<float>.One + exp;
            var s = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(x, Vector<float>.Zero), Vector<float>.One / denominator, exp / denominator);
            s.CopyTo(sigmoid, i); (x * s).CopyTo(output.Value.Data, i);
        }
        for (; i < sigmoid.Length; i++) { float x = input.Value.Data[i]; float s = Sigmoid(x); sigmoid[i] = s; output.Value.Data[i] = x * s; }
        if (training) tape.Record(() =>
        {
            if (!output.HasGradient) return;
            int p = 0; float[] dx = input.Gradient, dy = output.Gradient;
            for (; p <= sigmoid.Length - Vector<float>.Count; p += Vector<float>.Count)
            {
                var s = new Vector<float>(sigmoid, p); var x = new Vector<float>(input.Value.Data, p);
                (new Vector<float>(dx, p) + new Vector<float>(dy, p) * (s + x * s * (Vector<float>.One - s))).CopyTo(dx, p);
            }
            for (; p < sigmoid.Length; p++) { float s = sigmoid[p]; dx[p] += dy[p] * (s + input.Value.Data[p] * s * (1 - s)); }
        });
        return output;
    }
    internal static float Sigmoid(float value) => value >= 0 ? 1f / (1 + MathF.Exp(-value)) : MathF.Exp(value) / (1 + MathF.Exp(value));
    /// <summary>按缓存块转置行主序矩阵，可写入租用的精确长度工作区。</summary>
    /// <param name="input">待转置的矩阵。</param><param name="rows">原矩阵行数。</param><param name="columns">原矩阵列数。</param>
    /// <param name="destination">可选目标数组，为 null 时创建新数组。</param><returns>包含转置结果的目标数组。</returns>
    internal static float[] Transpose(float[] input, int rows, int columns, float[]? destination = null)
    {
        var output = destination ?? new float[input.Length];
        // 小块转置同时照顾两侧缓存行，避免大矩阵逐列跨页写入。
        for (int r0 = 0; r0 < rows; r0 += 32) for (int c0 = 0; c0 < columns; c0 += 32)
            for (int r = r0; r < Math.Min(r0 + 32, rows); r++) for (int c = c0; c < Math.Min(c0 + 32, columns); c++) output[c * rows + r] = input[r * columns + c];
        return output;
    }
    internal static void Add(float[] destination, float[] source)
    { int i = 0; for (; i <= source.Length - System.Numerics.Vector<float>.Count; i += System.Numerics.Vector<float>.Count) (new System.Numerics.Vector<float>(destination, i) + new System.Numerics.Vector<float>(source, i)).CopyTo(destination, i); for (; i < source.Length; i++) destination[i] += source[i]; }
}
