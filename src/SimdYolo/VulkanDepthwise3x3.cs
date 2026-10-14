namespace SimdYolo;

/// <summary>实现 Vulkan 3×3 深度卷积的四元素向量化路径。</summary>
public sealed unsafe partial class VulkanBackend
{
    private readonly bool depthwiseVector3x3 = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_DEPTHWISE_VECTOR") != "0";

    /// <summary>判断 3×3 深度卷积是否满足四元素向量读取、步长、对齐和可移植调度限制。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="stride">高、宽两个方向的卷积步长。</param>
    /// <param name="pads">上、左、下、右方向的填充量。</param>
    /// <param name="dilation">高、宽方向的卷积膨胀系数。</param>
    /// <param name="groups">卷积分组数，输入和输出通道须满足该分组布局。</param>
    /// <returns>当前输入满足此专用路径全部条件时为 true。</returns>
    private bool DepthwiseVectorEligible(Tensor x, Tensor w, int[] shape, int[] stride, int[] pads, int[] dilation, int groups)
        => depthwiseVector3x3 && w.IsConstant && groups == x.Shape[1] && w.Shape.SequenceEqual([groups, 1, 3, 3])
        && stride[0] is 1 or 2 && stride[0] == stride[1] && pads.All(p => p == 1) && dilation.All(d => d == 1)
        && shape[3] > 0 && shape[3] % 4 == 0 && x.Shape[3] == shape[3] * stride[0]
        && (long)Tensor.Count(shape) / 4 <= 65535L * 128
        && (long)x.Length * 4 + (long)(x.Storage?.Offset % 256 ?? 0) <= 128L * 1024 * 1024
        && (long)w.Length * 4 + (long)(w.Storage?.Offset % 256 ?? 0) <= 128L * 1024 * 1024
        && (x.Storage is not { } source || source.Offset % 16 == 0);

    /// <summary>执行四元素向量化 3×3 深度卷积，缺省偏置使用缓存的零张量。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="stride">高、宽两个方向的卷积步长。</param>
    /// <param name="silu">是否在偏置相加后融合 SiLU 激活。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    private Tensor DepthwiseVectorConv(Tensor x, Tensor w, Tensor? bias, int[] shape, int[] stride, bool silu)
    {
        int channels = x.Shape[1];
        if (bias is null && !zeros.TryGetValue(channels, out bias))
        {
            bias = new Tensor(channels) { IsConstant = true }; zeros[channels] = bias;
        }
        var input = Upload(x); var weights = Upload(w); var biases = Upload(bias!); var output = AllocateTensor(shape);
        uint[] parameters = [(uint)(output.Length / 4), (uint)channels, (uint)x.Shape[2], (uint)x.Shape[3], (uint)shape[2], (uint)shape[3],
            (uint)stride[0], silu ? 1u : 0u, (uint)(input.Offset % 256 / 16), (uint)(weights.Offset % 256 / 4), (uint)(biases.Offset % 256 / 4)];
        string key = "depthwise-vector3x3:" + string.Join(',', parameters);
        if (!convolutionPipelines.TryGetValue(key, out ulong selected))
            convolutionPipelines[key] = selected = CreateNamedPipeline("depthwise3x3_vector.spv", parameters);
        Dispatch(selected, input, weights, output.Storage!, biases, parameters, ((uint)(output.Length / 4) + 127) / 128);
        return output;
    }
}
