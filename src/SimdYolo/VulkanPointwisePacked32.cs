namespace SimdYolo;

/// <summary>实现 32 输出通道分块的 Vulkan 向量化逐点卷积。</summary>
public sealed unsafe partial class VulkanBackend
{
    private readonly bool pointwisePacked32 = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_POINTWISE_PACKED32") != "0";

    /// <summary>判断逐点卷积是否适合输出 32 通道分块，并避开已适合 64 通道分块的形状。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="stride">高、宽两个方向的卷积步长。</param>
    /// <param name="pads">上、左、下、右方向的填充量。</param>
    /// <param name="dilation">高、宽方向的卷积膨胀系数。</param>
    /// <param name="groups">卷积分组数，输入和输出通道须满足该分组布局。</param>
    /// <returns>当前输入满足此专用路径全部条件时为 true。</returns>
    private bool PointwisePacked32Eligible(Tensor x, Tensor w, int[] shape, int[] stride, int[] pads, int[] dilation, int groups)
    {
        long spatial = (long)x.Shape[2] * x.Shape[3];
        return pointwisePacked32 && w.IsConstant && groups == 1 && w.Shape[2] == 1 && w.Shape[3] == 1 && x.Shape[1] >= 16
            && shape[1] >= 16 && shape[1] % 4 == 0 && !(shape[1] >= 64 && shape[1] % 64 == 0) && spatial >= 64 && spatial % 4 == 0
            && stride.SequenceEqual([1, 1]) && pads.All(p => p == 0) && dilation.SequenceEqual([1, 1])
            && spatial <= 65535L * 64 && shape[0] <= 65535 && (shape[1] + 31L) / 32 <= 65535
            && (long)Math.Max(x.Length, Tensor.Count(shape)) * sizeof(float) + 256 <= 128L * 1024 * 1024
            && (long)w.Length * sizeof(float) <= 128L * 1024 * 1024
            && (x.Storage is not { } source || source.Offset % 16 == 0);
    }

    /// <summary>复用逐点权重重排缓存，按 32 通道分块执行向量化卷积并处理通道尾部。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="silu">是否在偏置相加后融合 SiLU 激活。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    private Tensor PointwisePacked32Conv(Tensor x, Tensor w, Tensor? bias, int[] shape, bool silu)
    {
        int ic = x.Shape[1], oc = shape[1], spatial = shape[2] * shape[3];
        if (!pointwisePackedWeights.TryGetValue(w, out Tensor? packed))
        {
            packed = new Tensor(ic, oc) { IsConstant = true };
            for (int ci = 0; ci < ic; ci++) for (int co = 0; co < oc; co++) packed.Data[ci * oc + co] = w.Data[co * ic + ci];
            pointwisePackedWeights[w] = packed;
        }
        if (bias is null && !zeros.TryGetValue(oc, out bias)) { bias = new Tensor(oc) { IsConstant = true }; zeros[oc] = bias; }
        var a = Upload(packed); var b = Upload(x); var d = Upload(bias!); var output = AllocateTensor(shape);
        uint[] parameters = [silu ? 1u : 0u, (uint)ic, (uint)spatial, (uint)oc, (uint)(b.Offset % 256 / 16), (uint)(d.Offset % 256 / 4)];
        string key = "pointwise-packed32:" + string.Join(',', parameters);
        if (!convolutionPipelines.TryGetValue(key, out ulong selected))
            convolutionPipelines[key] = selected = CreateNamedPipeline("pointwise_packed_vector32.spv", parameters);
        Dispatch(selected, a, b, output.Storage!, d, parameters, ((uint)spatial + 63) / 64, ((uint)oc + 31) / 32, (uint)shape[0]);
        return output;
    }
}
