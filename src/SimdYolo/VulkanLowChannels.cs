namespace SimdYolo;

/// <summary>实现输入三通道卷积的权重重排和低通道专用管线。</summary>
public sealed unsafe partial class VulkanBackend
{
    private readonly Dictionary<Tensor, Tensor> lowChannelWeights = [];

    /// <summary>判断输入 3 通道、输出 16 通道的 3×3 步长 2 卷积是否可走低通道专用管线。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="stride">高、宽两个方向的卷积步长。</param>
    /// <param name="pads">上、左、下、右方向的填充量。</param>
    /// <param name="dilation">高、宽方向的卷积膨胀系数。</param>
    /// <param name="groups">卷积分组数，输入和输出通道须满足该分组布局。</param>
    /// <returns>当前输入满足此专用路径全部条件时为 true。</returns>
    private static bool LowChannelsEligible(Tensor x, Tensor w, int[] stride, int[] pads, int[] dilation, int groups)
        => w.IsConstant && groups == 1 && x.Shape[1] == 3 && w.Shape.SequenceEqual([16, 3, 3, 3])
        && stride.SequenceEqual([2, 2]) && pads.SequenceEqual([1, 1, 1, 1]) && dilation.SequenceEqual([1, 1]);

    /// <summary>把固定低通道权重重排为归约维优先布局，并调度专用卷积管线。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="silu">是否在偏置相加后融合 SiLU 激活。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <exception cref="InvalidOperationException">上传后的权重、输入或偏置未按 256 字节对齐。</exception>
    private Tensor LowChannelsConv(Tensor x, Tensor w, Tensor? bias, int[] shape, bool silu)
    {
        if (!lowChannelWeights.TryGetValue(w, out var packed))
        {
            packed = new Tensor(27, 16) { IsConstant = true };
            for (int k = 0; k < 27; k++) for (int c = 0; c < 16; c++) packed.Data[k * 16 + c] = w.Data[c * 27 + k];
            lowChannelWeights[w] = packed;
        }
        if (bias is null && !zeros.TryGetValue(16, out bias)) { bias = new Tensor(16) { IsConstant = true }; zeros[16] = bias; }
        uint[] parameters = [silu ? 1u : 0u, 3, (uint)x.Shape[2], (uint)x.Shape[3], 16, (uint)shape[2], (uint)shape[3], 3, 3, 2, 2, 1, 1, 1, 1, 1];
        string key = "lowchannels:" + string.Join(',', parameters);
        if (!convolutionPipelines.TryGetValue(key, out ulong selected))
            convolutionPipelines[key] = selected = CreateNamedPipeline("conv_lowchannels.spv", parameters);
        var output = AllocateTensor(shape);
        var a = Upload(packed); var b = Upload(x); var d = Upload(bias!);
        if ((a.Offset | b.Offset | d.Offset) % 256 != 0) throw new InvalidOperationException("Unaligned convolution tensor.");
        Dispatch(selected, a, b, output.Storage!, d, parameters, (uint)(shape[2] * shape[3] + 127) / 128, 1, (uint)shape[0]);
        return output;
    }
    /// <summary>清除低通道卷积的托管重排权重缓存；设备常量由后端常量缓存统一释放。</summary>
    private void FreeLowChannelsResources() => lowChannelWeights.Clear();
}
