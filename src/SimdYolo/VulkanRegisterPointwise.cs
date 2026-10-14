namespace SimdYolo;

/// <summary>实现 Vulkan 256 线程寄存器逐点卷积及其形状筛选。</summary>
public sealed unsafe partial class VulkanBackend
{
    private readonly string? pointRegisterTile = Environment.GetEnvironmentVariable("SIMDYOLO_POINT_REGISTER_TILE");

    /// <summary>检查特化参数是否满足 256 线程、64 通道寄存器逐点卷积的形状要求。</summary>
    /// <param name="p">按卷积特化常量约定排列的参数。</param>
    /// <returns>当前输入满足此专用路径全部条件时为 true。</returns>
    private bool RegisterPointwiseShape(uint[] p)
        => pointRegisterTile != "0" && p.Length >= 16 && p[15] == 1 && p[1] >= 16 && p[4] >= 64 && p[4] % 64 == 0
        && p[7] == 1 && p[8] == 1 && p[9] == 1 && p[10] == 1 && p[11] == 0 && p[12] == 0 && p[13] == 1 && p[14] == 1
        && (ulong)p[5] * p[6] >= 32;

    /// <summary>判断当前设置与输入形状是否允许直接选择 256 线程寄存器逐点卷积。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="stride">高、宽两个方向的卷积步长。</param>
    /// <param name="pads">上、左、下、右方向的填充量。</param>
    /// <param name="dilation">高、宽方向的卷积膨胀系数。</param>
    /// <param name="groups">卷积分组数，输入和输出通道须满足该分组布局。</param>
    /// <returns>当前输入满足此专用路径全部条件时为 true。</returns>
    private bool RegisterPointwiseEligible(Tensor x, Tensor w, int[] stride, int[] pads, int[] dilation, int groups)
        => pointRegisterTile == "256" && w.Shape[0] % 64 == 0
        && w.IsConstant && groups == 1 && x.Shape[1] >= 16 && w.Shape[0] >= 32
        && w.Shape[2] == 1 && w.Shape[3] == 1 && x.Shape[2] * x.Shape[3] >= 32
        && stride.SequenceEqual([1, 1]) && pads.All(p => p == 0) && dilation.SequenceEqual([1, 1]);

    /// <summary>调度 64×64 输出分块、256 线程的寄存器逐点卷积管线。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="silu">是否在偏置相加后融合 SiLU 激活。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <exception cref="InvalidOperationException">上传后的权重、输入或偏置未按 256 字节对齐。</exception>
    private Tensor RegisterPointwiseConv(Tensor x, Tensor w, Tensor? bias, int[] shape, bool silu)
    {
        int oc = w.Shape[0], columns = 64, rows = 64;
        if (bias is null && !zeros.TryGetValue(oc, out bias)) { bias = new Tensor(oc) { IsConstant = true }; zeros[oc] = bias; }
        uint[] parameters = [silu ? 1u : 0u, (uint)x.Shape[1], (uint)x.Shape[2], (uint)x.Shape[3], (uint)oc, (uint)shape[2], (uint)shape[3], 1, 1, 1, 1, 0, 0, 1, 1, 1];
        string key = "point-register:" + pointRegisterTile + ":" + string.Join(',', parameters);
        if (!convolutionPipelines.TryGetValue(key, out ulong selected))
            convolutionPipelines[key] = selected = CreateNamedPipeline("point64threads256.spv", parameters);
        var output = AllocateTensor(shape); var a = Upload(w); var b = Upload(x); var d = Upload(bias!);
        if ((a.Offset | b.Offset | d.Offset) % 256 != 0) throw new InvalidOperationException("Unaligned convolution tensor.");
        Dispatch(selected, a, b, output.Storage!, d, parameters,
            (uint)(shape[2] * shape[3] + columns - 1) / (uint)columns, (uint)(oc + rows - 1) / (uint)rows, (uint)shape[0]);
        return output;
    }
}
