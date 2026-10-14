namespace SimdYolo;

/// <summary>通过逐点卷积与像素重排实现固定 2×2 转置卷积。</summary>
public sealed unsafe partial class VulkanBackend
{
    private readonly Dictionary<(Tensor Weight, Tensor? Bias), (Tensor Weight, Tensor? Bias)> transposeWeights = [];
    /// <summary>把固定 2×2 步长 2 转置卷积改写为逐点卷积和像素重排，缓存重排权重与偏置。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    private Tensor TransposeConv2x2(Tensor x, Tensor w, Tensor? bias, int[] shape)
    {
        int ic = x.Shape[1], oc = shape[1];
        if (!transposeWeights.TryGetValue((w, bias), out var packed))
        {
            var weight = new Tensor(oc * 4, ic, 1, 1) { IsConstant = true };
            Tensor? bv = bias is null ? null : new Tensor(oc * 4) { IsConstant = true };
            for (int co = 0; co < oc; co++) for (int pixel = 0; pixel < 4; pixel++)
            {
                for (int ci = 0; ci < ic; ci++) weight.Data[(co * 4 + pixel) * ic + ci] = w.Data[(ci * oc + co) * 4 + pixel];
                if (bv is not null) bv.Data[co * 4 + pixel] = bias!.Data[co];
            }
            packed = (weight, bv); transposeWeights[(w, bias)] = packed;
        }
        var product = ((IDeviceBackend)this).DeviceConv(x, packed.Weight, packed.Bias, [shape[0], oc * 4, x.Shape[2], x.Shape[3]], [1, 1], [0, 0, 0, 0], [1, 1], 1, false);
        return ((IDeviceBackend)this).Compute(35, [product], shape, DeviceOps.Parameters((uint)Tensor.Count(shape), (uint)shape[2], (uint)shape[3], (uint)oc), Tensor.Count(shape));
    }
}
