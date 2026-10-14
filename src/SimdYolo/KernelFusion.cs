namespace SimdYolo;

/// <summary>把重参数化卷积的并行常量分支合并为一个权重核。</summary>
internal static class KernelFusion
{
    /// <summary>将小卷积核居中加到大卷积核，并相加两个分支的偏置。</summary>
    /// <param name="large">空间尺寸较大的卷积权重。</param>
    /// <param name="largeBias">大卷积分支的偏置。</param>
    /// <param name="small">需要居中嵌入的小卷积权重。</param>
    /// <param name="smallBias">小卷积分支的偏置。</param>
    /// <returns>拥有独立存储、标记为常量的融合权重和偏置。</returns>
    /// <exception cref="ArgumentException">分支的输入输出通道不一致，或小核尺寸大于大核。</exception>
    public static (Tensor Weight, Tensor Bias) Merge(Tensor large, Tensor largeBias, Tensor small, Tensor smallBias)
    {
        if (large.Shape[0] != small.Shape[0] || large.Shape[1] != small.Shape[1] || large.Shape[2] < small.Shape[2] || large.Shape[3] < small.Shape[3]) throw new ArgumentException("Rep kernel shapes mismatch.");
        var merged = new Tensor((float[])large.Data.Clone(), large.Shape) { IsConstant = true };
        var bias = new Tensor(largeBias.Data.Zip(smallBias.Data, (a, b) => a + b).ToArray(), largeBias.Shape) { IsConstant = true };
        int ph = (large.Shape[2] - small.Shape[2]) / 2, pw = (large.Shape[3] - small.Shape[3]) / 2;
        for (int c = 0; c < large.Shape[0] * large.Shape[1]; c++) for (int y = 0; y < small.Shape[2]; y++) for (int x = 0; x < small.Shape[3]; x++)
            merged.Data[(c * large.Shape[2] + y + ph) * large.Shape[3] + x + pw] += small.Data[(c * small.Shape[2] + y) * small.Shape[3] + x];
        return (merged, bias);
    }
}
