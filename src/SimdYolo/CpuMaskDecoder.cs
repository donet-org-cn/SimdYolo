using System.Buffers;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>复用坐标表和临时原型平面，以SIMD完成掩码双线性插值及Sigmoid。</summary>
internal sealed class CpuMaskDecoder
{
    private readonly int mw, mh, plane;
    private readonly float[] prototype;
    private readonly int[] x0, x1, y0, y1;
    private readonly float[] fx, fy;

    /// <summary>为同一张图的全部实例准备共享且只读的原图到原型坐标映射。</summary>
    /// <param name="prototype">单图原型张量。</param><param name="transform">实际预处理变换。</param>
    internal CpuMaskDecoder(Tensor prototype, Letterbox transform)
    {
        if (prototype.Shape[0] != 1) throw new ArgumentException("Single-image mask decoding only.");
        mw = prototype.Shape[3]; mh = prototype.Shape[2]; plane = checked(mw * mh); this.prototype = prototype.Data;
        int left = Math.Clamp((int)MathF.Round((float)transform.PadX * mw / transform.TargetWidth), 0, mw - 1);
        int top = Math.Clamp((int)MathF.Round((float)transform.PadY * mh / transform.TargetHeight), 0, mh - 1);
        int right = Math.Clamp((int)MathF.Round((transform.PadX + transform.SourceWidth * transform.ScaleX) * mw / transform.TargetWidth), left + 1, mw);
        int bottom = Math.Clamp((int)MathF.Round((transform.PadY + transform.SourceHeight * transform.ScaleY) * mh / transform.TargetHeight), top + 1, mh);
        (x0, x1, fx) = Map(transform.SourceWidth, left, right);
        (y0, y1, fy) = Map(transform.SourceHeight, top, bottom);
    }

    /// <summary>按原有浮点运算顺序建立坐标表，保留填边ROI及半像素采样规则。</summary>
    /// <param name="length">原图轴长度。</param><param name="start">原型ROI起点。</param><param name="end">原型ROI终点，不包含。</param>
    /// <returns>相邻采样位置及小数权重。</returns>
    private static (int[] First, int[] Second, float[] Fraction) Map(int length, int start, int end)
    {
        var first = new int[length]; var second = new int[length]; var fraction = new float[length];
        for (int i = 0; i < length; i++) { float coordinate = start + (i + .5f) * (end - start) / length - .5f; coordinate = Math.Clamp(coordinate, start, end - 1); first[i] = (int)coordinate; second[i] = Math.Min(first[i] + 1, end - 1); fraction[i] = coordinate - first[i]; }
        return (first, second, fraction);
    }

    /// <summary>计算一个实例的裁剪概率图；临时logits归还池，公开概率数组保持独立所有权。</summary>
    /// <param name="box">原图坐标检测框。</param><param name="coefficients">按原型通道顺序的掩码系数。</param>
    /// <returns>与原实现相同尺寸和位置的实例掩码。</returns>
    internal unsafe InstanceMask Decode(Detection box, float[] coefficients)
    {
        int left = (int)MathF.Floor(box.X), top = (int)MathF.Floor(box.Y), right = (int)MathF.Ceiling(box.X + box.Width), bottom = (int)MathF.Ceiling(box.Y + box.Height), width = right - left, height = bottom - top;
        var values = new float[checked(width * height)];
        int beginX = Math.Max(left, (int)MathF.Ceiling(box.X)), beginY = Math.Max(top, (int)MathF.Ceiling(box.Y));
        float[] low = ArrayPool<float>.Shared.Rent(plane), horizontal = ArrayPool<float>.Shared.Rent(checked(width * 2));
        try
        {
            low.AsSpan(0, plane).Clear();
            for (int c = 0; c < coefficients.Length; c++)
            {
                float weight = coefficients[c]; var vw = new Vector<float>(weight); int i = 0, offset = c * plane;
                for (; i <= plane - Vector<float>.Count; i += Vector<float>.Count) (new Vector<float>(low, i) + vw * new Vector<float>(prototype, offset + i)).CopyTo(low, i);
                for (; i < plane; i++) low[i] += weight * prototype[offset + i];
            }
            fixed (float* lp = low, output = values, xp = fx, hp = horizontal)
            fixed (int* ax = x0, bx = x1)
            {
                // 放大时相邻原图行重复使用同一对原型行，避免为每一行执行四次Gather。
                float* upper = hp; float* lower = hp + width; int upperRow = -1, lowerRow = -1, columns = right - beginX;
                for (int y = beginY; y < bottom; y++)
                {
                    int firstRow = y0[y] * mw, secondRow = y1[y] * mw, destination = (y - top) * width + beginX - left;
                    if (firstRow == lowerRow) { float* swap = upper; upper = lower; lower = swap; (upperRow, lowerRow) = (lowerRow, upperRow); }
                    if (firstRow != upperRow) { InterpolateRow(lp + firstRow, ax, bx, xp, beginX, right, upper); upperRow = firstRow; }
                    if (secondRow != firstRow && secondRow != lowerRow) { InterpolateRow(lp + secondRow, ax, bx, xp, beginX, right, lower); lowerRow = secondRow; }
                    float* second = firstRow == secondRow ? upper : lower; float fractionY = fy[y]; int x = 0;
                    if (Avx2.IsSupported && Vector<float>.Count == 8)
                    {
                        var vy = Vector256.Create(fractionY); var uy = Vector256.Create(1 - fractionY);
                        for (; x <= columns - 8; x += 8) CpuActivation.Sigmoid(Avx.LoadVector256(upper + x) * uy + Avx.LoadVector256(second + x) * vy).Store(output + destination + x);
                    }
                    for (; x < columns; x++) values[destination + x] = TensorOps.Sigmoid(upper[x] * (1 - fractionY) + second[x] * fractionY);
                }
            }
            return new(left, top, width, height, values);
        }
        finally { ArrayPool<float>.Shared.Return(low); ArrayPool<float>.Shared.Return(horizontal); }
    }

    /// <summary>把一个原型行插值到实例的有效列，结果缓存供多条输出行重复使用。</summary>
    /// <param name="row">原型行首地址。</param><param name="first">左采样位置表。</param><param name="second">右采样位置表。</param>
    /// <param name="fractions">小数权重表。</param><param name="begin">原图有效列起点。</param><param name="end">原图有效列终点，不包含。</param><param name="output">连续行缓存。</param>
    private static unsafe void InterpolateRow(float* row, int* first, int* second, float* fractions, int begin, int end, float* output)
    {
        int x = begin;
        if (Avx2.IsSupported)
            for (; x <= end - 8; x += 8)
            {
                var weight = Avx.LoadVector256(fractions + x);
                (Avx2.GatherVector256(row, Avx.LoadVector256(first + x), 4) * (Vector256.Create(1f) - weight) + Avx2.GatherVector256(row, Avx.LoadVector256(second + x), 4) * weight).Store(output + x - begin);
            }
        for (; x < end; x++) output[x - begin] = row[first[x]] * (1 - fractions[x]) + row[second[x]] * fractions[x];
    }
}
