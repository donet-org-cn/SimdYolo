using System.Numerics;

namespace SimdYolo;

/// <summary>把浮点概率转换为适合JSON Base64和图像处理的8位掩码，保持中点取偶的舍入规则。</summary>
public static class MaskConverter
{
    /// <summary>使用托管SIMD批量转换概率，结果数组由调用方独立持有。</summary>
    /// <param name="probabilities">按行排列的概率；有限值裁剪到0到1，非有限值转换为零。</param>
    /// <returns>每像素一个字节的独立数组。</returns>
    public static byte[] ToBytes(ReadOnlySpan<float> probabilities)
    {
        var bytes = GC.AllocateUninitializedArray<byte>(probabilities.Length); int i = 0, block = Vector<float>.Count;
        if (Vector.IsHardwareAccelerated)
            for (; i <= probabilities.Length - block * 4; i += block * 4)
                Vector.Narrow(Vector.Narrow(Convert(new Vector<float>(probabilities[i..])), Convert(new Vector<float>(probabilities[(i + block)..]))),
                    Vector.Narrow(Convert(new Vector<float>(probabilities[(i + block * 2)..])), Convert(new Vector<float>(probabilities[(i + block * 3)..])))).CopyTo(bytes, i);
        for (; i < probabilities.Length; i++) bytes[i] = float.IsFinite(probabilities[i]) ? (byte)MathF.Round(Math.Clamp(probabilities[i], 0, 1) * 255) : (byte)0;
        return bytes;
    }

    /// <summary>把一组概率转换为0到255的整数，向量窄化时不发生饱和溢出。</summary>
    /// <param name="value">一组浮点概率。</param><returns>中点取偶后对应的无符号整数。</returns>
    private static Vector<uint> Convert(Vector<float> value)
    {
        var valid = Vector.LessThanOrEqual(Vector.Abs(value), new Vector<float>(float.MaxValue));
        value = Vector.ConditionalSelect(valid, Vector.Min(Vector.Max(value, Vector<float>.Zero), Vector<float>.One), Vector<float>.Zero);
        return Vector.AsVectorUInt32(Vector.ConvertToInt32(Vector.Round(value * new Vector<float>(255f))));
    }
}
