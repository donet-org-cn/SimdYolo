using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SimdYolo;

/// <summary>描述 PT 和 ONNX 共用的小端权重元素编码。</summary>
internal enum TensorStorageKind
{
    /// <summary>IEEE 754 单精度浮点编码。</summary>
    Float32,
    /// <summary>IEEE 754 半精度浮点编码。</summary>
    Float16,
    /// <summary>保留单精度高 16 位的 BF16 编码。</summary>
    BFloat16,
    /// <summary>IEEE 754 双精度浮点编码。</summary>
    Float64,
    /// <summary>有符号 64 位整数编码。</summary>
    Int64,
    /// <summary>有符号 32 位整数编码。</summary>
    Int32,
    /// <summary>有符号 8 位整数编码。</summary>
    Int8,
    /// <summary>无符号字节或逐字节布尔编码。</summary>
    UInt8
}

/// <summary>批量还原连续权重，按存储类型选择一次转换路径并保留 FP32 特殊值的位表示。</summary>
internal static class TensorStorageReader
{
    /// <summary>读取连续小端权重到独立的 FP32 存储；FP16 在可用时使用托管 SIMD。</summary>
    /// <param name="source">连续权重字节；长度应与目标元素数量及编码大小一致。</param>
    /// <param name="destination">接收每个权重值的 FP32 存储，方法会写满全部元素。</param>
    /// <param name="kind">权重字节的元素编码。</param>
    /// <exception cref="InvalidDataException">源字节长度与目标元素数量不匹配。</exception>
    /// <exception cref="NotSupportedException">元素编码不受支持。</exception>
    /// <exception cref="OverflowException">所需源字节数量超过 Int32 范围。</exception>
    internal static void Read(ReadOnlySpan<byte> source, Span<float> destination, TensorStorageKind kind)
    {
        int size = kind switch
        {
            TensorStorageKind.Float32 or TensorStorageKind.Int32 => 4,
            TensorStorageKind.Float16 or TensorStorageKind.BFloat16 => 2,
            TensorStorageKind.Float64 or TensorStorageKind.Int64 => 8,
            TensorStorageKind.Int8 or TensorStorageKind.UInt8 => 1,
            _ => throw new NotSupportedException($"Tensor storage encoding {kind} unsupported.")
        };
        if (source.Length != checked(destination.Length * size)) throw new InvalidDataException("Tensor bytes do not match shape.");
        switch (kind)
        {
            case TensorStorageKind.Float32:
                if (BitConverter.IsLittleEndian) source.CopyTo(MemoryMarshal.AsBytes(destination));
                else for (int i = 0; i < destination.Length; i++) destination[i] = BinaryPrimitives.ReadSingleLittleEndian(source.Slice(i * 4, 4));
                break;
            case TensorStorageKind.Float16:
                ReadHalf(source, destination);
                break;
            case TensorStorageKind.BFloat16:
                for (int i = 0; i < destination.Length; i++) destination[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(source.Slice(i * 2, 2)) << 16);
                break;
            case TensorStorageKind.Float64:
                for (int i = 0; i < destination.Length; i++) destination[i] = (float)BinaryPrimitives.ReadDoubleLittleEndian(source.Slice(i * 8, 8));
                break;
            case TensorStorageKind.Int64:
                for (int i = 0; i < destination.Length; i++) destination[i] = BinaryPrimitives.ReadInt64LittleEndian(source.Slice(i * 8, 8));
                break;
            case TensorStorageKind.Int32:
                for (int i = 0; i < destination.Length; i++) destination[i] = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(i * 4, 4));
                break;
            case TensorStorageKind.Int8:
                for (int i = 0; i < destination.Length; i++) destination[i] = (sbyte)source[i];
                break;
            case TensorStorageKind.UInt8:
                for (int i = 0; i < destination.Length; i++) destination[i] = source[i];
                break;
        }
    }

    /// <summary>把 FP16 还原为 FP32；SIMD 通过整数位运算精确扩大正规数，次正规数用精确的二进制缩放还原。</summary>
    /// <param name="source">小端 FP16 字节，已校验长度。</param>
    /// <param name="destination">接收 FP32 值的存储。</param>
    private static void ReadHalf(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int i = 0;
        if (BitConverter.IsLittleEndian && Vector256.IsHardwareAccelerated)
        {
            ReadOnlySpan<ushort> values = MemoryMarshal.Cast<byte, ushort>(source);
            ref ushort input = ref MemoryMarshal.GetReference(values);
            ref float output = ref MemoryMarshal.GetReference(destination);
            for (; i <= destination.Length - 16; i += 16)
            {
                var (lower, upper) = Vector256.Widen(Vector256.LoadUnsafe(ref input, (nuint)i));
                ExpandHalf(lower, ref output, i);
                ExpandHalf(upper, ref output, i + 8);
            }
        }
        for (; i < destination.Length; i++) destination[i] = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(source.Slice(i * 2, 2)));
    }

    /// <summary>精确展开八个 FP16 元素；无穷和 NaN 使用框架转换以保留其符号、载荷及静默位语义。</summary>
    /// <param name="values">零扩展后的八个 FP16 位表示。</param>
    /// <param name="destination">接收结果的 FP32 数组首元素引用。</param>
    /// <param name="offset">相对于首元素的目标偏移。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExpandHalf(Vector256<uint> values, ref float destination, int offset)
    {
        var exponent = values & Vector256.Create(0x7c00u);
        if (Vector256.EqualsAny(exponent, Vector256.Create(0x7c00u)))
        {
            for (int lane = 0; lane < 8; lane++) Unsafe.Add(ref destination, offset + lane) = (float)BitConverter.UInt16BitsToHalf((ushort)values[lane]);
            return;
        }
        var bits = ((values & Vector256.Create(0x7fffu)) << 13) + Vector256.Create(0x38000000u);
        if (Vector256.EqualsAny(exponent, Vector256<uint>.Zero))
        {
            // FP16 的次正规值在 FP32 中均为正规值；整数转换和乘以 2^-24 都精确，不依赖近似或查表。
            var subnormal = Vector256.ConvertToSingle(values & Vector256.Create(0x3ffu)) * Vector256.Create(5.9604644775390625e-8f);
            bits = Vector256.ConditionalSelect(Vector256.Equals(exponent, Vector256<uint>.Zero), subnormal.AsUInt32(), bits);
        }
        bits |= (values & Vector256.Create(0x8000u)) << 16;
        bits.AsSingle().StoreUnsafe(ref destination, (nuint)offset);
    }
}
