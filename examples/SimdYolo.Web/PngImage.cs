using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace SimdYolo.Web;

/// <summary>仅用框架 ZLib 将 RGB24 编码为 PNG，供内置样本预览。</summary>
internal static class PngImage
{
    internal static byte[] Encode(byte[] rgb, int width, int height)
    {
        using var output = new MemoryStream(); output.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });
        byte[] header = new byte[13]; BinaryPrimitives.WriteInt32BigEndian(header, width); BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height); header[8] = 8; header[9] = 2;
        Chunk(output, "IHDR", header);
        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, true))
            for (int row = 0; row < height; row++) { zlib.WriteByte(0); zlib.Write(rgb, row * width * 3, width * 3); }
        Chunk(output, "IDAT", compressed.ToArray()); Chunk(output, "IEND", []); return output.ToArray();
    }
    private static void Chunk(Stream stream, string type, byte[] data)
    {
        byte[] name = Encoding.ASCII.GetBytes(type); Span<byte> integer = stackalloc byte[4]; BinaryPrimitives.WriteInt32BigEndian(integer, data.Length); stream.Write(integer); stream.Write(name); stream.Write(data);
        uint crc = uint.MaxValue;
        foreach (byte b in name.Concat(data)) { crc ^= b; for (int bit = 0; bit < 8; bit++) crc = (crc >> 1) ^ ((crc & 1) == 0 ? 0 : 0xedb88320u); }
        BinaryPrimitives.WriteUInt32BigEndian(integer, ~crc); stream.Write(integer);
    }
}
