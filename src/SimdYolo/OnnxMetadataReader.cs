using System.Text;

namespace SimdYolo;

/// <summary>流式读取 ONNX 元数据和输入输出形状，跳过节点及权重字节。</summary>
internal static class OnnxMetadataReader
{
    /// <summary>读取模型元数据，不初始化执行后端，也不访问外部权重文件。</summary>
    /// <param name="path">ONNX 文件路径。</param>
    /// <returns>元数据、图像输入形状及输出形状组成的数据。</returns>
    /// <exception cref="InvalidDataException">protobuf 损坏，元数据超限或图像输入不唯一。</exception>
    /// <exception cref="IOException">文件无法读取或被截断。</exception>
    internal static (Dictionary<string, string> Metadata, int[] Input, int[][] Outputs) Read(string path)
    {
        using var stream = File.OpenRead(path);
        var metadata = new Dictionary<string, string>(); var inputs = new List<Proto>(); var outputs = new List<int[]>(); var weights = new HashSet<string>(); bool graph = false;
        while (stream.Position < stream.Length)
        {
            var field = Field(stream, stream.Length);
            if (field.Number == 14 && field.Wire == 2)
            {
                var entry = Proto.Parse(Bytes(stream, field.End));
                if (!metadata.TryAdd(entry.Text(1), entry.Text(2))) throw new InvalidDataException("Duplicate ONNX metadata key.");
            }
            else if (field.Number == 7 && field.Wire == 2)
            {
                if (graph) throw new InvalidDataException("Multiple ONNX graphs."); graph = true;
                while (stream.Position < field.End)
                {
                    var child = Field(stream, field.End);
                    if (child.Wire == 2 && child.Number is 11 or 12)
                    {
                        var info = Proto.Parse(Bytes(stream, child.End));
                        if (child.Number == 11) inputs.Add(info); else outputs.Add(Shape(info));
                    }
                    else if (child.Wire == 2 && child.Number == 5)
                    {
                        while (stream.Position < child.End)
                        {
                            var weight = Field(stream, child.End);
                            if (weight.Number == 8 && weight.Wire == 2) weights.Add(Encoding.UTF8.GetString(Bytes(stream, weight.End)));
                            stream.Position = weight.End;
                        }
                    }
                    stream.Position = child.End;
                }
            }
            stream.Position = field.End;
        }
        inputs.RemoveAll(input => weights.Contains(input.Text(1)));
        if (!graph || inputs.Count != 1) throw new InvalidDataException("ONNX requires one graph and one image input.");
        return (metadata, Shape(inputs[0]), outputs.ToArray());
    }

    /// <summary>解析 ValueInfoProto 的张量维度，符号维度使用 -1。</summary>
    /// <param name="info">图输入或输出的 ValueInfoProto。</param>
    /// <returns>静态和动态维度数组。</returns>
    /// <exception cref="OverflowException">静态维度超过 Int32 范围。</exception>
    internal static int[] Shape(Proto info)
    {
        var type = Proto.Parse(info.Byte(2) ?? []); var tensor = Proto.Parse(type.Byte(1) ?? []);
        return Proto.Parse(tensor.Byte(2) ?? []).Bytes(1).Select(Proto.Parse).Select(d => checked((int)d.Int(1, -1))).ToArray();
    }

    /// <summary>读取一个 protobuf 字段头，并检查它没有越过父消息边界。</summary>
    /// <param name="stream">可定位的模型文件流。</param>
    /// <param name="limit">父消息的结束位置。</param>
    /// <returns>字段编号、线型和字段结束位置。</returns>
    /// <exception cref="InvalidDataException">字段编号、线型或长度不合法。</exception>
    /// <exception cref="EndOfStreamException">字段头被截断。</exception>
    private static (int Number, int Wire, long End) Field(Stream stream, long limit)
    {
        ulong tag = Varint(stream, limit); int wire = (int)(tag & 7); ulong number = tag >> 3;
        if (number is 0 or > 536870911) throw new InvalidDataException("Invalid protobuf field number.");
        ulong length;
        switch (wire)
        {
            case 0: Varint(stream, limit); length = 0; break;
            case 1: length = 8; break;
            case 2: length = Varint(stream, limit); break;
            case 5: length = 4; break;
            default: throw new InvalidDataException("Unsupported protobuf wire type.");
        }
        if (length > (ulong)(limit - stream.Position)) throw new InvalidDataException("Protobuf field exceeds message.");
        return ((int)number, wire, stream.Position + (long)length);
    }

    /// <summary>读取有界的无符号变长整数。</summary>
    /// <param name="stream">模型文件流。</param>
    /// <param name="limit">父消息结束位置。</param>
    /// <returns>解码后的 64 位无符号整数。</returns>
    /// <exception cref="InvalidDataException">变长整数超过 64 位。</exception>
    /// <exception cref="EndOfStreamException">整数被截断。</exception>
    private static ulong Varint(Stream stream, long limit)
    {
        ulong value = 0;
        for (int shift = 0; shift < 70; shift += 7)
        {
            if (stream.Position >= limit) throw new EndOfStreamException("Truncated protobuf integer.");
            int b = stream.ReadByte(); if (b < 0) throw new EndOfStreamException();
            if (shift == 63 && (b & 0xfe) != 0) throw new InvalidDataException("Protobuf integer overflow.");
            value |= (ulong)(b & 127) << shift; if ((b & 128) == 0) return value;
        }
        throw new InvalidDataException("Protobuf integer overflow.");
    }

    /// <summary>读取小型元数据消息，限制分配大小为 4 MiB。</summary>
    /// <param name="stream">模型文件流。</param>
    /// <param name="end">当前消息结束位置。</param>
    /// <returns>消息原始字节。</returns>
    /// <exception cref="InvalidDataException">元数据消息超限。</exception>
    /// <exception cref="EndOfStreamException">消息被截断。</exception>
    private static byte[] Bytes(Stream stream, long end)
    {
        long count = end - stream.Position; if (count is < 0 or > 4 * 1024 * 1024) throw new InvalidDataException("ONNX metadata exceeds 4 MiB.");
        var data = new byte[(int)count]; stream.ReadExactly(data); return data;
    }
}
