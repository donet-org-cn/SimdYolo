using System.Buffers.Binary;
using System.Text;

namespace SimdYolo;

/// <summary>读取 ONNX 所需的 Protocol Buffers 字段，不依赖外部 protobuf 库。</summary>
internal sealed class Proto
{
    /// <summary>按字段号保存出现顺序，保留重复字段和原始数值或字节内容。</summary>
    public readonly Dictionary<int, List<object>> Fields = [];
    /// <summary>解析支持的 varint、32 位、64 位和长度分隔 protobuf 字段。</summary>
    /// <param name="bytes">待解析的 protobuf 消息字节内容。</param>
    /// <returns>以字段号组织的 protobuf 消息。</returns>
    /// <exception cref="InvalidDataException">字段号无效、wire type 不支持，或消息被截断。</exception>
    /// <exception cref="OverflowException">字段号或字段字节长度超过 Int32 范围。</exception>
    public static Proto Parse(byte[] bytes)
    {
        var p = new Proto(); int i = 0;
        while (i < bytes.Length)
        {
            ulong tag = Varint(bytes, ref i); int field = checked((int)(tag >> 3)); object value;
            if (field == 0) throw new InvalidDataException("Invalid protobuf tag.");
            switch (tag & 7)
            {
                case 0: value = Varint(bytes, ref i); break;
                case 1: Check(bytes, i, 8); value = bytes.AsSpan(i, 8).ToArray(); i += 8; break;
                case 2: int length = checked((int)Varint(bytes, ref i)); Check(bytes, i, length); value = bytes.AsSpan(i, length).ToArray(); i += length; break;
                case 5: Check(bytes, i, 4); value = bytes.AsSpan(i, 4).ToArray(); i += 4; break;
                default: throw new InvalidDataException("Unsupported protobuf wire type.");
            }
            if (!p.Fields.TryGetValue(field, out var list)) p.Fields[field] = list = [];
            list.Add(value);
        }
        return p;
    }
    /// <summary>确认即将读取的字节范围位于消息缓冲区内。</summary>
    /// <param name="data">消息字节缓冲区。</param>
    /// <param name="i">准备读取的起始字节位置。</param>
    /// <param name="n">需要读取的字节数量。</param>
    /// <exception cref="InvalidDataException">所请求的字节范围超出消息内容。</exception>
    private static void Check(byte[] data, int i, int n) { if (n < 0 || i < 0 || n > data.Length - i) throw new InvalidDataException("Truncated protobuf."); }
    /// <summary>读取最多 64 位的 protobuf 无符号变长整数，并更新读取位置。</summary>
    /// <param name="bytes">待解析的 protobuf 消息字节内容。</param>
    /// <param name="i">读取游标；方法完成时指向未消费的下一个字节。</param>
    /// <returns>解析的无符号整数。</returns>
    /// <exception cref="InvalidDataException">变长整数被截断或超过 64 位范围。</exception>
    internal static ulong Varint(byte[] bytes, ref int i)
    {
        ulong v = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            Check(bytes, i, 1); byte b = bytes[i++]; if (shift == 63 && b > 1) throw new InvalidDataException("Varint overflow.");
            v |= (ulong)(b & 127) << shift; if (b < 128) return v;
        }
        throw new InvalidDataException("Varint overflow.");
    }
    /// <summary>按保存顺序枚举指定字段的字节内容。</summary>
    /// <param name="f">protobuf 字段号。</param>
    /// <returns>字段的字节数组序列；字段缺失时为空。</returns>
    public IEnumerable<byte[]> Bytes(int f) => Fields.GetValueOrDefault(f, []).Cast<byte[]>();
    /// <summary>读取指定字段的第一个字节块。</summary>
    /// <param name="f">protobuf 字段号。</param>
    /// <returns>第一个字节数组；字段缺失时返回 null。</returns>
    public byte[]? Byte(int f) => Bytes(f).FirstOrDefault();
    /// <summary>把指定字段的第一个字节块按 UTF-8 解码。</summary>
    /// <param name="f">protobuf 字段号。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>字段文本；字段缺失时返回默认文本。</returns>
    public string Text(int f, string fallback = "") => Byte(f) is { } b ? Encoding.UTF8.GetString(b) : fallback;
    /// <summary>读取指定字段的第一个整数，并保留其二进制补码解释。</summary>
    /// <param name="f">protobuf 字段号。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>字段整数；字段缺失时返回默认值。</returns>
    public long Int(int f, long fallback = 0) => Fields.TryGetValue(f, out var a) ? unchecked((long)(ulong)a[0]) : fallback;
    /// <summary>读取重复 varint 或 packed varint 整数字段。</summary>
    /// <param name="f">protobuf 字段号。</param>
    /// <returns>按出现顺序展开的整数数组。</returns>
    public long[] Ints(int f)
    {
        var list = new List<long>();
        foreach (object a in Fields.GetValueOrDefault(f, []))
            if (a is ulong u) list.Add(unchecked((long)u));
            else { var b = (byte[])a; int i = 0; while (i < b.Length) list.Add(unchecked((long)Varint(b, ref i))); }
        return list.ToArray();
    }
    /// <summary>读取重复或 packed 小端 FP32 字段。</summary>
    /// <param name="f">protobuf 字段号。</param>
    /// <returns>按出现顺序展开的 FP32 数组。</returns>
    /// <exception cref="InvalidDataException">packed FP32 字节长度不是 4 的整数倍。</exception>
    public float[] Floats(int f)
    {
        var list = new List<float>();
        foreach (var b in Bytes(f))
        {
            if (b.Length % 4 != 0) throw new InvalidDataException("Invalid packed float.");
            for (int i = 0; i < b.Length; i += 4) list.Add(BinaryPrimitives.ReadSingleLittleEndian(b.AsSpan(i)));
        }
        return list.ToArray();
    }
}

/// <summary>保存一个 ONNX 算子的名称、类型、输入输出和值属性。</summary>
/// <param name="Name">记录或节点名称。</param>
/// <param name="Op">ONNX 算子类型。</param>
/// <param name="Inputs">按节点或图声明顺序排列的输入名称。</param>
/// <param name="Outputs">按节点或图声明顺序排列的输出名称。</param>
/// <param name="Attributes">按名称保存的 ONNX 属性原始消息。</param>
internal sealed record OnnxNode(string Name, string Op, string[] Inputs, string[] Outputs, Dictionary<string, Proto> Attributes)
{
    /// <summary>读取整数属性，属性缺失时使用默认值。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>属性整数或默认值。</returns>
    public long Int(string key, long fallback = 0) => Attributes.TryGetValue(key, out var p) ? p.Int(3, fallback) : fallback;
    /// <summary>读取整数列表属性，并检查元素是否可表示为 Int32。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>属性整数数组或指定默认数组。</returns>
    /// <exception cref="OverflowException">整数属性列表中有值超出 Int32 范围。</exception>
    public int[] Ints(string key, int[]? fallback = null) => Attributes.TryGetValue(key, out var p) ? p.Ints(8).Select(x => checked((int)x)).ToArray() : fallback ?? [];
    /// <summary>读取 FP32 属性的第一个值，未保存值时使用默认值。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>属性 FP32 值或默认值。</returns>
    public float Float(string key, float fallback = 0) => Attributes.TryGetValue(key, out var p) ? p.Floats(2).FirstOrDefault(fallback) : fallback;
    /// <summary>读取字符串属性，属性缺失时使用默认值。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>属性文本或默认值。</returns>
    public string Text(string key, string fallback = "") => Attributes.TryGetValue(key, out var p) ? p.Text(4, fallback) : fallback;
}

/// <summary>保存已解析的 ONNX 节点、常量权重、图输入输出、输入维度和元数据。</summary>
/// <param name="Nodes">按拓扑顺序排列的节点。</param>
/// <param name="Weights">按名称索引的常量初始化权重。</param>
/// <param name="Inputs">按节点或图声明顺序排列的输入名称。</param>
/// <param name="Outputs">按节点或图声明顺序排列的输出名称。</param>
/// <param name="InputShape">图像输入的静态或动态维度。</param>
/// <param name="Metadata">ONNX 模型元数据映射。</param>
/// <param name="Opset">标准 ONNX 域的算子集版本。</param>
internal sealed record OnnxGraph(List<OnnxNode> Nodes, Dictionary<string, Tensor> Weights, string[] Inputs, string[] Outputs, int[] InputShape, Dictionary<string, string> Metadata, int Opset)
{
    /// <summary>读取 ONNX 文件，校验标准域、opset、单个 FP32 图像输入，并加载权重。</summary>
    /// <param name="path">ONNX 模型文件路径；相对模型目录解析外部权重，拒绝绝对路径和 .. 越界。</param>
    /// <returns>可供托管解释器执行的 ONNX 图。</returns>
    /// <exception cref="InvalidDataException">ONNX 缺少 graph，或权重和 protobuf 数据损坏。</exception>
    /// <exception cref="NotSupportedException">域或 opset 不支持、图像输入不唯一，或输入类型不是 FP32。</exception>
    /// <exception cref="IOException">主模型文件或外部权重文件无法打开、读取，或数据被截断。</exception>
    /// <exception cref="FormatException">外部权重的 offset 或 length 文本不能解析为整数。</exception>
    /// <exception cref="OverflowException">张量尺寸、字节长度或外部 offset/length 超过受支持整数范围。</exception>
    public static OnnxGraph Load(string path)
    {
        var model = Proto.Parse(File.ReadAllBytes(path)); var graph = Proto.Parse(model.Byte(7) ?? throw new InvalidDataException("ONNX graph missing."));
        foreach (var op in model.Bytes(8).Select(Proto.Parse))
            if (op.Text(1) is not ("" or "ai.onnx")) throw new NotSupportedException($"ONNX domain {op.Text(1)} unsupported.");
        int opset = (int)model.Bytes(8).Select(Proto.Parse).Where(p => p.Text(1) is "" or "ai.onnx").Select(p => p.Int(2)).Single();
        if (opset is < 11 or > 21) throw new NotSupportedException($"ONNX opset {opset} unsupported. Use opset 11-21; opset 17 is reference-tested.");
        var weights = graph.Bytes(5).Select(Proto.Parse).ToDictionary(t => t.Text(8), t => ReadTensor(t, Path.GetDirectoryName(Path.GetFullPath(path))!));
        var inputs = graph.Bytes(11).Select(Proto.Parse).Where(p => !weights.ContainsKey(p.Text(1))).ToArray();
        if (inputs.Length != 1) throw new NotSupportedException("YOLO requires one image input.");
        var nodes = graph.Bytes(1).Select(Proto.Parse).Select(p =>
        {
            if (p.Text(7) is not ("" or "ai.onnx")) throw new NotSupportedException($"Node domain {p.Text(7)} unsupported.");
            return new OnnxNode(p.Text(3), p.Text(4), p.Bytes(1).Select(Encoding.UTF8.GetString).ToArray(), p.Bytes(2).Select(Encoding.UTF8.GetString).ToArray(), p.Bytes(5).Select(Proto.Parse).ToDictionary(a => a.Text(1)));
        }).ToList();
        var type = Proto.Parse(inputs[0].Byte(2) ?? []); var tensorType = Proto.Parse(type.Byte(1) ?? []);
        if (tensorType.Int(1) != 1) throw new NotSupportedException("Image input must be FP32.");
        var shape = Proto.Parse(tensorType.Byte(2) ?? []).Bytes(1).Select(Proto.Parse).Select(d => (int)d.Int(1, -1)).ToArray();
        return new(nodes, weights, inputs.Select(p => p.Text(1)).ToArray(), graph.Bytes(12).Select(Proto.Parse).Select(p => p.Text(1)).ToArray(), shape, model.Bytes(14).Select(Proto.Parse).ToDictionary(p => p.Text(1), p => p.Text(2)), opset);
    }
    /// <summary>把 ONNX TensorProto 的内嵌或受限外部存储转换为连续 FP32 常量。</summary>
    /// <param name="p">包含 ONNX TensorProto 字段的 protobuf 消息。</param>
    /// <param name="directory">模型所在目录，供受限外部权重加载；null 表示无法访问外部存储。</param>
    /// <returns>标记为常量且具有独立数据存储的张量。</returns>
    /// <exception cref="InvalidDataException">外部路径越过模型目录、字节范围不合法，或存储数量与形状不匹配。</exception>
    /// <exception cref="NotSupportedException">未提供外部存储目录，或张量元素类型不支持。</exception>
    /// <exception cref="OverflowException">张量维度、元素数量或字节长度超过 Int32 范围，或外部 offset/length 文本超过 Int64 范围。</exception>
    /// <exception cref="IOException">外部权重文件无法打开、读取，或读取时数据被截断。</exception>
    /// <exception cref="FormatException">外部权重的 offset 或 length 文本不能解析为整数。</exception>
    /// <exception cref="ArgumentException">张量形状包含负维度，或外部路径参数无效。</exception>
    internal static Tensor ReadTensor(Proto p, string? directory = null)
    {
        int[] shape = p.Ints(1).Select(x => checked((int)x)).ToArray(); int count = Tensor.Count(shape), type = (int)p.Int(2);
        byte[]? raw = p.Byte(9);
        if (p.Int(14) == 1)
        {
            if (directory is null) throw new NotSupportedException("External tensor unavailable.");
            var external = p.Bytes(13).Select(Proto.Parse).ToDictionary(e => e.Text(1), e => e.Text(2));
            string location = external.GetValueOrDefault("location") ?? throw new InvalidDataException("External tensor path missing.");
            string target = Path.GetFullPath(Path.Combine(directory, location));
            string relative = Path.GetRelativePath(directory, target);
            if (Path.IsPathRooted(location) || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar)) throw new InvalidDataException("External tensor must stay beside model.");
            using var f = File.OpenRead(target); long offset = long.Parse(external.GetValueOrDefault("offset", "0")); long length = long.Parse(external.GetValueOrDefault("length", (f.Length - offset).ToString()));
            if (offset < 0 || length < 0 || offset > f.Length - length) throw new InvalidDataException("External tensor range invalid.");
            f.Position = offset; raw = new byte[checked((int)length)]; f.ReadExactly(raw);
        }
        float[] data = new float[count];
        if (raw is not null)
        {
            int size = type switch { 1 or 6 => 4, 7 or 11 => 8, 10 or 16 => 2, 2 or 3 or 9 => 1, _ => throw new NotSupportedException($"ONNX tensor type {type} unsupported.") };
            if (raw.Length != checked(count * size)) throw new InvalidDataException("Tensor bytes do not match shape.");
            for (int i = 0; i < count; i++)
            {
                var b = raw.AsSpan(i * size, size);
                data[i] = type switch
                {
                    1 => BinaryPrimitives.ReadSingleLittleEndian(b),
                    6 => BinaryPrimitives.ReadInt32LittleEndian(b),
                    7 => BinaryPrimitives.ReadInt64LittleEndian(b),
                    11 => (float)BinaryPrimitives.ReadDoubleLittleEndian(b),
                    10 => (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(b)),
                    16 => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(b) << 16),
                    3 => (sbyte)b[0],
                    _ => b[0]
                };
            }
        }
        else
        {
            float[] values = type switch { 1 => p.Floats(4), 6 or 10 or 2 or 3 or 9 => p.Ints(5).Select(x => type == 10 ? (float)BitConverter.UInt16BitsToHalf((ushort)x) : (float)x).ToArray(), 7 => p.Ints(7).Select(x => (float)x).ToArray(), _ => throw new NotSupportedException($"ONNX tensor type {type} unsupported.") };
            if (values.Length != count) throw new InvalidDataException("Tensor values do not match shape."); data = values;
        }
        return new(data, shape) { IsConstant = true };
    }
}
