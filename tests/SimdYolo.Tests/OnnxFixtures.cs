using System.Buffers.Binary;
using System.Text;

// 独立的最小 protobuf 编码器，为优化回归用例构造模型；测试不依赖 Python、ONNX 包或参考推理运行库。
/// <summary>提供最小 ONNX protobuf 测试图的独立编码的回归验证。</summary>
internal static class OnnxFixtures
{
    /// <summary>保存 ONNX 测试图的权重定义。</summary>
    /// <param name="Name">测试张量或节点的名称。</param>
    /// <param name="Data">测试权重的浮点数据。</param>
    /// <param name="Shape">测试权重的逻辑维度。</param>
    /// <param name="Type">ONNX 张量元素类型编号。</param>
    internal sealed record Weight(string Name, float[] Data, int[] Shape, int Type = 1);
    /// <summary>保存 ONNX 测试图的操作节点定义。</summary>
    /// <param name="Operation">ONNX 操作名称。</param>
    /// <param name="Inputs">节点输入名称序列。</param>
    /// <param name="Outputs">节点输出名称序列。</param>
    /// <param name="Attributes">可选的 protobuf 属性字段。</param>
    internal sealed record Node(string Operation, string[] Inputs, string[] Outputs, byte[][]? Attributes = null);
    /// <summary>合并独立编码的 protobuf 字段，形成一条测试消息。</summary>
    /// <param name="fields">需要合并的 protobuf 字段字节串。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    private static byte[] Message(params byte[][] fields) => fields.SelectMany(x => x).ToArray();
    /// <summary>按 protobuf 的变长整数格式编码一个无符号整数。</summary>
    /// <param name="value">待编码、标记或比较的数值。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    private static byte[] Varint(ulong value)
    {
        var data = new List<byte>();
        do { byte b = (byte)(value & 127); value >>= 7; data.Add(value == 0 ? b : (byte)(b | 128)); } while (value != 0);
        return data.ToArray();
    }
    /// <summary>编码指定字段编号的 protobuf 整数值。</summary>
    /// <param name="field">protobuf 字段编号。</param>
    /// <param name="value">待编码、标记或比较的数值。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    private static byte[] Int(int field, long value) => Message(Varint((ulong)field * 8), Varint(unchecked((ulong)value)));
    /// <summary>编码指定字段编号的长度分隔字节串。</summary>
    /// <param name="field">protobuf 字段编号。</param>
    /// <param name="value">写入长度分隔字段的原始字节内容。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    private static byte[] Bytes(int field, byte[] value) => Message(Varint((ulong)field * 8 + 2), Varint((ulong)value.Length), value);
    /// <summary>将 UTF-8 文本编码为长度分隔的 protobuf 字段。</summary>
    /// <param name="field">protobuf 字段编号。</param>
    /// <param name="value">需要转换为 UTF-8 字节的文本。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    private static byte[] Text(int field, string value) => Bytes(field, Encoding.UTF8.GetBytes(value));
    /// <summary>编码 ONNX 整数序列属性。</summary>
    /// <param name="name">测试文件、张量或 ONNX 属性的名称。</param>
    /// <param name="values">测试浮点值或编码属性值序列。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    internal static byte[] Integers(string name, params int[] values) => Message(Text(1, name), Bytes(8, Message(values.Select(v => Varint(unchecked((ulong)(long)v))).ToArray())), Int(20, 7));
    /// <summary>编码 ONNX 单整数属性。</summary>
    /// <param name="name">测试文件、张量或 ONNX 属性的名称。</param>
    /// <param name="value">待编码、标记或比较的数值。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    internal static byte[] Integer(string name, long value) => Message(Text(1, name), Int(3, value), Int(20, 2));
    /// <summary>编码 ONNX 单精度浮点属性。</summary>
    /// <param name="name">测试文件、张量或 ONNX 属性的名称。</param>
    /// <param name="value">待编码、标记或比较的数值。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    internal static byte[] Real(string name, float value)
    {
        byte[] bytes = new byte[4]; BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        return Message(Text(1, name), Varint(2 * 8 + 5), bytes, Int(20, 1));
    }
    /// <summary>编码 ONNX 输入或输出张量的名称、类型和逻辑维度。</summary>
    /// <param name="name">测试文件、张量或 ONNX 属性的名称。</param>
    /// <param name="shape">测试张量的逻辑维度。</param>
    /// <param name="type">ONNX 张量元素类型编号。</param>
    /// <returns>编码完成的 protobuf 字节串。</returns>
    private static byte[] Value(string name, int[] shape, int type = 1)
    {
        byte[] dimensions = Message(shape.Select((d, i) => Bytes(1, d < 0 ? Text(2, "dynamic" + i) : Int(1, d))).ToArray());
        return Message(Text(1, name), Bytes(2, Bytes(1, Message(Int(1, type), Bytes(2, dimensions)))));
    }
    /// <summary>组装并写入最小 ONNX 模型，返回供加载器使用的测试文件路径。</summary>
    /// <param name="name">测试文件、张量或 ONNX 属性的名称。</param>
    /// <param name="inputShape">测试图输入张量的逻辑维度。</param>
    /// <param name="nodes">需要写入测试图的操作节点。</param>
    /// <param name="weights">待验证的权重数据或内部打包矩阵。</param>
    /// <param name="outputs">输出张量名称、维度和元素类型的声明。</param>
    /// <returns>写入的 ONNX 测试文件路径。</returns>
    internal static string Write(string name, int[] inputShape, Node[] nodes, Weight[] weights, params (string Name, int[] Shape, int Type)[] outputs)
    {
        var fields = new List<byte[]> { Text(2, name), Bytes(11, Value("images", inputShape)) };
        for (int i = 0; i < nodes.Length; i++)
        {
            var n = nodes[i];
            fields.Add(Bytes(1, Message(Message(n.Inputs.Select(s => Text(1, s)).ToArray()), Message(n.Outputs.Select(s => Text(2, s)).ToArray()), Text(3, "node" + i), Text(4, n.Operation), Message((n.Attributes ?? []).Select(a => Bytes(5, a)).ToArray()))));
        }
        foreach (var w in weights)
        {
            byte[] raw = new byte[w.Data.Length * (w.Type == 7 ? 8 : 4)];
            for (int i = 0; i < w.Data.Length; i++)
                if (w.Type == 7) BinaryPrimitives.WriteInt64LittleEndian(raw.AsSpan(i * 8), (long)w.Data[i]);
                else BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), w.Data[i]);
            fields.Add(Bytes(5, Message(Bytes(1, Message(w.Shape.Select(d => Varint((ulong)d)).ToArray())), Int(2, w.Type), Text(8, w.Name), Bytes(9, raw))));
        }
        fields.AddRange(outputs.Select(o => Bytes(12, Value(o.Name, o.Shape, o.Type))));
        byte[] model = Message(Int(1, 8), Bytes(7, Message(fields.ToArray())), Bytes(8, Message(Text(1, ""), Int(2, 17))));
        string directory = Path.GetFullPath(Path.Combine(".cache", "tests")); Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, name + ".onnx"); File.WriteAllBytes(path, model); return path;
    }
}
