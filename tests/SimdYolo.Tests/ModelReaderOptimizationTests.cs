using System.Buffers.Binary;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Text;
using SimdYolo;

/// <summary>验证连续权重批量加载、FP16 SIMD 位语义及 protobuf 借用视图的边界与所有权。</summary>
internal static class ModelReaderOptimizationTests
{
    /// <summary>注册独立于 Python 和模型下载的权重加载回归用例。</summary>
    /// <param name="test">执行命名断言的回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("Model readers FP16: exhaustive 65536 encodings, vector boundaries and singleton strides", Half);
        test("Model readers: all raw dtypes preserve special values, offsets and exact conversion", Dtypes);
        test("PT storage layouts: transpose, broadcast, empty, invalid strides and offset bounds", Layouts);
        test("ONNX protobuf views: zero-copy ownership, packed fields and malformed messages", Protobuf);
    }

    /// <summary>以框架逐元素转换为独立参考，核对全部 FP16 位模式、尾部及单元素维度的任意步长。</summary>
    /// <exception cref="Exception">加载结果的位模式或形状不符合独立参考时。</exception>
    private static void Half()
    {
        byte[] raw = new byte[65536 * 2];
        for (int bits = 0; bits < 65536; bits++) BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(bits * 2, 2), (ushort)bits);
        var pt = Pt("HalfStorage", raw, [1, 256, 256], [long.MaxValue, 256, 1]);
        var onnx = Onnx(10, raw, [256, 256]);
        for (int bits = 0; bits < 65536; bits++)
        {
            float expected = (float)BitConverter.UInt16BitsToHalf((ushort)bits);
            EqualBits(pt.Data[bits], expected, "PT FP16 " + bits); EqualBits(onnx.Data[bits], expected, "ONNX FP16 " + bits);
        }
        foreach (int length in new[] { 1, 7, 8, 15, 16, 17, 31, 32, 33, 65 })
        {
            int offset = 1020;
            pt = Pt("HalfStorage", raw, [length], [1], offset);
            onnx = Onnx(10, raw.AsSpan(offset * 2, length * 2).ToArray(), [length]);
            for (int i = 0; i < length; i++)
            {
                float expected = (float)BitConverter.UInt16BitsToHalf((ushort)(offset + i));
                EqualBits(pt.Data[i], expected, "PT FP16 tail"); EqualBits(onnx.Data[i], expected, "ONNX FP16 tail");
            }
        }
    }

    /// <summary>核对所有支持的原始存储类型，包含非标准 NaN 载荷、负零和大整数的框架舍入。</summary>
    /// <exception cref="Exception">任一存储类型改变数值、符号位或 NaN 载荷时。</exception>
    private static void Dtypes()
    {
        var cases = new List<(string Storage, int Onnx, byte[] Raw, float[] Expected)>();
        float[] floats = [BitConverter.Int32BitsToSingle(unchecked((int)0x7f812345)), BitConverter.Int32BitsToSingle(unchecked((int)0xffc23456)), -0f, 0f, float.Epsilon, float.PositiveInfinity, float.NegativeInfinity, -1.25f, float.MaxValue];
        byte[] bytes = new byte[floats.Length * 4];
        for (int i = 0; i < floats.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), floats[i]);
        cases.Add(("FloatStorage", 1, bytes, floats));
        ushort[] bf16 = [0, 0x8000, 1, 0x007f, 0x3f80, 0xff80, 0x7f80, 0x7f81, 0xffc7, 0x7fff];
        bytes = new byte[bf16.Length * 2];
        for (int i = 0; i < bf16.Length; i++) BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), bf16[i]);
        cases.Add(("BFloat16Storage", 16, bytes, bf16.Select(v => BitConverter.Int32BitsToSingle(v << 16)).ToArray()));
        double[] doubles = [-0d, double.Epsilon, double.PositiveInfinity, double.NegativeInfinity, double.NaN, -1.234567890123, double.MaxValue];
        bytes = new byte[doubles.Length * 8];
        for (int i = 0; i < doubles.Length; i++) BinaryPrimitives.WriteDoubleLittleEndian(bytes.AsSpan(i * 8), doubles[i]);
        cases.Add(("DoubleStorage", 11, bytes, doubles.Select(v => (float)v).ToArray()));
        long[] longs = [long.MinValue, long.MaxValue, 0, -1, 1, 16777217, 9007199254740993];
        bytes = new byte[longs.Length * 8];
        for (int i = 0; i < longs.Length; i++) BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(i * 8), longs[i]);
        cases.Add(("LongStorage", 7, bytes, longs.Select(v => (float)v).ToArray()));
        int[] ints = [int.MinValue, int.MaxValue, -16777217, 16777217, 0, 1, -1];
        bytes = new byte[ints.Length * 4];
        for (int i = 0; i < ints.Length; i++) BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4), ints[i]);
        cases.Add(("IntStorage", 6, bytes, ints.Select(v => (float)v).ToArray()));
        byte[] small = [0, 1, 127, 128, 255, 2, 254];
        cases.Add(("ByteStorage", 2, small, small.Select(v => (float)v).ToArray()));
        cases.Add(("CharStorage", 3, small, small.Select(v => (float)(sbyte)v).ToArray()));
        cases.Add(("BoolStorage", 9, small, small.Select(v => (float)v).ToArray()));
        foreach (var item in cases)
        {
            var pt = Pt(item.Storage, item.Raw, [item.Expected.Length], [1]);
            var onnx = Onnx(item.Onnx, item.Raw, [item.Expected.Length]);
            for (int i = 0; i < item.Expected.Length; i++)
            {
                EqualBits(pt.Data[i], item.Expected[i], item.Storage); EqualBits(onnx.Data[i], item.Expected[i], "ONNX " + item.Onnx);
            }
            int size = item.Raw.Length / item.Expected.Length;
            pt = Pt(item.Storage, item.Raw, [item.Expected.Length - 2], [1], offset: 1);
            onnx = Onnx(item.Onnx, item.Raw.AsSpan(size, (item.Expected.Length - 2) * size).ToArray(), [item.Expected.Length - 2]);
            for (int i = 0; i < pt.Length; i++) { EqualBits(pt.Data[i], item.Expected[i + 1], "PT offset"); EqualBits(onnx.Data[i], item.Expected[i + 1], "ONNX slice"); }
            if (item.Raw.Length > 1) Expect<InvalidDataException>(() => Onnx(item.Onnx, item.Raw[..^1], [item.Expected.Length]));
        }
    }

    /// <summary>核对连续切片和非连续视图，确保优化保留空张量及损坏存储的异常边界。</summary>
    /// <exception cref="Exception">视图地址、零长度行为或异常类型不符合预期时。</exception>
    private static void Layouts()
    {
        byte[] raw = new byte[40 * 4];
        for (int i = 0; i < 40; i++) BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), i + .25f);
        foreach (var (shape, strides, offset) in new[]
        {
            (new[] { 3, 5 }, new long[] { 5, 1 }, 5L),
            (new[] { 3, 4 }, new long[] { 7, 2 }, 2L),
            (new[] { 4, 3 }, new long[] { 1, 4 }, 3L),
            (new[] { 4, 5 }, new long[] { 0, 1 }, 1L),
            (new[] { 3, 1, 4 }, new long[] { 4, long.MaxValue, 1 }, 3L),
            (Array.Empty<int>(), Array.Empty<long>(), 39L)
        })
        {
            var tensor = Pt("FloatStorage", raw, shape, strides, offset);
            for (int i = 0; i < tensor.Length; i++)
            {
                long position = offset; int coordinate = i;
                for (int d = shape.Length - 1; d >= 0; d--) { position += coordinate % shape[d] * strides[d]; coordinate /= shape[d]; }
                EqualBits(tensor.Data[i], position + .25f, "PT strided");
            }
        }
        if (Pt("FloatStorage", raw, [2, 0, 3], [3, 3, 1], offset: long.MaxValue).Length != 0) throw new Exception("Empty PT tensor changed behavior.");
        Expect<InvalidDataException>(() => Pt("FloatStorage", raw, [3], [-1]));
        Expect<InvalidDataException>(() => Pt("FloatStorage", raw, [3], []));
        Expect<InvalidDataException>(() => Pt("FloatStorage", raw, [3], [1], offset: -1));
        Expect<InvalidDataException>(() => Pt("FloatStorage", raw, [3], [1], offset: 38));
        Expect<InvalidDataException>(() => Pt("FloatStorage", raw, [2, 2], [39, 2], offset: 1));
        Expect<InvalidDataException>(() => Pt("FloatStorage", raw, [3], [1], storageCount: 39));
        Expect<OverflowException>(() => Pt("FloatStorage", raw, [3], [long.MaxValue], offset: 1));
        Expect<NotSupportedException>(() => Pt("UnknownStorage", raw, [3], [1]));
    }

    /// <summary>核对 protobuf 视图不复制大字段，普通解析保留独立存储，且两条路径一致拒绝截断消息。</summary>
    /// <exception cref="Exception">字段所有权、特殊值或损坏消息的异常行为不符合预期时。</exception>
    private static void Protobuf()
    {
        Type protoType = typeof(Tensor).Assembly.GetType("SimdYolo.Proto")!;
        byte[] raw = new byte[4 * 1024 * 1024]; raw[0] = 17; raw[^1] = 29;
        byte[] message = Bytes(9, raw);
        object view = Invoke(protoType.GetMethod("ParseView", BindingFlags.NonPublic | BindingFlags.Static)!, null, [(ReadOnlyMemory<byte>)message])!;
        object owned = Invoke(protoType.GetMethod("Parse")!, null, [message])!;
        var buffer = (ReadOnlyMemory<byte>)Invoke(protoType.GetMethod("Buffer")!, view, [9])!;
        var copied = (ReadOnlyMemory<byte>)Invoke(protoType.GetMethod("Buffer")!, owned, [9])!;
        if (!MemoryMarshal.TryGetArray(buffer, out ArraySegment<byte> borrowed) || !ReferenceEquals(borrowed.Array, message)) throw new Exception("ONNX view copied raw tensor bytes.");
        if (!MemoryMarshal.TryGetArray(copied, out ArraySegment<byte> separate) || ReferenceEquals(separate.Array, message)) throw new Exception("Owning protobuf parse changed storage ownership.");
        message[borrowed.Offset] = 77;
        if (buffer.Span[0] != 77 || copied.Span[0] != 17 || buffer.Span[^1] != 29) throw new Exception("Protobuf field ownership mismatch.");
        foreach (byte[] bad in new[] { new byte[] { 0 }, new byte[] { 0x0a, 3, 1 }, new byte[] { 0x0d, 1, 2, 3 }, new byte[] { 8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 2 } })
        {
            Expect<InvalidDataException>(() => Invoke(protoType.GetMethod("Parse")!, null, [bad]));
            Expect<InvalidDataException>(() => Invoke(protoType.GetMethod("ParseView", BindingFlags.NonPublic | BindingFlags.Static)!, null, [(ReadOnlyMemory<byte>)bad]));
        }
        float[] expected = [-0f, float.PositiveInfinity, BitConverter.Int32BitsToSingle(unchecked((int)0x7f812345)), 1.25f];
        byte[] packed = new byte[expected.Length * 4];
        for (int i = 0; i < expected.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(packed.AsSpan(i * 4), expected[i]);
        byte[] tensorMessage = Message(Int(1, expected.Length), Int(2, 1), Bytes(4, packed[..8]), Bytes(4, packed[8..]));
        foreach (bool borrow in new[] { false, true })
        {
            object proto = borrow ? Invoke(protoType.GetMethod("ParseView", BindingFlags.NonPublic | BindingFlags.Static)!, null, [(ReadOnlyMemory<byte>)tensorMessage])! : Invoke(protoType.GetMethod("Parse")!, null, [tensorMessage])!;
            var tensor = ReadOnnxTensor(proto);
            for (int i = 0; i < tensor.Length; i++) EqualBits(tensor.Data[i], expected[i], "ONNX packed float");
            long[] integers = [int.MinValue, int.MaxValue, -16777217, 16777217, 0, -1];
            byte[] integerMessage = Message(Int(1, integers.Length), Int(2, 6), Bytes(5, Message(integers.Select(v => Varint(unchecked((ulong)v))).ToArray())));
            proto = borrow ? Invoke(protoType.GetMethod("ParseView", BindingFlags.NonPublic | BindingFlags.Static)!, null, [(ReadOnlyMemory<byte>)integerMessage])! : Invoke(protoType.GetMethod("Parse")!, null, [integerMessage])!;
            tensor = ReadOnnxTensor(proto);
            for (int i = 0; i < tensor.Length; i++) EqualBits(tensor.Data[i], (float)integers[i], "ONNX packed signed integer");
        }
    }

    /// <summary>通过独立 Pickle 编码器和内存 ZIP 构造单个张量，调用真实存储加载器。</summary>
    /// <param name="storage">Torch 存储类型名称。</param>
    /// <param name="raw">ZIP data/0 的存储字节。</param>
    /// <param name="shape">张量逻辑维度。</param>
    /// <param name="strides">张量存储步长。</param>
    /// <param name="offset">存储中的起始元素偏移。</param>
    /// <param name="storageCount">可选的显式存储元素数量，用于损坏存储测试。</param>
    /// <returns>真实 Pickle 加载器还原的 FP32 张量。</returns>
    private static Tensor Pt(string storage, byte[] raw, int[] shape, long[] strides, long offset = 0, long? storageCount = null)
    {
        int size = storage switch { "FloatStorage" or "IntStorage" => 4, "DoubleStorage" or "LongStorage" => 8, "HalfStorage" or "BFloat16Storage" => 2, _ => 1 };
        using var pickle = new MemoryStream(); using (var writer = new BinaryWriter(pickle, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(new byte[] { 0x80, 2, (byte)'c' }); writer.Write(Encoding.ASCII.GetBytes("torch._utils\n_rebuild_tensor_v2\n")); writer.Write((byte)'(');
            writer.Write((byte)'('); PickleText(writer, "storage"); writer.Write((byte)'c'); writer.Write(Encoding.ASCII.GetBytes("torch\n" + storage + "\n")); PickleText(writer, "0"); PickleText(writer, "cpu"); PickleInt(writer, storageCount ?? raw.Length / size); writer.Write((byte)'t'); writer.Write((byte)'Q');
            PickleInt(writer, offset); writer.Write((byte)'('); foreach (int dimension in shape) PickleInt(writer, dimension); writer.Write((byte)'t');
            writer.Write((byte)'('); foreach (long stride in strides) PickleInt(writer, stride); writer.Write((byte)'t');
            writer.Write(new byte[] { 0x89, (byte)'}', (byte)'t', (byte)'R', (byte)'.' });
        }
        using var archiveBytes = new MemoryStream();
        using (var zip = new ZipArchive(archiveBytes, ZipArchiveMode.Create, leaveOpen: true)) { using var stream = zip.CreateEntry("archive/data/0").Open(); stream.Write(raw); }
        archiveBytes.Position = 0;
        using var archive = new ZipArchive(archiveBytes, ZipArchiveMode.Read);
        Type readerType = typeof(Tensor).Assembly.GetType("SimdYolo.PickleReader")!;
        object reader = Activator.CreateInstance(readerType, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, [archive, "archive/", true], null)!;
        return (Tensor)Invoke(readerType.GetMethod("Load")!, reader, [pickle.ToArray()])!;
    }

    /// <summary>用独立 protobuf 编码器构造内嵌原始权重。</summary>
    /// <param name="type">ONNX 元素类型编号。</param>
    /// <param name="raw">小端原始权重字节。</param>
    /// <param name="shape">权重张量维度。</param>
    /// <returns>真实 ONNX 加载器还原的 FP32 张量。</returns>
    private static Tensor Onnx(int type, byte[] raw, int[] shape)
    {
        byte[] message = Message(Message(shape.Select(d => Int(1, d)).ToArray()), Int(2, type), Bytes(9, raw));
        Type protoType = typeof(Tensor).Assembly.GetType("SimdYolo.Proto")!;
        return ReadOnnxTensor(Invoke(protoType.GetMethod("ParseView", BindingFlags.NonPublic | BindingFlags.Static)!, null, [(ReadOnlyMemory<byte>)message])!);
    }

    /// <summary>反射调用内部 ONNX 权重读取器，测试不依赖其内部类型的编译访问权限。</summary>
    /// <param name="proto">权重的 protobuf 消息对象。</param>
    /// <returns>还原的 FP32 张量。</returns>
    private static Tensor ReadOnnxTensor(object proto) => (Tensor)Invoke(typeof(Tensor).Assembly.GetType("SimdYolo.OnnxGraph")!.GetMethod("ReadTensor", BindingFlags.NonPublic | BindingFlags.Static)!, null, [proto, null])!;

    /// <summary>调用内部实现并重新抛出原始异常，避免反射包装掩盖真实异常类型。</summary>
    /// <param name="method">待调用的方法。</param>
    /// <param name="target">实例方法的目标对象；静态方法为 null。</param>
    /// <param name="arguments">方法参数。</param>
    /// <returns>调用返回值。</returns>
    private static object? Invoke(MethodInfo method, object? target, object?[] arguments)
    {
        try { return method.Invoke(target, arguments); }
        catch (TargetInvocationException error) when (error.InnerException is not null) { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
    }

    /// <summary>比较 FP32 位表示，覆盖 NaN 载荷和负零。</summary>
    /// <param name="actual">加载器输出值。</param>
    /// <param name="expected">独立参考值。</param>
    /// <param name="context">失败位置说明。</param>
    /// <exception cref="Exception">位表示不一致时。</exception>
    private static void EqualBits(float actual, float expected, string context)
    {
        if (BitConverter.SingleToInt32Bits(actual) != BitConverter.SingleToInt32Bits(expected)) throw new Exception($"{context}: {BitConverter.SingleToInt32Bits(actual):x8} != {BitConverter.SingleToInt32Bits(expected):x8}.");
    }

    /// <summary>验证动作产生指定异常类型。</summary>
    /// <typeparam name="T">预期异常类型。</typeparam>
    /// <param name="action">预期失败的读取动作。</param>
    /// <exception cref="Exception">动作未抛出预期异常时。</exception>
    private static void Expect<T>(Action action) where T : Exception
    {
        try { action(); } catch (T) { return; }
        throw new Exception("Missing expected exception " + typeof(T).Name + ".");
    }

    /// <summary>写入 Pickle BINUNICODE 字段。</summary>
    /// <param name="writer">独立测试编码器。</param>
    /// <param name="value">要编码的文本。</param>
    private static void PickleText(BinaryWriter writer, string value) { byte[] bytes = Encoding.UTF8.GetBytes(value); writer.Write((byte)'X'); writer.Write(bytes.Length); writer.Write(bytes); }
    /// <summary>写入定长带符号 Pickle 整数，覆盖 Int64 布局异常。</summary>
    /// <param name="writer">独立测试编码器。</param>
    /// <param name="value">整数值。</param>
    private static void PickleInt(BinaryWriter writer, long value) { writer.Write((byte)0x8a); writer.Write((byte)8); writer.Write(value); }
    /// <summary>拼接独立 protobuf 字段。</summary>
    /// <param name="fields">字段字节。</param>
    /// <returns>消息字节。</returns>
    private static byte[] Message(params byte[][] fields) => fields.SelectMany(f => f).ToArray();
    /// <summary>编码无符号 protobuf 变长整数。</summary>
    /// <param name="value">整数值。</param>
    /// <returns>编码字节。</returns>
    private static byte[] Varint(ulong value)
    {
        var bytes = new List<byte>();
        do { byte next = (byte)(value & 127); value >>= 7; bytes.Add(value == 0 ? next : (byte)(next | 128)); } while (value != 0);
        return bytes.ToArray();
    }
    /// <summary>编码 protobuf 整数字段。</summary>
    /// <param name="field">字段编号。</param>
    /// <param name="value">字段整数。</param>
    /// <returns>字段字节。</returns>
    private static byte[] Int(int field, long value) => Message(Varint((ulong)field * 8), Varint(unchecked((ulong)value)));
    /// <summary>编码 protobuf 长度分隔字段。</summary>
    /// <param name="field">字段编号。</param>
    /// <param name="value">字段字节。</param>
    /// <returns>编码字段字节。</returns>
    private static byte[] Bytes(int field, byte[] value) => Message(Varint((ulong)field * 8 + 2), Varint((ulong)value.Length), value);
}
