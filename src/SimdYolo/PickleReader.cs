using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace SimdYolo;

/// <summary>保存 Pickle GLOBAL 的模块与名称，仅作为符号数据使用。</summary>
/// <param name="Module">Python 符号所属模块名称。</param>
/// <param name="Name">Python 符号的类型或函数名称。</param>
internal sealed record PyGlobal(string Module, string Name);
/// <summary>保存 Python 对象的符号类型、构造参数和状态，避免实例化任意 Python 代码。</summary>
/// <param name="type">对象的符号 Python 模块和类型。</param>
/// <param name="args">保存的构造参数或节点输入。</param>
internal sealed class PyObject(PyGlobal type, object?[]? args = null)
{
    /// <summary>Pickle 声明的符号模块和类型名称。</summary>
    /// <value>Pickle 声明的符号模块和类型名称。</value>
    public PyGlobal Type { get; } = type;
    /// <summary>Pickle 保存的构造参数数组。</summary>
    /// <value>Pickle 保存的构造参数数组。</value>
    public object?[] Args { get; } = args ?? [];
    /// <summary>通过 Pickle BUILD 还原的对象属性映射。</summary>
    /// <value>通过 Pickle BUILD 还原的对象属性映射。</value>
    public Dictionary<object, object?> State { get; } = [];
    /// <summary>读取对象状态中的指定属性。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <returns>属性值；不存在时返回 null。</returns>
    public object? Get(string key) => State.GetValueOrDefault(key);
    /// <summary>读取指定属性中的字典，未保存字典时使用空映射。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <returns>状态字典或新的空字典。</returns>
    public Dictionary<object, object?> Map(string key) => Get(key) as Dictionary<object, object?> ?? [];
    /// <summary>从 _modules 中取出指定子模块。</summary>
    /// <param name="name">子模块、参数或持久缓冲区名称。</param>
    /// <returns>对应的符号子模块对象。</returns>
    /// <exception cref="InvalidDataException">指定子模块不存在。</exception>
    public PyObject Child(string name) => Map("_modules").GetValueOrDefault(name) as PyObject ?? throw new InvalidDataException($"{Type.Name}.{name} missing.");
    /// <summary>检查 _modules 是否包含指定符号子模块。</summary>
    /// <param name="name">子模块、参数或持久缓冲区名称。</param>
    /// <returns>存在指定子模块时返回 true。</returns>
    public bool Has(string name) => Map("_modules").GetValueOrDefault(name) is PyObject;
    /// <summary>优先从 _parameters、随后从 _buffers 读取命名张量。</summary>
    /// <param name="name">子模块、参数或持久缓冲区名称。</param>
    /// <returns>指定权重或持久化缓冲区张量。</returns>
    /// <exception cref="InvalidDataException">参数表和缓冲区表都未保存指定张量。</exception>
    public Tensor Weight(string name) => (Map("_parameters").GetValueOrDefault(name) ?? Map("_buffers").GetValueOrDefault(name)) as Tensor ?? throw new InvalidDataException($"{Type.Name}.{name} tensor missing.");
    /// <summary>读取可转换为 Int32 的属性，属性缺失时使用默认值。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>转换后的整数或默认值。</returns>
    public int Int(string key, int fallback = 0) => Get(key) is { } v ? Convert.ToInt32(v) : fallback;
    /// <summary>读取可转换为布尔值的属性，属性缺失时使用默认值。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>转换后的布尔值或默认值。</returns>
    public bool Bool(string key, bool fallback = false) => Get(key) is { } v ? Convert.ToBoolean(v) : fallback;
    /// <summary>读取可转换为 FP32 的属性，属性缺失时使用默认值。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>转换后的 FP32 值或默认值。</returns>
    public float Float(string key, float fallback = 0) => Get(key) is { } v ? Convert.ToSingle(v) : fallback;
    /// <summary>将标量、元组或列表形式的空间参数转换为整数数组，标量会复制到两个维度。</summary>
    /// <param name="key">要读取的属性或子模块名称。</param>
    /// <param name="fallback">属性未声明或值缺失时使用的默认值。</param>
    /// <returns>空间参数数组；未声明时返回两个默认值。</returns>
    public int[] Pair(string key, int fallback = 1) => Get(key) switch { object?[] a => a.Select(Convert.ToInt32).ToArray(), List<object?> a => a.Select(Convert.ToInt32).ToArray(), null => [fallback, fallback], var v => [Convert.ToInt32(v), Convert.ToInt32(v)] };
    /// <summary>按 _modules 的保存顺序返回所有符号子模块。</summary>
    /// <value>按 _modules 的保存顺序返回所有符号子模块。</value>
    public PyObject[] Children => Map("_modules").Values.OfType<PyObject>().ToArray();
}

/// <summary>只解释数据的 Pickle 虚拟机；GLOBAL 和 REDUCE 保留为符号对象，不导入或执行 Python。</summary>
/// <param name="zip">包含 data.pkl 与张量数据的 Torch ZIP 归档。</param>
/// <param name="prefix">归档内 data.pkl 所属目录前缀。</param>
internal sealed class PickleReader(ZipArchive zip, string prefix)
{
    /// <summary>Pickle 操作数栈。</summary>
    private readonly List<object?> stack = [];
    /// <summary>Pickle 对象引用表，用于还原共享引用。</summary>
    private readonly Dictionary<int, object?> memo = [];
    /// <summary>按存储键缓存已读取的 PT ZIP 张量字节。</summary>
    private readonly Dictionary<string, byte[]> storage = [];
    /// <summary>区分可变长度集合起点的栈标记对象。</summary>
    private readonly object mark = new();
    /// <summary>当前 Pickle 元数据缓冲区。</summary>
    private byte[] bytes = [];
    /// <summary>下一个待读取字节的位置。</summary>
    private int pos;
    /// <summary>弹出操作数栈顶元素。</summary>
    /// <returns>被移除的栈顶对象，可为 null。</returns>
    /// <exception cref="InvalidDataException">Pickle 操作数栈为空。</exception>
    private object? Pop() { if (stack.Count == 0) throw new InvalidDataException("Pickle stack underflow."); var v = stack[^1]; stack.RemoveAt(stack.Count - 1); return v; }
    /// <summary>取出最后一个 MARK 之后的元素，并连同 MARK 从栈中移除。</summary>
    /// <returns>按原栈顺序排列的集合元素。</returns>
    /// <exception cref="InvalidDataException">栈中没有集合所需的 MARK。</exception>
    private object?[] MarkItems()
    {
        int index = stack.LastIndexOf(mark); if (index < 0) throw new InvalidDataException("Pickle mark missing.");
        var items = stack.Skip(index + 1).ToArray(); stack.RemoveRange(index, stack.Count - index); return items;
    }
    /// <summary>读取指定数量的 Pickle 字节并推进游标。</summary>
    /// <param name="count">需要从当前 Pickle 游标读取的字节数。</param>
    /// <returns>具有独立存储的字节数组。</returns>
    /// <exception cref="InvalidDataException">读取长度为负或超过剩余元数据。</exception>
    private byte[] Read(int count)
    {
        if (count < 0 || count > bytes.Length - pos) throw new InvalidDataException("Truncated pickle.");
        var b = bytes.AsSpan(pos, count).ToArray(); pos += count; return b;
    }
    /// <summary>从当前游标读取小端有符号 32 位整数。</summary>
    /// <returns>解析的整数值。</returns>
    private int I32() => BinaryPrimitives.ReadInt32LittleEndian(Read(4));
    /// <summary>读取换行终止的 UTF-8 文本并越过换行字节。</summary>
    /// <returns>不包含结尾换行的文本。</returns>
    /// <exception cref="InvalidDataException">剩余元数据中没有行终止换行。</exception>
    private string Line()
    {
        int end = Array.IndexOf(bytes, (byte)'\n', pos); if (end < 0) throw new InvalidDataException("Pickle line truncated.");
        string line = Encoding.UTF8.GetString(bytes, pos, end - pos); pos = end + 1; return line;
    }
    /// <summary>解释受支持的 Pickle 操作码，重建符号对象和张量，并限制对象数量。</summary>
    /// <param name="data">完整的 data.pkl 元数据字节。</param>
    /// <returns>STOP 操作码对应的唯一根对象。</returns>
    /// <exception cref="NotSupportedException">协议高于 5、整数超过 Int64，或操作码与 BUILD 目标不受支持。</exception>
    /// <exception cref="InvalidDataException">栈、字典、终止符或对象数量不合法，或元数据被截断。</exception>
    /// <exception cref="OverflowException">受检字节长度、数字文本或还原张量尺寸超过对应整数范围。</exception>
    /// <exception cref="FormatException">数字操作码包含无效的整数或浮点文本。</exception>
    public object? Load(byte[] data)
    {
        bytes = data; pos = 0;
        while (pos < bytes.Length)
        {
            byte op = Read(1)[0];
            switch (op)
            {
                case 0x80: if (Read(1)[0] > 5) throw new NotSupportedException("Pickle protocol > 5."); break;
                case 0x95: Read(8); break; // FRAME 仅描述字节块，不改变后续操作码的解释规则。
                case (byte)'.': if (stack.Count != 1) throw new InvalidDataException("Invalid pickle final stack."); return Pop();
                case (byte)'(': stack.Add(mark); break;
                case (byte)'N': stack.Add(null); break;
                case 0x88: stack.Add(true); break;
                case 0x89: stack.Add(false); break;
                case (byte)'K': stack.Add((long)Read(1)[0]); break;
                case (byte)'M': stack.Add((long)BinaryPrimitives.ReadUInt16LittleEndian(Read(2))); break;
                case (byte)'J': stack.Add((long)I32()); break;
                case 0x8a:
                case 0x8b:
                    int size = op == 0x8a ? Read(1)[0] : I32(); var big = Read(size); if (size > 8) throw new NotSupportedException("Pickle integer > Int64.");
                    long integer = size > 0 && (big[^1] & 128) != 0 ? -1 : 0;
                    for (int i = big.Length - 1; i >= 0; i--) integer = (integer << 8) | big[i]; stack.Add(integer); break;
                case (byte)'G': stack.Add(BinaryPrimitives.ReadDoubleBigEndian(Read(8))); break;
                case (byte)'F': stack.Add(double.Parse(Line(), System.Globalization.CultureInfo.InvariantCulture)); break;
                case (byte)'I': string number = Line(); stack.Add(number == "00" ? false : number == "01" ? true : long.Parse(number)); break;
                case (byte)'X': stack.Add(Encoding.UTF8.GetString(Read(I32()))); break;
                case 0x8c: stack.Add(Encoding.UTF8.GetString(Read(Read(1)[0]))); break;
                case 0x8d: stack.Add(Encoding.UTF8.GetString(Read(checked((int)BinaryPrimitives.ReadUInt64LittleEndian(Read(8)))))); break;
                case (byte)'U': stack.Add(Encoding.Latin1.GetString(Read(Read(1)[0]))); break;
                case (byte)'T': stack.Add(Encoding.Latin1.GetString(Read(I32()))); break;
                case (byte)'C': stack.Add(Read(Read(1)[0])); break;
                case (byte)'B': stack.Add(Read(I32())); break;
                case 0x8e: case 0x96: stack.Add(Read(checked((int)BinaryPrimitives.ReadUInt64LittleEndian(Read(8))))); break;
                case (byte)')': stack.Add(Array.Empty<object?>()); break;
                case (byte)'t': stack.Add(MarkItems()); break;
                case 0x85: var one = Pop(); stack.Add(new object?[] { one }); break;
                case 0x86: var second = Pop(); var first = Pop(); stack.Add(new object?[] { first, second }); break;
                case 0x87: var third = Pop(); second = Pop(); first = Pop(); stack.Add(new object?[] { first, second, third }); break;
                case (byte)']': stack.Add(new List<object?>()); break;
                case (byte)'l': stack.Add(MarkItems().ToList()); break;
                case (byte)'a': var item = Pop(); ((List<object?>)stack[^1]!).Add(item); break;
                case (byte)'e': var items = MarkItems(); ((List<object?>)stack[^1]!).AddRange(items); break;
                case (byte)'}': stack.Add(new Dictionary<object, object?>()); break;
                case (byte)'d': var pairs = MarkItems(); var dict = new Dictionary<object, object?>(); SetPairs(dict, pairs); stack.Add(dict); break;
                case (byte)'s': var value = Pop(); var key = Pop() ?? throw new InvalidDataException("Null dictionary key."); ((Dictionary<object, object?>)stack[^1]!)[key] = value; break;
                case (byte)'u': pairs = MarkItems(); SetPairs((Dictionary<object, object?>)stack[^1]!, pairs); break;
                case 0x8f: stack.Add(new List<object?>()); break;
                case 0x90: items = MarkItems(); ((List<object?>)stack[^1]!).AddRange(items); break;
                case 0x91: stack.Add(MarkItems().ToList()); break;
                case (byte)'q': memo[Read(1)[0]] = stack[^1]; break;
                case (byte)'r': memo[I32()] = stack[^1]; break;
                case 0x94: memo[memo.Count] = stack[^1]; break;
                case (byte)'h': stack.Add(memo[Read(1)[0]]); break;
                case (byte)'j': stack.Add(memo[I32()]); break;
                case (byte)'c': stack.Add(new PyGlobal(Line(), Line())); break;
                case 0x93: string name = (string)Pop()!; string module = (string)Pop()!; stack.Add(new PyGlobal(module, name)); break;
                case 0x81: var args = (object?[])Pop()!; var cls = (PyGlobal)Pop()!; stack.Add(new PyObject(cls, args)); break;
                case (byte)'R': args = (object?[])Pop()!; cls = (PyGlobal)Pop()!; stack.Add(Reduce(cls, args)); break;
                case (byte)'b':
                    var state = Pop(); var target = stack[^1];
                    if (target is PyObject py)
                    {
                        if (state is Dictionary<object, object?> attrs) foreach (var pair in attrs) py.State[pair.Key] = pair.Value;
                        else if (state is object?[] states) foreach (var s in states.OfType<Dictionary<object, object?>>()) foreach (var pair in s) py.State[pair.Key] = pair.Value;
                        else py.State["__state__"] = state;
                    }
                    else if (target is not Dictionary<object, object?>) throw new NotSupportedException("Pickle BUILD target unsupported.");
                    break;
                case (byte)'Q': stack.Add(Persistent((object?[])Pop()!)); break;
                case (byte)'0': Pop(); break;
                case (byte)'1': MarkItems(); break;
                case (byte)'2': stack.Add(stack[^1]); break;
                default: throw new NotSupportedException($"Pickle opcode 0x{op:x2} at {pos - 1} unsupported.");
            }
            if (stack.Count > 1_000_000 || memo.Count > 1_000_000) throw new InvalidDataException("Pickle object limit exceeded.");
        }
        throw new InvalidDataException("Pickle STOP missing.");
    }
    /// <summary>按交替键和值填充 Pickle 字典。</summary>
    /// <param name="dict">接收键值对的字典。</param>
    /// <param name="pairs">按键、值交替排列的对象数组。</param>
    /// <exception cref="InvalidDataException">键值数量不是偶数，或出现 null 字典键。</exception>
    private static void SetPairs(Dictionary<object, object?> dict, object?[] pairs)
    {
        if (pairs.Length % 2 != 0) throw new InvalidDataException("Pickle dictionary pairs invalid.");
        for (int i = 0; i < pairs.Length; i += 2) dict[pairs[i] ?? throw new InvalidDataException("Null dictionary key.")] = pairs[i + 1];
    }
    /// <summary>描述 Torch 持久存储的元素类型、ZIP 数据键及元素数量。</summary>
    /// <param name="Kind">Torch 存储的元素类型名称。</param>
    /// <param name="Key">归档 data 目录中的存储键。</param>
    /// <param name="Count">持久存储中的元素数量。</param>
    private sealed record Storage(string Kind, string Key, long Count);
    /// <summary>校验持久引用是否为 Torch 存储，并保存存储描述符。</summary>
    /// <param name="id">Pickle 的持久化引用元组。</param>
    /// <returns>包含存储类型、ZIP 键和元素数量的描述符。</returns>
    /// <exception cref="NotSupportedException">持久引用不是受支持的 Torch storage 元组。</exception>
    private object Persistent(object?[] id)
    {
        if (id.Length < 5 || id[0] is not "storage" || id[1] is not PyGlobal g || g.Module != "torch") throw new NotSupportedException("Unsupported persistent pickle reference.");
        return new Storage(g.Name, (string)id[2]!, Convert.ToInt64(id[4]));
    }
    /// <summary>解释已知的数据重建符号；未知元数据类保持为惰性符号对象。</summary>
    /// <param name="g">REDUCE 操作对应的符号全局对象。</param>
    /// <param name="args">Pickle REDUCE 保存的调用参数，仅作为数据解释。</param>
    /// <returns>还原的字典、集合、张量或未执行的符号对象。</returns>
    private object? Reduce(PyGlobal g, object?[] args)
    {
        if (g is { Module: "collections", Name: "OrderedDict" })
        {
            var dict = new Dictionary<object, object?>();
            if (args.Length > 0 && args[0] is List<object?> pairs) foreach (var item in pairs) { var p = (object?[])item!; dict[p[0]!] = p[1]; }
            return dict;
        }
        if (g.Module is "__builtin__" or "builtins" && g.Name is "set" or "frozenset") return args.Length == 0 ? new List<object?>() : ((IEnumerable<object?>)args[0]!).ToList();
        if (g.Module == "torch" && g.Name == "Size") return args[0];
        if (g.Module == "torch._utils")
        {
            if (g.Name is "_rebuild_parameter" or "_rebuild_parameter_with_state") return args[0];
            if (g.Name is "_rebuild_tensor_v2" or "_rebuild_tensor") return TensorFromStorage(args);
        }
        // NumPy 标量、优化器类等元数据仅保存为符号对象，不执行其构造或还原代码。
        return new PyObject(g, args);
    }
    /// <summary>从缓存的 ZIP 存储按偏移和步长读取元素，并转为连续 FP32 张量。</summary>
    /// <param name="args">Torch 张量重建参数：存储、偏移、形状与步长。</param>
    /// <returns>拥有独立连续 FP32 数据且标记为常量的张量。</returns>
    /// <exception cref="InvalidDataException">存储布局、数量、大小、偏移不合法，或 ZIP 中缺少数据条目。</exception>
    /// <exception cref="NotSupportedException">Torch 存储元素类型不受支持。</exception>
    /// <exception cref="OverflowException">形状、地址或存储索引计算超出受支持整数范围。</exception>
    /// <exception cref="ArgumentException">还原张量的形状包含负维度。</exception>
    private Tensor TensorFromStorage(object?[] args)
    {
        var s = (Storage)args[0]!; long offset = Convert.ToInt64(args[1]); int[] shape = ((object?[])args[2]!).Select(Convert.ToInt32).ToArray(); long[] strides = ((object?[])args[3]!).Select(Convert.ToInt64).ToArray();
        if (shape.Length != strides.Length || strides.Any(v => v < 0) || offset < 0) throw new InvalidDataException("Invalid tensor storage layout.");
        int count = Tensor.Count(shape); if (count > 268435456) throw new InvalidDataException("Tensor too large.");
        if (!storage.TryGetValue(s.Key, out byte[]? data))
        {
            var entry = zip.GetEntry(prefix + "data/" + s.Key) ?? throw new InvalidDataException($"Storage {s.Key} missing.");
            if (entry.Length > 1_073_741_824) throw new InvalidDataException("Storage too large.");
            data = new byte[(int)entry.Length]; using var stream = entry.Open(); stream.ReadExactly(data); storage[s.Key] = data;
        }
        int elementSize = s.Kind switch { "FloatStorage" or "IntStorage" => 4, "HalfStorage" or "BFloat16Storage" => 2, "DoubleStorage" or "LongStorage" => 8, "ByteStorage" or "CharStorage" or "BoolStorage" => 1, _ => throw new NotSupportedException($"Storage {s.Kind} unsupported.") };
        if (s.Count < 0 || s.Count != data.Length / elementSize || data.Length % elementSize != 0) throw new InvalidDataException("Storage size mismatch.");
        var result = new Tensor(shape);
        for (int i = 0; i < count; i++)
        {
            long index = offset; int t = i;
            for (int d = shape.Length - 1; d >= 0; d--) { index = checked(index + t % shape[d] * strides[d]); t /= shape[d]; }
            if (index < 0 || index >= s.Count) throw new InvalidDataException("Tensor storage offset invalid."); var b = data.AsSpan(checked((int)index * elementSize), elementSize);
            result.Data[i] = s.Kind switch
            {
                "FloatStorage" => BinaryPrimitives.ReadSingleLittleEndian(b),
                "HalfStorage" => (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(b)),
                "BFloat16Storage" => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(b) << 16),
                "DoubleStorage" => (float)BinaryPrimitives.ReadDoubleLittleEndian(b),
                "LongStorage" => BinaryPrimitives.ReadInt64LittleEndian(b),
                "IntStorage" => BinaryPrimitives.ReadInt32LittleEndian(b),
                "CharStorage" => (sbyte)b[0],
                _ => b[0]
            };
        }
        result.IsConstant = true; return result;
    }
}
