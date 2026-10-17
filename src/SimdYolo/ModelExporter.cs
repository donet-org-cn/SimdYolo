using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SimdYolo;

/// <summary>纯.NET模型导出；生成标准ONNX算子与嵌入权重，不需要Python或protobuf包。</summary>
public static class ModelExporter
{
    /// <summary>将完整PT或实际训练快照导出为静态batch=1、opset17的ONNX。</summary>
    /// <param name="modelPath">PT/PTH/SYTRAIN路径。</param><param name="outputPath">目标ONNX路径。</param>
    /// <param name="imageSize">普通PT导出的正方形尺寸；SYTRAIN使用其保存尺寸。</param>
    /// <returns>实际输入尺寸、输出形状、节点数量和文件摘要。</returns>
    /// <exception cref="ArgumentException">尺寸或输出扩展名非法。</exception><exception cref="NotSupportedException">源模型或运算无法标准化导出。</exception>
    /// <exception cref="InvalidDataException">训练快照损坏。</exception><exception cref="IOException">文件读取或原子写入失败。</exception>
    public static OnnxExportResult ExportOnnx(string modelPath, string outputPath, int imageSize = 640)
    {
        if (imageSize is < 64 or > 1280 || imageSize % 32 != 0 || !Path.GetExtension(outputPath).Equals(".onnx", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("导出需要.onnx路径，尺寸64到1280且为32倍数。");
        if (Path.GetFullPath(modelPath).Equals(Path.GetFullPath(outputPath), StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("不能覆盖源模型。");
        using var trace = new OnnxTraceBackend(); var ops = new TensorOps(trace);
        PtModel model;
        if (Path.GetExtension(modelPath).Equals(".sytrain", StringComparison.OrdinalIgnoreCase))
        {
            var state = TrainingModelState.Load(modelPath); model = new(new MemoryStream(state.Source, false), ops, null, state);
        }
        else if (Path.GetExtension(modelPath).ToLowerInvariant() is ".pt" or ".pth") model = new(modelPath, ops);
        else throw new NotSupportedException("导出入口需要完整PT或SYTRAIN。");
        int size = model.InputShape[2] > 0 ? model.InputShape[2] : imageSize; Tensor[] outputs;
        var previous = DeviceScope.Current; var previousPool = CpuTensorPool.Current;
        try { DeviceScope.Current = trace; CpuTensorPool.Current = null; outputs = model.Run(trace.Input([1, 3, size, size])); }
        finally { DeviceScope.Current = previous; CpuTensorPool.Current = previousPool; }
        byte[] bytes = trace.Serialize(outputs, model, modelPath);
        string full = Path.GetFullPath(outputPath); Directory.CreateDirectory(Path.GetDirectoryName(full)!); string temporary = full + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try { File.WriteAllBytes(temporary, bytes); File.Move(temporary, full, true); }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
        return new(full, [1, 3, size, size], outputs.Select(t => t.Shape.ToArray()).ToArray(), trace.NodeCount, Convert.ToHexStringLower(SHA256.HashData(bytes)));
    }
}

/// <summary>一次标准ONNX导出的可验证结果。</summary>
/// <param name="Path">输出路径。</param><param name="InputShape">实际静态输入形状。</param><param name="OutputShapes">输出形状。</param>
/// <param name="Nodes">标准算子数量。</param><param name="Sha256">文件摘要。</param>
public sealed record OnnxExportResult(string Path, int[] InputShape, int[][] OutputShapes, int Nodes, string Sha256);

/// <summary>符号设备后端，仅传递形状与图边；禁止下载，防止把依赖输入的数据烘焙成常量。</summary>
internal sealed class OnnxTraceBackend : IComputeBackend, IDeviceBackend
{
    private sealed record Value(string Name, int[] Shape);
    private readonly List<byte[]> nodes = [], constants = [];
    private readonly Dictionary<Tensor, string> names = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<string, string> views = [];
    private int serial;
    private int[] inputShape = [];
    public string Name => "ONNX symbolic exporter";
    public bool FrameActive => true;
    public long FrameId => 1;
    internal int NodeCount => nodes.Count;
    public void BeginFrame() { }
    public void EndFrame() { }
    public void Flush() { }
    public void Dispose() { }
    public void MatMul(float[] a, float[] b, float[] c, int m, int n, int k) => throw new NotSupportedException("导出遇到直接访问数据的矩阵路径。");
    public float[] Download(DeviceStorage storage) => throw new NotSupportedException("该PT运算直接读取依赖输入的主机数据，不能可靠导出。");
    internal Tensor Input(int[] shape) { inputShape = shape; return Wrap("images", shape); }
    private Tensor Wrap(string name, int[] shape) => new(new DeviceStorage(this, new Value(name, shape.ToArray()), 0, Tensor.Count(shape), 1), shape);
    private string New() => "v" + ++serial;
    private string NameOf(Tensor tensor)
    {
        if (tensor.Storage is { Owner: OnnxTraceBackend, Buffer: Value value })
        {
            if (value.Shape.SequenceEqual(tensor.Shape)) return value.Name;
            string key = value.Name + ":" + string.Join(',', tensor.Shape);
            if (!views.TryGetValue(key, out string? name)) views[key] = name = Reshape(value.Name, tensor.Shape);
            return name;
        }
        if (!names.TryGetValue(tensor, out string? constant))
        {
            constant = New(); constants.Add(TensorProto(constant, tensor.Shape, 1, MemoryMarshal.AsBytes(tensor.Data.AsSpan()).ToArray())); names[tensor] = constant;
        }
        return constant;
    }
    private string Float(float value) => Constant([value], []);
    private string Constant(float[] data, int[] shape) { string name = New(); constants.Add(TensorProto(name, shape, 1, MemoryMarshal.AsBytes(data.AsSpan()).ToArray())); return name; }
    private string Ints(params int[] data) { string name = New(); long[] values = data.Select(x => (long)x).ToArray(); constants.Add(TensorProto(name, [data.Length], 7, MemoryMarshal.AsBytes(values.AsSpan()).ToArray())); return name; }
    private string Node(string op, string[] inputs, params (string Key, object Value)[] attributes) => NodeMany(op, inputs, 1, attributes)[0];
    private string[] NodeMany(string op, string[] inputs, int outputs, params (string Key, object Value)[] attributes)
    {
        string[] names = Enumerable.Range(0, outputs).Select(_ => New()).ToArray(); var node = new Wire();
        foreach (string input in inputs) node.Text(1, input); foreach (string output in names) node.Text(2, output);
        node.Text(3, names[0]); node.Text(4, op);
        foreach (var (key, value) in attributes)
        {
            var attr = new Wire(); attr.Text(1, key);
            switch (value)
            {
                case int number: attr.Int(3, number); attr.Int(20, 2); break;
                case int[] array: foreach (int number in array) attr.Int(8, number); attr.Int(20, 7); break;
                case float number: attr.Float(2, number); attr.Int(20, 1); break;
                case string text: attr.Text(4, text); attr.Int(20, 3); break;
                default: throw new NotSupportedException("未知ONNX属性类型。");
            }
            node.Bytes(5, attr.Data());
        }
        nodes.Add(node.Data()); return names;
    }
    private string Reshape(string x, params int[] shape) => Node("Reshape", [x, Ints(shape)]);
    private string Slice(string x, int axis, int start, int end) => Node("Slice", [x, Ints(start), Ints(end), Ints(axis)]);
    private string Binary(string op, string a, string b) => Node(op, [a, b]);
    private string Cat(int axis, params string[] values) => Node("Concat", values, ("axis", axis));
    private string Cast(string x, int type) => Node("Cast", [x], ("to", type));

    public Tensor DeviceConv(Tensor x, Tensor w, Tensor? bias, int[] shape, int[] stride, int[] pads, int[] dilation, int groups, bool silu)
    {
        string value = Node("Conv", bias is null ? [NameOf(x), NameOf(w)] : [NameOf(x), NameOf(w), NameOf(bias)], ("strides", stride), ("pads", pads), ("dilations", dilation), ("group", groups));
        if (silu) value = Binary("Mul", value, Node("Sigmoid", [value])); return Wrap(value, shape);
    }
    public Tensor DeviceMatMul(Tensor a, Tensor b, int[] shape) => Wrap(Node("MatMul", [NameOf(a), NameOf(b)]), shape);
    public Tensor DeviceConcat(IReadOnlyList<Tensor> tensors, int axis, int[] shape) => Wrap(Cat(axis, tensors.Select(NameOf).ToArray()), shape);
    public Tensor DeviceSlice(Tensor x, int[] shape, int axis, int start) => Wrap(Slice(NameOf(x), axis, start, start + shape[axis]), shape);
    internal Tensor Transpose(Tensor x, int[] perm, int[] shape) => Wrap(Node("Transpose", [NameOf(x)], ("perm", perm)), shape);

    public Tensor Compute(int operation, Tensor[] inputs, int[] shape, uint[] p, int workItems)
    {
        string x = NameOf(inputs[0]), value;
        switch (operation)
        {
            case 1: value = Binary("Mul", x, Node("Sigmoid", [x])); break;
            case 2: value = Node("Sigmoid", [x]); break;
            case 3: value = Node("Relu", [x]); break;
            case 4: value = Binary("Mul", x, Float(BitConverter.UInt32BitsToSingle(p[1]))); break;
            case 5: value = Node("Exp", [x]); break;
            case 6: value = Node("Sqrt", [x]); break;
            case 7: value = Node("Neg", [x]); break;
            case 8: value = Node("Abs", [x]); break;
            case >= 10 and <= 13: value = Binary(new[] { "Add", "Sub", "Mul", "Div" }[operation - 10], x, NameOf(inputs[1])); break;
            case 20: value = Node("Resize", [x, "", "", Ints(shape)], ("mode", "nearest"), ("coordinate_transformation_mode", "asymmetric"), ("nearest_mode", "floor")); break;
            case 21: case 22:
                value = Node(operation == 21 ? "MaxPool" : "AveragePool", [x], ("kernel_shape", new[] { (int)p[5], (int)p[6] }), ("strides", new[] { (int)p[7], (int)p[8] }), ("pads", new[] { (int)p[9], (int)p[10], (int)p[9], (int)p[10] }),
                    operation == 22 ? ("count_include_pad", (object)(int)p[11]) : ("ceil_mode", (object)0)); break;
            case 25:
                int inner = (int)p[2], axis = inputs[0].Rank - 1;
                while (axis > 0 && Tensor.Count(inputs[0].Shape[(axis + 1)..]) != inner) axis--;
                value = Node("Softmax", [x], ("axis", axis)); break;
            case 30: value = DecodeHead(inputs, p); break;
            case 31: value = DecodePose(inputs[0], p); break;
            case 33:
                value = Node("ConvTranspose", inputs.Select(NameOf).ToArray(), ("strides", new[] { (int)p[9], (int)p[10] }), ("pads", new[] { (int)p[11], (int)p[12], (int)p[11], (int)p[12] }), ("dilations", new[] { (int)p[13], (int)p[14] }), ("group", (int)p[15])); break;
            default: throw new NotSupportedException($"导出暂不识别设备算子{operation}，拒绝生成不完整模型。");
        }
        return Wrap(value, shape);
    }

    private string DecodeHead(Tensor[] inputs, uint[] p)
    {
        int spatial = (int)p[1], width = (int)p[2], bins = (int)p[3], nc = (int)p[4]; float stride = BitConverter.UInt32BitsToSingle(p[5]);
        string distances = NameOf(inputs[0]);
        if (bins > 1)
        {
            distances = Node("Softmax", [Reshape(distances, 1, 4, bins, spatial)], ("axis", 2));
            distances = Binary("Mul", distances, Constant(Enumerable.Range(0, bins).Select(r => (float)r).ToArray(), [1, 1, bins, 1]));
            distances = Node("ReduceSum", [distances, Ints(2)], ("keepdims", 0));
        }
        else distances = Reshape(distances, 1, 4, spatial);
        distances = Binary("Mul", distances, Float(stride));
        string anchorsX = Constant(Enumerable.Range(0, spatial).Select(i => (i % width + .5f) * stride).ToArray(), [1, 1, spatial]);
        string anchorsY = Constant(Enumerable.Range(0, spatial).Select(i => (i / width + .5f) * stride).ToArray(), [1, 1, spatial]);
        string left = Binary("Sub", anchorsX, Slice(distances, 1, 0, 1)), top = Binary("Sub", anchorsY, Slice(distances, 1, 1, 2));
        string right = Binary("Add", anchorsX, Slice(distances, 1, 2, 3)), bottom = Binary("Add", anchorsY, Slice(distances, 1, 3, 4));
        string coordinates = p[6] != 0 ? Cat(1, left, top, right, bottom) : Cat(1, Binary("Mul", Binary("Add", left, right), Float(.5f)), Binary("Mul", Binary("Add", top, bottom), Float(.5f)), Binary("Sub", right, left), Binary("Sub", bottom, top));
        // 显式1/(1+exp(-x))保留低分区精度，避免部分运行时Sigmoid内核的相消舍入改变TopK边界。
        string logits = Reshape(NameOf(inputs[1]), 1, nc, spatial);
        string probabilities = Binary("Div", Float(1), Binary("Add", Float(1), Node("Exp", [Node("Neg", [logits])])));
        return Cat(1, coordinates, probabilities);
    }
    private string DecodePose(Tensor input, uint[] p)
    {
        int spatial = (int)p[1], width = (int)p[2], dimensions = (int)p[3], count = input.Shape[1] / dimensions; float stride = BitConverter.UInt32BitsToSingle(p[4]); bool newer = p[5] != 0;
        string raw = Reshape(NameOf(input), 1, count, dimensions, spatial);
        string x = Slice(raw, 2, 0, 1), y = Slice(raw, 2, 1, 2);
        if (!newer) { x = Binary("Mul", x, Float(2)); y = Binary("Mul", y, Float(2)); }
        x = Binary("Mul", Binary("Add", x, Constant(Enumerable.Range(0, spatial).Select(i => i % width + (newer ? .5f : 0)).ToArray(), [1, 1, 1, spatial])), Float(stride));
        y = Binary("Mul", Binary("Add", y, Constant(Enumerable.Range(0, spatial).Select(i => i / width + (newer ? .5f : 0)).ToArray(), [1, 1, 1, spatial])), Float(stride));
        return Reshape(dimensions == 3 ? Cat(2, x, y, Node("Sigmoid", [Slice(raw, 2, 2, 3)])) : Cat(2, x, y), 1, count * dimensions, spatial);
    }

    internal Tensor EndToEnd(Tensor input, int maximum, int classes)
    {
        if (input.Shape[0] != 1) throw new NotSupportedException("ONNX导出当前使用静态batch=1。");
        int anchors = input.Shape[2], k = Math.Min(maximum, anchors), extra = input.Shape[1] - 4 - classes;
        string rows = Node("Transpose", [NameOf(input)], ("perm", new[] { 0, 2, 1 })); rows = Reshape(rows, anchors, 4 + classes + extra);
        string scores = Slice(rows, 1, 4, 4 + classes);
        string maxima = Node("ReduceMax", [scores], ("axes", new[] { 1 }), ("keepdims", 0));
        string[] anchorTop = NodeMany("TopK", [maxima, Ints(k)], 2, ("axis", 0), ("largest", 1), ("sorted", 1));
        // 原锚点升序作为相同分数时的稳定次序，与PT执行器的anchor/class规则一致。
        string[] orderedAnchors = NodeMany("TopK", [anchorTop[1], Ints(k)], 2, ("axis", 0), ("largest", 0), ("sorted", 1));
        string selected = Node("Gather", [rows, orderedAnchors[0]], ("axis", 0));
        string selectedScores = Reshape(Slice(selected, 1, 4, 4 + classes), k * classes);
        string[] classTop = NodeMany("TopK", [selectedScores, Ints(k)], 2, ("axis", 0), ("largest", 1), ("sorted", 1));
        string index = Cast(classTop[1], 1), selectedIndex = Cast(Binary("Div", index, Float(classes)), 7);
        string classIndex = Binary("Sub", index, Binary("Mul", Cast(selectedIndex, 1), Float(classes)));
        string best = Node("Gather", [selected, selectedIndex], ("axis", 0));
        var fields = new List<string> { Slice(best, 1, 0, 4), Reshape(classTop[0], k, 1), Reshape(classIndex, k, 1) };
        if (extra > 0) fields.Add(Slice(best, 1, 4 + classes, 4 + classes + extra));
        return Wrap(Reshape(Cat(1, fields.ToArray()), 1, k, 6 + extra), [1, k, 6 + extra]);
    }

    internal byte[] Serialize(Tensor[] outputs, IModel model, string sourcePath)
    {
        string[] outputNames = outputs.Select(NameOf).ToArray(); var graph = new Wire();
        foreach (byte[] node in nodes) graph.Bytes(1, node); graph.Text(2, "SimdYolo"); foreach (byte[] constant in constants) graph.Bytes(5, constant);
        graph.Bytes(11, ValueInfo("images", inputShape)); for (int i = 0; i < outputs.Length; i++) graph.Bytes(12, ValueInfo(outputNames[i], outputs[i].Shape));
        var result = new Wire(); result.Int(1, 9); result.Text(2, "SimdYolo.NET"); result.Text(3, typeof(ModelExporter).Assembly.GetName().Version!.ToString(3)); result.Bytes(7, graph.Data()); var opset = new Wire(); opset.Int(2, 17); result.Bytes(8, opset.Data());
        var metadata = new Dictionary<string, string> { ["names"] = JsonSerializer.Serialize(model.Names), ["task"] = model.Task switch { YoloTask.Segmentation => "segment", YoloTask.Pose => "pose", _ => "detect" },
            ["imgsz"] = JsonSerializer.Serialize(inputShape[2..]), ["end2end"] = model.EndToEnd ? "True" : "False", ["simdyolo.source_sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(sourcePath))), ["simdyolo.export"] = "pure-dotnet/opset17/static-batch1" };
        if (model.Task == YoloTask.Pose) metadata["kpt_shape"] = JsonSerializer.Serialize(model.KeypointShape);
        foreach (var pair in metadata) { var entry = new Wire(); entry.Text(1, pair.Key); entry.Text(2, pair.Value); result.Bytes(14, entry.Data()); }
        return result.Data();
    }
    private static byte[] TensorProto(string name, int[] shape, int type, byte[] raw) { var p = new Wire(); foreach (int d in shape) p.Int(1, d); p.Int(2, type); p.Text(8, name); p.Bytes(9, raw); return p.Data(); }
    private static byte[] ValueInfo(string name, int[] shape)
    {
        var dimensions = new Wire(); foreach (int d in shape) { var dimension = new Wire(); dimension.Int(1, d); dimensions.Bytes(1, dimension.Data()); }
        var tensor = new Wire(); tensor.Int(1, 1); tensor.Bytes(2, dimensions.Data()); var type = new Wire(); type.Bytes(1, tensor.Data()); var info = new Wire(); info.Text(1, name); info.Bytes(2, type.Data()); return info.Data();
    }
    private sealed class Wire
    {
        private readonly MemoryStream stream = new();
        private void Var(ulong value) { while (value >= 128) { stream.WriteByte((byte)(value | 128)); value >>= 7; } stream.WriteByte((byte)value); }
        internal void Int(int field, long value) { Var((ulong)field << 3); Var(unchecked((ulong)value)); }
        internal void Bytes(int field, byte[] value) { Var(((ulong)field << 3) | 2); Var((ulong)value.Length); stream.Write(value); }
        internal void Text(int field, string value) => Bytes(field, Encoding.UTF8.GetBytes(value));
        internal void Float(int field, float value) { Var(((ulong)field << 3) | 5); stream.Write(BitConverter.GetBytes(value)); }
        internal byte[] Data() => stream.ToArray();
    }
}
