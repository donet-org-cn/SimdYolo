using System.Collections.ObjectModel;
using System.Globalization;
using System.IO.Compression;

namespace SimdYolo;

/// <summary>从 PT、ONNX 或原生训练归档独立读取类别标签及推理元数据，不加载权重数据或创建计算后端。</summary>
public sealed class ModelInfo
{
    /// <summary>保存输入维度，防止调用者改变元数据。</summary>
    private readonly int[] inputShape;
    /// <summary>保存关键点维度，防止调用者改变元数据。</summary>
    private readonly int[] keypointShape;
    /// <summary>模型文件格式，值为 PT、ONNX 或 SYTRAIN。</summary>
    /// <value>从文件扩展名确定的模型格式。</value>
    public string Format { get; }
    /// <summary>模型原始类别编号到名称的只读映射；模型未保存名称时为空。</summary>
    /// <value>保留原始编号及 Unicode 名称的不可变映射。</value>
    public IReadOnlyDictionary<int, string> ClassNames { get; }
    /// <summary>模型的类别通道数量，不能用已提供标签的条目数量替代。</summary>
    /// <value>检测头 nc、元数据 nc 或名称最大编号加一；未知时为 0。</value>
    public int ClassCount { get; }
    /// <summary>模型是否提供至少一个类别名称。</summary>
    /// <value>类别名称映射是否非空。</value>
    public bool HasClassNames => ClassNames.Count != 0;
    /// <summary>模型的检测、分割或姿态任务。</summary>
    /// <value>从检测头或 ONNX task 元数据识别的任务。</value>
    public YoloTask Task { get; }
    /// <summary>模型是否直接输出经过 TopK 筛选的结果。</summary>
    /// <value>端到端模式；分割和姿态在六字段之后附加实例数据。</value>
    public bool EndToEnd { get; }
    /// <summary>图像输入维度的副本；-1 表示动态维度。</summary>
    /// <value>PT 使用 [1,3,-1,-1]，ONNX 使用图声明；SYTRAIN 使用保存时的固定训练尺寸。</value>
    public int[] InputShape => (int[])inputShape.Clone();
    /// <summary>关键点数量与分量数的副本；非姿态模型为空。</summary>
    /// <value>姿态模型通常使用 [17,3]。</value>
    public int[] KeypointShape => (int[])keypointShape.Clone();

    /// <summary>创建与模型加载器共用的不可变元数据快照。</summary>
    /// <param name="format">模型格式名称。</param>
    /// <param name="names">原始类别标签。</param>
    /// <param name="classes">类别通道数量。</param>
    /// <param name="task">任务类型。</param>
    /// <param name="endToEnd">是否使用端到端输出。</param>
    /// <param name="input">输入维度。</param>
    /// <param name="keypoints">关键点维度。</param>
    internal ModelInfo(string format, IReadOnlyDictionary<int, string> names, int classes, YoloTask task, bool endToEnd, int[] input, int[] keypoints)
    {
        // 标签集合很小且在加载时只构建一次；普通只读映射避免首次创建 Frozen 特化类型的 JIT 开销。
        Format = format; ClassNames = new ReadOnlyDictionary<int, string>(new Dictionary<int, string>(names)); ClassCount = classes; Task = task; EndToEnd = endToEnd;
        inputShape = (int[])input.Clone(); keypointShape = task == YoloTask.Pose ? (int[])keypoints.Clone() : [];
    }

    /// <summary>读取模型标签和元数据；PT 仅解释 Pickle 数据，ONNX 流式跳过权重和节点。</summary>
    /// <param name="path">扩展名为 .pt、.pth、.onnx 或 .sytrain 的模型路径。</param>
    /// <returns>包含原始类别名称的元数据快照；缺少名称时返回空映射，不预设 COCO 标签。</returns>
    /// <exception cref="ArgumentException">路径为空或包含无效字符。</exception>
    /// <exception cref="NotSupportedException">文件格式不是 PT/ONNX/SYTRAIN，或 PT 使用不支持的保存格式。</exception>
    /// <exception cref="InvalidDataException">模型元数据、标签编号或归档损坏。</exception>
    /// <exception cref="IOException">文件无法打开、读取或被截断。</exception>
    /// <exception cref="UnauthorizedAccessException">没有读取模型文件的权限。</exception>
    /// <exception cref="OverflowException">模型声明的维度或类别数量超出 Int32 范围。</exception>
    public static ModelInfo Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".pt" or ".pth" => ReadPt(path), ".onnx" => ReadOnnx(path), ".sytrain" => ReadTraining(path),
            _ => throw new NotSupportedException("模型元数据需要 .pt、.onnx 或 .sytrain 文件。")
        };
    }

    /// <summary>读取原生训练归档中的类别和任务，不读取参数或初始化设备。</summary>
    /// <param name="path">包含训练元数据的归档路径。</param>
    /// <returns>检测任务、训练时尺寸及保存类别的快照。</returns>
    /// <exception cref="InvalidDataException">归档或训练元数据损坏。</exception>
    private static ModelInfo ReadTraining(string path)
    {
        using var stream = File.OpenRead(path); using var zip = new ZipArchive(stream, ZipArchiveMode.Read);
        var manifest = TrainingModelState.ReadManifest(zip);
        return new("SYTRAIN", manifest.ClassNames.Select((name, index) => (name, index)).ToDictionary(p => p.index, p => p.name), manifest.ClassNames.Length, YoloTask.Detection, manifest.EndToEnd, [1, 3, manifest.ImageSize, manifest.ImageSize], [17, 3]);
    }

    /// <summary>读取 PT 的符号对象及标签，使用不读取张量存储的 Pickle 模式。</summary>
    /// <param name="path">Torch ZIP 检查点路径。</param>
    /// <returns>检查点提供的元数据。</returns>
    /// <exception cref="NotSupportedException">检查点没有 data.pkl。</exception>
    /// <exception cref="InvalidDataException">标签或元数据损坏、超限。</exception>
    private static ModelInfo ReadPt(string path) => ReadPt(path, out _);

    /// <summary>在一次元数据解析中同时取得不可变信息与符号检测头，供训练能力检查复用。</summary>
    /// <param name="path">完整 PT/PTH 检查点路径。</param>
    /// <param name="head">解析得到的最终符号模块；无完整模型结构时为 null。</param>
    /// <returns>模型的不可变标签、任务和尺寸元数据。</returns>
    /// <exception cref="InvalidDataException">归档或元数据损坏、超限。</exception>
    /// <exception cref="NotSupportedException">文件不是可解析的 Torch ZIP 检查点。</exception>
    /// <exception cref="IOException">文件无法读取或被截断。</exception>
    internal static ModelInfo ReadPt(string path, out PyObject? head)
    {
        using var stream = File.OpenRead(path); using var zip = new ZipArchive(stream, ZipArchiveMode.Read);
        var entry = zip.Entries.SingleOrDefault(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal)) ?? throw new NotSupportedException("PT metadata requires a torch.save ZIP checkpoint.");
        if (entry.Length > 64 * 1024 * 1024) throw new InvalidDataException("PT metadata exceeds 64 MiB.");
        string prefix = entry.FullName[..^"data.pkl".Length]; var bytes = new byte[(int)entry.Length]; using (var data = entry.Open()) data.ReadExactly(bytes);
        object? root = new PickleReader(zip, prefix, false).Load(bytes);
        var checkpoint = root as Dictionary<object, object?>; object? candidate = checkpoint is null ? root : checkpoint.GetValueOrDefault("ema") ?? checkpoint.GetValueOrDefault("model");
        var model = candidate as PyObject; head = model is { } && model.Has("model") ? model.Child("model").Children.LastOrDefault() : null;
        var names = ReadNames(model?.Get("names") ?? checkpoint?.GetValueOrDefault("names"));
        var task = head?.Type.Name switch { "Segment" or "Segment26" => YoloTask.Segmentation, "Pose" or "Pose26" => YoloTask.Pose, _ => YoloTask.Detection };
        int classes = head?.Int("nc", LabelClassCount(names)) ?? LabelClassCount(names);
        int[] points = head?.Get("kpt_shape") is IEnumerable<object?> shape ? shape.Select(Convert.ToInt32).ToArray() : [17, 3];
        return new("PT", names, classes, task, head is not null && IsEndToEnd(head, model), [1, 3, -1, -1], points);
    }

    /// <summary>流式读取 ONNX 元数据并推导输出布局。</summary>
    /// <param name="path">ONNX 路径。</param>
    /// <returns>图声明的元数据快照。</returns>
    private static ModelInfo ReadOnnx(string path)
    {
        var info = OnnxMetadataReader.Read(path); return FromOnnx(info.Metadata, info.Input, info.Outputs);
    }

    /// <summary>由已读出的 ONNX 字段构建元数据，与执行加载器使用相同的解析规则。</summary>
    /// <param name="metadata">ONNX 键值元数据。</param>
    /// <param name="input">图输入维度。</param>
    /// <param name="outputs">按图输出顺序排列的维度。</param>
    /// <returns>解析后的元数据。</returns>
    /// <exception cref="InvalidDataException">类别或关键点元数据不合法。</exception>
    internal static ModelInfo FromOnnx(Dictionary<string, string> metadata, int[] input, int[][] outputs)
    {
        var names = ReadNames(metadata.GetValueOrDefault("names"));
        var task = metadata.GetValueOrDefault("task") switch { "segment" => YoloTask.Segmentation, "pose" => YoloTask.Pose, _ => outputs.Length > 1 && outputs[1].Length == 4 ? YoloTask.Segmentation : YoloTask.Detection };
        int[] points = metadata.TryGetValue("kpt_shape", out string? text) && MetadataLiteral.Parse(text) is IEnumerable<object?> values ? values.Select(Convert.ToInt32).ToArray() : [17, 3];
        if (task == YoloTask.Pose && (points.Length != 2 || points[0] <= 0 || points[1] is not (2 or 3))) throw new InvalidDataException("Invalid keypoint metadata.");
        int classes = metadata.TryGetValue("nc", out string? nc) && int.TryParse(nc, NumberStyles.None, CultureInfo.InvariantCulture, out int count) ? count : LabelClassCount(names);
        int extra = task == YoloTask.Pose ? checked(points[0] * points[1]) : task == YoloTask.Segmentation && outputs.Length > 1 && outputs[1].Length == 4 ? outputs[1][1] : 0;
        bool? declared = metadata.TryGetValue("end2end", out string? mode) ? ParseBool(mode) : null;
        if (declared is null && metadata.TryGetValue("args", out string? args) && MetadataLiteral.Parse(args) is Dictionary<object, object?> options && options.GetValueOrDefault("end2end") is bool end2end) declared = end2end;
        int[] head = outputs.FirstOrDefault() ?? [];
        bool end = declared ?? (classes != 2 && head.Length == 3 && head[2] == 6 + extra && head[1] != 4 + classes + extra);
        return new("ONNX", names, classes, task, end, input, points);
    }

    /// <summary>解析显式的端到端布尔元数据。</summary>
    /// <param name="text">布尔值文本。</param>
    /// <returns>显式声明的布尔值。</returns>
    /// <exception cref="InvalidDataException">文本不是 true、false、0 或 1。</exception>
    private static bool ParseBool(string text) => text.Trim().ToLowerInvariant() switch { "true" or "1" => true, "false" or "0" => false, _ => throw new InvalidDataException("Invalid end2end metadata.") };

    /// <summary>依据检测头状态、融合分支与模型 YAML 识别新旧 PT 的端到端模式。</summary>
    /// <param name="head">最终检测、分割或姿态头。</param>
    /// <param name="model">可选根模型，用于读取老版 YOLO26 保存在 YAML 中的标记。</param>
    /// <returns>是否使用 one2one 分支与端到端筛选。</returns>
    internal static bool IsEndToEnd(PyObject head, PyObject? model = null)
    {
        if (head.Type.Name == "v10Detect") return true;
        if (!head.Has("one2one_cv2")) return false;
        if (!head.Has("cv2")) return true;
        if (head.Get("_end2end") is { } mode) return Convert.ToBoolean(mode);
        if (head.Get("end2end") is { } oldMode) return Convert.ToBoolean(oldMode);
        return model?.Get("yaml") is Dictionary<object, object?> yaml && yaml.GetValueOrDefault("end2end") is { } configured && Convert.ToBoolean(configured);
    }

    /// <summary>解析字典或列表形式的标签，并冻结映射以防止外部修改。</summary>
    /// <param name="value">PT names 对象，或 ONNX 的 JSON/Python 字面量文本。</param>
    /// <returns>保留原始编号的标签；缺失时为空映射。</returns>
    /// <exception cref="InvalidDataException">标签形式、编号、重复编号或名称类型不合法。</exception>
    internal static IReadOnlyDictionary<int, string> ReadNames(object? value)
    {
        if (value is string text) value = string.IsNullOrWhiteSpace(text) ? null : MetadataLiteral.Parse(text);
        var labels = new Dictionary<int, string>();
        if (value is Dictionary<object, object?> map)
        {
            foreach (var pair in map)
            {
                string key = Convert.ToString(pair.Key, CultureInfo.InvariantCulture) ?? "";
                if (!int.TryParse(key, NumberStyles.None, CultureInfo.InvariantCulture, out int id) || id == int.MaxValue || pair.Value is not string name || !labels.TryAdd(id, name)) throw new InvalidDataException("Invalid or duplicate class label.");
            }
        }
        else if (value is IEnumerable<object?> list)
        {
            foreach (object? item in list)
            { if (item is not string name) throw new InvalidDataException("Class names must be strings."); labels.Add(labels.Count, name); }
        }
        else if (value is not null) throw new InvalidDataException("Class names require a dictionary or list.");
        return new ReadOnlyDictionary<int, string>(labels);
    }

    /// <summary>从标签最大编号推导通道数量，保留稀疏编号的含义。</summary>
    /// <param name="names">原始编号的标签映射。</param>
    /// <returns>最大编号加一；没有标签时为 0。</returns>
    internal static int LabelClassCount(IReadOnlyDictionary<int, string> names) => names.Count == 0 ? 0 : checked(names.Keys.Max() + 1);
}
