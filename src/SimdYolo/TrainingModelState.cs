using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace SimdYolo;

/// <summary>描述纯 C# 检测头微调归档；该格式不冒充 PyTorch 检查点。</summary>
internal sealed class TrainingManifest
{
    public string Format { get; set; } = "SimdYolo.Training/1";
    public string[] ClassNames { get; set; } = [];
    public string SourceSha256 { get; set; } = "";
    public List<TrainingTensorEntry> Parameters { get; set; } = [];
    public int Epoch { get; set; }
    public int Step { get; set; }
    public int Seed { get; set; }
    public int ImageSize { get; set; }
    public int BatchSize { get; set; } = 1;
    public float LearningRate { get; set; }
    public float Momentum { get; set; }
    public float WeightDecay { get; set; }
    public string DatasetSha256 { get; set; } = "";
    /// <summary>保存所用检测头损失策略，恢复时与原始 PT 核对。</summary><value>旧版归档未声明时为空。</value>
    public string? HeadKind { get; set; }
    /// <summary>记录每条边的原始回归通道数。</summary><value>传统 DFL 为十六，YOLO26 为一。</value>
    public int RegMax { get; set; } = 16;
    /// <summary>记录推理是否使用一对一分支及 TopK 输出。</summary><value>传统单头为假，双头端到端模型为真。</value>
    public bool EndToEnd { get; set; }
    /// <summary>首次创建时固定的渐进损失调度长度，恢复延长训练时保持不变。</summary><value>旧版无调度时为零。</value>
    public int LossScheduleEpochs { get; set; }
}

/// <summary>记录检测头参数名称、形状和归档内索引。</summary>
/// <param name="Name">相对检测头的完整参数或缓冲区路径。</param>
/// <param name="Shape">FP32 张量维度，空数组表示标量。</param>
internal sealed record TrainingTensorEntry(string Name, int[] Shape);

/// <summary>读取自包含的训练快照，在推理预融合之前恢复检测头权重。</summary>
internal sealed class TrainingModelState
{
    internal TrainingManifest Manifest { get; }
    internal byte[] Source { get; }
    internal Dictionary<string, Tensor> Parameters { get; }
    internal TrainingModelState(TrainingManifest manifest, byte[] source, Dictionary<string, Tensor> parameters)
    { Manifest = manifest; Source = source; Parameters = parameters; }

    /// <summary>仅读取小型元数据，不读取骨干或参数存储。</summary>
    /// <param name="zip">调用方持有的训练 ZIP 归档。</param>
    /// <returns>通过格式、类别和大小校验的元数据。</returns>
    /// <exception cref="InvalidDataException">存在重复项或元数据损坏、超限。</exception>
    internal static TrainingManifest ReadManifest(ZipArchive zip)
    {
        if (zip.Entries.GroupBy(e => e.FullName).Any(g => g.Count() != 1)) throw new InvalidDataException("训练归档存在重复记录。");
        var entry = zip.GetEntry("manifest.json") ?? throw new InvalidDataException("缺少训练元数据。");
        if (entry.Length > 2 * 1024 * 1024) throw new InvalidDataException("训练元数据过大。");
        using var data = entry.Open();
        var manifest = JsonSerializer.Deserialize<TrainingManifest>(data) ?? throw new InvalidDataException("训练元数据为空。");
        if (manifest.Format is not ("SimdYolo.Training/1" or "SimdYolo.Training/2") || manifest.ClassNames is null || manifest.ClassNames.Length is < 1 or > 1000 || manifest.ClassNames.Any(string.IsNullOrWhiteSpace) || manifest.ClassNames.Distinct(StringComparer.Ordinal).Count() != manifest.ClassNames.Length || manifest.Parameters is null || manifest.Parameters.Count is < 1 or > 1024 || manifest.ImageSize is < 64 or > 1280 || manifest.ImageSize % 32 != 0 || manifest.Epoch < 0 || manifest.Step < 0)
            throw new InvalidDataException("训练归档格式、类别或参数清单无效。");
        if (manifest.Format == "SimdYolo.Training/2" && (manifest.LossScheduleEpochs is < 1 or > 10000 || manifest.HeadKind switch { "Dfl" => manifest.EndToEnd || manifest.RegMax != 16, "LegacyEndToEnd" => !manifest.EndToEnd || manifest.RegMax != 16, "ProgressiveEndToEnd" => !manifest.EndToEnd || manifest.RegMax != 1, _ => true }))
            throw new InvalidDataException("训练归档的头类型、回归通道或损失调度无效。");
        return manifest;
    }

    /// <summary>加载并校验原 PT 字节及每项 FP32 头参数；不在磁盘解压文件。</summary>
    /// <param name="path">自包含训练归档路径。</param>
    /// <returns>经过摘要、参数清单、形状和有限性校验的状态。</returns>
    /// <exception cref="InvalidDataException">参数缺失、摘要错误或归档超限。</exception>
    internal static TrainingModelState Load(string path)
    {
        using var file = File.OpenRead(path); using var zip = new ZipArchive(file, ZipArchiveMode.Read);
        var manifest = ReadManifest(zip);
        byte[] source = ReadEntry(zip, "source.pt", 256 * 1024 * 1024);
        if (Convert.ToHexStringLower(SHA256.HashData(source)) != manifest.SourceSha256) throw new InvalidDataException("骨干 PT 摘要不匹配。");
        var parameters = new Dictionary<string, Tensor>(StringComparer.Ordinal); long total = source.Length;
        for (int i = 0; i < manifest.Parameters.Count; i++)
        {
            var item = manifest.Parameters[i];
            if (item is null || item.Shape is null || item.Shape.Length > 4 || item.Shape.Any(d => d < 1) || item.Name is null || item.Name.Length > 160 || !(item.Name.StartsWith("cv", StringComparison.Ordinal) || item.Name.StartsWith("one2one_cv", StringComparison.Ordinal))) throw new InvalidDataException("头参数名称或形状无效。");
            int count = Tensor.Count(item.Shape); if (count > 20_000_000) throw new InvalidDataException("头参数过大。");
            byte[] bytes = ReadEntry(zip, $"parameters/{i}.f32", checked(count * 4));
            total += bytes.Length; if (total > 384 * 1024 * 1024 || bytes.Length != (long)count * 4) throw new InvalidDataException("参数长度或归档总量超限。");
            float[] values = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
            if (values.Any(v => !float.IsFinite(v)) || !parameters.TryAdd(item.Name, new Tensor(values, item.Shape))) throw new InvalidDataException("参数非有限或名称重复。");
        }
        return new(manifest, source, parameters);
    }

    /// <summary>按符号路径替换原始检测头参数，并同步自定义类别输出通道。</summary>
    /// <param name="head">尚未执行推理 BN 融合的原始检测头。</param>
    /// <exception cref="NotSupportedException">检测头不属于受支持的三尺度单头或双头检测布局。</exception>
    /// <exception cref="InvalidDataException">参数名称或维度与原头不一致。</exception>
    internal void Apply(PyObject head)
    {
        var schema = TrainingHeadSchema.Read(head);
        if (Manifest.Format == "SimdYolo.Training/1" && schema.EndToEnd || Manifest.Format == "SimdYolo.Training/2" && (schema.Kind != Manifest.HeadKind || schema.RegMax != Manifest.RegMax || schema.EndToEnd != Manifest.EndToEnd)) throw new InvalidDataException("训练元数据与原始检测头不一致。");
        var expected = new HashSet<string>(StringComparer.Ordinal);
        void Collect(PyObject module, string prefix)
        {
            foreach (string map in new[] { "_parameters", "_buffers" }) foreach (var pair in module.Map(map))
                if (pair.Value is Tensor) expected.Add(prefix + "." + pair.Key);
            foreach (var pair in module.Map("_modules")) if (pair.Value is PyObject child) Collect(child, prefix + "." + pair.Key);
        }
        foreach (string prefix in schema.Prefixes) Collect(head.Child(prefix), prefix);
        if (!expected.SetEquals(Parameters.Keys)) throw new InvalidDataException("训练归档必须包含完整检测头参数和 BN 统计。");
        foreach (var pair in Parameters)
        {
            string[] parts = pair.Key.Split('.'); PyObject current = head;
            for (int i = 0; i < parts.Length - 1; i++) current = current.Child(parts[i]);
            string name = parts[^1]; var map = current.Map("_parameters");
            if (!map.ContainsKey(name)) map = current.Map("_buffers");
            if (!map.ContainsKey(name)) throw new InvalidDataException("检测头参数路径不存在：" + pair.Key);
            var previous = current.Weight(name);
            bool finalClass = parts.Length == 4 && parts[0] is "cv3" or "one2one_cv3" && parts[2] == "2";
            if (!previous.Shape.SequenceEqual(pair.Value.Shape) && !(finalClass && pair.Value.Shape[0] == Manifest.ClassNames.Length && previous.Shape.AsSpan(1).SequenceEqual(pair.Value.Shape.AsSpan(1))))
                throw new InvalidDataException("检测头参数形状与骨干不兼容：" + pair.Key);
            map[name] = pair.Value;
        }
        head.State["nc"] = Manifest.ClassNames.Length; head.State["no"] = Manifest.ClassNames.Length + schema.RegMax * 4;
        foreach (string prefix in schema.Prefixes.Where(p => p.EndsWith("cv3", StringComparison.Ordinal)))
            foreach (var branch in head.Child(prefix).Children) branch.Children[^1].State["out_channels"] = Manifest.ClassNames.Length;
    }

    /// <summary>以长度上限读取单个 ZIP 项，防止损坏归档导致无界分配。</summary>
    /// <param name="zip">已打开的训练归档。</param>
    /// <param name="name">归档中的完整项名称。</param>
    /// <param name="limit">允许解压的最大字节数。</param>
    /// <returns>实际项内容的独立字节数组。</returns>
    /// <exception cref="InvalidDataException">项缺失或超过限制。</exception>
    /// <exception cref="EndOfStreamException">实际数据短于声明长度。</exception>
    internal static byte[] ReadEntry(ZipArchive zip, string name, int limit)
    {
        var entry = zip.GetEntry(name) ?? throw new InvalidDataException("训练归档缺少 " + name);
        if (entry.Length < 0 || entry.Length > limit) throw new InvalidDataException("归档项过大：" + name);
        byte[] bytes = new byte[(int)entry.Length]; using var stream = entry.Open(); stream.ReadExactly(bytes); return bytes;
    }
}
