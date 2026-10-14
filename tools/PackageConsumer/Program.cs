using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;

// 只通过 NuGet 包加载主库，指纹必须等于正式计时程序集。
var assembly = typeof(YoloDetector).Assembly;
string dllHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(assembly.Location)));
if (dllHash != "74a09c65fd9b535746ca1e6fc56e62cedac6ef0c27d3f38421f7c98ab221ab91")
    throw new InvalidDataException("消费程序加载了不同的主库。");
var labels = new List<object>();
using var manifest = JsonDocument.Parse(File.ReadAllText("models/manifest.json"));
foreach (var row in manifest.RootElement.GetProperty("Models").EnumerateArray())
{
    string path = row.GetProperty("File").GetString()!;
    var info = ModelInfo.Read(path);
    var expected = ModelInfo.Read(row.GetProperty("SourceWeight").GetString()!);
    if (info.ClassCount != expected.ClassCount || info.ClassNames.Count != expected.ClassNames.Count ||
        info.ClassNames.Any(p => !expected.ClassNames.TryGetValue(p.Key, out var name) || name != p.Value))
        throw new InvalidDataException("标签与源权重不一致：" + path);
    labels.Add(new { Model = path, info.ClassCount, LabelCount = info.ClassNames.Count, Passed = true });
}

// 烟雾验证覆盖三种任务、两种格式及两个后端，检查调用方持有的输出不会被复用覆盖。
var configurations = new List<object>();
foreach (var device in new[] { ComputeDevice.Cpu, ComputeDevice.Vulkan })
foreach (string name in new[] { "yolo26n", "yolo26n-seg", "yolo26n-pose" })
foreach (string format in new[] { "pt", "onnx" })
{
    string path = format == "pt" ? $"models/pt/{name}.pt" : $"models/onnx/{name}-dynamic.onnx";
    using var detector = new YoloDetector(path, new() { Device = device, CpuThreads = 12 });
    var input = new Tensor(1, 3, 128, 128);
    var random = new Random(263);
    Tensor[]? previous = null;
    byte[][]? snapshots = null;
    object[]? shapes = null;
    for (int frame = 0; frame < 3; frame++)
    {
        for (int i = 0; i < input.Length; i++) input.Data[i] = (float)random.NextDouble();
        var outputs = detector.PredictRaw(input);
        if (previous is not null)
            for (int i = 0; i < previous.Length; i++)
                if (!MemoryMarshal.AsBytes(previous[i].Data.AsSpan()).SequenceEqual(snapshots![i]))
                    throw new InvalidDataException("已返回的输出被覆盖：" + path);
        if (outputs.Length != (name.EndsWith("-seg") ? 2 : 1) || outputs.Any(t => t.Data.Any(v => !float.IsFinite(v))))
            throw new InvalidDataException("输出数量或数值无效：" + path);
        if (outputs[0].Shape[1] != 300 || outputs[0].Shape[2] != (name.EndsWith("-seg") ? 38 : name.EndsWith("-pose") ? 57 : 6))
            throw new InvalidDataException("端到端输出形状无效：" + path);
        previous = outputs;
        snapshots = outputs.Select(t => MemoryMarshal.AsBytes(t.Data.AsSpan()).ToArray()).ToArray();
        shapes = outputs.Select(t => (object)new { Shape = t.Shape, t.Length }).ToArray();
    }
    configurations.Add(new { Model = path, Backend = device.ToString(), Frames = 3, Outputs = shapes, DetachedOutputs = true, Passed = true });
    Console.WriteLine($"PASS NuGet {path} / {device}");
}
var report = new
{
    Status = "passed", Version = assembly.GetName().Version!.ToString(), DllSha256 = dllHash,
    PackageSha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes("artifacts/SimdYolo.0.8.1.nupkg"))),
    Scope = "只引用本地 NuGet 包，无项目引用；33 标签映射、12 个 CPU/Vulkan 模型配置及输出所有权烟雾验证，不替代独立原版数值对照。",
    LabelMaps = labels, Configurations = configurations
};
File.WriteAllText("artifacts/validation/0.8.1/package-consumer.json", JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
