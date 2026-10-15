using System.Collections.Concurrent;
using System.Text.Json;
using SimdYolo.Training;

namespace SimdYolo.Web;

/// <summary>维护本地示例的模型、数据集和生成文件，仅以随机ID对外引用。</summary>
internal sealed class Workspace
{
    internal string RepositoryRoot { get; }
    internal string DataRoot { get; }
    internal ConcurrentDictionary<string, WebModel> Models { get; } = new(StringComparer.Ordinal);
    internal ConcurrentDictionary<string, (string Name, TrainingDataset Dataset)> Datasets { get; } = new(StringComparer.Ordinal);

    public Workspace(IConfiguration configuration, IWebHostEnvironment environment)
    {
        RepositoryRoot = configuration["SimdYolo:RepositoryRoot"] is { Length: > 0 } root ? Path.GetFullPath(root) : FindRepository(environment.ContentRootPath);
        DataRoot = Path.GetFullPath(configuration["SimdYolo:DataRoot"] ?? Path.Combine(RepositoryRoot, ".cache", "web-example")); Directory.CreateDirectory(DataRoot);
        foreach (string directory in new[] { "models", "datasets", "runs" }) Directory.CreateDirectory(Path.Combine(DataRoot, directory));
        foreach (string path in Directory.EnumerateFiles(Path.Combine(RepositoryRoot, "models", "pt"), "*.pt").Concat(Directory.EnumerateFiles(Path.Combine(RepositoryRoot, "models", "onnx"), "*.onnx")))
            RegisterModel(path, Path.GetFileName(path));
        foreach (string path in Directory.EnumerateFiles(Path.Combine(DataRoot, "models")))
            if (Path.GetExtension(path).ToLowerInvariant() is ".pt" or ".pth" or ".onnx" or ".sytrain")
            {
                string name = Path.GetFileName(path);
                if (name.Length > 33 && name[32] == '_' && Guid.TryParseExact(name[..32], "N", out _)) name = name[33..];
                try { RegisterModel(path, name); }
                catch (Exception error) when (error is IOException or InvalidDataException or ArgumentException or NotSupportedException) { Console.Error.WriteLine($"跳过损坏模型 {name}：{error.Message}"); }
            }
        Datasets["demo-squares"] = ("内置方块示例（用于流程验证）", TrainingDataset.CreateDemo());
        foreach (string path in Directory.EnumerateFiles(Path.Combine(DataRoot, "datasets"), "*.json"))
        {
            try
            {
                var saved = JsonSerializer.Deserialize<DatasetRequest>(File.ReadAllText(path), JsonOptions);
                if (saved is not null) Datasets[Path.GetFileNameWithoutExtension(path)] = (saved.Name, CreateDataset(saved));
            }
            catch (Exception error) when (error is IOException or JsonException or ArgumentException or InvalidDataException) { Console.Error.WriteLine($"跳过损坏数据集 {Path.GetFileName(path)}：{error.Message}"); }
        }
    }

    internal WebModel RegisterModel(string path, string name)
    {
        var info = ModelInfo.Read(path); string id = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(Path.GetFullPath(path))))[..20];
        bool trainable = info.Task == YoloTask.Detection && (info.Format == "SYTRAIN" || info.Format == "PT" && name.StartsWith("yolov8", StringComparison.OrdinalIgnoreCase));
        var model = new WebModel(id, name, path, info.Format, info.Task.ToString(), trainable); Models[id] = model; return model;
    }

    internal static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web) { WriteIndented = true };
    internal static TrainingDataset CreateDataset(DatasetRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Name) || request.Name.Length > 100 || request.Images is null || request.Images.Length is < 1 or > 200 || request.Classes is null) throw new ArgumentException("数据集名称、类别或图像数量无效（最多200张）。");
        long total = 0; var images = new List<TrainingImage>();
        foreach (var image in request.Images)
        {
            byte[] pixels = DecodePixels(image.Width, image.Height, image.Pixels); total += pixels.Length;
            if (total > 48 * 1024 * 1024) throw new ArgumentException("示例数据集的解码像素总量不能超过48 MiB。");
            images.Add(new(Path.GetFileName(image.Name), pixels, image.Width, image.Height, image.Labels ?? ""));
        }
        return new(request.Classes, images);
    }

    internal static byte[] DecodePixels(int width, int height, string? encoded)
    {
        if (width is < 1 or > 2048 || height is < 1 or > 2048 || encoded is null || encoded.Length > 2048 * 2048 * 4) throw new ArgumentException("图像尺寸或 RGB 数据超限，最大2048×2048。");
        byte[] bytes = Convert.FromBase64String(encoded);
        if (bytes.Length != checked(width * height * 3)) throw new ArgumentException("RGB24 字节数与图像尺寸不一致。");
        return bytes;
    }

    internal static ComputeDevice Device(string name, bool training = false)
    {
        if (!Enum.TryParse<ComputeDevice>(name, true, out var device) || !Enum.IsDefined(device) || training && device is not (ComputeDevice.Cpu or ComputeDevice.Vulkan)) throw new ArgumentException("计算设备不受支持。");
        return device;
    }

    private static string FindRepository(string start)
    {
        for (var directory = new DirectoryInfo(start); directory is not null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, "SimdYolo.sln"))) return directory.FullName;
        throw new DirectoryNotFoundException("找不到仓库，请配置 SimdYolo:RepositoryRoot。");
    }
}
