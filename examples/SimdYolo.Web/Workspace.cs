using System.Collections.Concurrent;
using System.Text.Json;
using SimdYolo.Training;

namespace SimdYolo.Web;

/// <summary>维护本地示例的模型、数据集和生成文件，仅以随机ID对外引用。</summary>
internal sealed class Workspace
{
    /// <summary>浏览器预览和 RGB 上传的最长边；限制中间缓冲区，不限制原照片为此尺寸。</summary>
    internal const int MaxUploadEdge = 2048;
    /// <summary>原照片允许的最大单边长度，实际解码能力还取决于浏览器。</summary>
    internal const int MaxOriginalEdge = 65535;
    /// <summary>原照片最多包含256 Mi个像素；服务器只接收缩放后的RGB。</summary>
    internal const long MaxOriginalPixels = 268435456;
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
        bool trainable = YoloTrainer.CanTrain(path, out var info);
        string id = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(Path.GetFullPath(path))))[..20];
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

    /// <summary>校验原照片尺寸，不为原照片分配 RGB 或掩码缓冲区。</summary>
    /// <param name="width">原照片宽度。</param>
    /// <param name="height">原照片高度。</param>
    /// <exception cref="ArgumentException">尺寸非正、超过单边长度或总像素上限。</exception>
    internal static void ValidateOriginalSize(int width, int height)
    {
        if (width < 1 || height < 1 || width > MaxOriginalEdge || height > MaxOriginalEdge || (long)width * height > MaxOriginalPixels)
            throw new ArgumentException($"原图尺寸无效，最长边不能超过{MaxOriginalEdge}，总像素不能超过256 Mi像素。");
    }

    /// <summary>解码已缩放的RGB24，先限制编码长度，避免按原照片尺寸分配服务端内存。</summary>
    /// <param name="width">上传图像宽度。</param>
    /// <param name="height">上传图像高度。</param>
    /// <param name="encoded">RGB24的Base64编码。</param>
    /// <returns>字节数与上传尺寸一致的RGB24数据。</returns>
    /// <exception cref="ArgumentException">上传尺寸、编码长度或解码字节数无效。</exception>
    /// <exception cref="FormatException">数据不是有效的Base64编码。</exception>
    internal static byte[] DecodePixels(int width, int height, string? encoded)
    {
        if (width < 1 || width > MaxUploadEdge || height < 1 || height > MaxUploadEdge || encoded is null)
            throw new ArgumentException($"上传RGB最长边为{MaxUploadEdge}，网页会自动缩放大图；直接调用API时请先缩放并携带originalWidth/originalHeight。");
        int byteCount = checked(width * height * 3);
        if (encoded.Length != (byteCount + 2) / 3 * 4) throw new ArgumentException("RGB24 编码长度与图像尺寸不一致。");
        byte[] bytes = Convert.FromBase64String(encoded);
        if (bytes.Length != byteCount) throw new ArgumentException("RGB24 字节数与图像尺寸不一致。");
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
