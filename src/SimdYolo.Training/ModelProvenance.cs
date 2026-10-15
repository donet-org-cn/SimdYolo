using System.IO.Compression;
using System.Text.Json;

namespace SimdYolo.Training;

/// <summary>根据已核验的原始权重摘要保留模型来源和原许可证，不改变项目源码的许可状态。</summary>
internal static class ModelProvenance
{
    /// <summary>匹配官方清单的原始 SHA256，并将模型来源和适用的原许可证写入检查点。</summary>
    /// <param name="zip">正在写入的自包含训练归档。</param><param name="sourceSha256">原始 PT 的小写 SHA256。</param>
    internal static void Write(ZipArchive zip, string sourceSha256)
    {
        var assembly = typeof(ModelProvenance).Assembly;
        using var manifest = assembly.GetManifestResourceStream("SimdYolo.ModelManifest.json")!;
        using var document = JsonDocument.Parse(manifest);
        foreach (var model in document.RootElement.GetProperty("Models").EnumerateArray())
        {
            if (!model.GetProperty("Sha256").GetString()!.Equals(sourceSha256, StringComparison.OrdinalIgnoreCase)) continue;
            using (var entry = zip.CreateEntry("source-provenance.json").Open())
                JsonSerializer.Serialize(entry, new { SourceSha256 = sourceSha256, SourceUrl = model.GetProperty("SourceUrl").GetString(), License = model.GetProperty("License").GetString() });
            using var license = assembly.GetManifestResourceStream("SimdYolo.ModelLicense.txt")!;
            using var target = zip.CreateEntry("LICENSE-AGPL-3.0.txt").Open(); license.CopyTo(target);
            return;
        }
        // 自定义权重无法按摘要识别许可证，避免擅自将其标记为官方 AGPL 模型。
        using var unknown = zip.CreateEntry("source-provenance.json").Open();
        JsonSerializer.Serialize(unknown, new { SourceSha256 = sourceSha256, Notice = "Custom source weights; retain the original supplier's license and attribution separately." });
    }
}
