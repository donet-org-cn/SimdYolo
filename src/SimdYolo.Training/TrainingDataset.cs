using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace SimdYolo.Training;

/// <summary>保存经过严格校验的 RGB 图像、类别名称和 YOLO 检测标注。</summary>
public sealed class TrainingDataset
{
    /// <summary>类别名称的独立副本。</summary><value>按零起始类别编号排列。</value>
    public IReadOnlyList<string> ClassNames { get; }
    /// <summary>训练图像列表。</summary><value>构造时复制图像和标签，调用方不应修改其字节。</value>
    public IReadOnlyList<TrainingImage> Images { get; }
    /// <summary>类别、图像尺寸、像素及标注共同形成的 SHA256。</summary><value>用于拒绝数据集不同的断点恢复。</value>
    public string Sha256 { get; }

    /// <summary>按固定种子拆分训练与验证样本；拒绝内容重复，避免相同照片泄漏到两侧。</summary>
    /// <param name="validationFraction">验证比例，必须大于零且小于一。</param><param name="seed">洗牌种子。</param>
    /// <returns>包含相同类别、互不重叠照片的训练和验证集。</returns>
    /// <exception cref="ArgumentException">样本不足、比例非法或存在重复照片。</exception>
    public (TrainingDataset Training, TrainingDataset Validation) Split(double validationFraction = .2, int seed = 26)
    {
        if (Images.Count < 2 || !double.IsFinite(validationFraction) || validationFraction <= 0 || validationFraction >= 1) throw new ArgumentException("拆分需要至少两张图，验证比例在0到1之间。");
        if (Images.Select(ImageHash).Distinct(StringComparer.Ordinal).Count() != Images.Count) throw new ArgumentException("数据集含重复像素内容，请先去重再拆分，避免验证泄漏。");
        int[] order = Enumerable.Range(0, Images.Count).ToArray(); new Random(seed).Shuffle(order);
        int count = Math.Clamp((int)Math.Round(Images.Count * validationFraction), 1, Images.Count - 1);
        return (new(ClassNames, order[count..].Select(i => Images[i])), new(ClassNames, order[..count].Select(i => Images[i])));
    }

    /// <summary>校验独立验证集的类别和照片内容。</summary>
    /// <param name="validation">验证集。</param>
    /// <exception cref="ArgumentException">类别不同或两侧含相同照片。</exception>
    public void ValidateHoldout(TrainingDataset validation)
    {
        ArgumentNullException.ThrowIfNull(validation);
        if (!ClassNames.SequenceEqual(validation.ClassNames)) throw new ArgumentException("训练和验证集类别名称及顺序必须一致。");
        var hashes = Images.Select(ImageHash).ToHashSet(StringComparer.Ordinal);
        if (validation.Images.Any(image => hashes.Contains(ImageHash(image)))) throw new ArgumentException("训练和验证集包含相同照片，请使用独立数据或先拆分。");
    }

    private static string ImageHash(TrainingImage image) => $"{image.Width}x{image.Height}:" + Convert.ToHexStringLower(SHA256.HashData(image.Pixels));

    /// <summary>构造检测数据集并校验所有归一化标签。</summary>
    /// <param name="classNames">类别名称。</param><param name="images">RGB24 图像及 YOLO 标签。</param>
    /// <exception cref="ArgumentException">类别或图像为空、尺寸不合法、像素长度错误。</exception>
    /// <exception cref="InvalidDataException">类别编号、框范围或标签格式无效。</exception>
    public TrainingDataset(IEnumerable<string> classNames, IEnumerable<TrainingImage> images)
    {
        string[] names = classNames.ToArray(); TrainingImage[] samples = images.ToArray();
        if (names.Length is < 1 or > 1000 || names.Any(string.IsNullOrWhiteSpace) || names.Distinct(StringComparer.Ordinal).Count() != names.Length || samples.Length is < 1 or > 10000)
            throw new ArgumentException("需要非空、无重复的类别和至少一张图像。");
        ClassNames = Array.AsReadOnly(names);
        var copies = new List<TrainingImage>();
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData(Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(names)));
        foreach (var sample in samples)
        {
            if (sample.Width is < 1 or > 8192 || sample.Height is < 1 or > 8192 || sample.Pixels.Length != checked(sample.Width * sample.Height * 3)) throw new ArgumentException("RGB 图像尺寸或字节数无效。");
            _ = ParseLabels(sample.Labels, names.Length, new(sample.Width, sample.Height, sample.Width, sample.Height, 1, 1, 0, 0));
            var copy = sample with { Pixels = (byte[])sample.Pixels.Clone() }; copies.Add(copy);
            hash.AppendData(BitConverter.GetBytes(copy.Width)); hash.AppendData(BitConverter.GetBytes(copy.Height));
            hash.AppendData(copy.Pixels); hash.AppendData(Encoding.UTF8.GetBytes(copy.Labels)); hash.AppendData([0]);
        }
        Images = copies.AsReadOnly(); Sha256 = Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    /// <summary>将 YOLO 归一化框转换为实际 letterbox 后的像素边界。</summary>
    /// <param name="labels">每行五列的 YOLO 检测标签，允许空文本。</param><param name="classes">合法类别数量。</param>
    /// <param name="transform">保持比例缩放和填边的实际变换。</param>
    /// <returns>训练输入像素坐标内的有效目标框。</returns>
    /// <exception cref="InvalidDataException">类别、数值、框边界或目标数量无效。</exception>
    internal static TrainingBox[] ParseLabels(string labels, int classes, Letterbox transform)
    {
        var result = new List<TrainingBox>();
        foreach (string line in labels.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 5 || !int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int cls) || cls < 0 || cls >= classes) throw new InvalidDataException("检测标签必须为 class cx cy w h，类别编号从零开始。");
            float[] v = parts.Skip(1).Select(s => float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && float.IsFinite(value) ? value : throw new InvalidDataException("标签数值无效。")).ToArray();
            if (v[0] is < 0 or > 1 || v[1] is < 0 or > 1 || v[2] is <= 0 or > 1 || v[3] is <= 0 or > 1 || v[0] - v[2] / 2 < -1e-5f || v[0] + v[2] / 2 > 1.00001f || v[1] - v[3] / 2 < -1e-5f || v[1] + v[3] / 2 > 1.00001f)
                throw new InvalidDataException("检测框须完全位于图像的归一化范围内。");
            result.Add(new(cls, (v[0] - v[2] / 2) * transform.SourceWidth * transform.ScaleX + transform.PadX, (v[1] - v[3] / 2) * transform.SourceHeight * transform.ScaleY + transform.PadY,
                (v[0] + v[2] / 2) * transform.SourceWidth * transform.ScaleX + transform.PadX, (v[1] + v[3] / 2) * transform.SourceHeight * transform.ScaleY + transform.PadY));
            if (result.Count > 1000) throw new InvalidDataException("单张图像标注超过1000个框。");
        }
        return result.ToArray();
    }

    /// <summary>生成可离线运行的彩色方块示例数据，用于验证训练流程而非代表真实数据集精度。</summary>
    /// <param name="count">示例图像数量，范围1到128。</param><param name="seed">生成种子。</param>
    /// <returns>包含 square 类别的检测数据集。</returns>
    /// <exception cref="ArgumentOutOfRangeException">数量超出范围。</exception>
    public static TrainingDataset CreateDemo(int count = 8, int seed = 26)
    {
        if (count is < 1 or > 128) throw new ArgumentOutOfRangeException(nameof(count));
        var random = new Random(seed); var images = new List<TrainingImage>(); const int size = 128;
        for (int i = 0; i < count; i++)
        {
            int boxSize = random.Next(32, 61), x = random.Next(8, size - boxSize - 8), y = random.Next(8, size - boxSize - 8); byte[] pixels = new byte[size * size * 3];
            for (int p = 0; p < size * size; p++) { int py = p / size, px = p % size; bool inside = px >= x && px < x + boxSize && py >= y && py < y + boxSize; pixels[p * 3] = (byte)(inside ? 235 : 25 + px / 8); pixels[p * 3 + 1] = (byte)(inside ? 120 + i % 3 * 30 : 30 + py / 8); pixels[p * 3 + 2] = (byte)(inside ? 48 : 45); }
            string label = FormattableString.Invariant($"0 {(x + boxSize / 2f) / size} {(y + boxSize / 2f) / size} {boxSize / (float)size} {boxSize / (float)size}");
            images.Add(new($"square-{i}.png", pixels, size, size, label));
        }
        return new(["square"], images);
    }
}
