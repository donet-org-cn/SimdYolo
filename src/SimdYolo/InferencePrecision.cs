using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;

namespace SimdYolo;

/// <summary>选择推理卷积和矩阵乘法的操作数精度；非线性、归约与输出保持FP32。</summary>
public enum InferencePrecision
{
    /// <summary>原有FP32执行器。</summary>
    Float32,
    /// <summary>FP16操作数、FP32累加；CPU使用托管SIMD混合精度实现。</summary>
    Float16,
    /// <summary>对称INT8操作数、INT32累加、逐输出通道权重尺度；支持静态或动态激活尺度。</summary>
    Int8
}

/// <summary>用于校准的RGB24照片。</summary>
/// <param name="Pixels">RGB24像素。</param><param name="Width">照片宽度。</param><param name="Height">照片高度。</param>
public sealed record CalibrationImage(byte[] Pixels, int Width, int Height);

/// <summary>绑定具体模型、输入尺寸和卷积顺序的INT8激活校准。</summary>
public sealed class CalibrationProfile
{
    /// <summary>校准文件格式。</summary><value>固定为SimdYolo.Calibration/1格式。</value>
    public string Format { get; set; } = "SimdYolo.Calibration/1";
    /// <summary>实际模型文件摘要，训练权重变化后必须重新校准。</summary><value>模型文件的SHA256摘要。</value>
    public string ModelSha256 { get; set; } = "";
    /// <summary>实际NCHW输入形状。</summary><value>默认为空。</value>
    public int[] InputShape { get; set; } = [];
    /// <summary>参与校准的实际照片数。</summary><value>默认零。</value>
    public int Images { get; set; }
    /// <summary>按卷积位置/权重形状标识的最大绝对激活。</summary><value>用于对称零点0的INT8尺度。</value>
    public Dictionary<string, float> ActivationMaxima { get; set; } = new(StringComparer.Ordinal);
}

/// <summary>纯.NET代表性照片校准，保留模型及输入绑定，不执行量化训练。</summary>
public static class ModelCalibration
{
    /// <summary>以FP32运行代表性照片，记录每个卷积输入的最大绝对值并保存JSON。</summary>
    /// <param name="modelPath">模型路径。</param><param name="images">代表性照片。</param><param name="outputPath">校准JSON路径。</param>
    /// <param name="imageSize">目标输入尺寸；静态模型优先使用自身尺寸。</param><param name="threads">CPU线程数。</param><param name="cancellationToken">取消标记。</param>
    /// <returns>绑定模型与实际输入的校准结果。</returns>
    /// <exception cref="ArgumentException">没有照片或图像非法。</exception><exception cref="IOException">模型或JSON无法读写。</exception>
    /// <exception cref="OperationCanceledException">用户取消。</exception>
    public static CalibrationProfile Calibrate(string modelPath, IEnumerable<CalibrationImage> images, string outputPath, int imageSize = 640, int threads = 4, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (Path.GetFullPath(modelPath).Equals(Path.GetFullPath(outputPath), StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("校准文件不能覆盖源模型。");
        var profile = new CalibrationProfile { ModelSha256 = Hash(modelPath) };
        using var detector = new YoloDetector(modelPath, new() { Width = imageSize, Height = imageSize, CpuThreads = threads, CalibrationCollector = profile });
        foreach (var image in images) { cancellationToken.ThrowIfCancellationRequested(); detector.Detect(image.Pixels, image.Width, image.Height); profile.Images++; }
        if (profile.Images == 0 || profile.ActivationMaxima.Count == 0) throw new ArgumentException("校准需要至少一张实际照片和可执行卷积。");
        string full = Path.GetFullPath(outputPath); Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        string temporary = full + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try { File.WriteAllText(temporary, JsonSerializer.Serialize(profile, new JsonSerializerOptions { WriteIndented = true })); File.Move(temporary, full, true); }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
        return profile;
    }
    internal static string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(file)); }
}

/// <summary>纯C#低精度CPU计算；INT8使用AVX2整数点积，FP16操作数通过FP32 SIMD执行累加。</summary>
internal sealed class PrecisionCpuBackend : IComputeBackend
{
    internal CpuBackend Cpu { get; }
    private readonly InferencePrecision precision;
    private readonly CalibrationProfile? profile, collector;
    private readonly Dictionary<Tensor, (sbyte[] Data, float[] Scales)> packed = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<Tensor, Tensor> halves = new(ReferenceEqualityComparer.Instance);
    private int convolutionIndex;
    internal long Convolutions { get; private set; }
    internal long Matrices { get; private set; }
    public string Name => collector is not null ? "CPU FP32 calibration" : precision == InferencePrecision.Float16 ? "CPU FP16 operands / FP32 SIMD accumulation" : $"CPU INT8 {(Avx2.IsSupported ? "AVX2" : "managed scalar")} / INT32 accumulation ({(profile is null ? "dynamic" : "calibrated")})";
    internal PrecisionCpuBackend(InferencePrecision precision, int threads, CalibrationProfile? profile, CalibrationProfile? collector)
    { this.precision = precision; this.profile = profile; this.collector = collector; Cpu = new(threads); }
    internal void Begin(int[] shape)
    {
        convolutionIndex = 0;
        if (profile is not null && !profile.InputShape.SequenceEqual(shape)) throw new ArgumentException("INT8校准尺寸与当前输入不同，请重新校准。");
        if (collector is not null) { if (collector.InputShape.Length > 0 && !collector.InputShape.SequenceEqual(shape)) throw new ArgumentException("同一校准不能混用输入尺寸。"); collector.InputShape = shape.ToArray(); }
    }
    public void Dispose() { packed.Clear(); halves.Clear(); Cpu.Dispose(); }
    private static float Maximum(float[] values)
    {
        float maximum = 0;
        foreach (float value in values) { if (!float.IsFinite(value)) throw new ArithmeticException("低精度输入包含非有限值。"); maximum = MathF.Max(maximum, MathF.Abs(value)); }
        return maximum;
    }
    private static float Scale(float maximum) => maximum > 0 ? MathF.Max(maximum / 127, 1e-30f) : 1;
    private static sbyte Quantize(float value, float scale) => (sbyte)(int)Math.Clamp(MathF.Round(value / scale), -127f, 127f);
    private static Tensor HalfOperands(Tensor input)
    {
        var output = new Tensor(input.Shape); for (int i = 0; i < input.Length; i++) { float value = (float)(Half)input.Data[i]; if (!float.IsFinite(value)) throw new ArithmeticException("FP16操作数溢出，请使用FP32。"); output.Data[i] = value; }
        output.IsConstant = input.IsConstant; return output;
    }
    internal Tensor Convolve(Tensor x, Tensor w, Tensor? bias, int[] shape, int[] strides, int[] pads, int[] dilation, int groups, bool silu)
    {
        string key = convolutionIndex++ + ":" + string.Join('x', w.Shape);
        Convolutions++;
        if (collector is not null)
        {
            collector.ActivationMaxima[key] = MathF.Max(collector.ActivationMaxima.GetValueOrDefault(key), Maximum(x.Data));
            return new TensorOps(Cpu).Conv(x, w, bias, strides, pads, dilation, groups, silu);
        }
        if (precision == InferencePrecision.Float16)
        {
            if (!halves.TryGetValue(w, out var weight)) { weight = HalfOperands(w); halves[w] = weight; }
            return new TensorOps(Cpu).Conv(HalfOperands(x), weight, bias, strides, pads, dilation, groups, silu);
        }
        float actualMaximum = Maximum(x.Data);
        float maximum = profile is null ? actualMaximum : profile.ActivationMaxima.TryGetValue(key, out float calibrated) ? calibrated : throw new InvalidDataException("校准文件缺少卷积：" + key);
        float xs = Scale(maximum); int oc = w.Shape[0], icg = w.Shape[1], kh = w.Shape[2], kw = w.Shape[3], k = icg * kh * kw;
        if (!packed.TryGetValue(w, out var weights))
        {
            _ = Maximum(w.Data);
            var q = new sbyte[w.Length]; var scales = new float[oc];
            for (int co = 0; co < oc; co++) { float max = 0; for (int j = 0; j < k; j++) max = MathF.Max(max, MathF.Abs(w.Data[co * k + j])); scales[co] = Scale(max); for (int j = 0; j < k; j++) q[co * k + j] = Quantize(w.Data[co * k + j], scales[co]); }
            weights = (q, scales); packed.Add(w, weights);
        }
        var y = Tensor.TemporaryUninitialized(shape); int spatial = shape[2] * shape[3], blocks = (spatial + 31) / 32, cog = oc / groups;
        var wd = weights; float[] source = x.Data, destination = y.Data;
        CpuScheduling.For(0, shape[0] * groups * blocks, Cpu.Parallel, block =>
        {
            int group = block / blocks % groups, batch = block / (blocks * groups), start = block % blocks * 32, count = Math.Min(32, spatial - start);
            sbyte[] patch = ArrayPool<sbyte>.Shared.Rent(checked(count * k));
            try
            {
                for (int pos = 0; pos < count; pos++)
                {
                    int oy = (start + pos) / shape[3], ox = (start + pos) % shape[3], index = pos * k;
                    for (int ci = 0; ci < icg; ci++) for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++)
                    {
                        int iy = oy * strides[0] - pads[0] + ky * dilation[0], ix = ox * strides[1] - pads[1] + kx * dilation[1];
                        patch[index++] = (uint)iy < x.Shape[2] && (uint)ix < x.Shape[3] ? Quantize(source[((batch * x.Shape[1] + group * icg + ci) * x.Shape[2] + iy) * x.Shape[3] + ix], xs) : (sbyte)0;
                    }
                }
                for (int co = group * cog; co < (group + 1) * cog; co++) for (int pos = 0; pos < count; pos++)
                {
                    float value = Dot(patch, pos * k, wd.Data, co * k, k) * xs * wd.Scales[co] + (bias?.Data[co] ?? 0);
                    destination[(batch * oc + co) * spatial + start + pos] = silu ? value / (1 + MathF.Exp(-value)) : value;
                }
            }
            finally { ArrayPool<sbyte>.Shared.Return(patch); }
        });
        return y;
    }
    public void MatMul(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        Matrices++;
        if (precision == InferencePrecision.Float16 || collector is not null)
        {
            if (collector is not null) Cpu.MatMul(a, b, c, m, n, k);
            else { var qa = a.Select(x => (float)(Half)x).ToArray(); var qb = b.Select(x => (float)(Half)x).ToArray(); if (qa.Any(x => !float.IsFinite(x)) || qb.Any(x => !float.IsFinite(x))) throw new ArithmeticException("FP16矩阵溢出。"); Cpu.MatMul(qa, qb, c, m, n, k); }
            return;
        }
        _ = Maximum(b); float sa = Scale(Maximum(a)); var qa8 = a.Select(x => Quantize(x, sa)).ToArray(); var qb8 = new sbyte[checked(n * k)]; var sb = new float[n];
        for (int col = 0; col < n; col++) { float max = 0; for (int row = 0; row < k; row++) max = MathF.Max(max, MathF.Abs(b[row * n + col])); sb[col] = Scale(max); for (int row = 0; row < k; row++) qb8[col * k + row] = Quantize(b[row * n + col], sb[col]); }
        CpuScheduling.For(0, m, Cpu.Parallel, row => { for (int col = 0; col < n; col++) c[row * n + col] = Dot(qa8, row * k, qb8, col * k, k) * sa * sb[col]; });
    }
    internal static unsafe long Dot(sbyte[] a, int ao, sbyte[] b, int bo, int length)
    {
        // 每段最多4096个乘积，避免INT32向量累加溢出；跨段使用INT64。
        long total = 0; int offset = 0;
        fixed (sbyte* ap = a, bp = b)
        {
            while (offset < length)
            {
                int end = Math.Min(length, offset + 4096); var sum = Vector256<int>.Zero;
                if (Avx2.IsSupported) for (; offset <= end - 16; offset += 16)
                    sum = Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Sse2.LoadVector128(ap + ao + offset)), Avx2.ConvertToVector256Int16(Sse2.LoadVector128(bp + bo + offset))));
                for (int i = 0; i < 8; i++) total += sum.GetElement(i);
                for (; offset < end; offset++) total += (int)ap[ao + offset] * bp[bo + offset];
            }
        }
        return total;
    }
}
