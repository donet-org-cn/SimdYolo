using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;

// 串行分段测量实际照片的预处理、网络、后处理和完整Predict，不把分段中位数相加冒充总时间。
string model = args[0], imagePath = args[1], output = args[2];
int width = int.Parse(args[3]), height = int.Parse(args[4]), threads = args.Length > 6 ? int.Parse(args[6]) : 4;
ComputeDevice device = Enum.Parse<ComputeDevice>(args.Length > 5 ? args[5] : "Cpu");
byte[] rgb = File.ReadAllBytes(imagePath);
if (rgb.Length != width * height * 3) throw new Exception("RGB fixture length mismatch");
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
var load = Stopwatch.StartNew();
using var detector = new YoloDetector(model, new() { Device = device, CpuThreads = threads, Width = 640, Height = 640, MaxDetections = 100 });
load.Stop();
var first = Stopwatch.StartNew(); var firstResult = detector.Predict(rgb, width, height); first.Stop();
var (input, transform) = ImagePreprocessor.Letterbox(rgb, width, height, 640, 640);
Tensor[] raw = detector.PredictRaw(input);
var options = new DetectorOptions { Task = detector.Task, ClassCount = detector.ClassNames.Count, OutputLayout = detector.Info.EndToEnd ? OutputLayout.EndToEnd : OutputLayout.ChannelsFirst, CpuThreads = threads, MaxDetections = 100 };
int totalPixels = 0; IReadOnlyList<YoloInstance> instances = [];
var stages = new Dictionary<string, object>();
Measure("preprocess", () => ImagePreprocessor.Letterbox(rgb, width, height, 640, 640));
Measure("forward", () => detector.PredictRaw(input));
Measure("decode", () => instances = VisionDecoder.Decode(raw, transform, options));
Measure("predict", () => instances = detector.Predict(rgb, width, height));
var codecMethod = typeof(YoloDetector).Assembly.GetType("SimdYolo.MaskConverter")?.GetMethod("ToBytes");
var codec = codecMethod?.CreateDelegate<EncodeMask>();
if (instances.Any(value => value.Mask is not null)) Measure("maskBytes", () => { foreach (var value in instances) if (value.Mask is { } mask) _ = codec is null ? mask.Probabilities.Select(p => (byte)Math.Clamp((int)MathF.Round(p * 255), 0, 255)).ToArray() : codec(mask.Probabilities); });
for (int i = 0; i < instances.Count; i++) if (instances[i].Mask is { } mask) { totalPixels += mask.Probabilities.Length; File.WriteAllBytes(output + $".mask-{i}.f32", MemoryMarshal.AsBytes(mask.Probabilities.AsSpan()).ToArray()); }
File.WriteAllText(output + ".instances.json", JsonSerializer.Serialize(instances.Select(instance => new { instance.Box, mask = instance.Mask is { } mask ? new { mask.X, mask.Y, mask.Width, mask.Height } : null, instance.Keypoints })));
File.WriteAllText(output + ".json", JsonSerializer.Serialize(new { model, imagePath, width, height, threads, backend = detector.BackendName, inputSha256 = Hash(imagePath), coreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), modelSha256 = Hash(model), loadMs = load.Elapsed.TotalMilliseconds, firstPredictMs = first.Elapsed.TotalMilliseconds, instances = instances.Count, totalMaskPixels = totalPixels, stages }, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(File.ReadAllText(output + ".json"));

void Measure(string name, Action action)
{
    for (int i = 0; i < 12; i++) action();
    var samples = new List<double>(); using var process = Process.GetCurrentProcess(); var cpu = process.TotalProcessorTime;
    long allocated = GC.GetTotalAllocatedBytes(true); var window = Stopwatch.StartNew();
    do { var clock = Stopwatch.StartNew(); action(); samples.Add(clock.Elapsed.TotalMilliseconds); } while (samples.Count < 40 || window.Elapsed.TotalSeconds < 2);
    window.Stop(); var duration = process.TotalProcessorTime - cpu; allocated = GC.GetTotalAllocatedBytes(true) - allocated;
    var ordered = samples.Order().ToArray();
    stages[name] = new { medianMs = ordered[ordered.Length / 2], p95Ms = ordered[(int)(ordered.Length * .95)], frames = samples.Count, cpuCorePercent = duration.TotalSeconds / window.Elapsed.TotalSeconds * 100, allocatedBytesPerFrame = allocated / samples.Count, samplesMs = samples };
}
static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));

delegate byte[] EncodeMask(ReadOnlySpan<float> probabilities);
