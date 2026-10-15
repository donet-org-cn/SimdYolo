using System.Reflection.PortableExecutable;
using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;
using SimdYolo.Training;

// 从本地 NuGet 包消费训练库及其传递依赖，避免 ProjectReference 掩盖缺少包资源或依赖。
if (args.Length != 3) throw new ArgumentException("参数为：原始 PT 路径、输出目录、Cpu 或 Vulkan。");
var device = Enum.Parse<ComputeDevice>(args[2]);
string output = Path.GetFullPath(args[1]); var dataset = TrainingDataset.CreateDemo(1);
using (var trainer = new YoloTrainer(args[0], dataset, new() { ImageSize = 64, Epochs = 1, CpuThreads = 2, Device = device })) trainer.Train(output);
using (var resumed = YoloTrainer.Resume(Path.Combine(output, "last.sytrain"), dataset, 2, device, 2)) resumed.Train(output);
string checkpoint = Path.Combine(output, "last.sytrain");
using var detector = new YoloDetector(checkpoint, new() { Device = device });
var sample = dataset.Images[0]; var (input, _) = ImagePreprocessor.Letterbox(sample.Pixels, sample.Width, sample.Height, 64, 64);
var prediction = detector.PredictRaw(input);
if (prediction[0].Shape[1] != 5 || prediction.Any(t => t.Data.Any(v => !float.IsFinite(v))) || ModelInfo.Read(checkpoint).ClassNames[0] != "square") throw new Exception("打包训练库恢复或推理结果不正确。");
var assembly = typeof(YoloTrainer).Assembly;
using var file = File.OpenRead(assembly.Location); using var pe = new PEReader(file);
bool managed = pe.PEHeaders.CorHeader is { } cor && (cor.Flags & CorFlags.ILOnly) != 0;
var references = assembly.GetReferencedAssemblies().Select(a => a.Name!).ToArray();
if (!managed || references.Any(n => n != "SimdYolo" && !n.StartsWith("System", StringComparison.Ordinal))) throw new Exception("训练包包含非预期程序集依赖。");
var report = new { Status = "passed", Device = device.ToString(), Backend = detector.BackendName, TrainingManagedIlOnly = managed, TrainingReferences = references,
    CoreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), TrainingAssemblySha256 = Hash(assembly.Location), CheckpointInference = true, RestoredEpoch = 2, Classes = detector.ClassNames };
File.WriteAllText(Path.Combine(output, "consumer.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(JsonSerializer.Serialize(report));
static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
