using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;
using SimdYolo.Training;

// 只引用本地NuGet包及其传递依赖，实际执行新增API、PT/ONNX分割和训练保存/恢复。
var device = Enum.Parse<ComputeDevice>(args.Length > 0 ? args[0] : "Cpu");
string output = args.Length > 1 ? args[1] : ".cache/image-package-consumer/" + device;
Directory.CreateDirectory(output); var cases = new List<object>();
// 新包验证单次模型检查；通过反射保留此工具对0.11.2历史包的复现能力。
bool singlePassModelInspection = false;
if (typeof(YoloTrainer).Assembly.GetName().Version! >= new Version(0, 11, 3))
{
    var inspect = typeof(YoloTrainer).GetMethod("CanTrain", [typeof(string), typeof(ModelInfo).MakeByRefType()]) ?? throw new Exception("新包缺少单次模型检查API。");
    foreach (string path in Directory.EnumerateFiles("models/pt", "*.pt").Concat(Directory.EnumerateFiles("models/onnx", "*.onnx")))
    {
        object?[] arguments = [path, null]; bool trainable = (bool)inspect.Invoke(null, arguments)!; var info = (ModelInfo)arguments[1]!;
        if (trainable != YoloTrainer.CanTrain(path) || info.Task != ModelInfo.Read(path).Task) throw new Exception("包内模型检查结果不一致。");
    }
    singlePassModelInspection = true;
}
byte[] pixels = File.ReadAllBytes(".cache/image-optimization/fixtures/bus.rgb");
foreach (string name in new[] { "yolov8n-seg", "yolo26n-seg" })
foreach (string format in new[] { "pt", "onnx" })
{
    string model = $"models/{format}/{name}.{format}";
    using var detector = new YoloDetector(model, new() { Device = device, CpuThreads = 4, Confidence = 1 });
    if (detector.Predict(pixels, 810, 1080).Count != 0) throw new Exception("默认阈值错误。");
    var actual = detector.Predict(pixels, 810, 1080, .25f);
    if (actual.Count == 0 || actual.Any(item => item.Mask is null || item.Mask.Probabilities.Any(p => !float.IsFinite(p)))) throw new Exception("分割没有有效实例。");
    foreach (var item in actual)
    {
        float[] values = item.Mask!.Probabilities; byte[] bytes = MaskConverter.ToBytes(values);
        for (int i = 0; i < bytes.Length; i++) if (bytes[i] != (byte)MathF.Round(Math.Clamp(values[i], 0, 1) * 255)) throw new Exception("掩码字节转换改变舍入规则。");
    }
    if (detector.Predict(pixels, 810, 1080).Count != 0) throw new Exception("调用阈值修改了会话默认值。");
    cases.Add(new { model, detector.BackendName, instances = actual.Count, maskPixels = actual.Sum(item => item.Mask!.Probabilities.Length) });
}
var data = TrainingDataset.CreateDemo(1, 26); string training = Path.Combine(output, "training");
using (var trainer = new YoloTrainer("models/pt/yolo26n.pt", data, new() { Device = device, CpuThreads = 2, ImageSize = 64, Epochs = 1 })) trainer.Train(training);
using (var resumed = YoloTrainer.Resume(Path.Combine(training, "last.sytrain"), data, 2, device, 2)) resumed.Train(training);
using (var detector = new YoloDetector(Path.Combine(training, "last.sytrain"), new() { Device = device, Width = 64, Height = 64, CpuThreads = 2 }))
{
    var sample = data.Images[0]; _ = detector.Predict(sample.Pixels, sample.Width, sample.Height, .001f);
}
ModelExporter.ExportOnnx(Path.Combine(training, "last.sytrain"), Path.Combine(training, "export.onnx"), 64);
var report = new { status = "passed", device = device.ToString(), version = typeof(YoloDetector).Assembly.GetName().Version!.ToString(3),
    coreSha256 = Hash(typeof(YoloDetector).Assembly.Location), trainingSha256 = Hash(typeof(YoloTrainer).Assembly.Location),
    projectReference = false, singlePassModelInspection, cases, trainingSavedResumedAndInferred = true, exportedTrainedOnnx = true };
File.WriteAllText(Path.Combine(output, "consumer.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(JsonSerializer.Serialize(report));
static string Hash(string file) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(file)));
