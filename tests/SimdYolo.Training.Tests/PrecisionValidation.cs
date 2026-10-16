using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;
using SimdYolo.Training;

/// <summary>在已有真实照片、真值和固定拆分上串行验证低精度，所有推理和指标均由.NET计算。</summary>
internal static class PrecisionValidation
{
    internal static void Run(string[] args)
    {
        string fixture = args.Length > 1 ? args[1] : ".cache/training-011/outcomes/dataset.json";
        string output = args.Length > 2 ? args[2] : "artifacts/validation/0.11.0/precision.json";
        string cache = ".cache/precision-011"; Directory.CreateDirectory(cache); Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var images = document.RootElement.GetProperty("images").EnumerateArray().ToArray();
        TrainingImage Read(JsonElement image) => new(image.GetProperty("id").GetString()!, File.ReadAllBytes(image.GetProperty("rgbPath").GetString()!), image.GetProperty("width").GetInt32(), image.GetProperty("height").GetInt32(), image.GetProperty("labels").GetString()!);
        var validationImages = images.Where(image => image.GetProperty("split").GetString() == "val").Select(Read).ToArray();
        var calibrationImages = images.Where(image => image.GetProperty("split").GetString() == "train").Take(8).Select(Read).ToArray();
        var runs = new List<object>();
        foreach (string name in new[] { "yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo26n" })
        {
            string model = "models/pt/" + name + ".pt";
            var dataset = new TrainingDataset(ModelInfo.Read(model).ClassNames.OrderBy(pair => pair.Key).Select(pair => pair.Value), validationImages);
            string calibration = Path.Combine(cache, name + ".json");
            ModelCalibration.Calibrate(model, calibrationImages.Select(image => new CalibrationImage(image.Pixels, image.Width, image.Height)), calibration, 320, 4);
            var sample = validationImages[0]; var (input, _) = ImagePreprocessor.Letterbox(sample.Pixels, sample.Width, sample.Height, 320, 320);
            foreach (string mode in new[] { "FP32", "FP16-mixed", "INT8-dynamic", "INT8-calibrated" })
            {
                var precision = mode.StartsWith("INT8") ? InferencePrecision.Int8 : mode == "FP16-mixed" ? InferencePrecision.Float16 : InferencePrecision.Float32;
                string? profile = mode == "INT8-calibrated" ? calibration : null;
                var metrics = DetectionEvaluator.Evaluate(model, dataset, new() { ImageSize = 320, CpuThreads = 4, Precision = precision, CalibrationPath = profile });
                var load = Stopwatch.StartNew();
                using var detector = new YoloDetector(model, new() { Width = 320, Height = 320, CpuThreads = 4, Precision = precision, CalibrationPath = profile });
                load.Stop(); var first = Stopwatch.StartNew(); detector.PredictRaw(input); first.Stop();
                for (int i = 0; i < 20; i++) detector.PredictRaw(input);
                long allocated = GC.GetTotalAllocatedBytes(true); var times = new double[15];
                for (int i = 0; i < times.Length; i++) { var clock = Stopwatch.StartNew(); var result = detector.PredictRaw(input); clock.Stop(); times[i] = clock.Elapsed.TotalMilliseconds; if (result.Any(t => t.Data.Any(v => !float.IsFinite(v)))) throw new Exception("Nonfinite actual model output"); }
                allocated = GC.GetTotalAllocatedBytes(true) - allocated; Array.Sort(times);
                runs.Add(new { model = name, modelSha256 = Hash(model), mode, backend = detector.BackendName, validationImages = validationImages.Length, calibrationImages = mode == "INT8-calibrated" ? calibrationImages.Length : 0,
                    metrics.Precision, metrics.Recall, metrics.Map50, metrics.Map50To95, modelLoadMs = load.Elapsed.TotalMilliseconds, firstForwardMs = first.Elapsed.TotalMilliseconds,
                    medianForwardMs = times[7], samplesMs = times, allocatedBytesPerForward = allocated / times.Length });
                Console.WriteLine($"{name} {mode}: mAP50:95={metrics.Map50To95:F6}, forward={times[7]:F3}ms");
            }
        }
        File.WriteAllText(output, JsonSerializer.Serialize(new { status = "completed", input = new[] { 1, 3, 320, 320 }, threads = 4, warmup = 20, iterations = 15,
            scope = "COCO128 implementation diagnostic; pretraining saw COCO train2017. Quantization calibration uses 8 training images; evaluation uses fixed 32 holdout images. Latency excludes preprocessing/NMS/model loading. No performance or INT8 accuracy superiority claim.",
            datasetManifestSha256 = Hash(fixture), source = document.RootElement.GetProperty("source").GetString(), coreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), trainingAssemblySha256 = Hash(typeof(YoloTrainer).Assembly.Location), runs }, new JsonSerializerOptions { WriteIndented = true }));
    }
    private static string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(file)); }
}
