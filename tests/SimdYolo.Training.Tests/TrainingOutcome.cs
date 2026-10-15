using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;
using SimdYolo.Training;

/// <summary>导出真实图片的统一微调输入，经过公开训练入口后验证完整快照及保存模型的实际预测。</summary>
internal static class TrainingOutcome
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { WriteIndented = true };

    internal static void Run(string[] args)
    {
        string Option(string name, string fallback) { int i = Array.IndexOf(args, name); return i < 0 ? fallback : args[i + 1]; }
        if (args[0] == "outcome-emit")
        {
            string directory = Path.GetFullPath(args[1]); Directory.CreateDirectory(directory);
            string source = Path.GetFullPath(Option("--model", "models/pt/yolov8n.pt"));
            using var datasetDocument = JsonDocument.Parse(File.ReadAllText(Option("--dataset", ".cache/training-011/dataset.json")));
            int size = int.Parse(Option("--size", "256")), epochs = int.Parse(Option("--epochs", "2"));
            using var extractor = new FrozenFeatureExtractor(File.ReadAllBytes(source), new() { ImageSize = size, CpuThreads = 4 });
            string[] names = extractor.Model.Names.OrderBy(p => p.Key).Select(p => p.Value).ToArray();
            var initialHead = new DetectionHead(extractor.Model.DetectionHead, names, extractor.Model.Names, 26);
            var parameters = DescribeState(initialHead, directory, "initial");
            var images = new List<object>(); int index = 0;
            foreach (var image in datasetDocument.RootElement.GetProperty("images").EnumerateArray())
            {
                string rgb = image.GetProperty("rgbPath").GetString()!, labels = image.GetProperty("labels").GetString()!;
                int width = image.GetProperty("width").GetInt32(), height = image.GetProperty("height").GetInt32();
                var (input, transform) = ImagePreprocessor.Letterbox(File.ReadAllBytes(rgb), width, height, size, size);
                var features = extractor.Extract(input); string id = image.GetProperty("id").GetString()!;
                var truth = TrainingDataset.ParseLabels(labels, names.Length, transform);
                images.Add(new { Id = id, Split = image.GetProperty("split").GetString(), RgbPath = rgb, Width = width, Height = height, Labels = labels, Truth = truth,
                    Input = Write(directory, $"image-{index}-input.f32", input), Features = features.Select((t, i) => Write(directory, $"image-{index}-feature-{i}.f32", t)).ToArray() });
                index++;
            }
            int count = datasetDocument.RootElement.GetProperty("images").EnumerateArray().Count(i => i.GetProperty("split").GetString() == "train");
            int[][] order = Enumerable.Range(0, epochs).Select(epoch => { int[] v = Enumerable.Range(0, count).ToArray(); new Random(unchecked(26 + epoch * 104729)).Shuffle(v); return v; }).ToArray();
            var fixture = new { SourceModel = source, SourceSha256 = Hash(source), ClassNames = names, Strides = initialHead.Strides, RegMax = initialHead.Schema.RegMax, HeadKind = initialHead.Schema.Kind,
                Size = size, Epochs = epochs, LearningRate = .0001f, Momentum = .9f, WeightDecay = .0005f, Seed = 26, Threads = 4, TrainOrder = order, Parameters = parameters, Images = images,
                DatasetSha256 = Hash(Option("--dataset", ".cache/training-011/dataset.json")), CoreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), TrainingAssemblySha256 = Hash(typeof(YoloTrainer).Assembly.Location) };
            File.WriteAllText(Path.Combine(directory, "fixture.json"), JsonSerializer.Serialize(fixture, Json));
            Console.WriteLine($"Prepared {Path.GetFileName(source)}: {count} train / {index - count} validation images.");
            return;
        }
        string fixtureDirectory = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]); Directory.CreateDirectory(output);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixtureDirectory, "fixture.json"))); var f = document.RootElement;
        Tensor Read(JsonElement d) => new(MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(fixtureDirectory, d.GetProperty("path").GetString()!))).ToArray(), d.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray());
        var allImages = f.GetProperty("images").EnumerateArray().ToArray(); var train = allImages.Where(i => i.GetProperty("split").GetString() == "train").ToArray();
        var dataset = new TrainingDataset(f.GetProperty("classNames").EnumerateArray().Select(n => n.GetString()!), train.Select(i => new TrainingImage(i.GetProperty("id").GetString()!, File.ReadAllBytes(i.GetProperty("rgbPath").GetString()!), i.GetProperty("width").GetInt32(), i.GetProperty("height").GetInt32(), i.GetProperty("labels").GetString()!)));
        var options = new TrainingOptions { ImageSize = f.GetProperty("size").GetInt32(), Epochs = f.GetProperty("epochs").GetInt32(), LearningRate = f.GetProperty("learningRate").GetSingle(), Momentum = f.GetProperty("momentum").GetSingle(), WeightDecay = f.GetProperty("weightDecay").GetSingle(), Seed = f.GetProperty("seed").GetInt32(), CpuThreads = 4, Device = args.Contains("--vulkan") ? ComputeDevice.Vulkan : ComputeDevice.Cpu };
        using var trainer = new YoloTrainer(f.GetProperty("sourceModel").GetString()!, dataset, options);
        // 测试探针只访问本程序集已知字段，使两边使用逐字节相同的冻结特征，并仍执行公开 Train/Save/Load 路径。
        var headField = typeof(YoloTrainer).GetField("head", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var head = (DetectionHead)headField.GetValue(trainer)!;
        head.Restore(f.GetProperty("parameters").EnumerateArray().ToDictionary(p => p.GetProperty("name").GetString()!, Read));
        var cache = (Dictionary<int, Tensor[]>)typeof(YoloTrainer).GetField("featureCache", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(trainer)!;
        for (int i = 0; i < train.Length; i++) cache[i] = train[i].GetProperty("features").EnumerateArray().Select(Read).ToArray();
        var history = new List<TrainingProgress>(); var snapshots = new List<object>();
        var initialPredictions = Evaluate("before", null);
        trainer.Train(Path.Combine(output, "native"), p =>
        {
            history.Add(p);
            if (p.Step == 1 || p.Step == 16 || p.Step % train.Length == 0) snapshots.Add(new { p.Step, Parameters = DescribeState(head, output, "step-" + p.Step) });
        });
        string native = Path.Combine(output, "native/last.sytrain");
        using var detector = new YoloDetector(native, new() { Device = options.Device, CpuThreads = 4, Width = options.ImageSize, Height = options.ImageSize });
        var finalPredictions = Evaluate("after", detector);
        var info = ModelInfo.Read(native);
        if (info.ClassNames.Count != dataset.ClassNames.Count || trainer.CompletedSteps != train.Length * options.Epochs) throw new Exception("保存模型的类别或训练步数不符。");
        File.WriteAllText(Path.Combine(output, "outcome.json"), JsonSerializer.Serialize(new { Status = "completed", Device = options.Device.ToString(), Steps = trainer.CompletedSteps, FixtureSha256 = Hash(Path.Combine(fixtureDirectory, "fixture.json")), History = history, Snapshots = snapshots,
            Before = initialPredictions, After = finalPredictions, NativeCheckpointSha256 = Hash(native), ClassNames = info.ClassNames, CoreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), TrainingAssemblySha256 = Hash(typeof(YoloTrainer).Assembly.Location) }, Json));
        Console.WriteLine($"Completed {Path.GetFileName(f.GetProperty("sourceModel").GetString())} {options.Device}: {trainer.CompletedSteps} updates, native validation predictions saved.");

        object[] Evaluate(string stage, YoloDetector? saved)
        {
            using IComputeBackend backend = options.Device == ComputeDevice.Vulkan ? new VulkanBackend() : new CpuBackend(4);
            var ops = new HeadOperations(backend, 4); var results = new List<object>();
            foreach (var i in allImages.Where(i => i.GetProperty("split").GetString() == "val"))
            {
                string id = i.GetProperty("id").GetString()!; var features = i.GetProperty("features").EnumerateArray().Select(Read).ToArray();
                var tape = new GradientTape(); var values = head.Forward(features, ops, tape, false, head.Schema.EndToEnd);
                var boxes = values.Boxes.Select((v, n) => Write(output, $"{stage}-{id}-boxes-{n}.f32", v.Value)).ToArray();
                var classes = values.Classes.Select((v, n) => Write(output, $"{stage}-{id}-classes-{n}.f32", v.Value)).ToArray();
                tape.Backward(); // 未连接梯度，只归还卷积临时矩阵，不改变参数或 BN。
                object? nativeRaw = saved is null ? null : Write(output, $"{stage}-{id}-native.f32", saved.PredictRaw(Read(i.GetProperty("input")))[0]);
                results.Add(new { Id = id, Boxes = boxes, Classes = classes, Native = nativeRaw });
            }
            return results.ToArray();
        }
    }

    private static object[] DescribeState(DetectionHead head, string directory, string prefix)
    {
        return head.Snapshot().Select((p, i) => new { Name = p.Key, Path = ((TensorFile)Write(directory, $"{prefix}-parameter-{i}.f32", p.Value)).Path, Shape = p.Value.Shape,
            VelocityPath = head.Parameters.FirstOrDefault(v => v.Name == p.Key) is { } parameter ? ((TensorFile)Write(directory, $"{prefix}-velocity-{i}.f32", new Tensor((float[])parameter.Velocity.Clone(), parameter.Shape))).Path : null }).Cast<object>().ToArray();
    }

    private sealed record TensorFile(string Path, int[] Shape);
    private static object Write(string directory, string name, Tensor value)
    { File.WriteAllBytes(Path.Combine(directory, name), MemoryMarshal.AsBytes(value.Data.AsSpan()).ToArray()); return new TensorFile(name, value.Shape); }
    private static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
}
