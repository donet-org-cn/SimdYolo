using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using SimdYolo;
using SimdYolo.Training;

/// <summary>对五代真实检测权重执行独立对照夹具、完整训练生命周期与缓存特征性能测量。</summary>
internal static class TrainingMatrix
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { WriteIndented = true };
    private static readonly string[] Models = ["yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo26n"];
    internal static void Run(string[] args)
    {
        string Option(string key, string fallback) { int i = Array.IndexOf(args, key); return i >= 0 ? args[i + 1] : fallback; }
        string directory = Path.GetFullPath(args[1]); Directory.CreateDirectory(directory);
        var device = args.Contains("--vulkan") ? ComputeDevice.Vulkan : ComputeDevice.Cpu;
        if (args[0] == "test-matrix") { Integration(directory, device, int.Parse(Option("--epochs", "100"))); return; }
        string source = Path.GetFullPath(Option("--model", "models/pt/yolov8n.pt")); int size = int.Parse(Option("--size", "64"));
        int epoch = int.Parse(Option("--epoch", "0")), schedule = int.Parse(Option("--schedule-epochs", "5"));
        using var extractor = new FrozenFeatureExtractor(File.ReadAllBytes(source), new() { ImageSize = size, CpuThreads = int.Parse(Option("--threads", "4")), Device = device });
        var image = new Tensor(1, 3, size, size); var random = new Random(27); for (int i = 0; i < image.Length; i++) image.Data[i] = (float)random.NextDouble();
        var features = extractor.Extract(image); var head = new DetectionHead(extractor.Model.DetectionHead, ["object-a", "object-b"], extractor.Model.Names, 26);
        var operations = new HeadOperations(extractor.Backend, int.Parse(Option("--threads", "4")));
        float ratio = size / 64f;
        TrainingBox[] truth = args.Contains("--small-boxes") ? [new(0, 21, 21, 25, 25), new(1, 28, 26, 33, 31)] : args.Contains("--empty") ? [] : [new(0, 10 * ratio, 9 * ratio, 43 * ratio, 46 * ratio), new(1, 25 * ratio, 22 * ratio, 58 * ratio, 59 * ratio)];
        if (args[0] == "bench-head")
        {
            string fixturePath = Option("--fixture", "");
            if (fixturePath.Length != 0)
            {
                using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixturePath, "fixture.json"))); var benchmarkFixture = document.RootElement;
                if (Hash(source) != Hash(benchmarkFixture.GetProperty("sourceModel").GetString()!)) throw new InvalidDataException("性能夹具模型不同。");
                Tensor Read(JsonElement d) => new(MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(fixturePath, d.GetProperty("path").GetString()!))).ToArray(), d.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
                features = benchmarkFixture.GetProperty("features").EnumerateArray().Select(Read).ToArray();
                head.Restore(benchmarkFixture.GetProperty("parameters").EnumerateArray().ToDictionary(d => d.GetProperty("name").GetString()!, Read));
                truth = JsonSerializer.Deserialize<TrainingBox[]>(benchmarkFixture.GetProperty("truth"), Json)!;
            }
            Benchmark(directory, source, features, head, operations, truth, device, size, int.Parse(Option("--threads", "4")), int.Parse(Option("--steps", "128")), fixturePath); return;
        }
        var before = head.Snapshot(); var tape = new GradientTape(); var branches = new List<object>();
        int group = 0; float manyWeight = head.Schema.Progressive ? DetectionHead.ProgressiveWeight(epoch, schedule) : 1;
        foreach (bool one in head.Schema.EndToEnd ? new[] { false, true } : [false])
        {
            var values = head.Forward(features, operations, tape, true, one);
            int topk = one ? head.Schema.Progressive ? 7 : 1 : 10, topk2 = one ? 1 : 10;
            float weight = one && head.Schema.Progressive ? 1 - manyWeight : manyWeight;
            var loss = DetectionLoss.Compute(values.Boxes, values.Classes, head.Strides, truth, 2, true, head.Schema.RegMax, topk, topk2, weight);
            branches.Add(new { Name = one ? "one2one" : "one2many", Prefix = one ? "one2one_" : "", Weight = weight, Topk = topk, Topk2 = topk2,
                Boxes = values.Boxes.Select((v, i) => Describe(v, $"branch-{group}-boxes-{i}")).ToArray(), Classes = values.Classes.Select((v, i) => Describe(v, $"branch-{group}-classes-{i}")).ToArray(),
                Loss = new { loss.Box, loss.Classification, loss.Dfl, loss.Total, loss.Positives } });
            group++;
        }
        tape.Backward(); var after = head.Snapshot();
        var descriptors = before.Select((p, i) => new { Name = p.Key, Shape = p.Value.Shape, Path = Write($"parameter-{i}.f32", p.Value.Data),
            Connected = head.Parameters.FirstOrDefault(v => v.Name == p.Key)?.HasGradient,
            GradientPath = head.Parameters.FirstOrDefault(v => v.Name == p.Key) is { } parameter ? Write($"gradient-{i}.f32", parameter.Gradient) : null,
            AfterPath = Write($"after-{i}.f32", after[p.Key].Data) }).ToArray();
        var optimizer = new TrainingOptions { LearningRate = .003f, Momentum = .9f, WeightDecay = .0005f };
        foreach (var parameter in head.Parameters) parameter.Update(optimizer, 1);
        var updated = head.Parameters.Select((p, i) => new { Name = p.Name, Shape = p.Shape, Path = Write($"sgd-parameter-{i}.f32", p.Values), VelocityPath = Write($"sgd-velocity-{i}.f32", p.Velocity), p.Decay }).ToArray();
        var fixture = new { SourceModel = source, ClassNames = new[] { "object-a", "object-b" }, Strides = head.Strides, RegMax = head.Schema.RegMax, HeadKind = head.Schema.Kind, Epoch = epoch, ScheduleEpochs = schedule, Device = device.ToString(), Truth = truth, SgdParameters = updated,
            Optimizer = new { optimizer.LearningRate, optimizer.Momentum, optimizer.WeightDecay },
            Features = features.Select((t, i) => new { Path = Write($"feature-{i}.f32", t.Data), Shape = t.Shape }).ToArray(), Parameters = descriptors, Branches = branches,
            CoreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), TrainingAssemblySha256 = Hash(typeof(YoloTrainer).Assembly.Location) };
        File.WriteAllText(Path.Combine(directory, "fixture.json"), JsonSerializer.Serialize(fixture, Json)); Console.WriteLine($"Emitted {Path.GetFileName(source)} {device} {head.Schema.Kind}, {descriptors.Length} tensors.");
        object Describe(GradientValue value, string name) => new { Path = Write(name + ".f32", value.Value.Data), Shape = value.Value.Shape, GradientPath = Write(name + "-gradient.f32", value.GradientOrZero) };
        string Write(string name, float[] data) { File.WriteAllBytes(Path.Combine(directory, name), MemoryMarshal.AsBytes(data.AsSpan()).ToArray()); return name; }
    }

    private static void Integration(string directory, ComputeDevice device, int epochs)
    {
        var results = new List<object>(); var dataset = TrainingDataset.CreateDemo(1, 26);
        foreach (string model in Models)
        {
            string source = $"models/pt/{model}.pt", output = Path.Combine(directory, model); var options = new TrainingOptions { ImageSize = 64, Epochs = epochs, CpuThreads = 4, LearningRate = .005f, Device = device };
            if (!YoloTrainer.CanTrain(source)) throw new Exception("Capability probe failed: " + model);
            using var trainer = new YoloTrainer(source, dataset, options); float before = trainer.EvaluateLoss(); trainer.SaveCheckpoint(output + "-before.sytrain");
            var history = new List<TrainingProgress>(); trainer.Train(output, p => { history.Add(p); if (p.Step % 25 == 0) Console.WriteLine($"{model} {device} {p.Step}: {p.TotalLoss:F5}"); });
            float after = trainer.EvaluateLoss(); if (!(after < before * .9f)) throw new Exception($"{model} {before} -> {after}: loss did not decrease 10%.");
            string checkpoint = Path.Combine(output, "last.sytrain"); var start = TrainingModelState.Load(output + "-before.sytrain"); var state = TrainingModelState.Load(checkpoint);
            foreach (string prefix in state.Manifest.EndToEnd ? new[] { "cv2", "cv3", "one2one_cv2", "one2one_cv3" } : ["cv2", "cv3"])
                if (!state.Parameters.Any(p => p.Key.StartsWith(prefix + ".", StringComparison.Ordinal) && p.Key.EndsWith("weight", StringComparison.Ordinal) && !p.Value.Data.SequenceEqual(start.Parameters[p.Key].Data))) throw new Exception("Branch never updated: " + prefix);
            var image = dataset.Images[0]; var (input, _) = ImagePreprocessor.Letterbox(image.Pixels, image.Width, image.Height, 64, 64);
            var info = ModelInfo.Read(checkpoint); if (info.EndToEnd != state.Manifest.EndToEnd || info.ClassNames[0] != "square") throw new Exception("Metadata mismatch.");
            using var cpu = new YoloDetector(checkpoint, new() { Device = ComputeDevice.Cpu }); var expected = cpu.PredictRaw(input)[0];
            if (expected.Data.Any(v => !float.IsFinite(v)) || (info.EndToEnd ? expected.Shape[^1] != 6 : expected.Shape[1] != 5)) throw new Exception("Invalid native inference output.");
            float gpuError = 0;
            if (device == ComputeDevice.Vulkan)
            {
                using var gpu = new YoloDetector(checkpoint, new() { Device = ComputeDevice.Vulkan }); var actual = gpu.PredictRaw(input)[0];
                if (!actual.Shape.SequenceEqual(expected.Shape)) throw new Exception("CPU/GPU output shape mismatch.");
                // 比较带原始顺序的 TopK；若分数相等导致顺序变化，逐行按坐标匹配。
                if (info.EndToEnd)
                {
                    for (int i = 0; i < expected.Shape[1]; i++)
                    {
                        float best = float.PositiveInfinity;
                        for (int j = 0; j < actual.Shape[1]; j++) { float delta = 0; for (int k = 0; k < 6; k++) delta = MathF.Max(delta, MathF.Abs(expected.Data[i * 6 + k] - actual.Data[j * 6 + k])); best = MathF.Min(best, delta); }
                        gpuError = MathF.Max(gpuError, best);
                    }
                }
                else for (int i = 0; i < expected.Length; i++) gpuError = MathF.Max(gpuError, MathF.Abs(expected.Data[i] - actual.Data[i]));
                if (gpuError > .01f) throw new Exception("CPU/GPU native inference difference: " + gpuError);
            }
            // 中断时保留原始调度总轮数，恢复与同一调度的连续运行比较参数、BN 和动量。
            string a = output + "-continuous", b = output + "-resumed";
            using (var continuous = new YoloTrainer(source, dataset, options with { Epochs = 3 })) continuous.Train(a);
            using (var interrupted = new YoloTrainer(source, dataset, options with { Epochs = 3 }))
            {
                using var cancel = new CancellationTokenSource();
                try { interrupted.Train(b, p => { if (p.Step == 1) cancel.Cancel(); }, cancel.Token); throw new Exception("Cancellation not observed."); } catch (OperationCanceledException) { }
            }
            using (var resumed = YoloTrainer.Resume(b + "/last.sytrain", dataset, 3, device, 4))
            {
                resumed.SaveCheckpoint(b + "/restored.sytrain");
                using (var saved = ZipFile.OpenRead(b + "/last.sytrain")) using (var loaded = ZipFile.OpenRead(b + "/restored.sytrain"))
                    foreach (var entry in saved.Entries.Where(e => e.FullName.StartsWith("optimizer/", StringComparison.Ordinal)))
                        if (!TrainingModelState.ReadEntry(saved, entry.FullName, 80_000_000).SequenceEqual(TrainingModelState.ReadEntry(loaded, entry.FullName, 80_000_000))) throw new Exception("Loading changed momentum bytes.");
                resumed.Train(b);
            }
            var first = TrainingModelState.Load(a + "/last.sytrain"); var second = TrainingModelState.Load(b + "/last.sytrain"); float resumeError = 0;
            foreach (var pair in first.Parameters) for (int i = 0; i < pair.Value.Length; i++) resumeError = MathF.Max(resumeError, MathF.Abs(pair.Value.Data[i] - second.Parameters[pair.Key].Data[i]));
            if (resumeError > 2e-5f || second.Manifest.LossScheduleEpochs != 3) throw new Exception("Resume changed training state or loss schedule.");
            float momentumError = 0;
            using (var az = ZipFile.OpenRead(a + "/last.sytrain")) using (var bz = ZipFile.OpenRead(b + "/last.sytrain"))
                foreach (var entry in az.Entries.Where(e => e.FullName.StartsWith("optimizer/", StringComparison.Ordinal)))
                {
                    float[] av = MemoryMarshal.Cast<byte, float>(TrainingModelState.ReadEntry(az, entry.FullName, 80_000_000)).ToArray(), bv = MemoryMarshal.Cast<byte, float>(TrainingModelState.ReadEntry(bz, entry.FullName, 80_000_000)).ToArray();
                    for (int i = 0; i < av.Length; i++) { float error = MathF.Abs(av[i] - bv[i]); momentumError = MathF.Max(momentumError, error); if (error > 5e-4f + 2e-3f * MathF.Abs(av[i])) throw new Exception("Resumed SGD momentum differs beyond numerical tolerance."); }
                }
            results.Add(new { Model = model, Device = device.ToString(), Before = before, After = after, Epochs = epochs, Steps = trainer.CompletedSteps, AllBranchesUpdated = true, ResumeMaxAbs = resumeError, ResumeMomentumMaxAbs = momentumError, LoadedMomentumBitwiseEqual = true, NativeInference = true, NativeGpuMaxAbs = gpuError, CheckpointSha256 = Hash(checkpoint), History = history, Passed = true });
            File.WriteAllText(Path.Combine(directory, "integration.json"), JsonSerializer.Serialize(new { Status = results.Count == Models.Length ? "passed" : "running", CoreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), TrainingAssemblySha256 = Hash(typeof(YoloTrainer).Assembly.Location), Results = results }, Json));
            Console.WriteLine($"PASS {model} {device}: {before:F6} -> {after:F6}; resume max {resumeError}");
        }
        foreach (string model in new[] { "yolov8n-seg", "yolo11n-pose", "yolo26n-seg" }) if (YoloTrainer.CanTrain($"models/pt/{model}.pt")) throw new Exception("Unsupported task advertised as trainable.");
        if (YoloTrainer.CanTrain("models/onnx/yolo26n.onnx")) throw new Exception("Inference ONNX advertised as trainable.");
    }

    private static void Benchmark(string directory, string source, Tensor[] features, DetectionHead head, HeadOperations operations, TrainingBox[] truth, ComputeDevice device, int size, int threads, int steps, string fixturePath)
    {
        var options = new TrainingOptions { LearningRate = .0001f, CpuThreads = threads }; var durations = new List<double>();
        void Step() { var tape = new GradientTape(); var loss = head.Loss(features, operations, tape, truth, true, 0, 5); if (!float.IsFinite(loss.Total)) throw new ArithmeticException(); tape.Backward(); foreach (var p in head.Parameters) p.Update(options, 1); }
        for (int i = 0; i < 10; i++) Step();
        File.WriteAllText(Path.Combine(directory, "measuring"), "ready");
        using var process = Process.GetCurrentProcess(); var cpu = process.TotalProcessorTime; long allocated = GC.GetTotalAllocatedBytes(); var window = Stopwatch.StartNew();
        for (int i = 0; i < steps; i++) { long start = Stopwatch.GetTimestamp(); Step(); durations.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); }
        double elapsed = window.Elapsed.TotalMilliseconds; File.WriteAllText(Path.Combine(directory, "measured"), "done"); process.Refresh();
        durations.Sort(); var report = new { Engine = "SimdYolo", Model = Path.GetFileName(source), SourceSha256 = Hash(source), Device = device.ToString(), Size = size, Threads = threads, Warmup = 10, Steps = durations.Count,
            MedianMs = (durations[(durations.Count - 1) / 2] + durations[durations.Count / 2]) * .5, MeanMs = durations.Average(), P95Ms = durations[(int)(durations.Count * .95)], CpuPercentOfMachine = (process.TotalProcessorTime - cpu).TotalMilliseconds / elapsed / Environment.ProcessorCount * 100,
            WorkingSetMiB = process.WorkingSet64 / 1048576d, AllocatedMiB = (GC.GetTotalAllocatedBytes() - allocated) / 1048576d,
            CoreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), TrainingAssemblySha256 = Hash(typeof(YoloTrainer).Assembly.Location), FixtureSha256 = fixturePath.Length == 0 ? null : Hash(Path.Combine(fixturePath, "fixture.json")), Scope = "cached frozen features; head forward+loss+backward+SGD; excludes backbone, preprocessing, checkpoint IO", Passed = true };
        File.WriteAllText(Path.Combine(directory, "benchmark.json"), JsonSerializer.Serialize(report, Json)); Console.WriteLine(JsonSerializer.Serialize(report));
    }

    private static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
}
