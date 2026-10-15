using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text.Json;
using SimdYolo;
using SimdYolo.Training;

var json = new JsonSerializerOptions(JsonSerializerDefaults.Web) { WriteIndented = true };
if (args.FirstOrDefault() == "emit-oracle")
{
    string directory = Path.GetFullPath(args[1]); Directory.CreateDirectory(directory);
    using var extractor = new FrozenFeatureExtractor(File.ReadAllBytes("models/pt/yolov8n.pt"), new() { ImageSize = 64, CpuThreads = 2 });
    var image = new Tensor(1, 3, 64, 64); var rng = new Random(27); for (int i = 0; i < image.Length; i++) image.Data[i] = (float)rng.NextDouble();
    Tensor[] features = extractor.Extract(image); var head = new DetectionHead(extractor.Model.DetectionHead, ["object-a", "object-b"], extractor.Model.Names, 26);
    var before = head.Snapshot(); var tape = new GradientTape();
    using IComputeBackend backend = args.Contains("--vulkan") ? new VulkanBackend() : new CpuBackend(2);
    var values = head.Forward(features, new HeadOperations(backend), tape, true);
    TrainingBox[] truth = args.Contains("--small-boxes") ? [new(0, 21, 21, 25, 25), new(1, 28, 26, 33, 31)] : args.Contains("--empty") ? [] : [new(0, 10, 9, 43, 46), new(1, 25, 22, 58, 59)];
    var loss = DetectionLoss.Compute(values.Boxes, values.Classes, head.Strides, truth, 2); tape.Backward();
    var after = head.Snapshot();
    var descriptors = before.Select((p, i) =>
    {
        var parameter = head.Parameters.FirstOrDefault(v => v.Name == p.Key);
        return new { Name = p.Key, Shape = p.Value.Shape, Path = Write($"parameter-{i}.f32", p.Value.Data),
            GradientPath = parameter is null ? null : Write($"gradient-{i}.f32", parameter.Gradient), AfterPath = Write($"after-{i}.f32", after[p.Key].Data) };
    }).ToArray();
    var fixture = new { SourceModel = Path.GetFullPath("models/pt/yolov8n.pt"), ClassNames = new[] { "object-a", "object-b" }, Strides = head.Strides, Truth = truth,
        Features = features.Select((t, i) => new { Path = Write($"feature-{i}.f32", t.Data), Shape = t.Shape }).ToArray(), Parameters = descriptors,
        Boxes = values.Boxes.Select((t, i) => new { Path = Write($"boxes-{i}.f32", t.Value.Data), Shape = t.Value.Shape, GradientPath = Write($"boxes-gradient-{i}.f32", t.Gradient) }).ToArray(),
        Classes = values.Classes.Select((t, i) => new { Path = Write($"classes-{i}.f32", t.Value.Data), Shape = t.Value.Shape, GradientPath = Write($"classes-gradient-{i}.f32", t.Gradient) }).ToArray(),
        Loss = new { loss.Box, loss.Classification, loss.Dfl, loss.Total, loss.Positives },
        CoreAssemblySha256 = Hash(typeof(YoloDetector).Assembly.Location), TrainingAssemblySha256 = Hash(typeof(YoloTrainer).Assembly.Location) };
    File.WriteAllText(Path.Combine(directory, "fixture.json"), JsonSerializer.Serialize(fixture, json)); Console.WriteLine(JsonSerializer.Serialize(fixture.Loss)); return;
    string Write(string path, float[] data) { File.WriteAllBytes(Path.Combine(directory, path), MemoryMarshal.AsBytes(data.AsSpan()).ToArray()); return path; }
    static string Hash(string path) => Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(path)));
}

int passed = 0;
Run("卷积权重和输入梯度的独立有限差分", () =>
{
    using var backend = new CpuBackend(2); var ops = new HeadOperations(backend); var tape = new GradientTape();
    var input = new GradientValue(new Tensor(Enumerable.Range(0, 40).Select(i => MathF.Sin(i * .31f)).ToArray(), 1, 2, 4, 5));
    var weight = new TrainingParameter("w", Enumerable.Range(0, 54).Select(i => MathF.Cos(i * .23f) * .1f).ToArray(), [3, 2, 3, 3], true);
    var output = ops.Conv(tape, input, weight, null, 1); float[] upstream = Enumerable.Range(0, 60).Select(i => MathF.Sin(i * .17f)).ToArray(); upstream.CopyTo(output.Gradient, 0); tape.Backward();
    float Objective() { var y = ops.Conv(new(), new(input.Value), weight, null, 1).Value.Data; return y.Select((v, i) => v * upstream[i]).Sum(); }
    foreach (int index in new[] { 0, 13, 53 })
    { float saved = weight.Values[index]; const float e = .002f; weight.Values[index] = saved + e; float a = Objective(); weight.Values[index] = saved - e; float b = Objective(); weight.Values[index] = saved; Near(weight.Gradient[index], (a - b) / (2 * e), .002f); }
    foreach (int index in new[] { 0, 17, 39 })
    { float saved = input.Value.Data[index]; const float e = .002f; input.Value.Data[index] = saved + e; float a = Objective(); input.Value.Data[index] = saved - e; float b = Objective(); input.Value.Data[index] = saved; Near(input.Gradient[index], (a - b) / (2 * e), .002f); }
});
Run("SiLU 极值和有限差分", () =>
{
    float[] data = [-30, -2, -.3f, 0, .7f, 5, 30]; var x = new GradientValue(new Tensor(data, 1, 1, 1, data.Length)); var tape = new GradientTape(); var y = HeadOperations.Silu(tape, x, true); Array.Fill(y.Gradient, 1); tape.Backward();
    for (int i = 0; i < data.Length; i++) { float a = data[i] + .002f, b = data[i] - .002f; Near(x.Gradient[i], (a * HeadOperations.Sigmoid(a) - b * HeadOperations.Sigmoid(b)) / .004f, .002f); }
});
Run("BatchNorm 参数和输入反向、运行方差", () =>
{
    using var backend = new CpuBackend(1); var ops = new HeadOperations(backend);
    var data = new Tensor(Enumerable.Range(0, 24).Select(i => MathF.Sin(i * .7f)).ToArray(), 1, 2, 3, 4); var input = new GradientValue(data);
    var gamma = new TrainingParameter("gamma", [1.2f, .8f], [2], false); var beta = new TrainingParameter("beta", [.1f, -.2f], [2], false);
    var tape = new GradientTape(); var mean = new float[2]; var variance = new float[] { 1, 1 }; var output = ops.BatchNorm(tape, input, gamma, beta, mean, variance, .001f, .03f, true);
    float[] dy = Enumerable.Range(0, 24).Select(i => MathF.Cos(i * .2f)).ToArray(); dy.CopyTo(output.Gradient, 0); tape.Backward();
    float Objective() { var y = ops.BatchNorm(new(), new(data), gamma, beta, new float[2], [1, 1], .001f, .03f, true); return y.Value.Data.Select((v, i) => v * dy[i]).Sum(); }
    foreach (int index in new[] { 0, 13, 23 }) { float saved = data.Data[index]; data.Data[index] = saved + .002f; float a = Objective(); data.Data[index] = saved - .002f; float b = Objective(); data.Data[index] = saved; Near(input.Gradient[index], (a - b) / .004f, .002f); }
    for (int c = 0; c < 2; c++) Near(beta.Gradient[c], dy.Skip(c * 12).Take(12).Sum(), 1e-5f);
    if (variance.Any(v => !float.IsFinite(v) || v <= 0)) throw new Exception("BN 运行方差无效。");
});
Run("SGD 动量和权重衰减", () =>
{
    var parameter = new TrainingParameter("weight", [2], [1], true); var options = new TrainingOptions { LearningRate = .1f, Momentum = .9f, WeightDecay = .01f };
    parameter.Gradient[0] = 4; parameter.Update(options, 2); Near(parameter.Values[0], 1.798f, 1e-6f); Near(parameter.Velocity[0], 2.02f, 1e-6f);
    parameter.Gradient[0] = 2; parameter.Update(options, 1); Near(parameter.Values[0], 1.414402f, 1e-6f);
});
Run("空标签背景样本仍产生分类梯度", () =>
{
    var boxes = new[] { new GradientValue(new Tensor(1, 64, 2, 2)) }; var classes = new[] { new GradientValue(new Tensor(1, 1, 2, 2)) };
    var loss = DetectionLoss.Compute(boxes, classes, [8], [], 1); Near(loss.Classification, MathF.Log(2) * 2, 1e-6f); Near(classes[0].Gradient[0], .25f, 1e-6f); if (loss.Positives != 0 || loss.Box != 0 || loss.Dfl != 0) throw new Exception("背景样本产生正样本。");
});
Run("数据集拒绝坏标签和配置", () =>
{
    MustThrow<InvalidDataException>(() => new TrainingDataset(["x"], [new("bad", new byte[12], 2, 2, "1 .5 .5 .2 .2")]));
    MustThrow<ArgumentException>(() => new TrainingDataset(["x"], [new("bad", new byte[11], 2, 2, "")]));
    MustThrow<ArgumentException>(() => new TrainingOptions { ImageSize = 63 }.Validate());
    MustThrow<NotSupportedException>(() => new TrainingOptions { Device = ComputeDevice.Metal }.Validate());
});

if (args.Contains("--integration"))
{
    string output = ".cache/training-validation/integration"; Directory.CreateDirectory(output); var dataset = TrainingDataset.CreateDemo(1, 26);
    float before, after; var history = new List<TrainingProgress>();
    Run("真实 YOLOv8 微调损失下降与快照推理", () =>
    {
        using var trainer = new YoloTrainer("models/pt/yolov8n.pt", dataset, new() { ImageSize = 64, Epochs = 100, CpuThreads = 4, LearningRate = .005f });
        before = trainer.EvaluateLoss(); trainer.Train(output, item => { history.Add(item); if (item.Step % 10 == 0) Console.WriteLine($"step {item.Step}: {item.TotalLoss:F5}"); }); after = trainer.EvaluateLoss();
        Console.WriteLine($"eval loss {before:F6} -> {after:F6}"); if (!(after < before * .9f)) throw new Exception("真实微调没有达到10%损失下降。");
        using var detector = new YoloDetector(Path.Combine(output, "last.sytrain"), new() { Width = 64, Height = 64 });
        var image = dataset.Images[0]; var (input, _) = ImagePreprocessor.Letterbox(image.Pixels, image.Width, image.Height, 64, 64); var raw = detector.PredictRaw(input);
        if (raw[0].Shape[1] != 5 || raw[0].Data.Any(v => !float.IsFinite(v)) || detector.ClassNames[0] != "square") throw new Exception("训练模型推理或类别无效。");
        var info = ModelInfo.Read(Path.Combine(output, "last.sytrain")); if (info.ClassCount != 1 || info.Format != "SYTRAIN") throw new Exception("训练模型元数据无效。");
        File.WriteAllText(Path.Combine(output, "summary.json"), JsonSerializer.Serialize(new { Before = before, After = after, trainer.CompletedEpochs, trainer.CompletedSteps, Passed = true }, json));
    });
    Run("SGD 断点恢复与连续训练一致", () =>
    {
        var options = new TrainingOptions { ImageSize = 64, Epochs = 2, CpuThreads = 2, LearningRate = .003f };
        using (var continuous = new YoloTrainer("models/pt/yolov8n.pt", dataset, options)) continuous.Train(output + "-continuous");
        using (var first = new YoloTrainer("models/pt/yolov8n.pt", dataset, options with { Epochs = 1 })) first.Train(output + "-resume");
        using (var resumed = YoloTrainer.Resume(output + "-resume/last.sytrain", dataset, 2, cpuThreads: 2)) resumed.Train(output + "-resume");
        var expected = TrainingModelState.Load(output + "-continuous/last.sytrain"); var actual = TrainingModelState.Load(output + "-resume/last.sytrain");
        foreach (var pair in expected.Parameters) for (int i = 0; i < pair.Value.Length; i++) Near(actual.Parameters[pair.Key].Data[i], pair.Value.Data[i], 2e-5f);
        MustThrow<InvalidDataException>(() => YoloTrainer.Resume(output + "-resume/last.sytrain", TrainingDataset.CreateDemo(1, 99), 3));
    });
    Run("取消保存最近完整轮次", () =>
    {
        using var trainer = new YoloTrainer("models/pt/yolov8n.pt", dataset, new() { ImageSize = 64, Epochs = 3, CpuThreads = 2 }); using var cancel = new CancellationTokenSource();
        MustThrow<OperationCanceledException>(() => trainer.Train(output + "-cancel", _ => cancel.Cancel(), cancel.Token));
        var saved = TrainingModelState.Load(output + "-cancel/last.sytrain"); if (saved.Manifest.Epoch != 1) throw new Exception("取消没有保留最近完整轮次。");
        if (trainer.CompletedSteps != saved.Manifest.Step || trainer.CompletedEpochs != saved.Manifest.Epoch) throw new Exception("取消后的会话与检查点边界不同。");
    });
    Run("训练会话拒绝并发重入且从输出错误中恢复", () =>
    {
        using var trainer = new YoloTrainer("models/pt/yolov8n.pt", dataset, new() { ImageSize = 64, Epochs = 1, CpuThreads = 2 });
        string blocked = output + "/not-a-directory"; File.WriteAllText(blocked, "test");
        MustThrow<IOException>(() => trainer.Train(blocked));
        if (!float.IsFinite(trainer.EvaluateLoss())) throw new Exception("输出错误使会话卡在运行状态。");
        trainer.Train(output + "-reentry", _ =>
        {
            Task.Run(() => MustThrow<InvalidOperationException>(() => trainer.EvaluateLoss())).GetAwaiter().GetResult();
            MustThrow<InvalidOperationException>(() => trainer.Train(output + "-reentry"));
            MustThrow<InvalidOperationException>(() => trainer.Dispose());
        });
    });
    Run("原生训练快照保留模型许可证并拒绝缺参和错误尺寸", () =>
    {
        string checkpoint = Path.Combine(output, "last.sytrain");
        using (var stream = File.OpenRead(checkpoint)) using (var archive = new ZipArchive(stream, ZipArchiveMode.Read))
            if (archive.GetEntry("LICENSE-AGPL-3.0.txt") is null || archive.GetEntry("source-provenance.json") is null) throw new Exception("官方模型来源或许可证丢失。");
        using var detector = new YoloDetector(checkpoint);
        MustThrow<ArgumentException>(() => detector.PredictRaw(new Tensor(1, 3, 96, 96)));
        string corrupt = Path.Combine(output, "missing-parameter.sytrain"); File.Copy(checkpoint, corrupt, true);
        using (var archive = ZipFile.Open(corrupt, ZipArchiveMode.Update)) archive.GetEntry("parameters/0.f32")!.Delete();
        MustThrow<InvalidDataException>(() => new YoloDetector(corrupt));
    });
    if (args.Contains("--vulkan")) Run("Vulkan 真实训练更新及原生快照 CPU/GPU 推理一致", () =>
    {
        using (var trainer = new YoloTrainer("models/pt/yolov8n.pt", dataset, new() { ImageSize = 64, Epochs = 2, CpuThreads = 2, Device = ComputeDevice.Vulkan })) trainer.Train(output + "-vulkan");
        using var cpu = new YoloDetector(output + "-vulkan/last.sytrain", new() { Device = ComputeDevice.Cpu });
        using var gpu = new YoloDetector(output + "-vulkan/last.sytrain", new() { Device = ComputeDevice.Vulkan });
        var image = dataset.Images[0]; var (input, _) = ImagePreprocessor.Letterbox(image.Pixels, image.Width, image.Height, 64, 64);
        float[] expected = cpu.PredictRaw(input)[0].Data, actual = gpu.PredictRaw(input)[0].Data;
        for (int i = 0; i < expected.Length; i++) Near(actual[i], expected[i], .001f);
    });
}
Console.WriteLine($"{passed} training tests passed.");
void Run(string name, Action test) { test(); passed++; Console.WriteLine("PASS " + name); }
static void Near(float actual, float expected, float tolerance) { if (!float.IsFinite(actual) || MathF.Abs(actual - expected) > tolerance * (1 + MathF.Abs(expected))) throw new Exception($"Expected {expected}, actual {actual}, tolerance {tolerance}"); }
static void MustThrow<T>(Action action) where T : Exception { try { action(); } catch (T) { return; } throw new Exception("Expected " + typeof(T).Name); }
