using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace SimdYolo.Training;

/// <summary>以纯 C# 微调 YOLOv8/v9/v10/YOLO11/YOLO26 检测头；冻结骨干与 FPN，使用真实反向传播和带动量 SGD。</summary>
public sealed class YoloTrainer : IDisposable
{
    private readonly TrainingDataset dataset;
    private readonly TrainingOptions options;
    private readonly byte[] source;
    private readonly FrozenFeatureExtractor extractor;
    private readonly DetectionHead head;
    private readonly HeadOperations operations;
    private readonly int lossScheduleEpochs;
    private readonly Dictionary<int, Tensor[]> featureCache = [];
    private readonly object gate = new();
    private long cachedBytes;
    private bool disposed, running;
    private int completedEpochs, completedSteps;
    private double? bestValidationMap;
    private int bestValidationEpoch, staleValidationEpochs, lastValidatedEpoch = -1;
    private string? resumeBestPath;
    /// <summary>本次是否因验证指标连续无提升而正常早停。</summary><value>默认假。</value>
    public bool StoppedEarly { get; private set; }
    /// <summary>最近完成轮次的验证摘要。</summary><value>未启用验证时为空。</value>
    public EpochValidation? LastValidation { get; private set; }

    /// <summary>当前冻结骨干及矩阵后端的实际名称。</summary><value>Vulkan 表示混合训练，BN、损失和 SGD 仍在 CPU。</value>
    public string BackendName => options.Device == ComputeDevice.Vulkan ? "Vulkan large GEMM + CPU SIMD small GEMM/depthwise/BN/loss/SGD" : extractor.Backend.Name;
    /// <summary>已经完整保存的训练轮数。</summary><value>取消恢复以最近完成轮次为边界。</value>
    public int CompletedEpochs => Volatile.Read(ref completedEpochs);
    /// <summary>已经完成的 SGD 更新次数。</summary><value>按微批次梯度累加后计数。</value>
    public int CompletedSteps => Volatile.Read(ref completedSteps);

    /// <summary>按 PT 的真实模块结构探测检测头微调能力，不依赖文件名、不加载权重或创建设备。</summary>
    /// <param name="modelPath">PT/PTH 或本项目训练归档路径；普通 ONNX 返回 false。</param>
    /// <returns>含完整受支持训练分支时返回 true；不支持的任务或头结构返回 false。</returns>
    /// <exception cref="IOException">文件无法读取或被截断。</exception><exception cref="InvalidDataException">ZIP 或元数据损坏、超限。</exception>
    /// <exception cref="ArgumentException">路径无效。</exception>
    public static bool CanTrain(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        string extension = Path.GetExtension(modelPath).ToLowerInvariant();
        if (extension is not (".pt" or ".pth" or ".sytrain")) return false;
        using var file = File.OpenRead(modelPath); using var zip = new ZipArchive(file, ZipArchiveMode.Read);
        if (extension == ".sytrain") { TrainingModelState.ReadManifest(zip); return true; }
        var entry = zip.Entries.SingleOrDefault(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal));
        if (entry is null) return false;
        if (entry.Length > 64 * 1024 * 1024) throw new InvalidDataException("PT 元数据超过64 MiB。");
        byte[] bytes = new byte[(int)entry.Length]; using (var stream = entry.Open()) stream.ReadExactly(bytes);
        object? root = new PickleReader(zip, entry.FullName[..^"data.pkl".Length], false).Load(bytes);
        if (root is Dictionary<object, object?> checkpoint) root = checkpoint.GetValueOrDefault("ema") ?? checkpoint.GetValueOrDefault("model");
        if (root is not PyObject model || !model.Has("model") || model.Child("model").Children.LastOrDefault() is not { } head) return false;
        try { TrainingHeadSchema.Read(head); return true; }
        catch (NotSupportedException) { return false; }
    }

    /// <summary>从原 PT 或自包含训练快照初始化检测头；自定义类别按名称复用已有分类通道。</summary>
    /// <param name="modelPath">五代支持的检测 PT 或 .sytrain 文件。</param><param name="dataset">检测数据集。</param><param name="options">训练选项。</param>
    /// <exception cref="ArgumentException">配置无效。</exception><exception cref="NotSupportedException">模型或设备不属于首期支持范围。</exception><exception cref="InvalidDataException">模型或快照不完整。</exception><exception cref="IOException">无法读取模型文件。</exception>
    public YoloTrainer(string modelPath, TrainingDataset dataset, TrainingOptions? options = null)
    {
        this.dataset = dataset ?? throw new ArgumentNullException(nameof(dataset)); this.options = options ?? new(); this.options.Validate();
        if (this.options.ValidationDataset is { } validation) dataset.ValidateHoldout(validation);
        TrainingModelState? state = null;
        if (Path.GetExtension(modelPath).Equals(".sytrain", StringComparison.OrdinalIgnoreCase)) { state = TrainingModelState.Load(modelPath); source = state.Source; }
        else if (Path.GetExtension(modelPath).ToLowerInvariant() is ".pt" or ".pth")
        {
            if (new FileInfo(modelPath).Length > 256L * 1024 * 1024) throw new InvalidDataException("首期训练入口的 PT 文件不能超过 256 MiB。");
            source = File.ReadAllBytes(modelPath);
        }
        else throw new NotSupportedException("训练入口支持五代检测 PT 或 .sytrain，不能从普通推理 ONNX 恢复训练结构。");
        extractor = new(source, this.options);
        try
        {
            head = new(extractor.Model.DetectionHead, dataset.ClassNames.ToArray(), extractor.Model.Names, this.options.Seed);
            lossScheduleEpochs = state?.Manifest.LossScheduleEpochs is > 0 ? state.Manifest.LossScheduleEpochs : this.options.Epochs;
            operations = new(extractor.Backend, this.options.CpuThreads);
            if (state is not null)
            {
                if (!state.Manifest.ClassNames.SequenceEqual(dataset.ClassNames)) throw new InvalidDataException("训练模型类别与数据集不同。");
                if (state.Manifest.Format == "SimdYolo.Training/1" && head.Schema.EndToEnd) throw new InvalidDataException("旧版训练归档不能声明双检测头。");
                if (state.Manifest.Format == "SimdYolo.Training/2" && (state.Manifest.HeadKind != head.Schema.Kind || state.Manifest.RegMax != head.Schema.RegMax || state.Manifest.EndToEnd != head.Schema.EndToEnd)) throw new InvalidDataException("快照的检测头类型与原始模型不同。");
                head.Restore(state.Parameters);
            }
        }
        catch { extractor.Dispose(); throw; }
    }

    /// <summary>恢复完整轮次边界、动量和数据顺序；拒绝更换数据集或训练超参数。</summary>
    /// <param name="checkpointPath">带 SGD 动量的训练快照。</param><param name="dataset">与保存时一致的数据集。</param><param name="totalEpochs">恢复后的总轮数，必须大于已完成轮数。</param><param name="device">本次计算设备，可以更换。</param><param name="cpuThreads">本次 CPU 线程预算。</param>
    /// <returns>恢复后的训练会话，调用方负责释放。</returns>
    /// <exception cref="InvalidDataException">快照、动量或数据摘要不匹配。</exception><exception cref="ArgumentException">目标总轮数不合法。</exception>
    /// <param name="validationDataset">与检查点相同的独立验证集；无验证旧归档使用空值。</param>
    public static YoloTrainer Resume(string checkpointPath, TrainingDataset dataset, int totalEpochs, ComputeDevice device = ComputeDevice.Cpu, int cpuThreads = 0, TrainingDataset? validationDataset = null)
    {
        var state = TrainingModelState.Load(checkpointPath); var manifest = state.Manifest;
        if (manifest.DatasetSha256 != dataset.Sha256) throw new InvalidDataException("恢复训练的数据集摘要不同。");
        if (totalEpochs <= manifest.Epoch) throw new ArgumentException("恢复后的总轮数必须大于已完成轮数。");
        if (manifest.ValidationDatasetSha256 != validationDataset?.Sha256) throw new InvalidDataException("恢复训练必须使用相同的验证集，不能移除或替换验证数据。");
        var trainer = new YoloTrainer(checkpointPath, dataset, new() { ImageSize = manifest.ImageSize, Epochs = totalEpochs, BatchSize = manifest.BatchSize, Seed = manifest.Seed, LearningRate = manifest.LearningRate, Momentum = manifest.Momentum, WeightDecay = manifest.WeightDecay, Device = device, CpuThreads = cpuThreads,
            ValidationDataset = validationDataset, EarlyStoppingPatience = manifest.EarlyStoppingPatience, MinimumImprovement = manifest.MinimumImprovement });
        try
        {
            using var stream = File.OpenRead(checkpointPath); using var zip = new ZipArchive(stream, ZipArchiveMode.Read);
            foreach (var parameter in trainer.head.Parameters)
            {
                int index = manifest.Parameters.FindIndex(item => item.Name == parameter.Name);
                byte[] bytes = TrainingModelState.ReadEntry(zip, $"optimizer/{index}.f32", checked(parameter.Values.Length * 4));
                if (bytes.Length != parameter.Values.Length * 4) throw new InvalidDataException("SGD 动量长度不匹配。");
                MemoryMarshal.Cast<byte, float>(bytes).CopyTo(parameter.Velocity);
                if (parameter.Velocity.Any(v => !float.IsFinite(v))) throw new InvalidDataException("SGD 动量包含非有限值。");
            }
            trainer.completedEpochs = manifest.Epoch; trainer.completedSteps = manifest.Step;
            trainer.bestValidationMap = manifest.BestValidationMap; trainer.bestValidationEpoch = manifest.BestValidationEpoch; trainer.staleValidationEpochs = manifest.StaleValidationEpochs;
            trainer.lastValidatedEpoch = manifest.LastValidatedEpoch;
            string best = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(checkpointPath))!, "best.sytrain");
            if (File.Exists(best))
            {
                using var bestFile = File.OpenRead(best); using var bestZip = new ZipArchive(bestFile, ZipArchiveMode.Read); var bestManifest = TrainingModelState.ReadManifest(bestZip);
                if (bestManifest.SourceSha256 == manifest.SourceSha256 && bestManifest.DatasetSha256 == manifest.DatasetSha256 && bestManifest.ValidationDatasetSha256 == manifest.ValidationDatasetSha256 && bestManifest.ClassNames.SequenceEqual(manifest.ClassNames) && bestManifest.Epoch == manifest.BestValidationEpoch && bestManifest.BestValidationMap == manifest.BestValidationMap) trainer.resumeBestPath = best;
            }
            return trainer;
        }
        catch { trainer.Dispose(); throw; }
    }

    /// <summary>同步执行训练，每完成一轮原子保存 last.sytrain；取消保留最近已完成轮次的快照。</summary>
    /// <param name="outputDirectory">输出目录，将创建 last.sytrain 与 history.jsonl。</param><param name="progress">每次实际 SGD 更新后的同步回调。</param><param name="cancellationToken">取消后抛出异常，已保存快照仍有效。</param>
    /// <exception cref="OperationCanceledException">用户取消训练。</exception><exception cref="InvalidOperationException">会话已在训练。</exception><exception cref="ObjectDisposedException">会话已释放。</exception><exception cref="ArithmeticException">损失或梯度出现非有限值。</exception><exception cref="IOException">无法写入快照或历史。</exception>
    /// <param name="validationProgress">每轮实际验证完成后的指标和早停回调。</param>
    public void Train(string outputDirectory, Action<TrainingProgress>? progress = null, CancellationToken cancellationToken = default, Action<EpochValidation>? validationProgress = null)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (running) throw new InvalidOperationException("同一训练会话不能重入。");
            running = true;
        }
        try
        {
            Directory.CreateDirectory(outputDirectory); var elapsed = Stopwatch.StartNew();
            SaveCheckpointCore(Path.Combine(outputDirectory, "last.sytrain"));
            string bestPath = Path.Combine(outputDirectory, "best.sytrain");
            if (options.ValidationDataset is not null)
            {
                if (resumeBestPath is not null && !Path.GetFullPath(resumeBestPath).Equals(Path.GetFullPath(bestPath), StringComparison.OrdinalIgnoreCase)) File.Copy(resumeBestPath, bestPath, true);
                if (!bestValidationMap.HasValue || !File.Exists(bestPath)) ValidateEpoch(outputDirectory, true, validationProgress, cancellationToken);
                else if (lastValidatedEpoch < completedEpochs) { ValidateEpoch(outputDirectory, false, validationProgress, cancellationToken); if (StoppedEarly) return; }
            }
            using var history = new StreamWriter(Path.Combine(outputDirectory, "history.jsonl"), append: true) { AutoFlush = true };
            for (int epoch = completedEpochs; epoch < options.Epochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int[] order = Enumerable.Range(0, dataset.Images.Count).ToArray(); new Random(unchecked(options.Seed + epoch * 104729)).Shuffle(order);
                foreach (int[] batch in order.Chunk(options.BatchSize))
                {
                    float box = 0, cls = 0, dfl = 0;
                    foreach (int index in batch)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var (features, truth) = Features(index); var tape = new GradientTape();
                        var loss = head.Loss(features, operations, tape, truth, true, epoch, lossScheduleEpochs);
                        if (!float.IsFinite(loss.Total)) throw new ArithmeticException("检测损失出现非有限值。");
                        tape.Backward(); box += loss.Box; cls += loss.Classification; dfl += loss.Dfl;
                    }
                    if (head.Parameters.Any(p => !TrainingParameter.AllFinite(p.Gradient))) throw new ArithmeticException("检测头梯度出现非有限值。");
                    foreach (var parameter in head.Parameters) parameter.Update(options, batch.Length);
                    completedSteps++; var item = new TrainingProgress(completedSteps, epoch + 1, (box + cls + dfl) / batch.Length, box / batch.Length, cls / batch.Length, dfl / batch.Length, elapsed.Elapsed.TotalMilliseconds);
                    history.WriteLine(JsonSerializer.Serialize(item)); progress?.Invoke(item);
                }
                completedEpochs = epoch + 1; SaveCheckpointCore(Path.Combine(outputDirectory, "last.sytrain"));
                if (options.ValidationDataset is not null)
                {
                    ValidateEpoch(outputDirectory, false, validationProgress, cancellationToken);
                    if (StoppedEarly) break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            // 丢弃未完成轮次的参数、BN 和动量，确保当前对象与磁盘检查点处于同一边界。
            string checkpoint = Path.Combine(outputDirectory, "last.sytrain");
            if (File.Exists(checkpoint)) RestoreCheckpoint(checkpoint);
            throw;
        }
        finally { lock (gate) running = false; }
    }

    private void ValidateEpoch(string directory, bool baseline, Action<EpochValidation>? callback, CancellationToken cancellation)
    {
        var result = DetectionEvaluator.Evaluate(Path.Combine(directory, "last.sytrain"), options.ValidationDataset!, new() { Device = options.Device, ImageSize = options.ImageSize, CpuThreads = options.CpuThreads }, cancellationToken: cancellation);
        bool improved = baseline || !bestValidationMap.HasValue || result.Map50To95 > bestValidationMap.Value + options.MinimumImprovement;
        if (improved) { bestValidationMap = result.Map50To95; bestValidationEpoch = completedEpochs; staleValidationEpochs = 0; } else staleValidationEpochs++;
        StoppedEarly = !baseline && options.EarlyStoppingPatience > 0 && staleValidationEpochs >= options.EarlyStoppingPatience;
        LastValidation = new(completedEpochs, result.Precision, result.Recall, result.Map50, result.Map50To95, improved, bestValidationEpoch, staleValidationEpochs, StoppedEarly);
        lastValidatedEpoch = completedEpochs;
        File.WriteAllText(Path.Combine(directory, baseline ? "baseline-validation.json" : "last-validation.json"), JsonSerializer.Serialize(result));
        File.AppendAllText(Path.Combine(directory, "validation-history.jsonl"), JsonSerializer.Serialize(LastValidation) + "\n");
        SaveCheckpointCore(Path.Combine(directory, "last.sytrain"));
        if (improved) { File.Copy(Path.Combine(directory, "last.sytrain"), Path.Combine(directory, "best.sytrain"), true); File.WriteAllText(Path.Combine(directory, "best-validation.json"), JsonSerializer.Serialize(result)); }
        callback?.Invoke(LastValidation);
    }

    /// <summary>使用 BN 运行统计计算数据集平均损失；YOLO26 固定初始双头权重，使训练前后可比。</summary>
    /// <returns>各图像的加权总损失平均值。</returns>
    /// <exception cref="InvalidOperationException">训练仍在运行。</exception><exception cref="ObjectDisposedException">会话已释放。</exception>
    public float EvaluateLoss()
    {
        lock (gate)
        {
        ObjectDisposedException.ThrowIf(disposed, this); if (running) throw new InvalidOperationException("训练过程中不能重入验证。");
        float sum = 0;
        for (int i = 0; i < dataset.Images.Count; i++) { var (features, truth) = Features(i); sum += head.Loss(features, operations, new(), truth, false, 0, lossScheduleEpochs).Total; }
        return sum / dataset.Images.Count;
        }
    }

    /// <summary>将当前完整状态保存为可被 YoloDetector 直接加载的自包含归档。</summary>
    /// <param name="path">目标 .sytrain 路径。</param>
    /// <exception cref="InvalidOperationException">训练仍在运行。</exception><exception cref="IOException">无法原子写入目标文件。</exception>
    public void SaveCheckpoint(string path)
    { lock (gate) { ObjectDisposedException.ThrowIf(disposed, this); if (running) throw new InvalidOperationException("训练运行时只由训练循环保存快照。"); SaveCheckpointCore(path); } }

    private (Tensor[] Features, TrainingBox[] Truth) Features(int index)
    {
        var image = dataset.Images[index]; var (input, transform) = ImagePreprocessor.Letterbox(image.Pixels, image.Width, image.Height, options.ImageSize, options.ImageSize);
        if (!featureCache.TryGetValue(index, out var features))
        {
            features = extractor.Extract(input); long bytes = features.Sum(t => (long)t.Length * 4);
            if (options.CacheFeatures && cachedBytes + bytes <= 512L * 1024 * 1024) { featureCache.Add(index, features); cachedBytes += bytes; }
        }
        return (features, TrainingDataset.ParseLabels(image.Labels, head.ClassCount, transform));
    }

    private void SaveCheckpointCore(string path)
    {
        var snapshot = head.Snapshot(); var entries = snapshot.Select(p => new TrainingTensorEntry(p.Key, p.Value.Shape)).ToList();
        var manifest = new TrainingManifest { Format = "SimdYolo.Training/2", HeadKind = head.Schema.Kind, RegMax = head.Schema.RegMax, EndToEnd = head.Schema.EndToEnd, LossScheduleEpochs = lossScheduleEpochs,
            ClassNames = dataset.ClassNames.ToArray(), SourceSha256 = Convert.ToHexStringLower(SHA256.HashData(source)), Parameters = entries, Epoch = completedEpochs, Step = completedSteps,
            Seed = options.Seed, ImageSize = options.ImageSize, BatchSize = options.BatchSize, LearningRate = options.LearningRate, Momentum = options.Momentum, WeightDecay = options.WeightDecay, DatasetSha256 = dataset.Sha256,
            ValidationDatasetSha256 = options.ValidationDataset?.Sha256, BestValidationMap = bestValidationMap, BestValidationEpoch = bestValidationEpoch, StaleValidationEpochs = staleValidationEpochs,
            EarlyStoppingPatience = options.EarlyStoppingPatience, MinimumImprovement = options.MinimumImprovement, LastValidatedEpoch = lastValidatedEpoch };
        string full = Path.GetFullPath(path); Directory.CreateDirectory(Path.GetDirectoryName(full)!); string temporary = full + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var file = File.Create(temporary)) using (var zip = new ZipArchive(file, ZipArchiveMode.Create))
            {
                using (var data = zip.CreateEntry("manifest.json", CompressionLevel.Optimal).Open()) JsonSerializer.Serialize(data, manifest);
                using (var data = zip.CreateEntry("source.pt", CompressionLevel.NoCompression).Open()) data.Write(source);
                ModelProvenance.Write(zip, manifest.SourceSha256);
                for (int i = 0; i < entries.Count; i++)
                {
                    using (var data = zip.CreateEntry($"parameters/{i}.f32", CompressionLevel.Optimal).Open()) data.Write(MemoryMarshal.AsBytes(snapshot[entries[i].Name].Data.AsSpan()));
                    var parameter = head.Parameters.FirstOrDefault(p => p.Name == entries[i].Name);
                    if (parameter is not null) { using var state = zip.CreateEntry($"optimizer/{i}.f32", CompressionLevel.Optimal).Open(); state.Write(MemoryMarshal.AsBytes(parameter.Velocity.AsSpan())); }
                }
            }
            File.Move(temporary, full, true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    private void RestoreCheckpoint(string path)
    {
        var state = TrainingModelState.Load(path); head.Restore(state.Parameters);
        using var stream = File.OpenRead(path); using var zip = new ZipArchive(stream, ZipArchiveMode.Read);
        foreach (var parameter in head.Parameters)
        {
            int index = state.Manifest.Parameters.FindIndex(item => item.Name == parameter.Name);
            byte[] bytes = TrainingModelState.ReadEntry(zip, $"optimizer/{index}.f32", checked(parameter.Values.Length * 4));
            MemoryMarshal.Cast<byte, float>(bytes).CopyTo(parameter.Velocity); Array.Clear(parameter.Gradient);
        }
        completedEpochs = state.Manifest.Epoch; completedSteps = state.Manifest.Step;
        bestValidationMap = state.Manifest.BestValidationMap; bestValidationEpoch = state.Manifest.BestValidationEpoch; staleValidationEpochs = state.Manifest.StaleValidationEpochs;
        lastValidatedEpoch = state.Manifest.LastValidatedEpoch;
    }

    /// <summary>释放冻结骨干、设备资源和缓存特征；运行时不可释放。</summary>
    /// <exception cref="InvalidOperationException">训练仍在运行。</exception>
    public void Dispose() { lock (gate) { if (disposed) return; if (running) throw new InvalidOperationException("训练运行期间不能释放会话。"); disposed = true; featureCache.Clear(); extractor.Dispose(); } }
}
