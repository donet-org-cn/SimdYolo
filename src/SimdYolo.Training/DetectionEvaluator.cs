using System.Diagnostics;

namespace SimdYolo.Training;

/// <summary>检测验证集选项；AP使用全部低阈值预测，P/R使用单独的工作阈值。</summary>
public sealed record EvaluationOptions
{
    /// <summary>网络输入边长。</summary><value>默认640；原生训练归档优先使用保存尺寸。</value>
    public int ImageSize { get; init; } = 640;
    /// <summary>推理计算设备。</summary><value>默认CPU。</value>
    public ComputeDevice Device { get; init; } = ComputeDevice.Cpu;
    /// <summary>CPU线程预算。</summary><value>默认四线程。</value>
    public int CpuThreads { get; init; } = 4;
    /// <summary>P/R和混淆矩阵采用的置信度。</summary><value>默认0.25；不用于截断AP排序。</value>
    public float OperatingConfidence { get; init; } = .25f;
    /// <summary>推理精度。</summary><value>默认FP32。</value>
    public InferencePrecision Precision { get; init; } = InferencePrecision.Float32;
    /// <summary>INT8静态校准文件；空值使用动态量化。</summary><value>默认空。</value>
    public string? CalibrationPath { get; init; }
}

/// <summary>一张图像的真值和模型预测，均使用原图像素坐标。</summary>
/// <param name="Name">样本名称。</param><param name="Truth">真值边框。</param><param name="Predictions">预测边框。</param>
public sealed record EvaluationSample(string Name, TrainingBox[] Truth, Detection[] Predictions);
/// <summary>单类别指标；没有真值的类别不参加宏平均AP。</summary>
/// <param name="ClassId">类别编号。</param><param name="Name">类别名称。</param><param name="GroundTruth">真值数量。</param>
/// <param name="TruePositive">工作阈值下正确预测数。</param><param name="FalsePositive">误检数。</param><param name="FalseNegative">漏检数。</param>
/// <param name="Precision">精确率。</param><param name="Recall">召回率。</param><param name="Map50">IoU0.50处的类别平均精度。</param><param name="Map50To95">十个IoU阈值的平均AP。</param>
public sealed record ClassEvaluation(int ClassId, string Name, int GroundTruth, int TruePositive, int FalsePositive, int FalseNegative, double Precision, double Recall, double Map50, double Map50To95);
/// <summary>验证结果及逐图诊断；混淆矩阵行是真值、列是预测，最后一行/列表示背景。</summary>
/// <param name="Images">图像数。</param><param name="Precision">宏平均精确率。</param><param name="Recall">宏平均召回率。</param>
/// <param name="Map50">宏平均AP50。</param><param name="Map50To95">宏平均AP50:95。</param><param name="Classes">逐类指标。</param>
/// <param name="ConfusionMatrix">含背景的混淆矩阵。</param><param name="Samples">逐图真值和预测。</param><param name="ElapsedMs">总验证耗时。</param>
public sealed record EvaluationResult(int Images, double Precision, double Recall, double Map50, double Map50To95, ClassEvaluation[] Classes, int[][] ConfusionMatrix, EvaluationSample[] Samples, double ElapsedMs);

/// <summary>纯C#检测验证器，使用逐图一对一匹配和101点插值AP，不调用Python或原生评估库。</summary>
public static class DetectionEvaluator
{
    /// <summary>重新加载实际模型，在验证集上执行完整预处理、网络和检测后处理。</summary>
    /// <param name="modelPath">PT、ONNX或SYTRAIN。</param><param name="dataset">独立验证集。</param><param name="options">验证配置。</param>
    /// <param name="progress">每完成一张图后的进度。</param><param name="cancellationToken">取消标记。</param>
    /// <returns>可序列化的完整指标和逐图结果。</returns>
    /// <exception cref="ArgumentException">配置或模型任务不合法。</exception><exception cref="OperationCanceledException">用户取消。</exception>
    /// <exception cref="IOException">模型无法读取。</exception>
    public static EvaluationResult Evaluate(string modelPath, TrainingDataset dataset, EvaluationOptions? options = null, Action<int, int>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(dataset); options ??= new();
        if (!float.IsFinite(options.OperatingConfidence) || options.OperatingConfidence is < 0 or > 1) throw new ArgumentException("P/R置信度必须在0到1之间。");
        var clock = Stopwatch.StartNew();
        using var detector = new YoloDetector(modelPath, new() { Device = options.Device, Width = options.ImageSize, Height = options.ImageSize, CpuThreads = options.CpuThreads,
            Confidence = .001f, IouThreshold = .7f, MaxDetections = 300, Precision = options.Precision, CalibrationPath = options.CalibrationPath });
        if (detector.Task != YoloTask.Detection) throw new ArgumentException("检测评估需要检测模型，不能将分割或姿态指标冒充检测指标。");
        var indices = dataset.ClassNames.Select((name, index) => (name, index)).ToDictionary(p => p.name, p => p.index, StringComparer.Ordinal);
        var samples = new List<EvaluationSample>();
        foreach (var image in dataset.Images)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var truth = TrainingDataset.ParseLabels(image.Labels, dataset.ClassNames.Count, new(image.Width, image.Height, image.Width, image.Height, 1, 1, 0, 0));
            // 通过实际类别名映射预训练模型与自定义模型，避免把person的编号0当作自定义类别0。
            var predictions = detector.Detect(image.Pixels, image.Width, image.Height)
                .Where(p => detector.ClassNames.TryGetValue(p.ClassId, out string? name) && indices.ContainsKey(name))
                .Select(p => p with { ClassId = indices[detector.ClassNames[p.ClassId]] }).ToArray();
            samples.Add(new(image.Name, truth, predictions)); progress?.Invoke(samples.Count, dataset.Images.Count);
        }
        return Calculate(dataset.ClassNames, samples, options.OperatingConfidence) with { ElapsedMs = clock.Elapsed.TotalMilliseconds };
    }

    /// <summary>从已经产生的预测计算指标；同置信度使用图像/预测原始顺序保持确定性。</summary>
    /// <param name="classNames">类别名称。</param><param name="samples">逐图真值与预测。</param><param name="operatingConfidence">P/R工作阈值。</param>
    /// <returns>IoU 0.50到0.95、步长0.05的101点插值AP及诊断。</returns>
    /// <exception cref="ArgumentException">类别、阈值或框包含非法值。</exception>
    public static EvaluationResult Calculate(IReadOnlyList<string> classNames, IEnumerable<EvaluationSample> samples, float operatingConfidence = .25f)
    {
        ArgumentNullException.ThrowIfNull(classNames); ArgumentNullException.ThrowIfNull(samples);
        if (classNames.Count is < 1 or > 1000 || !float.IsFinite(operatingConfidence) || operatingConfidence is < 0 or > 1) throw new ArgumentException("评估类别或置信度无效。");
        var data = samples.ToArray(); int nc = classNames.Count;
        foreach (var sample in data)
        {
            if (sample.Truth.Any(t => t.ClassId < 0 || t.ClassId >= nc || !new[] { t.X1, t.Y1, t.X2, t.Y2 }.All(float.IsFinite) || t.X2 <= t.X1 || t.Y2 <= t.Y1) ||
                sample.Predictions.Any(p => p.ClassId < 0 || p.ClassId >= nc || !new[] { p.X, p.Y, p.Width, p.Height, p.Confidence }.All(float.IsFinite) || p.Width <= 0 || p.Height <= 0 || p.Confidence is < 0 or > 1)) throw new ArgumentException("评估样本含非法类别或边框。");
        }
        var metrics = new List<ClassEvaluation>();
        for (int cls = 0; cls < nc; cls++)
        {
            int truths = data.Sum(s => s.Truth.Count(t => t.ClassId == cls)), tp = 0, fp = 0; var aps = new double[10];
            var ranked = data.SelectMany((s, image) => s.Predictions.Select((p, index) => (p, image, index)).Where(x => x.p.ClassId == cls))
                .OrderByDescending(x => x.p.Confidence).ThenBy(x => x.image).ThenBy(x => x.index).ToArray();
            for (int threshold = 0; threshold < 10; threshold++)
            {
                var used = data.Select(s => new bool[s.Truth.Length]).ToArray(); var matches = new bool[ranked.Length];
                for (int i = 0; i < ranked.Length; i++)
                {
                    var (prediction, image, _) = ranked[i]; float best = .5f + threshold * .05f; int chosen = -1;
                    for (int j = 0; j < data[image].Truth.Length; j++)
                    {
                        var truth = data[image].Truth[j]; if (truth.ClassId != cls || used[image][j]) continue;
                        float overlap = Iou(prediction, truth);
                        if (overlap >= best && (chosen < 0 || overlap > best)) { best = overlap; chosen = j; }
                    }
                    if (chosen >= 0) { matches[i] = true; used[image][chosen] = true; }
                }
                aps[threshold] = AveragePrecision(matches, truths);
                if (threshold == 0) for (int i = 0; i < ranked.Length; i++) if (ranked[i].p.Confidence >= operatingConfidence) { if (matches[i]) tp++; else fp++; }
            }
            metrics.Add(new(cls, classNames[cls], truths, tp, fp, Math.Max(0, truths - tp), tp + fp > 0 ? (double)tp / (tp + fp) : 0, truths > 0 ? (double)tp / truths : 0, aps[0], aps.Average()));
        }
        var matrix = Enumerable.Range(0, nc + 1).Select(_ => new int[nc + 1]).ToArray();
        foreach (var sample in data)
        {
            var used = new bool[sample.Truth.Length];
            foreach (var prediction in sample.Predictions.Where(p => p.Confidence >= operatingConfidence).OrderByDescending(p => p.Confidence))
            {
                int chosen = -1; float best = .5f;
                for (int j = 0; j < used.Length; j++) if (!used[j]) { float overlap = Iou(prediction, sample.Truth[j]); if (overlap >= best && (chosen < 0 || overlap > best)) { best = overlap; chosen = j; } }
                if (chosen < 0) matrix[nc][prediction.ClassId]++; else { used[chosen] = true; matrix[sample.Truth[chosen].ClassId][prediction.ClassId]++; }
            }
            for (int j = 0; j < used.Length; j++) if (!used[j]) matrix[sample.Truth[j].ClassId][nc]++;
        }
        var present = metrics.Where(m => m.GroundTruth > 0).ToArray();
        return new(data.Length, present.Length > 0 ? present.Average(m => m.Precision) : 0, present.Length > 0 ? present.Average(m => m.Recall) : 0,
            present.Length > 0 ? present.Average(m => m.Map50) : 0, present.Length > 0 ? present.Average(m => m.Map50To95) : 0, metrics.ToArray(), matrix, data, 0);
    }

    private static float Iou(Detection p, TrainingBox t)
    {
        float intersection = MathF.Max(0, MathF.Min(p.X + p.Width, t.X2) - MathF.Max(p.X, t.X1)) * MathF.Max(0, MathF.Min(p.Y + p.Height, t.Y2) - MathF.Max(p.Y, t.Y1));
        return intersection / MathF.Max(1e-12f, p.Width * p.Height + (t.X2 - t.X1) * (t.Y2 - t.Y1) - intersection);
    }
    private static double AveragePrecision(bool[] matches, int truths)
    {
        if (truths == 0 || matches.Length == 0) return 0;
        var precision = new double[matches.Length]; var recall = new double[matches.Length]; int tp = 0;
        for (int i = 0; i < matches.Length; i++) { if (matches[i]) tp++; precision[i] = (double)tp / (i + 1); recall[i] = (double)tp / truths; }
        for (int i = precision.Length - 2; i >= 0; i--) precision[i] = Math.Max(precision[i], precision[i + 1]);
        double sum = 0; int index = 0;
        for (int r = 0; r <= 100; r++) { while (index < recall.Length && recall[index] < r / 100.0) index++; if (index < recall.Length) sum += precision[index]; }
        return sum / 101;
    }
}
