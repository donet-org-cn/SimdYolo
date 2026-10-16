using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json;
using System.Threading.Channels;
using SimdYolo.Training;

namespace SimdYolo.Web;

/// <summary>串行执行可取消的评估、比较、ONNX导出、校准和预标注，持久化实际结果。</summary>
internal sealed class WorkflowService : BackgroundService
{
    private readonly Workspace workspace;
    private readonly ConcurrentDictionary<string, Job> jobs = new(StringComparer.Ordinal);
    private readonly Channel<Job> queue = Channel.CreateBounded<Job>(new BoundedChannelOptions(4) { SingleReader = true });
    public WorkflowService(Workspace workspace)
    {
        this.workspace = workspace; Directory.CreateDirectory(Path.Combine(workspace.DataRoot, "workflows"));
        foreach (string file in Directory.EnumerateFiles(Path.Combine(workspace.DataRoot, "workflows"), "status.json", SearchOption.AllDirectories))
        {
            try
            {
                var saved = JsonSerializer.Deserialize<SavedJob>(File.ReadAllText(file), Workspace.JsonOptions);
                if (saved is null || !Guid.TryParseExact(saved.Id, "N", out _) || Path.GetFileName(Path.GetDirectoryName(file)) != saved.Id) continue;
                var job = new Job(saved.Id, saved.Request, Path.GetDirectoryName(file)!) { Status = saved.Status is "running" or "queued" ? "cancelled" : saved.Status, Done = saved.Done, Total = saved.Total, Error = saved.Error, Result = saved.Result, CreatedUtc = saved.CreatedUtc ?? DateTimeOffset.UtcNow };
                if (job.Status == "completed" && saved.Request.Kind == "export" && File.Exists(job.Output)) workspace.RegisterModel(job.Output, "导出模型 " + job.Id[..8] + ".onnx");
                jobs[job.Id] = job;
            }
            catch (Exception error) when (error is IOException or JsonException or ArgumentException) { Console.Error.WriteLine("跳过工作流记录：" + error.Message); }
        }
    }
    internal object[] List() => jobs.Values.OrderByDescending(j => j.CreatedUtc).Select(View).ToArray();
    internal Job Find(string id) => jobs.TryGetValue(id, out var job) ? job : throw new KeyNotFoundException("评估/模型任务不存在。");
    internal object View(Job job) { lock (job.Gate) return new { job.Id, job.Request, job.Status, job.Done, job.Total, job.Error, job.Result, job.CreatedUtc, downloadUrl = job.Status == "completed" && File.Exists(job.Output) ? $"/api/workflows/{job.Id}/download" : null }; }
    internal Job Enqueue(WorkflowRequest request)
    {
        if (request.Kind is not ("evaluate" or "compare" or "export" or "calibrate" or "prelabel") || !workspace.Models.TryGetValue(request.ModelId, out var model)) throw new ArgumentException("任务类型或模型不存在。");
        if (request.Kind != "export" && (request.DatasetId is null || !workspace.Datasets.ContainsKey(request.DatasetId))) throw new ArgumentException("请选择数据集。");
        if (request.Kind == "compare" && (request.OtherModelId is null || !workspace.Models.ContainsKey(request.OtherModelId))) throw new ArgumentException("请选择另一个比较模型。");
        if (request.Kind == "compare" && request.CalibrationId is not null && request.OtherModelId != request.ModelId) throw new ArgumentException("不同模型比较不能共用一份静态校准，请使用FP32或动态INT8。");
        if (request.Kind == "export" && model.Format is not ("PT" or "SYTRAIN")) throw new ArgumentException("ONNX导出入口需要完整PT或SYTRAIN。");
        if (request.ImageSize is < 64 or > 1280 || request.ImageSize % 32 != 0 || !float.IsFinite(request.Confidence) || request.Confidence is < .001f or > .99f) throw new ArgumentException("输入尺寸或置信度无效。");
        var precision = ParsePrecision(request.Precision); var device = Workspace.Device(request.Backend);
        if (precision != InferencePrecision.Float32 && device != ComputeDevice.Cpu) throw new ArgumentException("FP16混合/INT8当前使用纯.NET CPU后端，请选择CPU。");
        if (request.CalibrationId is not null) _ = Calibration(request.CalibrationId, request.ModelId);
        if (request.Kind == "prelabel" && (request.ImageIndex < 0 || request.ImageIndex >= workspace.Datasets[request.DatasetId!].Dataset.Images.Count)) throw new ArgumentException("预标注图像编号不存在。");
        if (jobs.Count >= 500) throw new InvalidOperationException("工作流历史已满，请先归档本地记录。");
        string id = Guid.NewGuid().ToString("N"); var job = new Job(id, request, Path.Combine(workspace.DataRoot, "workflows", id)); jobs[id] = job;
        if (!queue.Writer.TryWrite(job)) { jobs.TryRemove(id, out _); throw new InvalidOperationException("后台任务队列已满。"); }
        Directory.CreateDirectory(job.Directory); Save(job); return job;
    }
    internal string? Calibration(string? id, string modelId)
    {
        if (id is null) return null; var job = Find(id);
        if (job.Status != "completed" || job.Request.Kind != "calibrate" || job.Request.ModelId != modelId || !File.Exists(job.Output)) throw new ArgumentException("校准任务未完成或不属于当前模型。");
        return job.Output;
    }
    internal static InferencePrecision ParsePrecision(string value) => Enum.TryParse<InferencePrecision>(value, true, out var precision) && Enum.IsDefined(precision) ? precision : throw new ArgumentException("推理精度无效。");
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    { await foreach (var job in queue.Reader.ReadAllAsync(stoppingToken)) await Task.Run(() => Execute(job, stoppingToken), stoppingToken); }
    private void Execute(Job job, CancellationToken stopping)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(stopping, job.Cancellation.Token); var token = linked.Token;
        try
        {
            token.ThrowIfCancellationRequested(); lock (job.Gate) job.Status = "running"; Save(job);
            var request = job.Request; var model = workspace.Models[request.ModelId];
            var options = new EvaluationOptions { Device = Workspace.Device(request.Backend), ImageSize = request.ImageSize, Precision = ParsePrecision(request.Precision), CalibrationPath = Calibration(request.CalibrationId, request.ModelId) };
            object result, summary;
            if (request.Kind == "export")
            {
                var exported = ModelExporter.ExportOnnx(model.Path, job.Output, request.ImageSize); token.ThrowIfCancellationRequested();
                var registered = workspace.RegisterModel(job.Output, "导出模型 " + job.Id[..8] + ".onnx");
                result = summary = new { modelId = registered.Id, format = "ONNX", exported.InputShape, exported.OutputShapes, exported.Nodes, exported.Sha256, sizeBytes = new FileInfo(job.Output).Length };
                File.WriteAllText(Path.Combine(job.Directory, "export-info.json"), JsonSerializer.Serialize(result, Workspace.JsonOptions));
            }
            else if (request.Kind == "calibrate")
            {
                var data = workspace.Datasets[request.DatasetId!].Dataset;
                result = ModelCalibration.Calibrate(model.Path, data.Images.Select((image, index) => { Progress(job, index, data.Images.Count); token.ThrowIfCancellationRequested(); return new CalibrationImage(image.Pixels, image.Width, image.Height); }), job.Output, request.ImageSize, cancellationToken: token);
                var profile = (CalibrationProfile)result; summary = new { profile.Images, profile.ModelSha256, profile.InputShape, layers = profile.ActivationMaxima.Count };
            }
            else if (request.Kind == "prelabel")
            {
                var data = workspace.Datasets[request.DatasetId!].Dataset; var image = data.Images[request.ImageIndex];
                using var detector = new YoloDetector(model.Path, new() { Device = options.Device, Width = request.ImageSize, Height = request.ImageSize, CpuThreads = 4, Confidence = request.Confidence, Precision = options.Precision, CalibrationPath = options.CalibrationPath });
                var names = data.ClassNames.Select((name, index) => (name, index)).ToDictionary(p => p.name, p => p.index, StringComparer.Ordinal);
                var labels = new List<string>();
                foreach (var box in detector.Detect(image.Pixels, image.Width, image.Height))
                {
                    if (!detector.ClassNames.TryGetValue(box.ClassId, out string? name) || !names.TryGetValue(name, out int cls)) continue;
                    float left = Math.Clamp(box.X, 0, image.Width), top = Math.Clamp(box.Y, 0, image.Height), right = Math.Clamp(box.X + box.Width, 0, image.Width), bottom = Math.Clamp(box.Y + box.Height, 0, image.Height);
                    if (right <= left || bottom <= top) continue;
                    labels.Add(string.Create(CultureInfo.InvariantCulture, $"{cls} {(left + right) / (2 * image.Width)} {(top + bottom) / (2 * image.Height)} {(right - left) / image.Width} {(bottom - top) / image.Height}"));
                }
                token.ThrowIfCancellationRequested(); result = summary = new { request.DatasetId, request.ImageIndex, revision = data.Sha256, labels = string.Join('\n', labels), count = labels.Count, requiresReview = true };
                File.WriteAllText(job.Output, JsonSerializer.Serialize(result, Workspace.JsonOptions));
            }
            else
            {
                var data = workspace.Datasets[request.DatasetId!].Dataset;
                var before = DetectionEvaluator.Evaluate(model.Path, data, options, (done, total) => Progress(job, done, total * (request.Kind == "compare" ? 2 : 1)), token);
                if (request.Kind == "compare")
                {
                    var after = DetectionEvaluator.Evaluate(workspace.Models[request.OtherModelId!].Path, data, options, (done, total) => Progress(job, total + done, total * 2), token);
                    result = new { datasetSha256 = data.Sha256, before, after }; summary = new { datasetSha256 = data.Sha256, before = Metrics(before), after = Metrics(after) };
                }
                else { result = new { datasetSha256 = data.Sha256, metrics = before }; summary = new { datasetSha256 = data.Sha256, metrics = Metrics(before) }; }
                File.WriteAllText(job.Output, JsonSerializer.Serialize(result, Workspace.JsonOptions));
            }
            lock (job.Gate) { job.Result = JsonSerializer.SerializeToElement(summary, Workspace.JsonOptions); job.Done = Math.Max(job.Total, 1); job.Total = job.Done; job.Status = "completed"; }
        }
        catch (OperationCanceledException) { lock (job.Gate) job.Status = "cancelled"; }
        catch (Exception error) { lock (job.Gate) { job.Status = "failed"; job.Error = error.Message; } }
        finally { Save(job); }
    }
    private static object Metrics(EvaluationResult value) => new { value.Images, value.Precision, value.Recall, value.Map50, value.Map50To95, value.Classes, value.ConfusionMatrix, value.ElapsedMs };
    private void Progress(Job job, int done, int total) { lock (job.Gate) { job.Done = done; job.Total = total; } if (Environment.TickCount64 - job.LastSaved > 500) Save(job); }
    private static void Save(Job job)
    {
        lock (job.Gate)
        {
            Directory.CreateDirectory(job.Directory); string path = Path.Combine(job.Directory, "status.json");
            File.WriteAllText(path + ".tmp", JsonSerializer.Serialize(new SavedJob(job.Id, job.Request, job.Status, job.Done, job.Total, job.Error, job.Result, job.CreatedUtc), Workspace.JsonOptions)); File.Move(path + ".tmp", path, true); job.LastSaved = Environment.TickCount64;
        }
    }
    internal sealed class Job(string id, WorkflowRequest request, string directory)
    {
        internal object Gate { get; } = new();
        internal string Id { get; } = id;
        internal WorkflowRequest Request { get; } = request;
        internal string Directory { get; } = directory;
        internal string Output => Path.Combine(Directory, Request.Kind == "export" ? "model.onnx" : "output.json");
        internal CancellationTokenSource Cancellation { get; } = new();
        internal string Status = "queued";
        internal int Done, Total;
        internal long LastSaved;
        internal string? Error;
        internal JsonElement? Result;
        internal DateTimeOffset CreatedUtc { get; set; } = DateTimeOffset.UtcNow;
    }
    private sealed record SavedJob(string Id, WorkflowRequest Request, string Status, int Done, int Total, string? Error, JsonElement? Result, DateTimeOffset? CreatedUtc = null);
}
