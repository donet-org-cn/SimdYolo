using System.Collections.Concurrent;
using System.Text.Json;
using System.Threading.Channels;
using SimdYolo.Training;

namespace SimdYolo.Web;

/// <summary>串行执行真实训练任务，保存状态并保留取消时最近完整轮次的检查点。</summary>
internal sealed class TrainingService : BackgroundService
{
    private readonly Workspace workspace;
    private readonly Channel<TrainingJob> queue = Channel.CreateBounded<TrainingJob>(new BoundedChannelOptions(4) { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
    private readonly ConcurrentDictionary<string, TrainingJob> jobs = new(StringComparer.Ordinal);

    public TrainingService(Workspace workspace)
    {
        this.workspace = workspace;
        foreach (string directory in Directory.EnumerateDirectories(Path.Combine(workspace.DataRoot, "runs")))
        {
            string path = Path.Combine(directory, "status.json"); if (!File.Exists(path)) continue;
            try
            {
                var saved = JsonSerializer.Deserialize<SavedJob>(File.ReadAllText(path), Workspace.JsonOptions);
                if (saved is null || !Guid.TryParseExact(saved.Id, "N", out _) || Path.GetFileName(directory) != saved.Id) continue;
                var job = new TrainingJob(saved.Id, saved.Request, directory) { Status = saved.Status, Epoch = saved.Epoch, Step = saved.Step, StartedUtc = saved.StartedUtc, Error = saved.Error, ActualBackend = saved.ActualBackend };
                job.History.AddRange(saved.History.TakeLast(2000));
                job.ValidationHistory.AddRange((saved.ValidationHistory ?? []).TakeLast(501)); job.StoppedEarly = saved.StoppedEarly;
                if (job.Status is "queued" or "running") { job.Status = "cancelled"; job.Error = "进程已重启，可从最近完成轮次恢复。"; }
                if (File.Exists(job.Checkpoint)) job.OutputModelId = workspace.RegisterModel(job.Checkpoint, "训练模型 " + job.Id[..8] + ".sytrain").Id;
                if (File.Exists(job.BestCheckpoint)) job.BestModelId = workspace.RegisterModel(job.BestCheckpoint, "最佳模型 " + job.Id[..8] + ".sytrain").Id;
                jobs[job.Id] = job;
            }
            catch (Exception error) when (error is IOException or JsonException or InvalidDataException or ArgumentException) { Console.Error.WriteLine("跳过损坏训练记录：" + error.Message); }
        }
    }

    internal object[] List() => jobs.Values.OrderByDescending(j => j.StartedUtc).Select(View).ToArray();
    /// <summary>训练及验证数据在排队或执行期间不可修改，保证快照摘要与实际使用的数据一致。</summary>
    internal bool DatasetBusy(string id) => jobs.Values.Any(job => job.Status is "queued" or "running" && (job.Request.DatasetId == id || job.Request.ValidationDatasetId == id));
    internal TrainingJob Find(string id) => jobs.TryGetValue(id, out var job) ? job : throw new KeyNotFoundException("训练任务不存在。");
    internal object View(TrainingJob job)
    {
        lock (job.Gate)
        {
            int count = workspace.Datasets.TryGetValue(job.Request.DatasetId, out var data) ? data.Dataset.Images.Count : 0;
            string modelName = workspace.Models.TryGetValue(job.Request.ModelId, out var model) ? model.Name : "原始模型不可用";
            string datasetName = count > 0 ? data.Name : "原始数据集不可用";
            return new { job.Id, job.Status, job.Epoch, job.Step, job.Request.Epochs, TotalSteps = (int)Math.Ceiling(count / (double)job.Request.BatchSize) * job.Request.Epochs,
                Name = modelName + " · " + datasetName, ModelName = modelName, DatasetName = datasetName,
                History = job.History.ToArray(), job.Error, CheckpointUrl = File.Exists(job.Checkpoint) ? $"/api/training/{job.Id}/checkpoint" : null,
                ValidationHistory = job.ValidationHistory.ToArray(), job.StoppedEarly, job.BestModelId,
                BestCheckpointUrl = File.Exists(job.BestCheckpoint) ? $"/api/training/{job.Id}/best" : null,
                ModelId = job.OutputModelId ?? job.Request.ModelId, Backend = job.ActualBackend ?? job.Request.Backend, job.StartedUtc };
        }
    }

    internal TrainingJob Enqueue(TrainRequest request)
    {
        if (!workspace.Models.TryGetValue(request.ModelId, out var model) || !model.Trainable) throw new ArgumentException("请选择可训练的 YOLOv8/v9/v10/YOLO11/YOLO26 检测 PT 或本程序训练快照。");
        if (!workspace.Datasets.ContainsKey(request.DatasetId)) throw new ArgumentException("数据集不存在。");
        if (request.ValidationDatasetId is { } validationId)
        {
            if (!workspace.Datasets.TryGetValue(validationId, out var validation)) throw new ArgumentException("验证集不存在。");
            workspace.Datasets[request.DatasetId].Dataset.ValidateHoldout(validation.Dataset);
        }
        if (request.EarlyStoppingPatience is < 0 or > 500 || request.EarlyStoppingPatience > 0 && request.ValidationDatasetId is null) throw new ArgumentException("早停耐心值需在0到500之间，并需要独立验证集。");
        if (request.Epochs is < 1 or > 500 || request.BatchSize is < 1 or > 32 || request.ImageSize is < 64 or > 640 || request.ImageSize % 32 != 0 || !float.IsFinite(request.LearningRate) || request.LearningRate is <= 0 or > 1) throw new ArgumentException("训练参数超出本地示例范围。");
        _ = Workspace.Device(request.Backend, true);
        if (jobs.Count >= 1000) throw new InvalidOperationException("任务历史已满，请归档本地 runs 目录后重启。");
        if (request.ResumeId is not null)
        {
            var previous = Find(request.ResumeId);
            lock (previous.Gate) if (previous.Status is "running" or "queued" || !File.Exists(previous.Checkpoint)) throw new InvalidOperationException("该任务仍在执行，或尚无完整检查点。");
        }
        string id = Guid.NewGuid().ToString("N"); var job = new TrainingJob(id, request, Path.Combine(workspace.DataRoot, "runs", id));
        jobs[id] = job;
        if (!queue.Writer.TryWrite(job)) { jobs.TryRemove(id, out _); throw new InvalidOperationException("训练队列已满，请稍后再试。"); }
        Directory.CreateDirectory(job.Directory); Save(job); return job;
    }

    internal TrainingJob Resume(string id, int epochs)
    { var job = Find(id); return Enqueue(job.Request with { Epochs = epochs, ResumeId = id }); }
    internal TrainingJob Cancel(string id)
    {
        var job = Find(id); lock (job.Gate) if (job.Status is "queued" or "running") job.Cancellation.Cancel(); return job;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await foreach (var job in queue.Reader.ReadAllAsync(stoppingToken))
            await Task.Run(() => Execute(job, stoppingToken), stoppingToken);
    }

    private void Execute(TrainingJob job, CancellationToken stoppingToken)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(job.Cancellation.Token, stoppingToken);
        try
        {
            linked.Token.ThrowIfCancellationRequested();
            var data = workspace.Datasets[job.Request.DatasetId].Dataset; var model = workspace.Models[job.Request.ModelId];
            var device = Workspace.Device(job.Request.Backend, true); int threads = Math.Min(4, Environment.ProcessorCount);
            var validation = job.Request.ValidationDatasetId is { } validationId ? workspace.Datasets[validationId].Dataset : null;
            using var trainer = job.Request.ResumeId is { } id ? YoloTrainer.Resume(Find(id).Checkpoint, data, job.Request.Epochs, device, threads, validation) :
                new YoloTrainer(model.Path, data, new() { Device = device, CpuThreads = threads, Epochs = job.Request.Epochs, ImageSize = job.Request.ImageSize, BatchSize = job.Request.BatchSize, LearningRate = job.Request.LearningRate, Seed = job.Request.Seed, ValidationDataset = validation, EarlyStoppingPatience = job.Request.EarlyStoppingPatience });
            lock (job.Gate) { job.Status = "running"; job.ActualBackend = trainer.BackendName; job.Epoch = trainer.CompletedEpochs; job.Step = trainer.CompletedSteps; }
            Save(job);
            try
            {
                trainer.Train(job.Directory, item =>
                {
                    lock (job.Gate) { job.Epoch = item.Epoch; job.Step = item.Step; job.History.Add(item); if (job.History.Count > 2000) job.History.RemoveAt(0); }
                    if (Environment.TickCount64 - job.LastSavedTick >= 1000) Save(job);
                }, linked.Token, validationProgress: item =>
                {
                    lock (job.Gate) { job.ValidationHistory.Add(item); job.StoppedEarly = item.StoppedEarly; }
                    Save(job);
                });
                lock (job.Gate) job.Status = "completed";
            }
            catch (OperationCanceledException)
            { lock (job.Gate) { job.Status = "cancelled"; job.Epoch = trainer.CompletedEpochs; job.Step = trainer.CompletedSteps; } }
            if (File.Exists(job.Checkpoint)) job.OutputModelId = workspace.RegisterModel(job.Checkpoint, "训练模型 " + job.Id[..8] + ".sytrain").Id;
            if (File.Exists(job.BestCheckpoint)) job.BestModelId = workspace.RegisterModel(job.BestCheckpoint, "最佳模型 " + job.Id[..8] + ".sytrain").Id;
        }
        catch (OperationCanceledException) { lock (job.Gate) job.Status = "cancelled"; }
        catch (Exception error) { lock (job.Gate) { job.Status = "failed"; job.Error = error.Message; } }
        finally { Save(job); }
    }

    private void Save(TrainingJob job)
    {
        lock (job.Gate)
        {
            Directory.CreateDirectory(job.Directory);
            var saved = new SavedJob(job.Id, job.Request, job.Status, job.Epoch, job.Step, job.StartedUtc, job.Error, job.ActualBackend, job.History.ToArray(), job.ValidationHistory.ToArray(), job.StoppedEarly);
            string temporary = Path.Combine(job.Directory, "status.json.tmp");
            File.WriteAllText(temporary, JsonSerializer.Serialize(saved, Workspace.JsonOptions)); File.Move(temporary, Path.Combine(job.Directory, "status.json"), true);
            job.LastSavedTick = Environment.TickCount64;
        }
    }

    internal sealed class TrainingJob(string id, TrainRequest request, string directory)
    {
        internal object Gate { get; } = new();
        internal string Id { get; } = id;
        internal TrainRequest Request { get; } = request;
        internal string Directory { get; } = directory;
        internal string Checkpoint => Path.Combine(Directory, "last.sytrain");
        internal string BestCheckpoint => Path.Combine(Directory, "best.sytrain");
        internal CancellationTokenSource Cancellation { get; } = new();
        internal string Status = "queued";
        internal int Epoch, Step;
        internal long LastSavedTick;
        internal string? Error, ActualBackend, OutputModelId, BestModelId;
        internal bool StoppedEarly;
        internal DateTimeOffset StartedUtc = DateTimeOffset.UtcNow;
        internal List<TrainingProgress> History { get; } = [];
        internal List<EpochValidation> ValidationHistory { get; } = [];
    }
    internal sealed record SavedJob(string Id, TrainRequest Request, string Status, int Epoch, int Step, DateTimeOffset StartedUtc, string? Error, string? ActualBackend, TrainingProgress[] History, EpochValidation[]? ValidationHistory = null, bool StoppedEarly = false);
}
