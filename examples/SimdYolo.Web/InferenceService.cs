using System.Diagnostics;

namespace SimdYolo.Web;

/// <summary>串行执行推理，复用最近两个模型会话，减少重复权重解析、上传和设备初始化。</summary>
internal sealed class InferenceService(Workspace workspace, WorkflowService workflows) : IDisposable
{
    private readonly SemaphoreSlim gate = new(1, 1);
    private readonly LinkedList<Session> sessions = new();

    /// <summary>在有界上传图像上推理，把框、关键点和掩码显示范围映射回原图；掩码像素保持上传分辨率。</summary>
    /// <param name="request">模型、RGB图像、后端及可选原图尺寸。</param>
    /// <param name="cancellation">等待队列及后台任务的取消标记。</param>
    /// <returns>使用原图坐标的结果、掩码位图及其显示尺寸，以及实际推理耗时。</returns>
    /// <exception cref="KeyNotFoundException">模型不存在。</exception>
    /// <exception cref="ArgumentException">原图尺寸、RGB数据、后端或置信度无效。</exception>
    /// <exception cref="FormatException">RGB编码无效。</exception>
    /// <exception cref="OperationCanceledException">等待或执行前任务被取消。</exception>
    internal async Task<object> Predict(InferenceRequest request, CancellationToken cancellation)
    {
        if (!workspace.Models.TryGetValue(request.ModelId, out var model)) throw new KeyNotFoundException("模型不存在。");
        if (request.OriginalWidth.HasValue != request.OriginalHeight.HasValue) throw new ArgumentException("originalWidth与originalHeight必须同时提供。");
        int originalWidth = request.OriginalWidth ?? request.Width, originalHeight = request.OriginalHeight ?? request.Height;
        Workspace.ValidateOriginalSize(originalWidth, originalHeight);
        var device = Workspace.Device(request.Backend);
        var precision = WorkflowService.ParsePrecision(request.Precision);
        string? calibration = workflows.Calibration(request.CalibrationId, request.ModelId);
        if (request.Confidence is < 0 or > 1 || !float.IsFinite(request.Confidence)) throw new ArgumentException("置信度必须在0到1之间。");
        await gate.WaitAsync(cancellation);
        try
        {
            return await Task.Run<object>(() =>
            {
                // 排队后才解码RGB，避免多个等待请求同时持有解码后的像素数组。
                byte[] pixels = Workspace.DecodePixels(request.Width, request.Height, request.Pixels);
                float scaleX = (float)originalWidth / request.Width, scaleY = (float)originalHeight / request.Height;
                var key = (model.Id, device, request.Confidence, precision, request.CalibrationId);
                var node = sessions.First;
                while (node is not null && node.Value.Key != key) node = node.Next;
                bool reusedSession = node is not null; double modelLoadMs = 0;
                if (node is null)
                {
                    // 先释放最旧会话，避免暂时持有第三份模型和 GPU 权重。
                    if (sessions.Count == 2) { sessions.Last!.Value.Detector.Dispose(); sessions.RemoveLast(); }
                    var loading = Stopwatch.StartNew();
                    int width = 640, height = 640;
                    if (calibration is not null) { var profile = System.Text.Json.JsonSerializer.Deserialize<CalibrationProfile>(File.ReadAllText(calibration))!; width = profile.InputShape[3]; height = profile.InputShape[2]; }
                    var detector = new YoloDetector(model.Path, new() { Device = device, Width = width, Height = height, Confidence = request.Confidence, CpuThreads = Math.Min(4, Environment.ProcessorCount), MaxDetections = 100, Precision = precision, CalibrationPath = calibration });
                    modelLoadMs = loading.Elapsed.TotalMilliseconds;
                    node = sessions.AddFirst(new Session(key, detector));
                }
                else { sessions.Remove(node); sessions.AddFirst(node); }
                var active = node.Value.Detector;
                var watch = Stopwatch.StartNew(); var instances = active.Predict(pixels, request.Width, request.Height); watch.Stop();
                return new { elapsedMs = watch.Elapsed.TotalMilliseconds, modelLoadMs, reusedSession, backend = active.BackendName, task = active.Task.ToString(),
                    image = new { originalWidth, originalHeight, processedWidth = request.Width, processedHeight = request.Height, resized = originalWidth != request.Width || originalHeight != request.Height, coordinateSystem = "original" },
                    instances = instances.Select(instance => new
                {
                    box = instance.Box with { X = instance.Box.X * scaleX, Y = instance.Box.Y * scaleY, Width = instance.Box.Width * scaleX, Height = instance.Box.Height * scaleY },
                    className = active.ClassNames.GetValueOrDefault(instance.Box.ClassId),
                    keypoints = instance.Keypoints.Select(point => point with { X = point.X * scaleX, Y = point.Y * scaleY }).ToArray(),
                    // Width/Height是位图尺寸，DisplayWidth/DisplayHeight是原图中的显示范围，避免放大概率数组。
                    mask = instance.Mask is { } mask ? new { x = mask.X * scaleX, y = mask.Y * scaleY, mask.Width, mask.Height, displayWidth = mask.Width * scaleX, displayHeight = mask.Height * scaleY,
                        probabilities = Convert.ToBase64String(mask.Probabilities.Select(p => (byte)Math.Clamp((int)MathF.Round(p * 255), 0, 255)).ToArray()) } : null
                }).ToArray() };
            }, cancellation);
        }
        finally { gate.Release(); }
    }

    /// <summary>在主机停止后释放缓存的模型与设备资源。</summary>
    public void Dispose() { foreach (var session in sessions) session.Detector.Dispose(); sessions.Clear(); gate.Dispose(); }
    private sealed record Session((string ModelId, ComputeDevice Device, float Confidence, InferencePrecision Precision, string? CalibrationId) Key, YoloDetector Detector);
}
