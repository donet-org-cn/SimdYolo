using System.Diagnostics;

namespace SimdYolo.Web;

/// <summary>串行执行推理，复用最近两个模型会话，减少重复权重解析、上传和设备初始化。</summary>
internal sealed class InferenceService(Workspace workspace) : IDisposable
{
    private readonly SemaphoreSlim gate = new(1, 1);
    private readonly LinkedList<Session> sessions = new();

    internal async Task<object> Predict(InferenceRequest request, CancellationToken cancellation)
    {
        if (!workspace.Models.TryGetValue(request.ModelId, out var model)) throw new KeyNotFoundException("模型不存在。");
        byte[] pixels = Workspace.DecodePixels(request.Width, request.Height, request.Pixels);
        var device = Workspace.Device(request.Backend);
        if (request.Confidence is < 0 or > 1 || !float.IsFinite(request.Confidence)) throw new ArgumentException("置信度必须在0到1之间。");
        await gate.WaitAsync(cancellation);
        try
        {
            return await Task.Run<object>(() =>
            {
                var key = (model.Id, device, request.Confidence);
                var node = sessions.First;
                while (node is not null && node.Value.Key != key) node = node.Next;
                bool reusedSession = node is not null; double modelLoadMs = 0;
                if (node is null)
                {
                    // 先释放最旧会话，避免暂时持有第三份模型和 GPU 权重。
                    if (sessions.Count == 2) { sessions.Last!.Value.Detector.Dispose(); sessions.RemoveLast(); }
                    var loading = Stopwatch.StartNew();
                    var detector = new YoloDetector(model.Path, new() { Device = device, Width = 640, Height = 640, Confidence = request.Confidence, CpuThreads = Math.Min(4, Environment.ProcessorCount), MaxDetections = 100 });
                    modelLoadMs = loading.Elapsed.TotalMilliseconds;
                    node = sessions.AddFirst(new Session(key, detector));
                }
                else { sessions.Remove(node); sessions.AddFirst(node); }
                var active = node.Value.Detector;
                var watch = Stopwatch.StartNew(); var instances = active.Predict(pixels, request.Width, request.Height); watch.Stop();
                return new { elapsedMs = watch.Elapsed.TotalMilliseconds, modelLoadMs, reusedSession, backend = active.BackendName, task = active.Task.ToString(), instances = instances.Select(instance => new
                {
                    box = instance.Box, className = active.ClassNames.GetValueOrDefault(instance.Box.ClassId), keypoints = instance.Keypoints,
                    mask = instance.Mask is { } mask ? new { mask.X, mask.Y, mask.Width, mask.Height, probabilities = Convert.ToBase64String(mask.Probabilities.Select(p => (byte)Math.Clamp((int)MathF.Round(p * 255), 0, 255)).ToArray()) } : null
                }).ToArray() };
            }, cancellation);
        }
        finally { gate.Release(); }
    }

    /// <summary>在主机停止后释放缓存的模型与设备资源。</summary>
    public void Dispose() { foreach (var session in sessions) session.Detector.Dispose(); sessions.Clear(); gate.Dispose(); }
    private sealed record Session((string ModelId, ComputeDevice Device, float Confidence) Key, YoloDetector Detector);
}
