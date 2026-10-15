using System.Text.Json;
using SimdYolo;
using SimdYolo.Web;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = 72 * 1024 * 1024);
builder.Services.AddSingleton<Workspace>(); builder.Services.AddSingleton<InferenceService>(); builder.Services.AddSingleton<TrainingService>(); builder.Services.AddHostedService(provider => provider.GetRequiredService<TrainingService>());
var app = builder.Build(); var workspace = app.Services.GetRequiredService<Workspace>(); var training = app.Services.GetRequiredService<TrainingService>();
// 本示例绑定回环地址。拒绝跨站变更请求，避免本地训练被其他网页触发。
app.Use(async (context, next) =>
{
    if (!HttpMethods.IsGet(context.Request.Method) && context.Request.Headers.Origin.FirstOrDefault() is { Length: > 0 } origin &&
        (!Uri.TryCreate(origin, UriKind.Absolute, out var uri) || !uri.Authority.Equals(context.Request.Host.Value, StringComparison.OrdinalIgnoreCase)))
    { context.Response.StatusCode = 403; await context.Response.WriteAsJsonAsync(new { error = "仅接受当前示例页面的操作请求。" }); return; }
    try { await next(context); }
    catch (Exception error) when (error is ArgumentException or InvalidDataException or FormatException or JsonException or NotSupportedException or OverflowException or BadHttpRequestException or KeyNotFoundException or InvalidOperationException)
    {
        if (context.Response.HasStarted) throw;
        context.Response.StatusCode = error is KeyNotFoundException ? 404 : error is InvalidOperationException ? 409 : 400;
        await context.Response.WriteAsJsonAsync(new { error = error.Message });
    }
});
app.UseDefaultFiles(); app.UseStaticFiles();

bool vulkan = false; string? vulkanError = null;
try { using var backend = new VulkanBackend(); vulkan = true; }
catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or PlatformNotSupportedException or InvalidOperationException) { vulkanError = error.Message; }
object ModelView(WebModel model) => new { model.Id, model.Name, model.Format, model.Task, model.Trainable };
app.MapGet("/api/info", () => Results.Ok(new { version = "0.9.0", models = workspace.Models.Values.OrderBy(m => m.Name).Select(ModelView),
    backends = new[] { new { id = "Cpu", name = "CPU SIMD", inference = true, training = true }, new { id = "Vulkan", name = "Vulkan GPU", inference = vulkan, training = vulkan }, new { id = "Metal", name = "Metal（训练尚未支持）", inference = OperatingSystem.IsMacOS(), training = false } },
    trainingScope = "YOLOv8 检测头微调，骨干与特征金字塔冻结。梯度累加按图像平均；Vulkan 仅加速矩阵运算，BN、损失和 SGD 在 CPU。", vulkanError }));

app.MapGet("/api/sample", () =>
{
    var sample = workspace.Datasets["demo-squares"].Dataset.Images[0];
    return Results.File(PngImage.Encode(sample.Pixels, sample.Width, sample.Height), "image/png");
});

app.MapPost("/api/infer", async (InferenceRequest request, InferenceService inference, CancellationToken cancellation) => Results.Ok(await inference.Predict(request, cancellation)));

app.MapPost("/api/models", async (HttpRequest request, CancellationToken cancellation) =>
{
    string name = Path.GetFileName(Uri.UnescapeDataString(request.Headers["X-Filename"].ToString())); string extension = Path.GetExtension(name).ToLowerInvariant();
    if (name.Length is < 1 or > 120 || extension is not (".pt" or ".pth" or ".onnx" or ".sytrain")) throw new ArgumentException("支持上传 .pt、.onnx 或 .sytrain 模型。");
    if (request.ContentLength is < 1 or > 64 * 1024 * 1024) throw new ArgumentException("模型文件最大64 MiB。");
    string path = Path.Combine(workspace.DataRoot, "models", Guid.NewGuid().ToString("N") + "_" + name);
    try
    {
        await using (var file = File.Create(path)) await request.Body.CopyToAsync(file, cancellation);
        return Results.Ok(ModelView(workspace.RegisterModel(path, name)));
    }
    catch { if (File.Exists(path)) File.Delete(path); throw; }
});

app.MapGet("/api/datasets", () => Results.Ok(workspace.Datasets.Select(pair => new { id = pair.Key, name = pair.Value.Name, classes = pair.Value.Dataset.ClassNames, count = pair.Value.Dataset.Images.Count })));
app.MapPost("/api/datasets", async (DatasetRequest request, CancellationToken cancellation) =>
{
    var dataset = Workspace.CreateDataset(request); string id = Guid.NewGuid().ToString("N");
    await File.WriteAllTextAsync(Path.Combine(workspace.DataRoot, "datasets", id + ".json"), JsonSerializer.Serialize(request, Workspace.JsonOptions), cancellation);
    workspace.Datasets[id] = (request.Name, dataset); return Results.Ok(new { id, name = request.Name, classes = dataset.ClassNames, count = dataset.Images.Count });
});
app.MapGet("/api/training", () => Results.Ok(training.List()));
app.MapGet("/api/training/{id}", (string id) => Results.Ok(training.View(training.Find(id))));
app.MapPost("/api/training", (TrainRequest request) => Results.Ok(training.View(training.Enqueue(request))));
app.MapPost("/api/training/{id}/cancel", (string id) => Results.Ok(training.View(training.Cancel(id))));
app.MapPost("/api/training/{id}/resume", (string id, ResumeRequest request) => Results.Ok(training.View(training.Resume(id, request.Epochs))));
app.MapGet("/api/training/{id}/checkpoint", (string id) =>
{
    var job = training.Find(id); if (!File.Exists(job.Checkpoint)) return Results.NotFound(new { error = "尚无完整检查点。" });
    return Results.File(new FileStream(job.Checkpoint, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete), "application/octet-stream", $"simdyolo-{id[..8]}.sytrain");
});
app.Run();
