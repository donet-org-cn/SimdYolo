using System.IO.Compression;
using System.Text;
using System.Text.Json;
using SimdYolo.Training;

namespace SimdYolo.Web;

/// <summary>纯.NET数据集拆分、图像读取、标注持久化和YOLO ZIP导出。</summary>
internal static class DatasetEndpoints
{
    private static readonly object MutationGate = new();

    /// <summary>注册数据编辑接口；所有路径来自受控数据集ID，修改时核对内容版本。</summary>
    internal static void Map(WebApplication app, Workspace workspace, TrainingService training)
    {
        (string Name, TrainingDataset Dataset) Find(string id) => workspace.Datasets.TryGetValue(id, out var data) ? data : throw new KeyNotFoundException("数据集不存在。");
        app.MapGet("/api/datasets/{id}/images", (string id) =>
        {
            var data = Find(id); return Results.Ok(new { id, data.Name, classes = data.Dataset.ClassNames, revision = data.Dataset.Sha256,
                images = data.Dataset.Images.Select((image, index) => new { index, image.Name, image.Width, image.Height, image.Labels }) });
        });
        app.MapGet("/api/datasets/{id}/images/{index:int}", (string id, int index) =>
        {
            var data = Find(id); if ((uint)index >= data.Dataset.Images.Count) throw new ArgumentException("图像编号不存在。");
            var image = data.Dataset.Images[index]; return Results.Ok(new { index, image.Name, image.Width, image.Height, image.Labels, pixels = Convert.ToBase64String(image.Pixels), classes = data.Dataset.ClassNames, revision = data.Dataset.Sha256 });
        });
        app.MapPut("/api/datasets/{id}/images/{index:int}/labels", (string id, int index, AnnotationUpdateRequest request) =>
        {
            lock (MutationGate)
            {
                var data = Find(id);
                if (training.DatasetBusy(id)) throw new InvalidOperationException("数据集正在训练或验证，完成后再修改标签。");
                if (data.Dataset.Sha256 != request.Revision) throw new InvalidOperationException("数据集已被其他操作修改，请刷新后重新编辑。");
                if ((uint)index >= data.Dataset.Images.Count || request.Labels is null || request.Labels.Length > 128 * 1024) throw new ArgumentException("图像编号或标注长度无效。");
                var images = data.Dataset.Images.ToArray(); images[index] = images[index] with { Labels = request.Labels };
                var updated = new TrainingDataset(data.Dataset.ClassNames, images); Save(workspace, id, data.Name, updated);
                return Results.Ok(new { id, index, revision = updated.Sha256, labels = updated.Images[index].Labels });
            }
        });
        app.MapPost("/api/datasets/{id}/split", (string id, DatasetSplitRequest request) =>
        {
            lock (MutationGate)
            {
                var data = Find(id); var (train, validation) = data.Dataset.Split(request.ValidationFraction, request.Seed);
                string trainId = Guid.NewGuid().ToString("N"), valId = Guid.NewGuid().ToString("N");
                Save(workspace, trainId, data.Name + " · train", train); Save(workspace, valId, data.Name + " · validation", validation);
                return Results.Ok(new { trainingId = trainId, validationId = valId, trainingCount = train.Images.Count, validationCount = validation.Images.Count, request.Seed, request.ValidationFraction });
            }
        });
        app.MapGet("/api/datasets/{id}/export", (string id) =>
        {
            var data = Find(id); using var bytes = new MemoryStream();
            using (var zip = new ZipArchive(bytes, ZipArchiveMode.Create, true))
            {
                for (int i = 0; i < data.Dataset.Images.Count; i++)
                {
                    var image = data.Dataset.Images[i]; string stem = Path.GetFileNameWithoutExtension(image.Name); if (stem.Length > 60) stem = stem[..60]; stem = i.ToString("D6") + "_" + stem;
                    using (var stream = zip.CreateEntry("images/" + stem + ".png").Open()) stream.Write(PngImage.Encode(image.Pixels, image.Width, image.Height));
                    using var label = new StreamWriter(zip.CreateEntry("labels/" + stem + ".txt").Open(), new UTF8Encoding(false)); label.Write(image.Labels);
                }
                using (var names = new StreamWriter(zip.CreateEntry("classes.txt").Open(), new UTF8Encoding(false))) names.Write(string.Join('\n', data.Dataset.ClassNames));
                using var yaml = new StreamWriter(zip.CreateEntry("data.yaml").Open(), new UTF8Encoding(false)); yaml.Write("path: .\ntrain: images\nnames: " + JsonSerializer.Serialize(data.Dataset.ClassNames) + "\n");
            }
            return Results.File(bytes.ToArray(), "application/zip", "simdyolo-dataset-" + id + ".zip");
        });
    }
    private static void Save(Workspace workspace, string id, string name, TrainingDataset dataset)
    {
        if (!Guid.TryParseExact(id, "N", out _) && id != "demo-squares") throw new ArgumentException("数据集ID无效。");
        var request = new DatasetRequest(name, dataset.ClassNames.ToArray(), dataset.Images.Select(image => new DatasetImageRequest(image.Name, image.Width, image.Height, Convert.ToBase64String(image.Pixels), image.Labels)).ToArray());
        string path = Path.Combine(workspace.DataRoot, "datasets", id + ".json");
        File.WriteAllText(path + ".tmp", JsonSerializer.Serialize(request, Workspace.JsonOptions)); File.Move(path + ".tmp", path, true); workspace.Datasets[id] = (name, dataset);
    }
}
