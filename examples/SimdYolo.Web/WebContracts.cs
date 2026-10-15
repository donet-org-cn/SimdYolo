namespace SimdYolo.Web;

/// <summary>浏览器已解码的 RGB 图像，避免服务器依赖原生图像解码库。</summary>
internal sealed record InferenceRequest(string ModelId, string Backend, int Width, int Height, string Pixels, float Confidence = .25f);
/// <summary>浏览器上传的单张检测训练图像和 YOLO 文本标签。</summary>
internal sealed record DatasetImageRequest(string Name, int Width, int Height, string Pixels, string Labels);
/// <summary>浏览器目录导入后形成的检测数据集。</summary>
internal sealed record DatasetRequest(string Name, string[] Classes, DatasetImageRequest[] Images);
/// <summary>检测头微调请求，BatchSize 表示梯度累加。</summary>
internal sealed record TrainRequest(string ModelId, string DatasetId, string Backend, int ImageSize = 128, int Epochs = 10, int BatchSize = 1, float LearningRate = .01f, int Seed = 26, string? ResumeId = null);
/// <summary>恢复任务时指定新的总轮数。</summary>
internal sealed record ResumeRequest(int Epochs);
/// <summary>模型目录中的可选条目；绝不允许客户端直接指定服务器路径。</summary>
internal sealed record WebModel(string Id, string Name, string Path, string Format, string Task, bool Trainable);
