namespace SimdYolo.Web;

/// <summary>浏览器缩放后上传的 RGB 图像；可携带原图尺寸，使结果使用原图坐标。</summary>
/// <param name="ModelId">模型目录中的标识。</param>
/// <param name="Backend">CPU、Vulkan 或 Metal 后端名称。</param>
/// <param name="Width">实际上传的 RGB 图像宽度。</param>
/// <param name="Height">实际上传的 RGB 图像高度。</param>
/// <param name="Pixels">RGB24 数据的 Base64 编码。</param>
/// <param name="Confidence">检测置信度阈值。</param>
/// <param name="OriginalWidth">缩放前原图宽度；与原图高度同时省略时使用上传尺寸。</param>
/// <param name="OriginalHeight">缩放前原图高度；与原图宽度同时省略时使用上传尺寸。</param>
/// <param name="Precision">FP32、FP16混合或INT8推理模式。</param><param name="CalibrationId">可选已完成校准任务标识。</param>
internal sealed record InferenceRequest(string ModelId, string Backend, int Width, int Height, string Pixels, float Confidence = .25f, int? OriginalWidth = null, int? OriginalHeight = null, string Precision = "Float32", string? CalibrationId = null);
/// <summary>浏览器上传的单张检测训练图像和 YOLO 文本标签。</summary>
internal sealed record DatasetImageRequest(string Name, int Width, int Height, string Pixels, string Labels);
/// <summary>浏览器目录导入后形成的检测数据集。</summary>
internal sealed record DatasetRequest(string Name, string[] Classes, DatasetImageRequest[] Images);
/// <summary>检测头微调请求，BatchSize 表示梯度累加。</summary>
internal sealed record TrainRequest(string ModelId, string DatasetId, string Backend, int ImageSize = 128, int Epochs = 10, int BatchSize = 1, float LearningRate = .01f, int Seed = 26, string? ResumeId = null, string? ValidationDatasetId = null, int EarlyStoppingPatience = 0);
/// <summary>恢复任务时指定新的总轮数。</summary>
internal sealed record ResumeRequest(int Epochs);
/// <summary>模型目录中的可选条目；绝不允许客户端直接指定服务器路径。</summary>
internal sealed record WebModel(string Id, string Name, string Path, string Format, string Task, bool Trainable);
/// <summary>后台评估、比较、ONNX导出、校准或预标注任务。</summary>
internal sealed record WorkflowRequest(string Kind, string ModelId, string? DatasetId = null, string? OtherModelId = null, int ImageSize = 640, string Backend = "Cpu", string Precision = "Float32", string? CalibrationId = null, int ImageIndex = 0, float Confidence = .25f);
/// <summary>使用数据摘要进行乐观并发校验的标签更新。</summary>
internal sealed record AnnotationUpdateRequest(string Labels, string Revision);
/// <summary>确定性训练/验证拆分请求。</summary>
internal sealed record DatasetSplitRequest(double ValidationFraction = .2, int Seed = 26);
