# 验证、导出、标注与低精度（当前源码0.11.2）

四项均使用纯 C# / .NET 10 实现。应用运行不调用 Python、PyTorch、ORT、TorchSharp、BLAS 或第三方原生计算库。浏览器承担 JPG/PNG 解码、画框交互和透明图合成；GPU 仍通过系统 Vulkan/Metal API 工作。开发目录里的 Python/ORT 只用于独立检查，不是应用依赖。

本页功能自0.11.0加入，当前0.11.2继续提供，并优化图片推理、掩码及网页目标分离。[0.11.1使用与下载](README.md) · [优化表格与图](../artifacts/image-pipeline-optimization-0.11.1.md)。[历史0.10.1下载包](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1)不包含这些功能。启动：

~~~powershell
git lfs pull
dotnet run --project examples/SimdYolo.Web -c Release
~~~

打开 **http://127.0.0.1:5188**。新增“数据标注”和“评估与导出”页面，训练和推理页也增加了设置。

## 训练验证、最佳模型与早停

五代检测头训练均可设置独立验证集。训练前及每个完整轮次后，重新加载实际保存的模型，执行整网推理，计算 Precision、Recall、mAP50、mAP50:95、逐类 TP/FP/FN 和含背景的混淆矩阵。验证不使用训练 loss 推算指标。

- AP采用101点精确率包络插值，IoU 0.50–0.95，步长0.05；无真值类别不参加宏平均。
- P/R、误检漏检及混淆矩阵默认置信度0.25、IoU0.50。AP预测使用置信度0.001、NMS IoU0.7、每图最多300个结果；端到端模型保持本身的TopK和免NMS行为。
- 混淆矩阵行是真值、列是预测，最后一行/列为背景。预训练模型与数据集按类别名映射，不将预训练类别0直接当作自定义类别0。
- 当前是检测框指标，没有COCO crowd/ignore、面积分组、分割mask AP或关键点OKS AP，不能冒充完整的这些评估。

在网页导入数据后，点击“拆分当前数据：80%训练 / 20%验证”，或选择自己的独立验证集。两侧类别及顺序必须一致；相同尺寸和RGB内容的照片会拒绝。种子固定、拆分可复现；相似照片、视频相邻帧等语义泄漏仍需要使用者按场景划分。

~~~csharp
using SimdYolo;
using SimdYolo.Training;

// dataset由调用方准备，包含RGB24、类别名和YOLO检测标签。
var split = dataset.Split(validationFraction: 0.2, seed: 26);
using var trainer = new YoloTrainer("models/pt/yolo26n.pt", split.Training, new()
{
    ImageSize = 320, Epochs = 100, CpuThreads = 4,
    ValidationDataset = split.Validation,
    EarlyStoppingPatience = 10, MinimumImprovement = 0.0001
});
trainer.Train("runs/example",
    progress: step => Console.WriteLine(step.TotalLoss),
    validationProgress: epoch => Console.WriteLine(epoch.Map50To95));
~~~

last.sytrain保存最近完整轮次及SGD动量；best.sytrain按mAP50:95保存最佳状态，包含训练前基线。如果所有训练轮次都变差，最佳轮次可以为0。耐心值0关闭早停；连续未改善达到设置轮数后正常完成，StoppedEarly为真。

目录包含baseline-validation.json、last-validation.json、best-validation.json和validation-history.jsonl；后续轮次才生成last验证文件。取消恢复会补做已保存但尚未完成的验证轮次。恢复必须提供相同验证集；保留原目录best文件才能延续原最佳模型，否则以当前完整检查点重新建立基线。

~~~csharp
using var resumed = YoloTrainer.Resume("runs/example/last.sytrain",
    split.Training, totalEpochs: 150, cpuThreads: 4,
    validationDataset: split.Validation);
resumed.Train("runs/resumed");

var result = DetectionEvaluator.Evaluate("runs/example/best.sytrain",
    split.Validation, new() { ImageSize = 320, CpuThreads = 4 });
// 原模型与训练后模型各调用一次，采用同一验证集、设备及阈值。
~~~

网页“评估与导出”支持单模型评估或双模型对比，显示两行指标、对照模型逐类误检/漏检及混淆矩阵。下载JSON包含两边完整逐图真值和预测。训练任务自动注册“训练模型”和“最佳模型”供选择。

## 训练结果导出标准ONNX

~~~csharp
var exported = ModelExporter.ExportOnnx(
    "runs/example/best.sytrain", "exports/custom.onnx");
// 普通完整PT可指定64到1280、32倍数的正方形输入尺寸。
ModelExporter.ExportOnnx("models/pt/yolov8n.pt",
    "exports/yolov8n-640.onnx", imageSize: 640);
~~~

导出器用符号设备追踪完整网络，生成标准ONNX节点及嵌入式FP32权重，包含类别名、任务、输入形状、端到端标记和源模型SHA256。不执行外部程序，不需要protobuf NuGet包。输出为**静态batch=1、opset17、IR9**；SYTRAIN使用保存的训练尺寸。继续训练仍需SYTRAIN，ONNX不是训练快照。

五代检测原模型和实际训练后的自定义两类别模型共10份，另有v8/26分割及姿态4份，以及v10/26超过300锚点的两份320输入导出，共16份已通过本库回读和独立ONNX checker/ORT检查。范围以这些权重和已实现模块为准；无法追踪的算子明确拒绝，不将输入相关数值烘焙为常量。没有任意state_dict、训练图、动态batch及INT8 Q/DQ图导出。

网页选择PT/SYTRAIN后点击“PT / SYTRAIN → 标准 ONNX”。完成后可下载，导出模型也会加入推理列表。

## 网页标注与模型预标注

1. 训练页导入图像目录并填写类别。缺失同名TXT的图片以空标签导入，可到标注页画框。
2. 选择数据集和图片，空白处拖动画框，点框选中、拖动框体移动、拖动四角缩放。在已有框内新建框时按住Shift。
3. 下拉修改选中框类别，Delete删除，Ctrl+Z撤销；点击“保存标注”写入数据集。
4. 选择检测模型，点击“生成预标注建议”。只映射同名类别；建议替换当前画布框，**人工审核并点击保存后**才写入数据集。
5. 下载YOLO数据集ZIP，包含PNG图片、labels、classes.txt与data.yaml。PNG由.NET编码；data.yaml仅声明当前数据的train路径，不伪造独立val路径。

排队或训练任务使用中的数据集不可修改。保存必须匹配数据摘要，旧页面覆盖新标注返回409；切换图像前需保存当前编辑。后续新训练和导出使用新标签，已完成任务保留原指标。当前编辑对象是检测矩形框，没有多边形、实例掩码或关键点编辑。

## CPU FP16混合与INT8实验推理

~~~csharp
using var detector = new YoloDetector("models/pt/yolov8n.pt", new()
{
    Device = ComputeDevice.Cpu, Width = 640, Height = 640,
    CpuThreads = 4, Precision = InferencePrecision.Int8
});
var instances = detector.Predict(rgbBytes, width, height);
~~~

| 模式 | 实际计算 | 当前边界 |
| --- | --- | --- |
| Float32（默认） | 原FP32 CPU SIMD或Vulkan/Metal路径 | 保持默认行为 |
| Float16 | 卷积/矩阵操作数舍入到Half，FP32 CPU SIMD累加 | 混合精度；没有原生FP16乘加，不承诺降低整个会话内存 |
| Int8动态 | 对称有符号INT8、逐输出通道权重尺度、逐卷积动态激活尺度、AVX2整数点积 | 每段最多4096项INT32累加，跨段INT64防溢出；无AVX2时使用托管整数路径 |
| Int8校准 | 上述整数计算，卷积激活尺度改为代表性图片统计的最大绝对值 | 绑定模型SHA256、输入形状和卷积顺序；矩阵乘法仍用动态尺度 |

非线性、归约、解码及输出保持FP32。这是FP32图上的混合执行，不是全图8位存储、QAT或原生ONNX量化图支持；仅常量权重复用转换缓存，动态卷积权重每次重新转换。低精度明确拒绝GPU请求，网页切到CPU；**Vulkan/Metal目前仍为FP32**。普通PT/ONNX/SYTRAIN通过精度选项运行受支持运算。

静态校准使用与业务场景接近、不混入评估集的代表性照片：

~~~csharp
ModelCalibration.Calibrate("models/pt/yolov8n.pt",
    calibrationImages.Select(i => new CalibrationImage(i.Pixels, i.Width, i.Height)),
    "calibration.json", imageSize: 640, threads: 4);
using var calibrated = new YoloDetector("models/pt/yolov8n.pt", new()
{
    Device = ComputeDevice.Cpu, Width = 640, Height = 640,
    Precision = InferencePrecision.Int8, CalibrationPath = "calibration.json"
});
~~~

网页生成校准后，在推理页选择同一模型、INT8和该校准记录。网页按校准形状设置网络输入；模型或形状不同明确拒绝。更换训练权重需重新校准。

**本机低精度实现尚未取得速度优势。** INT8内核和转换开销高于现有FP32执行器，FP16混合也增加转换成本；默认保持FP32。五模型、32张真实照片的精度及320尺寸单进程延迟见[本轮验证](../artifacts/validation/0.11.0/README.md)。不以“INT8”名称推断更快或准确率不变，不改写历史640尺寸CPU/GPU排名。

## API与复现

网页使用受控ID，不接受任意服务器路径：

| API | 用途 |
| --- | --- |
| GET /api/datasets/{id}/images[/index] | 列表或RGB、标注、版本 |
| PUT /api/datasets/{id}/images/{index}/labels | labels + revision保存 |
| POST /api/datasets/{id}/split | validationFraction + seed拆分 |
| GET /api/datasets/{id}/export | YOLO数据集ZIP |
| POST /api/workflows | kind为evaluate / compare / export / calibrate / prelabel |
| GET /api/workflows[/{id}] | 进度、指标与下载URL |
| POST /api/workflows/{id}/cancel | 请求取消 |
| GET /api/workflows/{id}/download | 完成后下载ONNX或JSON |
| GET /api/training/{id}/best | 最佳SYTRAIN下载 |

DataRoot/workflows持久保存结果，重启时中断任务标为cancelled。训练与工作流各自串行、队列各4项，可能与推理争用资源；计时应停止其他任务。导出同步阶段无法中途打断，会在返回后检查取消，不注册被取消模型。

~~~powershell
dotnet run --project tests/SimdYolo.Training.Tests -c Release -- features .cache/features-011
dotnet run --project tests/SimdYolo.Training.Tests -c Release -- --integration --vulkan
dotnet run --project tests/SimdYolo.Tests -c Release -- --vulkan
# 仅开发环境的独立检查，需要onnx、onnxruntime、numpy：
python tools/validate-managed-onnx.py
# 已启动网页；Node和Edge仅驱动交互：
node tools/validate-web-browser.mjs --image examples/SimdYolo.Web/wwwroot/samples/bus.jpg --features --large-images --cutout --output .cache/current-web-validation
~~~

0.11.0独立ONNX及低精度精度/速度证据保留原版本；0.11.2推理116项、训练19项和完整网页41条检查见[当前验证](../artifacts/validation/0.11.2/README.md)。复现输出写入.cache，避免覆盖已经冻结的验证记录。

ONNX标准参考：[图、张量与元数据](https://onnx.ai/onnx/intro/concepts.html)、[Conv](https://onnx.ai/onnx/operators/onnx__Conv.html)、[TopK](https://onnx.ai/onnx/operators/onnx__TopK.html)。导出采用标准算子，独立运行检查用于验证实际兼容性。
