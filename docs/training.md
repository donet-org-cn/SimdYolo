# 纯 C# 五代 YOLO 检测头训练

**当前源码0.11.1**继续支持五代检测头微调，训练集成/Vulkan回归19项通过。[当前验证](../artifacts/validation/0.11.1/README.md) · [下载与文档导航](README.md)。0.11.0起加入独立验证集、训练前/逐轮P/R与mAP、best.sytrain、早停和验证状态恢复；训练结果可纯.NET导出ONNX，网页支持检测框标注和预标注。[完整API](model-workflows.md) · [0.11.0功能验证](../artifacts/validation/0.11.0/README.md)。

0.10.1 源码与发布包支持 **YOLOv8、Ultralytics YOLOv9、YOLOv10、YOLO11、YOLO26 的检测头微调**，CPU 使用 C# SIMD，Vulkan 加速冻结网络和较大的前后向 GEMM。无需 Python、PyTorch、TorchSharp、ONNX Runtime、BLAS 或 CUDA 运行依赖。

0.10.1训练优化包括SIMD激活/BN/SGD、直接点积反向、1×1零复制、有界卷积工作区与小矩阵设备选择。[训练速度、资源和训练后模型对比](../artifacts/training-optimization-0.10.1.md)对应0.10.1程序集；0.11.1没有重新测量训练速度。

| 系列 | 仓库实际训练权重 | 回归与分类头 | 训练目标 |
| --- | --- | --- | --- |
| YOLOv8 | `yolov8n.pt` | 普通 Conv-BN-SiLU | TAL top10 + BCE + CIoU + DFL |
| Ultralytics YOLOv9 | `yolov9t.pt` | 普通 Conv-BN-SiLU | TAL top10 + BCE + CIoU + DFL |
| YOLOv10 | `yolov10n.pt` | 含深度卷积，one2many/one2one 两套参数 | 两头相加，TAL top10 / top1，DFL |
| YOLO11 | `yolo11n.pt` | 分类分支含深度可分离卷积 | TAL top10 + BCE + CIoU + DFL |
| YOLO26 | `yolo26n.pt` | 深度可分离分类、双头，`reg_max=1` | TAL top10 / top7→1，渐进双头权重，CIoU + 归一化距离 L1 |

本次覆盖表中的五份权重。其他 n/s/m/l/x 尺度按受支持模块结构解析，但未逐份验证；YOLOv9 原仓库的 DualDDetect/TripleDDetect、只有推理分支的融合 PT 不在此训练支持表内。不能据系列名称推断所有衍生模型都兼容。

范围保持为**冻结骨干和 FPN、仅更新完整检测头**。分割/姿态训练、全参数训练、Metal 训练、AdamW/MuSGD、EMA与数据增强尚未实现。0.11.0已支持调用方检测数据集的托管mAP验证，指标范围见新增文档。YOLO26 的损失与固定参考对应，优化器仍为 SGD，不声称复刻官方完整训练配方。普通 ONNX 只支持推理，不能作为本训练器的初始化结构。

Vulkan 是混合训练：矩阵乘加项数大于4,000,000的普通卷积 GEMM在GPU，小矩阵使用CPU SIMD以避免提交/回读开销；头部深度/分组卷积、im2col、BN、激活、标签分配、损失和SGD在CPU。CPU线程遵守`CpuThreads`。[实际训练性能](../artifacts/training-optimization-0.10.1.md)不能以推理速度推断。

## 训练后模型对比

0.10.1补充真实图片训练结果：COCO128按种子2611划分96张微调/32张验证，80类、256×256、相同初始状态、固定顺序和冻结特征，各192步SGD。CPU与原版CPU、Vulkan与原版CUDA分别比较参数、BN、动量与逐步损失；重新加载两边保存的模型，执行完整网络并比较P/R、mAP50及mAP50–95。

原始COCO128的train=val设置未用于本次微调划分；预训练权重已见过COCO train2017，不能把这32张称为独立于预训练的泛化验证集。两边的短程微调都有验证精度下降，本轮不声称提升精度或复刻官方全参数训练配方。设置、训练前后指标、误差与复现方法见[0.10.1报告](../artifacts/training-optimization-0.10.1.md)及[验证目录](../artifacts/validation/0.10.1/README.md)。

![0.10.1 保存模型效果对照](../artifacts/training-results-0.10.1.png)

## 启动网页或 C# API

需要 .NET 10 SDK；Vulkan 需要兼容驱动及系统 loader。可下载[0.10.1 训练库与网页运行包](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1)，或从源码启动：

```powershell
git clone https://github.com/hnlyf168/SimdYolo.git
cd SimdYolo
git lfs pull
dotnet run --project examples/SimdYolo.Web -c Release
```

打开 **http://127.0.0.1:5188**，导入每行 `class cx cy w h` 的归一化 YOLO 检测标注，选择五份模型之一。支持真实损失曲线、取消、恢复、检查点下载/重新上传及推理。网页说明与数据限制见[示例文档](../examples/SimdYolo.Web/README.md)。

![0.10.0 网页中实际完成 YOLO26 检测头微调](../artifacts/validation/0.10.0/web-browser/web-training.png)

引用 `src/SimdYolo.Training/SimdYolo.Training.csproj`：

```csharp
using SimdYolo;
using SimdYolo.Training;

var dataset = TrainingDataset.CreateDemo(count: 8, seed: 26);
string path = "models/pt/yolo26n.pt"; // 也可以是 v8n、v9t、v10n、11n
Console.WriteLine(YoloTrainer.CanTrain(path)); // 按实际结构探测，不看文件名
using var trainer = new YoloTrainer(path, dataset, new()
{
    Device = ComputeDevice.Cpu, // 可显式选择 Vulkan
    ImageSize = 128,
    Epochs = 20,
    BatchSize = 1,
    LearningRate = .003f,
    CpuThreads = 4
});
float before = trainer.EvaluateLoss();
trainer.Train(".cache/my-training", p =>
    Console.WriteLine($"epoch={p.Epoch}, step={p.Step}, loss={p.TotalLoss:F5}"));
float after = trainer.EvaluateLoss();
```

`Train` 为同步方法，可以放入自己的后台线程；同一会话禁止并发训练、评估、保存或释放。自定义数据由调用方提供连续 RGB24 字节，不引入图片编解码依赖：

```csharp
var dataset = new TrainingDataset(
    ["person", "helmet"],
    [new TrainingImage("001.jpg", rgb24, width, height,
        "0 0.50 0.50 0.40 0.60\n1 0.50 0.25 0.20 0.15")]);
```

类别从 0 开始，框须位于原图内；letterbox 保持比例。自定义类别按名称复用原分类通道，否则按种子初始化。双头分别重建输出通道，训练与推理使用正确的分支。

`BatchSize` 是**梯度累加的图像数**，每图 BN 独立统计，再按图像数平均梯度；不等同联合批次 BN。全背景图像只训练分类头；与原版一样，框参数梯度未连接时跳过 SGD/衰减，有正样本时各尺度的显式零梯度仍参与 SGD/衰减。

## 损失和恢复语义

固定独立参考为 Ultralytics **8.4.0**。v10 使用保留的官方 `E2EDetectLoss` 兼容目标；YOLO26 使用 `E2ELoss`。前者两头直接相加，后者 one2many 从 0.8 按轮衰减至 0.1，one2one 为其补值。YOLO26 的 TAL 先 top7、处理目标冲突，再 top1，不能简单替换成首次 top1。

YOLO26 的第三损失分量是按输入宽高归一化的距离 L1，**没有 DFL softmax**。为保持客户端兼容，`TrainingProgress.DflLoss` 和网页 DFL 历史字段保留名称；对 YOLO26 表示该 L1 分量。`TotalLoss` 返回本次实际优化的加权目标。`EvaluateLoss()` 使用 BN 运行统计，且固定初始 0.8/0.2 权重，保证训练前后评估口径一致；它仍是训练数据损失，不是 mAP。

```csharp
using var resumed = YoloTrainer.Resume(
    ".cache/my-training/last.sytrain", dataset,
    totalEpochs: 30, device: ComputeDevice.Cpu, cpuThreads: 4);
resumed.Train(".cache/my-training");

using var detector = new YoloDetector(".cache/my-training/last.sytrain");
var instances = detector.Predict(rgb24, width, height);
var metadata = ModelInfo.Read(".cache/my-training/last.sytrain");
```

`totalEpochs` 是恢复后的总轮数，必须大于已完成轮数。恢复严格校验数据集 SHA256（类别、顺序、RGB 和标注）、参数形状及动量。学习率、种子、累加配置和图像尺寸来自检查点，可以切换 CPU/Vulkan 和线程数。**YOLO26 的损失调度总轮数在首次创建时固定并保存**；恢复时延长训练不会重置调度，超过原定轮数继续使用 0.1/0.9。

每个完整轮次原子保存 `last.sytrain`，另写 `history.jsonl`。取消抛出 `OperationCanceledException` 并回退到最近完整轮次，包括权重、BN、梯度连接状态、动量和计数；过程历史可保留未完成轮次的记录。

新归档为 `SimdYolo.Training/2`，包含原始 PT、全部单/双头参数、BN、动量、类别和损失调度；继续读取 0.9.0 的 `/1` v8 归档。`ModelInfo.EndToEnd` 对 v10/26 训练模型返回正确模式。`.sytrain` 不能重命名成 PyTorch `.pt`，尚不导出 ONNX。

官方权重按原始 SHA256 识别并保留 AGPL-3.0 及来源。自定义权重保留供应方许可要求；项目源码未新增许可证。

## 测试与评测

[完整评测与图表](../artifacts/training-evaluation-0.10.0.md)包含 CPU/Vulkan × 五模型数值对照、小目标/空标签、SGD/动量、真实损失下降、取消恢复、检查点推理、网页和训练速度。原始证据及复现命令见[本版验证目录](../artifacts/validation/0.10.0/README.md)。

```powershell
dotnet run --project tests/SimdYolo.Training.Tests -c Release -- --integration --vulkan
python tools/run-training-matrix.py
dotnet run --project tests/SimdYolo.Training.Tests -c Release -- test-matrix .cache/training-010/cpu --epochs 100
dotnet run --project tests/SimdYolo.Training.Tests -c Release -- test-matrix .cache/training-010/vulkan --epochs 100 --vulkan
python tools/benchmark-training.py
```

Python/PyTorch、Pillow、Node 仅用于开发验证，不进入库或网页运行依赖。参考源码需要放置在 `.cache/yolo26-release-reference`，对照不自动下载或执行任意远端权重。普通 ONNX Runtime 不是本轮训练基线，没有把推理延迟当作训练耗时。
