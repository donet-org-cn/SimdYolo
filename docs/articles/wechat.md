# 纯 C# 跑 YOLO：推理、检测头微调和网页版，这次一起交付

> 0.10.0 源码更新：检测头训练已扩展到 v8、v9、v10、YOLO11、YOLO26，网页也可选择五个系列。新增双分支、深度卷积、YOLO26 标签分配与渐进损失；[训练测试与评测](../../artifacts/training-evaluation-0.10.0.md)。下文推理图表与 0.9.0 下载链接继续对应当时的冻结版本，旧发布包只支持 v8 训练。

![五代检测头的实际训练曲线，合成单图流程验证](../../artifacts/training-loss-0.10.0.png)

![SimdYolo 0.9.0：纯 C# YOLO 推理与检测头微调](assets/0.9.0/01-wechat-cover.png)

**代码：** [hnlyf168/SimdYolo](https://github.com/hnlyf168/SimdYolo)

**下载：** [0.9.0 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.9.0) · [33个模型包](https://github.com/hnlyf168/SimdYolo/releases/download/v0.9.0/SimdYolo-models-v0.9.0.zip)

我希望在 .NET 应用里，模型加载、计算和后处理都能直接用 C# 完成。于是有了 SimdYolo：直接读支持范围内的 `.pt` 和 ONNX，CPU 用 SIMD，GPU 走 Vulkan/Metal API。

现在的 0.9.0 又增加了两个入口：**真实的 YOLOv8 检测头微调，以及能在浏览器里操作的推理、训练示例。** 这篇文章和配图全部换成当前版本，性能数据也重新测量，不再沿用 0.7.x 的表。

## 先看速度：把原版、C# 和 ORT 放在同一台机器上

测试时间：2026-10-15 10:53:49 +0800。硬件是 i7-8700K、GTX 1050 Ti、31.93 GiB 内存。11个模型、PT/ONNX、CPU/GPU，完成176个独立窗口、39,134次推理，224项输出校验通过。

统一 FP32、batch=1、640×640、CPU线程上限12。每配置两轮，每轮预热60次，再计至少4秒/40帧。GPU 包含输入上传和全部输出回读；图片预处理、NMS、最终掩码后处理、加载和冷首帧不计入下面的稳态数值。

## CPU / GPU 速度排名

每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗；重复窗口取各窗口中位数的中位数，按未舍入值排序。小幅差距不代表稳定胜出或统计显著性。

### CPU 速度排名

| 模型 | 原版 PyTorch CPU | 我们的 `.pt` CPU | 我们的 ONNX CPU | ORT CPU（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 44.98 ④ | **21.74 ①** | 22.24 ② | 24.45 ③ |
| yolov9t | 77.56 ④ | **30.20 ①** | 33.62 ③ | 31.05 ② |
| yolov10n | 49.19 ④ | **21.57 ①** | 22.64 ② | 22.92 ③ |
| yolo11n | 51.80 ④ | **21.71 ①** | 22.72 ② | 23.64 ③ |
| yolo26n | 54.27 ④ | 22.76 ③ | 22.66 ② | **18.54 ①** |
| yolov8n-seg | 57.56 ④ | **28.07 ①** | 28.61 ② | 34.38 ③ |
| yolo11n-seg | 65.14 ④ | **27.99 ①** | 29.48 ② | 33.52 ③ |
| yolo26n-seg | 69.02 ④ | 30.58 ② | 31.44 ③ | **29.57 ①** |
| yolov8n-pose | 56.94 ④ | **26.88 ①** | 28.04 ② | 31.25 ③ |
| yolo11n-pose | 66.76 ④ | **23.27 ①** | 29.65 ③ | 25.35 ② |
| yolo26n-pose | 59.78 ④ | 24.98 ② | 25.49 ③ | **21.91 ①** |
| 第一名次数（完整四配置 11/11 模型） | 0 | 8 | 0 | 3 |

### GPU 速度排名

| 模型 | 原版 PyTorch CUDA | 我们的 `.pt` Vulkan | 我们的 ONNX Vulkan | ORT CUDA（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | **9.83 ①** | 10.39 ② | 10.63 ③ | 12.69 ④ |
| yolov9t | 22.27 ④ | **12.87 ①** | 14.09 ② | 17.81 ③ |
| yolov10n | 10.67 ③ | **10.07 ①** | 10.28 ② | 13.52 ④ |
| yolo11n | **10.52 ①** | 10.62 ② | 10.66 ③ | 13.69 ④ |
| yolo26n | 12.37 ③ | **10.99 ①** | 11.08 ② | 13.21 ④ |
| yolov8n-seg | **12.83 ①** | 13.75 ② | 13.88 ③ | 16.56 ④ |
| yolo11n-seg | 13.61 ② | **13.50 ①** | 13.66 ③ | 17.58 ④ |
| yolo26n-seg | 16.44 ③ | 14.75 ② | **14.57 ①** | 17.23 ④ |
| yolov8n-pose | **10.23 ①** | 10.55 ② | 10.86 ③ | 13.17 ④ |
| yolo11n-pose | 11.18 ② | **11.08 ①** | 11.26 ③ | 14.33 ④ |
| yolo26n-pose | 13.52 ③ | **12.00 ①** | 12.22 ② | 14.11 ④ |
| 第一名次数（完整四配置 11/11 模型） | 4 | 6 | 1 | 0 |


①表示本次窗口汇总数字最快；小差距和两轮换位不能解释成稳定领先，也不能据此说 C# 在所有模型上都比 ORT 好。

![当前版本检测模型 CPU 实测](assets/0.9.0/04-cpu-detection.png)

![当前版本分割和姿态 CPU 实测](assets/0.9.0/05-cpu-vision.png)

![当前版本检测模型 GPU 实测](assets/0.9.0/06-gpu-detection.png)

图中细线为两个窗口中位数的范围，不是置信区间。对照使用原版 Ultralytics `fuse()`，ORT 开启图优化；没有拿这些结果外推到 TensorRT、`torch.compile` 或 FP16/INT8。

## 快，也要看第一次打开和内存

第一次使用模型要经历读取、解析、准备后端和首次计算。它和预热后的稳态延迟是两回事。

![模型加载、首次前向和稳态分开看](../../artifacts/performance-0.9.0-startup.png)

这张图把加载、首次前向和稳态拆开。首次前向包含 GPU 上传与回读，但不包含模型加载；新建进程也不代表系统文件缓存和 GPU 驱动缓存都已清空。

![CPU 内存实际采样](assets/0.9.0/09-cpu-memory.png)

内存图保留整个进程 RSS，包含宿主、模型和缓存。速度领先不代表内存更小。CPU% 要结合 CPU ms/帧理解；GPU 使用率和显存是整卡采样，包含桌面等其他进程，不能说成单模型占用。

[完整报告](https://github.com/hnlyf168/SimdYolo/blob/main/artifacts/runtime-comparison-0.9.0.md)保留每一轮、P95、资源数据和程序集摘要，[原始 JSON](https://github.com/hnlyf168/SimdYolo/blob/main/artifacts/runtime-comparison-0.9.0.json)可以重新取数画图。

## 不依赖 CUDA 的 Vulkan 路径

对做 .NET 应用的人，部署成本也很实际。

SimdYolo 的 Vulkan 路径不调用 CUDA、cuDNN、cuBLAS，不需要安装一套 CUDA Toolkit。运行时需要的是 .NET、模型、Vulkan loader 和兼容显卡驱动；包里已有预编译 SPIR-V，普通运行不需要 Vulkan SDK。

![Vulkan 部署需要什么、验证到了哪里](assets/0.9.0/10-vulkan-deployment.png)

Vulkan 采用跨平台、跨厂商 API，给复用计算实现提供了基础。**这不意味着仓库已把每种显卡、每个平台都测过。** 本轮是 Windows/NVIDIA/Vulkan 实测，Metal 有独立实现但没有 macOS 实机结果；部署到其他设备仍需验证。

## 推理之外，加入真正的检测头微调

当前训练核心同样使用纯 C#，没有让网页后台启动 Python。

训练从标准 YOLOv8 检测 PT 初始化，冻结骨干和 FPN，更新三尺度检测头的卷积、BN 和分类输出。支持自定义类别、TAL/BCE/CIoU/DFL、带动量的 SGD，以及完整轮次的保存和恢复。CPU 复用 SIMD 矩阵运算；Vulkan 加速冻结网络和头部前后向 GEMM，BN、损失和优化器仍在 CPU。

![推理与训练共享哪些能力](assets/0.9.0/03-architecture.png)

第一阶段的范围是**检测头微调**。全参数、YOLO26、分割/姿态和 Metal 训练还没有实现。梯度累加按图像平均，每图独立 BN，也不能当成多图联合 BN。

训练生成 `.sytrain`，包含原 PT、已更新参数、BN、SGD 动量、类别和数据集摘要。它可直接交给 `YoloDetector` 推理或继续训练；它不是 PyTorch `.pt`，也暂未提供 ONNX 导出。

目前12项训练测试、117项推理回归，以及与固定原版 PyTorch/Ultralytics 的640项数值对照通过。单图训练的损失下降只是流程验证，不能当作独立验证集精度，更没有据此宣称训练比 PyTorch 更快。

## 打开网页，上传图片，也上传自己的标注

```powershell
git clone https://github.com/hnlyf168/SimdYolo.git
cd SimdYolo
git lfs pull
dotnet run --project examples/SimdYolo.Web -c Release
```

打开 `http://127.0.0.1:5188`。或者下载 Release 的网页版运行包，配合33模型 ZIP 使用；包内有具体步骤。

![网页版实际图片推理](../../artifacts/validation/0.9.0/web-inference.png)

推理页支持检测框、分割掩码和姿态关键点。模型下拉框也能选择训练生成的结果，最近两个模型会话可以复用。

![真实训练曲线与模型下载](../../artifacts/validation/0.9.0/web-training.png)

训练页可以先用内置方块数据跑流程，再导入自己的图片和 YOLO 检测 `.txt`，填写类别列表，查看实际损失曲线，取消、恢复、下载模型。目录导入缺少标签的图像会作为背景图，需核对标注是否匹配。示例默认只绑定本地，没有公网账号或鉴权。

## 下载与使用边界

当前发布包含11个 PT、22个 ONNX，覆盖 YOLOv8/9/10/11/26 的已验证 nano/tiny 检测模型，以及 v8/11/26 的分割和姿态模型。模型标签可通过 `ModelInfo.Read(path).ClassNames` 独立读取。

`.pt` 要包含受支持完整 YOLO 结构；单独 `state_dict`、TorchScript、未知自定义模块不能按扩展名直接推理。其他尺度、自训练模型和设备应单独验证。

[0.9.0 下载](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.9.0)提供主库、训练库、33模型、网页版和本文图文包。模型保留上游 AGPL-3.0、原许可证和来源，项目源码暂未新增许可证。
