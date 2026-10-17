# 纯C# YOLO 0.11.2：和PyTorch、ORT重新跑一轮，人物车辆也能分离

**仓库：** [hnlyf168/SimdYolo](https://github.com/hnlyf168/SimdYolo) · **本版下载：** [0.11.2 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.11.2)

后续源码0.11.3已继续优化加载、首次推理、ONNX工作区及Vulkan会话生命周期，并增加可选快速启动发布。[新实测与图](../../artifacts/startup-optimization-0.11.3.md)；下列原版/ORT排名仍对应0.11.2，没有替换历史测量或发布附件。

实测日期：2026-10-17 11:50:41 +0800；Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz，NVIDIA GeForce GTX 1050 Ti。FP32、batch1、640×640、CPU12线程。11模型×8配置×两轮，共**176个独立窗口、39,264次推理、224/224项输出校验通过**。预热60次，每窗口至少4秒/40帧。

本表计入网络前向与输出头解码；GPU包含输入上传和全部输出回读，排除图片预处理、NMS、最终实例掩码、模型加载和首帧。原版PyTorch使用官方Ultralytics fuse()，ORT启用全部图优化，CUDA关闭TF32。每格为两个独立窗口中位数的中位数，①最快，按未舍入值排名；接近结果不能解释为稳定胜出。

## CPU / GPU 速度排名

每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗；重复窗口取各窗口中位数的中位数，按未舍入值排序。小幅差距不代表稳定胜出或统计显著性。

### CPU 速度排名

| 模型 | 原版 PyTorch CPU | 我们的 `.pt` CPU | 我们的 ONNX CPU | ORT CPU（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 44.23 ④ | **21.74 ①** | 22.19 ② | 24.23 ③ |
| yolov9t | 76.55 ④ | 29.16 ② | 30.90 ③ | **28.56 ①** |
| yolov10n | 48.15 ④ | **21.57 ①** | 22.22 ② | 22.43 ③ |
| yolo11n | 50.34 ④ | **21.69 ①** | 22.19 ② | 23.11 ③ |
| yolo26n | 54.13 ④ | 22.32 ② | 26.76 ③ | **18.04 ①** |
| yolov8n-seg | 58.21 ④ | **28.12 ①** | 28.73 ② | 33.86 ③ |
| yolo11n-seg | 71.59 ④ | **27.72 ①** | 29.97 ② | 33.49 ③ |
| yolo26n-seg | 68.51 ④ | 30.03 ② | 30.97 ③ | **28.98 ①** |
| yolov8n-pose | 47.32 ④ | **22.78 ①** | 23.59 ② | 25.06 ③ |
| yolo11n-pose | 54.52 ④ | **23.15 ①** | 23.91 ② | 24.85 ③ |
| yolo26n-pose | 58.29 ④ | 24.81 ② | 25.38 ③ | **21.43 ①** |
| 第一名次数（完整四配置 11/11 模型） | 0 | 7 | 0 | 4 |

### GPU 速度排名

| 模型 | 原版 PyTorch CUDA | 我们的 `.pt` Vulkan | 我们的 ONNX Vulkan | ORT CUDA（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | **9.77 ①** | 10.36 ② | 10.60 ③ | 12.52 ④ |
| yolov9t | 21.44 ④ | **12.82 ①** | 14.03 ② | 17.56 ③ |
| yolov10n | 10.72 ③ | **10.03 ①** | 10.23 ② | 13.27 ④ |
| yolo11n | **10.46 ①** | 10.55 ② | 10.61 ③ | 13.52 ④ |
| yolo26n | 12.13 ③ | **10.95 ①** | 11.05 ② | 13.05 ④ |
| yolov8n-seg | **12.94 ①** | 13.70 ② | 13.80 ③ | 16.39 ④ |
| yolo11n-seg | 13.51 ② | **13.46 ①** | 13.57 ③ | 17.32 ④ |
| yolo26n-seg | 16.12 ③ | 14.70 ② | **14.50 ①** | 16.95 ④ |
| yolov8n-pose | **10.12 ①** | 10.51 ② | 10.76 ③ | 12.87 ④ |
| yolo11n-pose | **10.94 ①** | 11.05 ② | 11.24 ③ | 14.05 ④ |
| yolo26n-pose | 13.25 ③ | **11.96 ①** | 12.15 ② | 13.92 ④ |
| 第一名次数（完整四配置 11/11 模型） | 5 | 5 | 1 | 0 |


![本轮CPU](../../artifacts/performance-0.11.2-cpu.png)

![本轮GPU](../../artifacts/performance-0.11.2-gpu.png)

[CPU、内存、GPU、显存、加载与首帧完整结果](../../artifacts/runtime-comparison-0.11.2.md)。本版仍保持核心纯C#/.NET，浏览器处理图片和交互；应用不依赖PyTorch/ORT/CUDA工具链，开发对照依赖不进入发布包。

![本版车辆分离](../../artifacts/validation/0.11.2/web/cutout-vehicles.png)

## 既有掩码优化与训练评测（历史版本保留）

**代码：** [hnlyf168/SimdYolo](https://github.com/hnlyf168/SimdYolo)

源码0.11.1继续优化图片推理与“目标分离 · 背景去除”，包括掩码SIMD、坐标与行缓存、多实例并行、二进制RGB上传及有界画布缓存。[实际图片优化前后结果](../../artifacts/image-pipeline-optimization-0.11.1.md)单独列出CPU、Vulkan与网页整轮耗时；下文训练数字仍属于0.10.1。

**当前使用：** [0.11.1源码、NuGet与启动步骤](../README.md)。**历史正式下载：** [0.10.1 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1) · [33模型](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-models-v0.10.1.zip)。历史网页和图文ZIP不包含0.11.x功能。

## 先看0.11.1的图片推理与目标分离

同一公交实拍图、同一权重，模型输入640、FP32、CPU四线程，比较0.11.0与0.11.1。下表是YOLOv8n-seg完整图片推理，包含预处理、网络、筛选和原图掩码，排除HTTP及浏览器合成，单位ms。

| 原图 / 后端 | 优化前 | 优化后 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 810×1080 / CPU | 54.11 | 40.40 | 25.3% |
| 810×1080 / Vulkan | 30.05 | 17.49 | 41.8% |
| 2048×1536 / CPU | 62.34 | 41.60 | 33.3% |
| 2048×1536 / Vulkan | 39.75 | 17.79 | 55.2% |

![0.11.1实际图片推理对照](../../artifacts/image-pipeline-0.11.1.png)

主要收益来自掩码SIMD、坐标/行缓存和多实例并行，纯检测控制组基本持平。本轮未重测PyTorch/ORT，不用这些完整图片数字替换后文历史前向排名。24个性能窗口与12组实际输出对照通过，约757万个概率没有0.5阈值翻转；详细采样与边界见[完整报告](../../artifacts/image-pipeline-optimization-0.11.1.md)。

网页版点击“人物 / 车辆示例”，左侧得到只保留目标的透明PNG，右侧得到移除目标的背景PNG；切换人物、车辆或某个类别会复用现有结果。新版本缓存RGB与掩码，并使用二进制上传，RGB数据体比Base64少25%。核心计算为纯.NET，浏览器只负责图片和交互；移除的位置透明，不生成被遮挡的背景。

![0.11.1车辆与背景分离](../../artifacts/validation/0.11.1/web/cutout-vehicles.png)

[实际目标PNG](../../artifacts/validation/0.11.1/web/cutout-foreground.png) · [实际背景PNG](../../artifacts/validation/0.11.1/web/cutout-background.png) · [网页操作步骤](../../examples/SimdYolo.Web/README.md)

## 0.10.1检测头训练评测（历史）

我希望在.NET应用中，模型加载、推理、训练和结果处理都能直接使用C#。SimdYolo 0.10.1把YOLOv8、Ultralytics YOLOv9、YOLOv10、YOLO11、YOLO26检测头微调放进同一套API和网页，并针对上一轮训练慢的问题继续优化。

![本版CPU训练速度](../../artifacts/training-performance-0.10.1-cpu.png)

## 0.10.1训练速度

本机i7-8700K、GTX1050Ti；128×128、FP32、2类、4线程，相同初始权重和冻结特征。计入检测头前向、损失、反向和SGD，排除冻结骨干、预处理和保存。各配置两个独立串行窗口，每窗预热10步，再更新512步；GPU包含上传回读，CUDA每步同步。它不是640尺寸全参数训练的速度。

| 模型 | 原版 PyTorch CPU | 我们 CPU SIMD | 原版 PyTorch CUDA | 我们 Vulkan 混合 |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 11.37 | **7.17** | 27.13 | **11.43** |
| yolov9t | 10.90 | **6.46** | 27.17 | **10.39** |
| yolov10n | 26.56 | **15.39** | 59.08 | **22.64** |
| yolo11n | 13.33 | **7.05** | 30.58 | **9.77** |
| yolo26n | 26.06 | **9.29** | 64.49 | **10.61** |

本机这组测试五模型均快于对应PyTorch参考。原版参考使用官方Ultralytics头和损失、PyTorch SGD，不含完整train数据管线。两窗口有波动，排名不代表所有设备或尺寸稳定领先。

![本版Vulkan混合训练与CUDA](../../artifacts/training-performance-0.10.1-gpu.png)

## 速度之外，训练后的模型要能对得上

训练结果另外用COCO128做固定96张微调/32张验证，80类、256×256、2轮、每模型192次SGD。两边保存的`.pt`与`.sytrain`重新加载，运行完整骨干和检测头。mAP50–95最大差约0.088个百分点；直接参数门槛9/10通过，v8 Vulkan/CUDA未通过项保留，PyTorch自身CPU/CUDA出现同量级漂移。固定CPU参数控制10/10、保存模型指标门槛10/10通过，不声称参数逐位一致。

![实际保存模型的精度对照](../../artifacts/training-results-0.10.1.png)

范围仍为冻结骨干/FPN的检测头微调；不含全参数、分割/姿态或Metal训练。小样本短程微调的验证精度有下降，两边趋势相同；COCO128来自预训练使用的COCO train2017，这32张只对本次微调留出，不能据此证明泛化或训练配方有效。

19项训练回归、117项推理回归、22,332项单步数值检查通过。多步参数误差、BN、动量、每步损失和资源采样见[完整报告](../../artifacts/training-optimization-0.10.1.md)，未通过项也在其中。

## 仍然是纯C#，GPU也不用装庞大的CUDA工具链

主库和训练核心没有Python、PyTorch、TorchSharp、ORT或BLAS运行依赖。CPU使用SIMD/intrinsics，GPU通过系统Vulkan/Metal API与包内着色器计算。Python与CUDA只出现在开发对照环境。

Vulkan提供跨平台、跨厂商的API基础，部署不需要CUDA Toolkit、cuDNN、cuBLAS，也不用为了运行安装Vulkan SDK；需要兼容驱动和Vulkan loader。Windows/NVIDIA/Vulkan已实测，Metal推理有实现但本轮没有macOS实机结果，其他设备仍应验证。

本版训练按矩阵大小选择设备：小GEMM使用CPU SIMD，较大GEMM交给Vulkan，减少提交和回读开销；深度卷积、BN、损失与SGD仍在CPU。它是明确标注的混合路径，阈值基于本机测试。

## 打开网页，上传图片与标注

```powershell
git clone https://github.com/hnlyf168/SimdYolo.git
cd SimdYolo
git lfs pull
dotnet run --project examples/SimdYolo.Web -c Release
```

打开`http://127.0.0.1:5188`。也可以下载同版网页包与33模型包，安装.NET10/ASP.NET Core运行时后启动。页面支持检测、分割、姿态推理，以及五代检测模型的检测头微调、自定义类别、真实损失曲线、取消、恢复与模型下载。

![实际YOLO26训练页面，截图采集于0.10.0](../../artifacts/validation/0.10.0/web-browser/web-training.png)

训练使用YOLO检测文本`class cx cy w h`。`.sytrain`包含原始PT、全部训练参数、BN、动量与类别，可重新加载推理和恢复；它不是PyTorch`.pt`。0.11.0源码已支持纯.NET导出标准ONNX，并增加独立验证集、早停和网页标注，详见[新增功能](../model-workflows.md)。模型标签可通过`ModelInfo.Read(path).ClassNames`独立读取。

## 推理速度保留历史版本，避免混淆

下面推理表是**0.9.0冻结程序集的历史实测**，不是0.10.1重测：640×640、FP32、batch1、12线程，176窗口、39,134次推理、224项输出校验。GPU含上传及输出回读，不含预处理、NMS、最终mask后处理与加载首帧。

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


![0.9.0历史推理CPU图](../../artifacts/performance-0.9.0-cpu.png)

![0.9.0历史推理GPU图](../../artifacts/performance-0.9.0-gpu.png)

历史输入是固定实拍图；随机输入下ORT CUDA的YOLO26分割超差[单独保留](../../artifacts/validation/0.9.0-release/README.md)，不混入正式图片基准。

模型包仍为11个PT、22个ONNX，原文件与SHA256不变，保留上游AGPL-3.0及来源；源码暂未新增许可证。[训练说明](../training.md) · [推理API](../inference.md) · [0.10.1 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1) · [33模型](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-models-v0.10.1.zip) · [网页版](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-Web-v0.10.1.zip) · [中文图文包](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-docs-v0.10.1.zip)
