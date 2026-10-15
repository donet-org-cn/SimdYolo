# SimdYolo

[网页推理/训练示例](examples/SimdYolo.Web/README.md) · [纯 C# 训练说明](docs/training.md) · [模型下载和使用](models/README.md) · [模型来源与许可证](THIRD_PARTY_NOTICES.md)

C# / .NET 10 的 YOLO 推理库，包含目标检测、实例分割和姿态估计。主库为纯托管 C# 程序集：模型解析、计算图、CPU 算子、图像预处理和结果解码均由 C# 实现，CPU 使用 .NET SIMD / intrinsics。没有外部 NuGet 运行依赖，不调用 Python、PyTorch、LibTorch、TorchSharp、ONNX Runtime、BLAS/MKL/OpenBLAS 或其他原生数学运行库，也不在运行时把 `.pt` 转成 ONNX。GPU 通过 P/Invoke 调用系统 Vulkan / Metal API，执行包内嵌入的 SPIR-V / MSL 着色器。

当前版本 **0.9.0** 新增独立纯 C# 训练模块和 ASP.NET Core 网页示例。第一阶段支持 **YOLOv8 检测头微调**：冻结骨干/FPN，支持自定义类别、真实反向传播、SGD 动量、CPU SIMD 与 Vulkan GEMM 混合训练、取消和断点恢复。生成的自包含 `.sytrain` 可直接交给 `YoloDetector` 推理。网页版提供 PT/ONNX 检测、分割、姿态推理，以及数据集导入、真实损失曲线、模型下载和重新加载。

```powershell
dotnet run --project examples/SimdYolo.Web -c Release
# 浏览器打开 http://127.0.0.1:5188
```

训练测试 **12 项通过**，与固定原版 PyTorch/Ultralytics 的独立数值对照 **640/640 检查通过**，既有 CPU/Vulkan 推理回归 **117 项通过**。本阶段尚无全参数、YOLO26、分割/姿态或 Metal 训练，也未进行训练性能或独立验证集 mAP 比较。详见[训练范围与使用](docs/training.md)、[网页示例](examples/SimdYolo.Web/README.md)、[0.9.0 验证](artifacts/validation/0.9.0/README.md)。下方 0.8.1 及更早速度表保留原版本和原始数据，不代表 0.9.0 的测量。

0.8.1 优化权重加载、会话首帧和稳态执行。PT 连续 storage 使用批量转换，ONNX 解析借用 protobuf 数据视图；PT 复用模块路由工作区，CPU 对完整写入的临时结果省去预清零，以弱引用缓存整组 Winograd 权重面板。注意力批次 MatMul 直接按广播偏移读取输入、写入输出，省去逐 head 的整矩阵复制；无 DFL 的单桶检测头直接使用回归值并以 SIMD 还原坐标。YOLO26 检测、分割、姿态 ONNX 输出头严格匹配后融合为两阶段 TopK，保留全部实例字段，并可在 Vulkan 执行和捕获命令重放。仍使用 FP32；主库没有增加 Python、ORT 或原生数学依赖。

0.8.0 新增 **YOLO26 检测、实例分割和姿态推理**，直接加载官方 `.pt` 或 ONNX，支持无 DFL 回归、SPPF 残差、Proto26 多尺度融合、Pose26 坐标公式和端到端实例附加字段。新增 `ModelInfo.Read`，不初始化 GPU、不加载权重存储即可读取类别标签。官方 nano 模型已在 CPU/Vulkan、静态/动态输入及 batch=2 上验证，记录见 [YOLO26 验证](artifacts/validation/0.8.0/yolo26.md)。上面的 0.7.1 Release 是旧版本，不包含这些新功能；下方标为 0.7.0 的表格保留当时的原始测量。

0.7.1 补齐中文代码文档：类、接口、构造、方法、参数、返回值、属性和实际异常，以及 CPU SIMD、模型解析和 GPU 资源生命周期说明。NuGet 包包含 `SimdYolo.xml`，安装后 IDE 可直接显示中文提示。计算代码保持 0.7.0 的实现；下方速度表及原始报告来自 0.7.0 冻结程序集，0.7.1 的文档覆盖、代码一致性与回归结果单独记录在 [中文注释验证](artifacts/validation/0.7.1/chinese-documentation.md)。

## 0.8.1 本轮实测

[0.8.0→0.8.1 优化对照](artifacts/optimization-0.8.1.md)完成 352 个独立进程窗口：11 模型、PT/ONNX、CPU/Vulkan，每个版本与配置各 4 窗口，采用 ABBA 顺序。统一 FP32、batch=1、640×640、12 线程，预热 60 帧，每窗口至少 4 秒、40 帧。下表为各窗口指标的中位数，箭头表示旧版→新版；单位为 ms，每帧托管分配单列为 MiB。

| 模型 / 后端 | 模型加载 ms | 首次前向 ms | 程序内至首输出 ms | 稳态 ms | 托管分配 MiB/帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n.pt / CPU | 181.20→92.91 | 265.38→265.64 | 456.68→368.10 | 27.14→26.18 | 2.934→2.915 |
| yolo26n.pt / CPU | 172.62→101.87 | 261.14→258.90 | 442.81→370.86 | 29.47→28.00 | 6.416→0.337 |
| yolo26n-seg.onnx / CPU | 116.56→172.34 | 338.89→318.40 | 469.70→559.96 | 41.13→36.73 | 10.128→3.731 |
| yolo26n-seg.onnx / Vulkan | 531.59→505.22 | 397.47→370.95 | 975.31→890.57 | 17.56→14.52 | 3.906→3.169 |

加载、首次前向及稳态分别计时，不能把加载收益说成首次前向收益：v8 PT 的首次前向基本不变；26 分割 ONNX CPU 的加载和程序内至首输出的汇总中位数反而增加，虽稳态更快。该配置四个新版窗口的加载为 102.46–267.00 ms、至首输出为 430.06–607.02 ms，波动大，不能说每个窗口都回退。程序内至首输出不含 .NET 宿主启动和入口方法执行前的 JIT；新进程不代表清空了系统文件缓存、GPU 驱动或管线缓存。托管分配也不等于进程 RSS 或显存。

[原版 PyTorch、我们的 PT/ONNX、ORT 的 CPU/GPU 对照与排名](artifacts/runtime-comparison-0.8.1.md)完成 **176/176 窗口、36,460 次推理，224/224 输出校验通过**。CPU ORT 两轮明显波动，例如 v8 为 50.94/25.47 ms、11n 为 24.38/47.81 ms；26n、26n-seg、26n-pose 的 CPU 汇总仍由 ORT 领先，不能宣称我们稳定全胜。GPU 数值来自本机 Vulkan/CUDA，**Metal 尚无硬件实测**。

本轮 Release 构建零警告、零错误；CPU/Vulkan 回归 **117 项通过**，关闭 AVX 与关闭全部硬件指令各 **94 项通过**。352 个生命周期窗口的首帧/末帧共 **896/896 输出校验通过**，包含 TopK 分数、唯一候选配对、分割系数及关键点。冻结程序集、原始 JSON、复现方法及其他动态/图像验证见 [VALIDATION.md](VALIDATION.md)；新版 [NuGet 包](artifacts/SimdYolo.0.8.1.nupkg)的内容、依赖和中文 XML 检查见[包审查记录](artifacts/validation/0.8.1/package-audit.json)，实际消费验证见[独立 NuGet 消费记录](artifacts/validation/0.8.1/package-consumer.json)。

## 0.7.0 历史实测

0.7.0 继续使用纯 C#、FP32 CPU 计算，新增实际地址对应的缓存行对齐、16/48 列点卷积调度、F4 卷积边界 SIMD、窄平面 depthwise padding SIMD、卷积尾部四/八元素 SIMD 激活、8×8 SIMD 转置、严格匹配的 ONNX 检测头融合及 2×2 平均池化专用循环。保持既有专属 CPU 执行器和 SIMD 指数路径。32 列内核、F6 和分块大小调参没有稳定收益，未设为发布路径。GPU 实现和着色器保持不变，本次重新测量 Vulkan、原版 PyTorch CUDA 和 ORT CUDA；Metal 尚无 macOS 硬件验证。

2026-10-14 14:31:43 +0800 同机正式比较完成 CPU/GPU 共 128/128 个独立窗口、57,628 次计时推理，160/160 个输出校验通过（20 个 PyTorch CPU 自对照、140 个跨实现对照）。硬件：Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz、NVIDIA GeForce GTX 1050 Ti。统一 FP32、batch=1、640×640、CPU 线程上限 12；各配置独立进程、串行运行两轮，预热 30 次后每窗口至少 8 秒、40 次。计时包含网络前向和输出头解码，GPU 包含上传及全部输出回读；排除图片预处理、NMS、最终 mask 后处理和冷首帧。

## 0.7.0 CPU 速度排名

每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗；重复窗口取各窗口中位数的中位数，按未舍入值排序。小幅差距不代表稳定胜出或统计显著性。

| 模型 | 原版 PyTorch CPU | 我们的 `.pt` CPU | 我们的 ONNX CPU | ORT CPU（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 44.87 ④ | **21.99 ①** | 22.64 ② | 24.25 ③ |
| yolov9t | 77.42 ④ | 29.80 ② | 31.45 ③ | **28.48 ①** |
| yolov10n | 60.57 ④ | **22.33 ①** | 22.74 ③ | 22.36 ② |
| yolo11n | 51.45 ④ | **22.21 ①** | 22.94 ② | 23.34 ③ |
| yolov8n-seg | 56.83 ④ | **28.41 ①** | 28.81 ② | 33.84 ③ |
| yolov8n-pose | 47.69 ④ | **23.14 ①** | 23.95 ② | 25.17 ③ |
| yolo11n-seg | 63.39 ④ | **28.29 ①** | 29.21 ② | 33.03 ③ |
| yolo11n-pose | 54.54 ④ | **23.80 ①** | 24.91 ③ | 24.81 ② |
| 第一名次数（完整四配置 8/8 模型） | 0 | 7 | 0 | 1 |

`.pt` CPU 比 原版 PyTorch CPU 更快 8/8 个模型，比 ORT CPU 更快 7/8 个模型；ONNX CPU 比 原版 PyTorch CPU 更快 8/8 个模型，比 ORT CPU 更快 5/8 个模型。名次基于本次窗口的数值，接近的结果不代表稳定领先。

YOLOv10 CPU `.pt` 相对 ORT 的汇总优势仅 0.030775 ms（0.138%），两轮顺序交换，可视为接近持平，不能声称稳定超越。YOLOv9 CPU 仍由 ORT 领先；完整表中保留该结果。

## 0.7.0 GPU 速度排名

每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗；重复窗口取各窗口中位数的中位数，按未舍入值排序。小幅差距不代表稳定胜出或统计显著性。

| 模型 | 原版 PyTorch CUDA | 我们的 `.pt` Vulkan | 我们的 ONNX Vulkan | ORT CUDA（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | **9.89 ①** | 10.46 ② | 10.55 ③ | 12.69 ④ |
| yolov9t | 21.76 ④ | **12.86 ①** | 14.12 ② | 17.73 ③ |
| yolov10n | 12.80 ③ | **10.08 ①** | 10.24 ② | 13.53 ④ |
| yolo11n | 10.61 ② | 10.64 ③ | **10.60 ①** | 13.72 ④ |
| yolov8n-seg | **12.89 ①** | 13.72 ③ | 13.58 ② | 16.53 ④ |
| yolov8n-pose | **10.17 ①** | 10.52 ② | 10.78 ③ | 13.05 ④ |
| yolo11n-seg | 13.79 ③ | **13.52 ①** | 13.65 ② | 17.56 ④ |
| yolo11n-pose | **10.95 ①** | 11.05 ② | 11.21 ③ | 14.23 ④ |
| 第一名次数（完整四配置 8/8 模型） | 4 | 3 | 1 | 0 |

## CPU 时间与内存

以下为逐模型比值的范围，不是把不同模型成本合并平均；大于 1 表示本库成本更高。CPU%按全机逻辑核心归一化，CPU ms/帧由进程累计 CPU 时间计算，RSS 为整个宿主进程的采样工作集。

| 路径 | CPU ms/帧 / PyTorch | CPU ms/帧 / ORT | RSS / PyTorch | RSS / ORT |
| --- | ---: | ---: | ---: | ---: |
| `.pt` CPU | 0.33–0.45× | 0.73–0.96× | 1.24–1.45× | 2.04–3.60× |
| ONNX CPU | 0.32–0.45× | 0.73–0.98× | 0.75–1.24× | 1.51–3.09× |

较高 CPU 占用不代表更高效率；较低 CPU 占用也不能单独证明效率优势。更快延迟可能伴随更多 CPU 时间，需要同时查看 CPU ms/帧和进程 RSS，不能据速度排名宣称资源成本全面领先。RSS 包含运行库、模型和缓存，不等于算子工作区大小。

GPU 使用率和显存是整卡采样，包含桌面及其他进程，不等同于单进程显存或功耗。CPU%按全机逻辑核心归一化，CPU ms/帧来自进程累计 CPU 时间；RSS 包含宿主、模型和缓存。速度名次与资源成本分别评估；两次窗口不足以证明小幅领先有跨环境统计显著性。PyTorch 使用原版 fuse()，ORT 开启全部图优化、默认自旋及自动 CUDA 算法；结果不能外推到 torch.compile、TensorRT、FP16/INT8 或 Metal。

[CPU 完整报告](artifacts/cpu-runtime-comparison.md)、[CPU/GPU 完整报告](artifacts/optimized-runtime-comparison.md)、[原始 JSON](artifacts/optimized-runtime-comparison.json)。每个重复窗口、资源采样、输入/模型/DLL 哈希和输出校验均保留。

## 0.6→0.7 CPU 历史对照

每格为 0.6→0.7 中位耗时 ms（耗时降低百分比，负值表示更慢）。这是两次历史测量，未在同一窗口做版本开关 A/B；系统热状态、线程策略和缓存可能不同，不能把差值全部归因于代码。

| 模型 | `.pt` CPU：0.6→0.7 | ONNX CPU：0.6→0.7 |
| --- | ---: | ---: |
| yolov8n | 26.35→21.99（16.6%） | 27.31→22.64（17.1%） |
| yolov9t | 34.96→29.80（14.8%） | 36.67→31.45（14.2%） |
| yolov10n | 29.90→22.33（25.3%） | 33.25→22.74（31.6%） |
| yolo11n | 27.03→22.21（17.8%） | 27.39→22.94（16.3%） |
| yolov8n-seg | 34.43→28.41（17.5%） | 35.51→28.81（18.9%） |
| yolov8n-pose | 29.24→23.14（20.9%） | 29.68→23.95（19.3%） |
| yolo11n-seg | 35.31→28.29（19.9%） | 36.62→29.21（20.2%） |
| yolo11n-pose | 29.53→23.80（19.4%） | 30.67→24.91（18.8%） |

## 验证和交付

Release 构建零警告、零错误；Vulkan/CPU 统一回归 86 项通过；关闭硬件 intrinsics 的可移植回归 70 项通过（直接 SIMD 专项受硬件 guard 限制，部分跳过）。冻结 DLL 的 CPU `.pt`/ONNX 20 个输出、Vulkan 改输入和重放 80 个输出、静态/动态 ONNX CPU/Vulkan 30 个输出全部通过。真实图像 5 个配置、24 个实例通过，11 个 mask 最低 IoU=1.000000。

[构建](artifacts/validation/0.7.0/build.log)、[统一回归](artifacts/validation/0.7.0/test-vulkan.log)、[可移植回归](artifacts/validation/0.7.0/test-portable.log)、[CPU oracle](artifacts/validation/0.7.0/cpu-packed-oracle.json)、[GPU 重放](artifacts/validation/0.7.0/real-model-replay.json)、[ONNX oracle](artifacts/validation/0.7.0/onnx-oracle.json)、[图像 oracle](artifacts/validation/0.7.0/image-oracle.json)。

[0.7 NuGet 包](artifacts/SimdYolo.0.7.0.nupkg)、[包审查](artifacts/validation/0.7.0/package-audit.json)、[最终审查](artifacts/validation/0.7.0/final-audit.json)、[CPU 消费者](artifacts/validation/0.7.0/consumer-cpu.log)、[Vulkan 消费者](artifacts/validation/0.7.0/consumer-vulkan.log)。包内、CLI、正式基准与独立消费者绑定同一主 DLL；包哈希以包审查为准。文档重打包改变包哈希，保持主 DLL 不变。

## 构建和使用

```powershell
dotnet build -c Release
dotnet run --project tests/SimdYolo.Tests -c Release -- --vulkan
.\tools\download-models.ps1 -Vision
dotnet run --project src/SimdYolo.Cli -c Release -- inspect .cache/models/yolov8n.pt
dotnet run --project src/SimdYolo.Cli -c Release -- bench .cache/models/yolov8n.pt vulkan 640 10
```

普通构建使用仓库内预编译的 SPIR-V，不需要 Vulkan SDK。重新生成着色器需要 glslang：

```powershell
.\tools\build-shaders.ps1 -Compiler glslangValidator
```

应用可以引用 `src/SimdYolo/SimdYolo.csproj`，或使用下面的命令生成本地 NuGet 包。包内主库没有外部运行依赖，开发用的 ORT 对照工具不随包发布：

```powershell
dotnet pack src/SimdYolo -c Release -o artifacts
```

交付包的 DLL 一致性、依赖/资产审查及独立本地 NuGet 消费结果见 [0.7 包审查](artifacts/validation/0.7.0/package-audit.json)、[CPU 消费日志](artifacts/validation/0.7.0/consumer-cpu.log)和 [Vulkan 消费日志](artifacts/validation/0.7.0/consumer-vulkan.log)，具体状态以记录为准。主库 DLL 与正式矩阵绑定；文档重打包会改变包哈希。

检测、分割和姿态使用同一个 API。图片解码由调用方完成，输入为 RGB24 / BGR24 字节；可接入现有图像库、摄像头或视频帧。

```csharp
using SimdYolo;

using var detector = new YoloDetector("yolo11n-seg.pt", new DetectorOptions
{
    Device = ComputeDevice.Vulkan, // macOS 使用 Metal；Auto 可回退 CPU
    Width = 640,
    Height = 640,
    Confidence = 0.25f,
    IouThreshold = 0.45f
});

IReadOnlyList<YoloInstance> instances = detector.Predict(rgbBytes, imageWidth, imageHeight);
foreach (var instance in instances)
{
    Detection box = instance.Box;
    InstanceMask? mask = instance.Mask;
    IReadOnlyList<Keypoint> points = instance.Keypoints;
}
```

换成 `yolo11n-pose.pt` 可返回关键点；换成检测权重可返回检测框。`.onnx` 同样直接传文件路径。

YOLO26 使用相同 API，路径可换成 `models/pt/yolo26n.pt`、`yolo26n-seg.pt`、`yolo26n-pose.pt` 或对应的 `models/onnx/*.onnx`。框、掩码、关键点随端到端 TopK 一起收集，端到端结果跳过 NMS。

### 读取模型标签

```csharp
using SimdYolo;

// .pt / .pth / .onnx 通用：只读取元数据，不创建推理后端。
ModelInfo info = ModelInfo.Read("models/pt/yolo26n.pt");
foreach (var label in info.ClassNames.OrderBy(pair => pair.Key))
    Console.WriteLine($"{label.Key}: {label.Value}");

using var detector = new YoloDetector("models/onnx/yolo26n.onnx");
// 已加载会话直接使用 ClassNames 或 Info，无需再次读取文件。
string? className = detector.ClassNames.GetValueOrDefault(0); // person
```

标签支持 PT 字典/列表，以及 ONNX `names` 中的 JSON、Python 字典/列表/元组，保留 Unicode、引号、反斜线及原始编号。`ClassCount` 是类别通道数，稀疏标签不能用 `ClassNames.Count` 替代。缺少名称时 `HasClassNames=false`，返回空映射；编号重复、负数或格式损坏会抛出 `InvalidDataException`。普通 `state_dict` 没有架构，依然不能直接推理，但检查点若另存 `names`，可以读取它。

```powershell
dotnet run --project src/SimdYolo.Cli -c Release -- inspect models/pt/yolo26n-pose.pt
dotnet run --project src/SimdYolo.Cli -c Release -- labels models/onnx/yolo26n.onnx
```

`inspect` 输出任务、输入维度、类别数、名称与端到端标记；`labels` 只输出编号到名称的 JSON。`predict` 导出的 JSON 也包含 `ClassName`。

`Detect(...)` 只返回框，跳过 mask 解码。`PredictRaw(new Tensor(data, batch, 3, height, width))` 返回原始张量，适合自行处理输出。图像 API 的 batch 为 1；原始推理允许模型支持的 batch。输入按 NCHW、RGB、0–1 排列。`.pt` 输入宽高必须是 32 的倍数。

`InstanceMask` 是原图坐标中的裁剪掩码：`X/Y` 表示裁剪起点，`Width/Height` 表示尺寸，`Probabilities` 按行保存概率；可用 `> 0.5f` 得到二值掩码。关键点使用原图坐标。预处理使用居中 letterbox、114 填充和浮点双线性插值；与先量化为 uint8 的 OpenCV resize 不要求逐位相同。

无 task/names 元数据的 ONNX 分割或姿态模型，请明确配置，例如：

```csharp
new DetectorOptions
{
    Device = ComputeDevice.Cpu,
    Task = YoloTask.Pose,
    ClassCount = 1,
    KeypointCount = 17,
    KeypointDimensions = 3,
    OutputLayout = OutputLayout.ChannelsFirst
};
```

输出维度有歧义时也应指定 `OutputLayout`。YOLOv10 的 `[1,N,6]` 端到端输出使用 `EndToEnd`，按分数筛选，跳过 NMS。检测采用每个 anchor 的最高类别分数和按类别 NMS。

## 模型范围

| 模型 | 检测 | 分割 | 姿态 | 已验证格式 |
| --- | --- | --- | --- | --- |
| YOLOv8n | 已验证 | 已验证 | 已验证 | `.pt`、opset 17 `.onnx` |
| YOLOv9t（Ultralytics） | 已验证 | 未验证 | 未验证 | `.pt`、opset 17 `.onnx` |
| YOLOv10n | 已验证 | 未验证 | 未验证 | `.pt`、opset 17 `.onnx` |
| YOLO11n | 已验证 | 已验证 | 已验证 | `.pt`、opset 17 `.onnx` |
| YOLO26n | 已验证 | 已验证 | 已验证 | 官方 `.pt`、静态/动态 opset 17 `.onnx`，CPU/Vulkan |

其他大小和自训练模型在使用已支持模块/算子时可以加载，但没有全部逐一验证。WongKinYiu 原仓库 YOLOv9 的 RepConvN、RepNCSP、CBLinear、CBFuse 和多分支 DDetect 已实现，原仓库检查点仍需单独验证。不会因为文件名包含 v8/v9/v10/v11 就假定模型兼容。

`.pt` 支持 `torch.save` 的 ZIP 检查点，检查点中必须包含完整 YOLO `nn.Module`（`model` 或 `ema`），也支持直接保存完整模型。解析器读取 pickle 数据和 storage，恢复模块连接和权重，不导入或执行 Python GLOBAL/REDUCE。支持 FP32、FP16、BF16 storage，内部计算使用 FP32。

单独的 `state_dict` 没有执行结构，不能自动推断任意自定义网络；这类文件需要完整模型或 ONNX。TorchScript、旧式非 ZIP pickle、未知自定义模块和量化计算图不在当前支持范围，加载时返回明确异常。PyTorch 的文件格式背景见 [官方序列化文档](https://docs.pytorch.org/docs/stable/notes/serialization.html)。

ONNX 支持普通 `ai.onnx` 域、opset 11–21 中已实现的算子；主要对照测试使用 opset 17。支持动态图像尺寸、FP32/FP16 权重和同目录 external data。算子包括 Conv、ConvTranspose、MatMul、Softmax、常见激活/逐元素运算、Pool、Resize、Concat、Split、Reshape、Transpose、Gather、Slice、Shape、Range、TopK 等。Resize 支持 YOLO 常用的 nearest/asymmetric/floor，其他插值模式会明确拒绝。带内嵌 NonMaxSuppression 的导出、分类/OBB、未知算子或其他 ONNX 域不支持。shape/index 路径使用 FP32，超过其精确整数范围的通用整数模型不适用。

## 后端和性能

CPU 的 GEMM 使用 AVX/FMA 寄存器分块、连续列块打包和分批调度，其他硬件使用 `System.Numerics.Vector<float>` 路径。适用的 3×3 卷积采用 F(2×2,3×3) / F(4×4,3×3) Winograd，并对输入和输出变换使用 SIMD；普通卷积按 16 个空间位置生成计算块，减少完整 im2col 的分配和内存往返。depthwise、池化、上采样、Softmax 和 2×2 转置卷积也有专用 SIMD / GEMM 路径。会话按计算图的存活引用及时回收中间数组，缓存保留上限为 256 MiB；返回的公开输出脱离池管理，后续推理和调用方修改输出不会覆盖模型常量。

固定权重可缓存 GEMM 的 A 矩阵寄存器布局，减少内层循环寻址；普通 packed-A 路径对少于 6 个输出行、无 AVX/FMA 或可变权重保留回退。可用 `SIMDYOLO_CPU_PACKED_A=0` 关闭，线程上限仍由调用方指定。

0.5 的逐点卷积使用 4 行×24 列 AVX/FMA 计算块，适用时缓存固定权重布局，并融合 bias / SiLU；不满足硬件/形状/权重条件时回退。Winograd 在同一线程池任务的局部计算块中完成输入变换、通道归约和逆变换，避免整幅中间结果和重复打包，较小深层仍用原路径。CPU 使用 backend 专属执行器，调用线程参与计算并计入 CpuThreads 上限，空闲线程在有界自旋后休眠。请使用 using 或 Dispose 释放 detector/backend 持有的工作线程。发布任务和休眠交接使用完整内存屏障，并有强制休眠压力回归；其他路径可使用常规 .NET 线程池回退。

CPU SiLU 的常见有限值采用 FP32 范围缩减和多项式近似，在 AVX2/FMA 上执行；NaN、极值和不满足硬件条件时保留原实现，没有截断输入或量化。可用 `SIMDYOLO_CPU_FAST_ACTIVATION=0` 关闭。数值精度以原版输出对照记录为准。

1×1 卷积在计算块内融合 bias / SiLU。适用的 batch=1、固定权重通道 Concat→1×1 Conv 直接从各输入打包 GEMM 计算块，避免分配和复制完整的拼接张量；其他形状、设备和可变权重使用普通 Concat/Conv 路径。`.pt` 的 C2f/C3/SPPF 等模块及 ONNX 单使用方模式均可使用该优化。

`Tensor.Shape` 被调用方改成与存储不一致的尺寸时，会在 unsafe SIMD / GPU 入口前拒绝。CPU 输出通过所有权转移发布，重复输出别名各自复制；CPU/GPU 嵌套会话结束后恢复外层的张量池和设备作用域。

`.pt` 加载时融合 Conv/BN、RepConv 和 RepVGGDW，检测头融合 DFL/坐标解码，v10 使用堆式 TopK。ONNX 加载时消除死节点、折叠常量，并在匹配且不破坏其他输出/使用方时融合 Conv/BN、卷积分支、SiLU、DFL、Sigmoid/Concat 和 CPU Concat/1×1 Conv。按输入形状缓存最多 8 组 shape/anchor 计算计划，数据依赖的形状仍逐次计算。连续 Slice 可在设备上执行，减少 YOLO 通道切片触发的 CPU 回读；TopK 使用堆选择，连续末轴 ReduceMax 使用 SIMD。

Vulkan 使用 F2/F4 Winograd、低通道卷积和多种列宽的分块内核，在首次遇到适用形状时用 GPU timestamp 选择卷积内核；无 timestamp 支持时使用计时回退。固定权重驻留设备，中间张量保留在复用的显存块中，多个算子记录到同一命令缓冲区。depthwise、转置卷积、池化、上采样、广播、转置、Slice、Softmax、检测/关键点解码均有设备路径；分割的 2×2 转置卷积使用矩阵计算和像素重排。多个公开输出共用一次批量读回，读回内存优先选择 HOST_CACHED。Metal 保留运行时 MSL、共享内存和固定权重缓存；没有 Metal 提速或正确性的硬件实测结论。

同一帧中，多个卷积可复用共享输入的 Winograd 变换；缓存区分设备存储偏移、形状和变换块，结束帧时清空，后续帧重新计算。

0.5 的 Vulkan 为适用固定权重逐点卷积使用 vec4 读写与 32/64 输出行分块；符合边界/对齐条件的 3×3 depthwise 也有 vec4 路径。Winograd 按形状在 32×32 / 64×32 / 64×64 计算块间选核，适用的 F4 变换使用向量化读写。形状、设备存储偏移、通道尾部或可变权重不适用时保留通用路径。

ONNX 的末轴 ReduceMax 和 GatherElements 有 Vulkan 路径；YOLOv10 和 YOLO26 导出的端到端头按精确数据流、常量和使用方匹配，在适用布局下融合为设备两阶段 TopK，默认启用，可用 `SIMDYOLO_GPU_ONNX_E2E=0` 关闭。YOLO26 模式保留分割系数和关键点，支持已验证的静态/动态批次；其他变体回到原算子图。`.pt` 的专用 TopK 默认在 Vulkan 执行，`SIMDYOLO_GPU_TOPK=0` 也会禁用专用头路径。通用 ONNX TopK 仍默认走 CPU，其独立 GPU 路径仅在显式设置 `SIMDYOLO_GPU_ONNX_TOPK=1` 或 `SIMDYOLO_GPU_TOPK=1` 时启用。

Vulkan 默认将驱动 pipeline cache 和形状选核结果保存到系统临时目录的 `SimdYolo/VulkanCache`，可用 `SIMDYOLO_VULKAN_CACHE_DIR` 指定目录，或以 `SIMDYOLO_VULKAN_CACHE=0` 禁用。缓存绑定设备/驱动标识，选核记录还绑定嵌入着色器哈希及路由选项；损坏、不兼容或无法读写时继续使用未缓存路径。缓存命中不代表新形状已编译/选核，首次遇到新形状仍须预热。会话释放时保存新增结果；`PipelineCacheLoaded` / `CachedConvolutionShapes` 可观察恢复状态。

固定形状、全设备执行的图可在预热后重放已记录的命令缓冲区，减少 C# / 驱动逐帧记录开销；每帧仍上传实际输入、回读全部输出并返回独立主机数组。第一次权重上传、CPU 数据依赖处理、其他动态上传或中途同步会阻止捕获；输入形状变化时重新记录。`ReplayedFrames` 表示实际重放帧数，`LayerCompleted` 和 GPU profiling 会禁用该路径；诊断时也可设置 `SIMDYOLO_VULKAN_REPLAY=0`。它不保证所有模型都重放，也不缓存预测结果。

Vulkan 需要 64 位进程和系统 Vulkan 驱动，Metal 需要 64 位 macOS。Vulkan 优先选择有 compute queue 的独立显卡，再考虑核显、虚拟设备等，同类型保留驱动枚举顺序。默认使用 CPU；显式选择 Vulkan/Metal 时失败会报错，`Auto` 初始化失败会回退 CPU，并通过 `FallbackReason` 提供原因。运行中 GPU 出错会报错。会话串行执行推理，建议复用会话并提前预热；不要每帧重新加载模型。

Vulkan 在需要 CPU 结果时才提交并等待；普通推理不逐算子同步。未覆盖的算子和不匹配端到端融合模式的 ONNX TopK 会产生 CPU 回退及传输。首次形状特化和内核选择会增加首帧耗时，应在处理视频前通过 `PredictRaw` 预热，并复用会话。`LayerCompleted` 仅用于诊断，会主动同步 GPU 测量每层耗时；性能测试时请关闭。内部计算为 FP32，FP16/INT8 算子和 Tensor Core 尚未实现，结论不能外推为所有硬件或 PyTorch/TensorRT 配置的结果。

0.5 缓存专项在 v8n / 640×640 上测得：空应用缓存首帧 5.373 秒，恢复已知应用缓存后 0.310 秒，同结构 ONNX 为 0.388 秒，恢复 12 个选核结果。未清空系统/驱动 shader cache，不能视为完全冷启动或与 0.4 的代码纯收益比较。0.5 当时正式矩阵的 32 个 Vulkan 窗口均命中应用缓存，首次推理为 1.022–1.906 秒；CPU 两种格式为 0.239–0.354 秒，均不含模型构造/加载。新设备、驱动或形状仍可能需要昂贵编译/选核，稳态耗时不能当作启动耗时。[缓存专项](artifacts/validation/0.5.0/pipeline-cache-final.json)。

```powershell
.\tools\benchmark.ps1 -ModelDirectory .cache/models -Iterations 10
dotnet run --project src/SimdYolo.Cli -c Release -- profile .cache/models/yolov8n.pt vulkan 640
```

与 Python + PyTorch 做同机速度对比：

```powershell
dotnet build -c Release
python tools/compare-python.py --model-dir .cache/models --vision --threads 12 --warmup 5 --iterations 20
```

该工具统一 FP32、batch=1、输入文件、预热次数和测量次数，串行测量 .NET CPU/Vulkan 与原版 PyTorch CPU 的未融合/融合模型。CUDA 可用时，还会测包含输入上传和输出回读的 GPU 耗时，以及输入输出驻留 GPU 的耗时。JSON 保存每次采样、P95、FPS 和模型/输入校验值；Markdown、PNG、SVG 报告写到 `artifacts/python-comparison.*`。这些是模型前向和输出头解码的基准，不包含图片预处理或 NMS/mask 后处理。

Python 对照依赖仅用于开发测试。未安装 Ultralytics 时，脚本可使用已有的 `.cache/reference`；脚本本身不安装任何包。也可以只测指定模型：

```powershell
python tools/compare-python.py --models yolov8n yolo11n --model-dir .cache --iterations 20
```

比较当前版本、原版 PyTorch 和 ONNX Runtime 的 CPU/GPU 速度、CPU 时间、进程内存及 GPU 使用率：

```powershell
dotnet build -c Release
dotnet build tools/OrtBenchmark/OrtBenchmark.csproj -c Release
python tools/compare-runtimes.py --model-dir models/pt --onnx-dir models/onnx --ultralytics-dir .cache/yolo26-release-reference --models yolov8n yolov9t yolov10n yolo11n yolo26n yolov8n-seg yolo11n-seg yolo26n-seg yolov8n-pose yolo11n-pose yolo26n-pose --threads 12 --warmup 60 --iterations 40 --duration 4 --repeats 2 --engines torch-cpu net-cpu net-onnx-cpu ort-net-cpu net-vulkan net-onnx-vulkan torch-cuda ort-net-cuda --input .cache/image-validation/input.f32 --output artifacts/runtime-comparison-0.8.1.json
```

该命令测量 11 个模型的四种 CPU 和四种 GPU 配置，两轮共 176 个窗口，写到 `artifacts/runtime-comparison-0.8.1.json`、`.md`、`.png` 和 `.svg`。`optimized-runtime-comparison.*`、`cpu-runtime-comparison.*` 保留 0.7.0 的原始记录。真实采样时间戳、CPU ms/帧、全进程 RSS、私有提交量及整卡 GPU/显存均保存在 JSON；加载、首帧、预热时间分别记录，不混进稳态排名。

正式 ORT 对照使用 .NET 宿主和官方 Managed 绑定，输入/输出均在主机，输出复制为托管数组，与 `PredictRaw` 的返回方式一致。开发环境需要已有的 `numpy`、`psutil`、CPU PyTorch、Ultralytics，以及 `.cache/torch-cu126` 中的 CUDA PyTorch、`.cache/ort-gpu` 中的官方 ORT GPU 包。此次使用 Ultralytics 8.4.0 兼容 YOLO26 官方检查点；可用 `--ultralytics-dir` 指定已安装目录、`--net-cli` 指定冻结 CLI。工具直接预加载 CUDA/cuDNN DLL，不启动 Python 来执行 ORT，主推理库也没有新增 ORT 依赖。本机 NuGet ORT GPU 原生构建有 Pascal 内核兼容问题，所以对照通过 .NET 调用官方 GPU wheel 中已验证的同版本 DLL；JSON 保存原生 DLL 哈希。

正式比较使用 bus 示例图预处理后的同一个张量；输入不存在时先生成，再运行上面的全后端命令：

```powershell
dotnet src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll preprocess .cache/image-validation/image.ppm .cache/image-validation/input.f32 640
```

不指定 `--input` 时使用固定种子的随机张量。可调整 `--repeats`，或用 `--engines torch-cpu net-onnx-cpu net-onnx-vulkan` 检查本库的 ONNX 路径。`--render-only --output artifacts/runtime-comparison-0.8.1.json` 只更新 Markdown/图表，保留原始 JSON，不重新计时。脚本逐个校验所有输出，端到端 TopK 核对完整候选、类别和分割/姿态附加字段；这些计时均是网络和解码头的基准，排除图片预处理、NMS 和最终 mask 后处理。Metal 尚未在这台 Windows 机器上测试。

版本优化收益使用同一 .NET 测量宿主，只替换主 DLL，并以旧版、新版、新版、旧版的 ABBA 次序启动独立进程：

```powershell
python tools/compare-lifecycle.py --baseline-dir <冻结的旧版目录> --candidate-dir src/SimdYolo/bin/Release/net10.0 --input .cache/image-validation/input.f32 --repeats 2 --warmup 60 --duration 4 --output .cache/lifecycle-comparison.json
```

这里的 `<冻结的旧版目录>` 必须含对应版本的 `SimdYolo.dll`，可以从旧 NuGet 包解压。工具保存加载、首帧、稳态、分配量和首帧/最后一帧原始输出；GPU 缓存状态逐窗口记录。`--cache-mode isolated` 使用每窗口独立应用缓存目录，仍不清空系统或驱动缓存。程序内至首输出的耗时不包含 dotnet 宿主启动；进程总耗时还包含预热、稳态和写盘，不能作为首输出耗时。

## CLI 与验证

CLI 接受 P6 RGB PPM 图片，`predict` 输出 JSON、关键点和 PGM 裁剪掩码。库的字节输入 API 可使用任意图片格式的解码结果。

```powershell
dotnet run --project src/SimdYolo.Cli -c Release -- predict yolo11n-seg.pt image.ppm vulkan predictions
dotnet run --project src/SimdYolo.Cli -c Release -- detect yolov8n.pt image.ppm cpu
dotnet run --project tests/SimdYolo.Tests -c Release -- --vulkan --models .cache/models
```

macOS 后端验证命令：

```sh
dotnet run --project tests/SimdYolo.Tests -c Release -- --metal
python tools/validate-reference.py yolo11n.pt --onnx --backends metal
```

`tools/validate-reference.py` 和 `tools/validate-image.py` 是可选开发对照工具，需要开发环境中的 PyTorch/Ultralytics（ONNX 导出还需要 onnx）。它们不属于运行时依赖。对照导出文件是测试夹具，`.pt` 的 .NET 加载路径始终独立解析原文件。

协议和算子参考：[ONNX 算子](https://onnx.ai/onnx/operators/)、[Vulkan 规范](https://docs.vulkan.org/spec/latest/chapters/pipelines.html)、[Metal compute](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder)、[Ultralytics 模块](https://docs.ultralytics.com/reference/nn/modules/head/)。

