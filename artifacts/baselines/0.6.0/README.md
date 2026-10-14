# SimdYolo

C# / .NET 10 的 YOLO 推理库，包含目标检测、实例分割和姿态估计。主库为纯托管 C# 程序集：模型解析、计算图、CPU 算子、图像预处理和结果解码均由 C# 实现，CPU 使用 .NET SIMD / intrinsics。没有外部 NuGet 运行依赖，不调用 Python、PyTorch、LibTorch、TorchSharp、ONNX Runtime、BLAS/MKL/OpenBLAS 或其他原生数学运行库，也不在运行时把 `.pt` 转成 ONNX。GPU 通过 P/Invoke 调用系统 Vulkan / Metal API，执行包内嵌入的 SPIR-V / MSL 着色器。

当前版本为 0.6.0，本轮集中优化 CPU，保持纯 C#、FP32：专用执行器减少并行调度成本，步长 2 卷积共享输入加载，Winograd 使用适用形状的 24 列计算块，检测/姿态头使用 SIMD，并补齐 YOLOv9 ONNX 卷积与 SiLU 融合。执行器修复了任务发布与休眠交接竞争，回归包含 20,000 次强制休眠压力检查。GPU 源码和嵌入着色器本轮冻结，下面的 GPU 排名仍来自 0.5.0 历史测量，不能当作 0.6.0 的计时结果。Metal 尚无 macOS 硬件验证。

2026-10-14 13:02:08 +0800 同机 CPU 正式比较已完成 64/64 个窗口、14,770 次计时推理，80/80 个输出校验通过（20 个 PyTorch CPU 自对照、60 个跨实现对照）。硬件为 Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz；统一 FP32、batch=1、640×640、同一输入张量、CPU 线程上限 12。每配置独立运行两轮，各预热 30 次后至少 8 秒、40 次。计时包含网络前向和输出头解码，排除图片预处理、NMS、最终 mask 后处理和冷首帧。本轮 GPU 只做正确性验证，没有重新计时。

## 0.6.0 CPU 速度排名

每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗；重复窗口取各窗口中位数的中位数，按未舍入值排序。小幅差距不代表稳定胜出或统计显著性。

| 模型 | 原版 PyTorch CPU | 我们的 `.pt` CPU | 我们的 ONNX CPU | ORT CPU（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 45.04 ④ | 26.35 ② | 27.31 ③ | **24.21 ①** |
| yolov9t | 77.60 ④ | 34.96 ② | 36.67 ③ | **28.34 ①** |
| yolov10n | 59.75 ④ | 29.90 ② | 33.25 ③ | **22.36 ①** |
| yolo11n | 50.99 ④ | 27.03 ② | 27.39 ③ | **23.19 ①** |
| yolov8n-seg | 57.04 ④ | 34.43 ② | 35.51 ③ | **33.70 ①** |
| yolov8n-pose | 47.79 ④ | 29.24 ② | 29.68 ③ | **24.94 ①** |
| yolo11n-seg | 62.50 ④ | 35.31 ② | 36.62 ③ | **33.15 ①** |
| yolo11n-pose | 54.65 ④ | 29.53 ② | 30.67 ③ | **24.79 ①** |
| 第一名次数（完整四配置 8/8 模型） | 0 | 0 | 0 | 8 |

`.pt` CPU 比 原版 PyTorch CPU 更快 8/8 个模型，比 ORT CPU 更快 0/8 个模型；ONNX CPU 比 原版 PyTorch CPU 更快 8/8 个模型，比 ORT CPU 更快 0/8 个模型。名次基于本次窗口的数值，接近的结果不代表稳定领先。

## CPU 时间与内存

以下为逐模型比值的范围，不是把不同模型成本合并平均；大于 1 表示本库成本更高。CPU%按全机逻辑核心归一化，CPU ms/帧由进程累计 CPU 时间计算，RSS 为整个宿主进程的采样工作集。

| 路径 | CPU ms/帧 / PyTorch | CPU ms/帧 / ORT | RSS / PyTorch | RSS / ORT |
| --- | ---: | ---: | ---: | ---: |
| `.pt` CPU | 0.43–0.55× | 0.89–1.11× | 1.13–1.54× | 2.12–3.49× |
| ONNX CPU | 0.44–0.55× | 0.90–1.13× | 0.79–1.44× | 1.56–3.48× |

较高 CPU 占用不代表更高效率；较低 CPU 占用也不能单独证明效率优势。更快延迟可能伴随更多 CPU 时间，需要同时查看 CPU ms/帧和进程 RSS，不能据速度排名宣称资源成本全面领先。RSS 包含运行库、模型和缓存，不等于算子工作区大小。

[0.6 CPU 完整报告](artifacts/cpu-runtime-comparison.md)、[原始 JSON](artifacts/cpu-runtime-comparison.json)、[验证记录](VALIDATION.md)保留全部重复窗口、资源、版本与输出校验。

## 0.5→0.6 CPU 历史速度对照

每格为 0.5→0.6 中位耗时 ms（耗时降低百分比，负值表示更慢）。这是两次历史测量，未在同一窗口做版本开关 A/B；系统热状态、线程策略和缓存可能不同，不能把差值全部归因于代码。

| 模型 | `.pt` CPU：0.5→0.6 | ONNX CPU：0.5→0.6 |
| --- | ---: | ---: |
| yolov8n | 30.87→26.35（14.6%） | 31.06→27.31（12.1%） |
| yolov9t | 45.36→34.96（22.9%） | 53.97→36.67（32.1%） |
| yolov10n | 35.66→29.90（16.2%） | 38.54→33.25（13.7%） |
| yolo11n | 32.69→27.03（17.3%） | 32.76→27.39（16.4%） |
| yolov8n-seg | 39.81→34.43（13.5%） | 40.18→35.51（11.6%） |
| yolov8n-pose | 37.49→29.24（22.0%） | 34.23→29.68（13.3%） |
| yolo11n-seg | 41.41→35.31（14.7%） | 42.04→36.62（12.9%） |
| yolo11n-pose | 38.87→29.53（24.0%） | 36.13→30.67（15.1%） |

## GPU 速度排名（0.5.0 历史，本轮未重新计时）

以下整张表来自 0.5.0 归档，未与 0.6 CPU 窗口混合。本轮 GPU 正确性回归通过不代表重新证明这些速度排名。

每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗；重复窗口取各窗口中位数的中位数，按未舍入值排序。小幅差距不代表稳定胜出或统计显著性。

| 模型 | 原版 PyTorch CUDA | 我们的 `.pt` Vulkan | 我们的 ONNX Vulkan | ORT CUDA（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | **9.92 ①** | 10.47 ② | 10.57 ③ | 12.74 ④ |
| yolov9t | 22.01 ④ | **12.88 ①** | 14.15 ② | 17.87 ③ |
| yolov10n | 12.86 ③ | **10.09 ①** | 10.26 ② | 13.59 ④ |
| yolo11n | 10.67 ③ | 10.66 ② | **10.60 ①** | 13.74 ④ |
| yolov8n-seg | **12.96 ①** | 13.77 ③ | 13.60 ② | 16.62 ④ |
| yolov8n-pose | **10.20 ①** | 10.52 ② | 10.79 ③ | 13.16 ④ |
| yolo11n-seg | 13.57 ② | **13.49 ①** | 13.69 ③ | 17.65 ④ |
| yolo11n-pose | 11.06 ② | **11.05 ①** | 11.21 ③ | 14.30 ④ |
| 第一名次数（完整四配置 8/8 模型） | 3 | 4 | 1 | 0 |

[0.5 全后端历史报告](artifacts/baselines/0.5.0/optimized-runtime-comparison.md)、[0.5 原始 JSON](artifacts/baselines/0.5.0/optimized-runtime-comparison.json)；主目录中的 [optimized-runtime-comparison.md](artifacts/optimized-runtime-comparison.md)同样保留为 0.5 历史记录。

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

交付包的 DLL 一致性、依赖/资产审查及独立本地 NuGet 消费结果见 [0.6 包审查](artifacts/validation/0.6.0/package-audit.json)、[CPU 消费日志](artifacts/validation/0.6.0/consumer-cpu.log)和 [Vulkan 消费日志](artifacts/validation/0.6.0/consumer-vulkan.log)，具体状态以记录为准。主库 DLL 与正式矩阵绑定；文档重打包会改变包哈希。

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

其他大小和自训练模型在使用已支持模块/算子时可以加载，但没有全部逐一验证。WongKinYiu 原仓库 YOLOv9 的 RepConvN、RepNCSP、CBLinear、CBFuse 和多分支 DDetect 已实现，原仓库检查点仍需单独验证。不会因为文件名包含 v8/v9/v10/v11 就假定模型兼容。

`.pt` 支持 `torch.save` 的 ZIP 检查点，检查点中必须包含完整 YOLO `nn.Module`（`model` 或 `ema`），也支持直接保存完整模型。解析器读取 pickle 数据和 storage，恢复模块连接和权重，不导入或执行 Python GLOBAL/REDUCE。支持 FP32、FP16、BF16 storage，内部计算使用 FP32。

单独的 `state_dict` 没有执行结构，不能自动推断任意自定义网络；这类文件需要完整模型或 ONNX。TorchScript、旧式非 ZIP pickle、未知自定义模块和量化计算图不在当前支持范围，加载时返回明确异常。PyTorch 的文件格式背景见 [官方序列化文档](https://docs.pytorch.org/docs/stable/notes/serialization.html)。

ONNX 支持普通 `ai.onnx` 域、opset 11–21 中已实现的算子；主要对照测试使用 opset 17。支持动态图像尺寸、FP32/FP16 权重和同目录 external data。算子包括 Conv、ConvTranspose、MatMul、Softmax、常见激活/逐元素运算、Pool、Resize、Concat、Split、Reshape、Transpose、Gather、Slice、Shape、Range、TopK 等。Resize 支持 YOLO 常用的 nearest/asymmetric/floor，其他插值模式会明确拒绝。带内嵌 NonMaxSuppression 的导出、分类/OBB、未知算子或其他 ONNX 域不支持。shape/index 路径使用 FP32，超过其精确整数范围的通用整数模型不适用。

## 后端和性能

CPU 的 GEMM 使用 AVX/FMA 寄存器分块、连续列块打包和分批调度，其他硬件使用 `System.Numerics.Vector<float>` 路径。适用的 3×3 卷积采用 F(2×2,3×3) / F(4×4,3×3) Winograd，并对输入和输出变换使用 SIMD；普通卷积按 16 个空间位置生成计算块，减少完整 im2col 的分配和内存往返。depthwise、池化、上采样、Softmax 和 2×2 转置卷积也有专用 SIMD / GEMM 路径。会话按计算图的存活引用及时回收中间数组，缓存保留上限为 256 MiB；返回的公开输出脱离池管理，后续推理和调用方修改输出不会覆盖模型常量。

固定权重可缓存 GEMM 的 A 矩阵寄存器布局，减少内层循环寻址；普通 packed-A 路径对少于 6 个输出行、无 AVX/FMA 或可变权重保留回退。可用 `SIMDYOLO_CPU_PACKED_A=0` 关闭，线程上限仍由调用方指定。

0.5 的逐点卷积使用 4 行×24 列 AVX/FMA 计算块，适用时缓存固定权重布局，并融合 bias / SiLU；不满足硬件/形状/权重条件时回退。Winograd 在同一线程池任务的局部计算块中完成输入变换、通道归约和逆变换，避免整幅中间结果和重复打包，较小深层仍用原路径。0.6 为 CPU 增加 backend 专属执行器，调用线程参与计算并计入 CpuThreads 上限，空闲线程在有界自旋后休眠。请使用 using 或 Dispose 释放 detector/backend 持有的工作线程。发布任务和休眠交接使用完整内存屏障，并有强制休眠压力回归；其他路径可使用常规 .NET 线程池回退。

CPU SiLU 的常见有限值采用 FP32 范围缩减和多项式近似，在 AVX2/FMA 上执行；NaN、极值和不满足硬件条件时保留原实现，没有截断输入或量化。可用 `SIMDYOLO_CPU_FAST_ACTIVATION=0` 关闭。数值精度以原版输出对照记录为准。

1×1 卷积在计算块内融合 bias / SiLU。适用的 batch=1、固定权重通道 Concat→1×1 Conv 直接从各输入打包 GEMM 计算块，避免分配和复制完整的拼接张量；其他形状、设备和可变权重使用普通 Concat/Conv 路径。`.pt` 的 C2f/C3/SPPF 等模块及 ONNX 单使用方模式均可使用该优化。

`Tensor.Shape` 被调用方改成与存储不一致的尺寸时，会在 unsafe SIMD / GPU 入口前拒绝。CPU 输出通过所有权转移发布，重复输出别名各自复制；CPU/GPU 嵌套会话结束后恢复外层的张量池和设备作用域。

`.pt` 加载时融合 Conv/BN、RepConv 和 RepVGGDW，检测头融合 DFL/坐标解码，v10 使用堆式 TopK。ONNX 加载时消除死节点、折叠常量，并在匹配且不破坏其他输出/使用方时融合 Conv/BN、卷积分支、SiLU、DFL、Sigmoid/Concat 和 CPU Concat/1×1 Conv。按输入形状缓存最多 8 组 shape/anchor 计算计划，数据依赖的形状仍逐次计算。连续 Slice 可在设备上执行，减少 YOLO 通道切片触发的 CPU 回读；TopK 使用堆选择，连续末轴 ReduceMax 使用 SIMD。

Vulkan 使用 F2/F4 Winograd、低通道卷积和多种列宽的分块内核，在首次遇到适用形状时用 GPU timestamp 选择卷积内核；无 timestamp 支持时使用计时回退。固定权重驻留设备，中间张量保留在复用的显存块中，多个算子记录到同一命令缓冲区。depthwise、转置卷积、池化、上采样、广播、转置、Slice、Softmax、检测/关键点解码均有设备路径；分割的 2×2 转置卷积使用矩阵计算和像素重排。多个公开输出共用一次批量读回，读回内存优先选择 HOST_CACHED。Metal 保留运行时 MSL、共享内存和固定权重缓存；没有 Metal 提速或正确性的硬件实测结论。

同一帧中，多个卷积可复用共享输入的 Winograd 变换；缓存区分设备存储偏移、形状和变换块，结束帧时清空，后续帧重新计算。

0.5 的 Vulkan 为适用固定权重逐点卷积使用 vec4 读写与 32/64 输出行分块；符合边界/对齐条件的 3×3 depthwise 也有 vec4 路径。Winograd 按形状在 32×32 / 64×32 / 64×64 计算块间选核，适用的 F4 变换使用向量化读写。形状、设备存储偏移、通道尾部或可变权重不适用时保留通用路径。

ONNX 的末轴 ReduceMax 和 GatherElements 有 Vulkan 路径；YOLOv10 导出的端到端头按精确数据流、常量和使用方匹配，在 batch=1 / 适用布局下融合为设备两阶段 TopK，默认启用，可用 `SIMDYOLO_GPU_ONNX_E2E=0` 关闭。其他变体回到原算子图。`.pt` 的专用 TopK 默认在 Vulkan 执行，`SIMDYOLO_GPU_TOPK=0` 也会禁用专用头路径。通用 ONNX TopK 仍默认走 CPU，其独立 GPU 路径仅在显式设置 `SIMDYOLO_GPU_ONNX_TOPK=1` 或 `SIMDYOLO_GPU_TOPK=1` 时启用。

Vulkan 默认将驱动 pipeline cache 和形状选核结果保存到系统临时目录的 `SimdYolo/VulkanCache`，可用 `SIMDYOLO_VULKAN_CACHE_DIR` 指定目录，或以 `SIMDYOLO_VULKAN_CACHE=0` 禁用。缓存绑定设备/驱动标识，选核记录还绑定嵌入着色器哈希及路由选项；损坏、不兼容或无法读写时继续使用未缓存路径。缓存命中不代表新形状已编译/选核，首次遇到新形状仍须预热。会话释放时保存新增结果；`PipelineCacheLoaded` / `CachedConvolutionShapes` 可观察恢复状态。

固定形状、全设备执行的图可在预热后重放已记录的命令缓冲区，减少 C# / 驱动逐帧记录开销；每帧仍上传实际输入、回读全部输出并返回独立主机数组。第一次权重上传、CPU 数据依赖处理、其他动态上传或中途同步会阻止捕获；输入形状变化时重新记录。`ReplayedFrames` 表示实际重放帧数，`LayerCompleted` 和 GPU profiling 会禁用该路径；诊断时也可设置 `SIMDYOLO_VULKAN_REPLAY=0`。它不保证所有模型都重放，也不缓存预测结果。

Vulkan 需要 64 位进程和系统 Vulkan 驱动，Metal 需要 64 位 macOS。Vulkan 优先选择有 compute queue 的独立显卡，再考虑核显、虚拟设备等，同类型保留驱动枚举顺序。默认使用 CPU；显式选择 Vulkan/Metal 时失败会报错，`Auto` 初始化失败会回退 CPU，并通过 `FallbackReason` 提供原因。运行中 GPU 出错会报错。会话串行执行推理，建议复用会话并提前预热；不要每帧重新加载模型。

Vulkan 在需要 CPU 结果时才提交并等待；普通推理不逐算子同步。未覆盖的算子和不匹配端到端融合模式的 ONNX TopK 会产生 CPU 回退及传输。首次形状特化和内核选择会增加首帧耗时，应在处理视频前通过 `PredictRaw` 预热，并复用会话。`LayerCompleted` 仅用于诊断，会主动同步 GPU 测量每层耗时；性能测试时请关闭。内部计算为 FP32，FP16/INT8 算子和 Tensor Core 尚未实现，结论不能外推为所有硬件或 PyTorch/TensorRT 配置的结果。

0.5 缓存专项在 v8n / 640×640 上测得：空应用缓存首帧 5.373 秒，恢复已知应用缓存后 0.310 秒，同结构 ONNX 为 0.388 秒，恢复 12 个选核结果。未清空系统/驱动 shader cache，不能视为完全冷启动或与 0.4 的代码纯收益比较。正式矩阵的 32 个 Vulkan 窗口均命中应用缓存，首次推理为 1.022–1.906 秒；CPU 两种格式为 0.239–0.354 秒，均不含模型构造/加载。新设备、驱动或形状仍可能需要昂贵编译/选核，稳态耗时不能当作启动耗时。[缓存专项](artifacts/validation/0.5.0/pipeline-cache-final.json)。

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

比较当前版本、原版 PyTorch 和 ONNX Runtime 的 CPU 速度、CPU 时间和进程内存：

```powershell
dotnet build -c Release
dotnet build tools/OrtBenchmark/OrtBenchmark.csproj -c Release
python tools/compare-runtimes.py --vision --threads 12 --warmup 30 --iterations 40 --duration 8 --repeats 2 --engines torch-cpu net-cpu net-onnx-cpu ort-net-cpu --input .cache/speed-comparison/bus-input.f32 --output artifacts/cpu-runtime-comparison.json
```

该命令只重测 8 个模型的四种 CPU 配置，两轮共 64 个窗口，写到 `artifacts/cpu-runtime-comparison.json`、`.md`、`.png` 和 `.svg`。统一 FP32、batch=1、640×640，各配置独立进程、串行运行。目标每 250 ms 采样，CPU 满载时可能延迟；真实时间戳、CPU ms/帧、全进程 RSS 和私有提交量保存在 JSON。0.5 的 `optimized-runtime-comparison.*` 保留为历史全后端报告；重新测试 GPU 时请使用不同输出文件名，避免覆盖历史证据。

正式 ORT 对照使用 .NET 宿主和官方 Managed 绑定，输入/输出均在主机，输出复制为托管数组，与 `PredictRaw` 的返回方式一致。开发环境需要已有的 `numpy`、`psutil`、CPU PyTorch、Ultralytics，以及 `.cache/torch-cu126` 中的 CUDA PyTorch、`.cache/ort-gpu` 中的官方 ORT GPU 包和 `.cache/validation` 中的静态 640 ONNX。工具直接预加载 CUDA/cuDNN DLL，不启动 Python 来执行 ORT，主推理库也没有新增 ORT 依赖。本机 NuGet ORT GPU 原生构建有 Pascal 内核兼容问题，所以对照通过 .NET 调用官方 GPU wheel 中已验证的同版本 DLL；JSON 保存原生 DLL 哈希。

正式比较使用 bus 示例图预处理后的同一个张量；输入不存在时先生成，再运行上面的全后端命令：

```powershell
dotnet src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll preprocess .cache/image-validation/image.ppm .cache/speed-comparison/bus-input.f32 640
```

不指定 `--input` 时使用固定种子的随机张量。可调整 `--repeats`，或用 `--engines torch-cpu net-onnx-cpu net-onnx-vulkan` 检查本库的 ONNX 路径。`--render-only --output artifacts/cpu-runtime-comparison.json` 只更新 Markdown/图表，保留原始 JSON，不重新计时。脚本逐个校验所有输出；这些计时均是网络和解码头的基准，排除图片预处理、NMS 和最终 mask 后处理。Metal 尚未在这台 Windows 机器上测试。

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
