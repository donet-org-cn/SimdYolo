# 推理 API 与模型兼容范围

以下命令从仓库根目录执行。当前版本和实测排名见 [README](../README.md)。

## 构建和使用

```powershell
dotnet build -c Release
dotnet run --project tests/SimdYolo.Tests -c Release -- --vulkan
git lfs pull
dotnet run --project src/SimdYolo.Cli -c Release -- inspect models/pt/yolov8n.pt
dotnet run --project src/SimdYolo.Cli -c Release -- bench models/pt/yolov8n.pt vulkan 640 10
```

普通构建使用仓库内预编译的 SPIR-V，不需要 Vulkan SDK。重新生成着色器需要 glslang：

```powershell
.\tools\build-shaders.ps1 -Compiler glslangValidator
```

应用可以引用 `src/SimdYolo/SimdYolo.csproj`，或使用下面的命令生成本地 NuGet 包。包内主库没有外部运行依赖，开发用的 ORT 对照工具不随包发布：

```powershell
dotnet pack src/SimdYolo -c Release -o artifacts
```

0.9.0交付包的 DLL 一致性、依赖和独立NuGet消费结果见[包审查](../artifacts/validation/0.9.0/package-audit.json)、[CPU消费](../artifacts/validation/0.9.0/package-consumer-cpu.json)和[Vulkan消费](../artifacts/validation/0.9.0/package-consumer-vulkan.json)。本次发布保持这些已验证二进制和摘要，最新README/图表另在源码和文档附件中更新。

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

0.5 缓存专项在 v8n / 640×640 上测得：空应用缓存首帧 5.373 秒，恢复已知应用缓存后 0.310 秒，同结构 ONNX 为 0.388 秒，恢复 12 个选核结果。未清空系统/驱动 shader cache，不能视为完全冷启动或与 0.4 的代码纯收益比较。0.5 当时正式矩阵的 32 个 Vulkan 窗口均命中应用缓存，首次推理为 1.022–1.906 秒；CPU 两种格式为 0.239–0.354 秒，均不含模型构造/加载。新设备、驱动或形状仍可能需要昂贵编译/选核，稳态耗时不能当作启动耗时。[缓存专项](../artifacts/validation/0.5.0/pipeline-cache-final.json)。

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
python tools/compare-runtimes.py --model-dir models/pt --onnx-dir models/onnx --ultralytics-dir .cache/yolo26-release-reference --models yolov8n yolov9t yolov10n yolo11n yolo26n yolov8n-seg yolo11n-seg yolo26n-seg yolov8n-pose yolo11n-pose yolo26n-pose --threads 12 --warmup 60 --iterations 40 --duration 4 --repeats 2 --engines torch-cpu net-cpu net-onnx-cpu ort-net-cpu net-vulkan net-onnx-vulkan torch-cuda ort-net-cuda --input .cache/image-validation/input.f32 --output artifacts/runtime-comparison-0.9.0.json
```

该命令测量 11 个模型的四种 CPU 和四种 GPU 配置，两轮共 176 个窗口，写到 `artifacts/runtime-comparison-0.9.0.json`、`.md`、`.png` 和 `.svg`。`optimized-runtime-comparison.*`、`cpu-runtime-comparison.*` 保留 0.7.0 的原始记录。真实采样时间戳、CPU ms/帧、全进程 RSS、私有提交量及整卡 GPU/显存均保存在 JSON；加载、首帧、预热时间分别记录，不混进稳态排名。

正式 ORT 对照使用 .NET 宿主和官方 Managed 绑定，输入/输出均在主机，输出复制为托管数组，与 `PredictRaw` 的返回方式一致。开发环境需要已有的 `numpy`、`psutil`、CPU PyTorch、Ultralytics，以及 `.cache/torch-cu126` 中的 CUDA PyTorch、`.cache/ort-gpu` 中的官方 ORT GPU 包。此次使用 Ultralytics 8.4.0 兼容 YOLO26 官方检查点；可用 `--ultralytics-dir` 指定已安装目录、`--net-cli` 指定冻结 CLI。工具直接预加载 CUDA/cuDNN DLL，不启动 Python 来执行 ORT，主推理库也没有新增 ORT 依赖。本机 NuGet ORT GPU 原生构建有 Pascal 内核兼容问题，所以对照通过 .NET 调用官方 GPU wheel 中已验证的同版本 DLL；JSON 保存原生 DLL 哈希。

正式比较使用 bus 示例图预处理后的同一个张量；输入不存在时先生成，再运行上面的全后端命令：

```powershell
dotnet src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll preprocess .cache/image-validation/image.ppm .cache/image-validation/input.f32 640
```

不指定 `--input` 时使用固定种子的随机张量。可调整 `--repeats`，或用 `--engines torch-cpu net-onnx-cpu net-onnx-vulkan` 检查本库的 ONNX 路径。`--render-only --output artifacts/runtime-comparison-0.9.0.json` 只更新 Markdown/图表，保留原始 JSON，不重新计时。脚本逐个校验所有输出，端到端 TopK 核对完整候选、类别和分割/姿态附加字段；这些计时均是网络和解码头的基准，排除图片预处理、NMS 和最终 mask 后处理。Metal 尚未在这台 Windows 机器上测试。

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
dotnet run --project tests/SimdYolo.Tests -c Release -- --vulkan --models models/pt
```

macOS 后端验证命令：

```sh
dotnet run --project tests/SimdYolo.Tests -c Release -- --metal
python tools/validate-reference.py yolo11n.pt --onnx --backends metal
```

`tools/validate-reference.py` 和 `tools/validate-image.py` 是可选开发对照工具，需要开发环境中的 PyTorch/Ultralytics（ONNX 导出还需要 onnx）。它们不属于运行时依赖。对照导出文件是测试夹具，`.pt` 的 .NET 加载路径始终独立解析原文件。

协议和算子参考：[ONNX 算子](https://onnx.ai/onnx/operators/)、[Vulkan 规范](https://docs.vulkan.org/spec/latest/chapters/pipelines.html)、[Metal compute](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder)、[Ultralytics 模块](https://docs.ultralytics.com/reference/nn/modules/head/)。
