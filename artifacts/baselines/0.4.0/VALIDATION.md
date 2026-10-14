# 验证记录

测试日期：2026-10-13–14（UTC+8）。运行环境：Windows x64，.NET SDK 10.0.401 / Runtime 10.0.12，Intel i7-8700K（12 逻辑核心）、31.93 GiB RAM、256 位 SIMD，NVIDIA GTX 1050 Ti（驱动 560.94），Vulkan 1.3。正式对照使用 PyTorch CPU 2.11.0+cpu / CUDA 2.11.0+cu126、Ultralytics 8.3.0 和 ORT 1.26.0（.NET 宿主）。模型来自 Ultralytics assets 的 v8.3.0 发布。

0.4.0 正式矩阵于 2026-10-14 完成 128/128 个独立窗口，共 51,154 次计时推理，160 个输出全部通过（20 个 PyTorch CPU 自对照、140 个跨实现对照），最大绝对误差为 0.008194；判定使用逐元素 `atol + rtol × |reference|`。统一 FP32、batch=1、640×640、bus 图像产生的同一输入张量、CPU 线程上限 12；每配置两轮，各预热 30 次后至少 8 秒、40 次。GPU 包含输入上传和全部输出回读；排除图片预处理、NMS、mask 后处理及冷首帧。

## 0.4.0 速度排名

每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗；重复窗口取各窗口中位数的中位数，按未舍入值排序。小幅差距不代表稳定胜出或统计显著性。

### CPU 速度排名

| 模型 | 原版 PyTorch CPU | 我们的 `.pt` CPU | 我们的 ONNX CPU | ORT CPU（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 52.94 ④ | 32.96 ② | 33.18 ③ | **24.36 ①** |
| yolov9t | 91.74 ④ | 45.22 ② | 56.31 ③ | **29.05 ①** |
| yolov10n | 60.15 ④ | 36.88 ② | 40.01 ③ | **22.85 ①** |
| yolo11n | 50.78 ④ | 33.31 ② | 34.04 ③ | **23.21 ①** |
| yolov8n-seg | 57.35 ④ | 40.90 ② | 41.47 ③ | **33.97 ①** |
| yolov8n-pose | 49.24 ④ | 38.51 ③ | 34.78 ② | **25.17 ①** |
| yolo11n-seg | 63.23 ④ | 43.37 ② | 43.48 ③ | **33.04 ①** |
| yolo11n-pose | 55.23 ④ | 40.44 ③ | 36.65 ② | **24.85 ①** |
| 第一名次数（完整四配置 8/8 模型） | 0 | 0 | 0 | 8 |

### GPU 速度排名

| 模型 | 原版 PyTorch CUDA | 我们的 `.pt` Vulkan | 我们的 ONNX Vulkan | ORT CUDA（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | **9.98 ①** | 11.50 ③ | 11.50 ② | 12.79 ④ |
| yolov9t | 21.96 ④ | **13.65 ①** | 14.93 ② | 17.81 ③ |
| yolov10n | 12.86 ② | **12.08 ①** | 14.40 ④ | 13.58 ③ |
| yolo11n | **10.57 ①** | 12.21 ② | 12.42 ③ | 13.78 ④ |
| yolov8n-seg | **12.99 ①** | 14.80 ② | 14.89 ③ | 16.61 ④ |
| yolov8n-pose | **10.22 ①** | 11.59 ② | 11.82 ③ | 13.22 ④ |
| yolo11n-seg | **13.70 ①** | 15.95 ③ | 15.71 ② | 17.58 ④ |
| yolo11n-pose | **11.09 ①** | 12.73 ② | 12.88 ③ | 14.32 ④ |
| 第一名次数（完整四配置 8/8 模型） | 6 | 2 | 0 | 0 |

CPU 的 `.pt` / ONNX 在全部 8 个模型上快于本次 PyTorch CPU，ORT CPU 则全部第一。GPU 的 `.pt` Vulkan 在全部 8 个模型上快于 ORT CUDA，在 v9t / v10n 上快于 PyTorch CUDA，其余 6 个模型仍由 PyTorch CUDA 领先；ONNX Vulkan 在 7/8 个模型上快于 ORT CUDA，v10n 仍慢。

跨轮保守比较取本库较慢窗口 / 对照较快窗口：CPU `.pt` 比 PyTorch 耗时低 20.00–41.65%，ONNX 低 23.52–32.61%；`.pt` Vulkan 比 ORT CUDA 低 8.33–23.05%。Vulkan 对 PyTorch CUDA 的两项领先分别为 v9t 至少 36.68%、v10n 至少 5.66%。没有丢弃窗口或为排名补选数据。

PyTorch CPU 的 v8n 两轮为 45.12205–60.75045 ms，v9t 为 77.88140–105.60820 ms，约 35% 波动，原因未确定。两轮汇总取两个窗口中位数的平均；进程累计 CPU 时间与外部采样一致，不能把波动直接归因于采样延迟。PyTorch CPU 固定每模型先测，其余 7 个后端随机，模型顺序固定，仍可能存在顺序/热状态偏差，两次重复不构成统计显著性检验。

近似并列且跨轮换位：v8n GPU `.pt` / ONNX 综合仅差 0.003075 ms（约 0.027%）；11n-seg CPU 仅差 0.11035 ms（约 0.25%）。保留按未舍入值生成的名次，不能宣称这些路径稳定胜出。[完整报告](artifacts/optimized-runtime-comparison.md)、[原始 JSON](artifacts/optimized-runtime-comparison.json)、[PNG](artifacts/optimized-runtime-comparison.png)、[SVG](artifacts/optimized-runtime-comparison.svg)保留全部窗口和资源明细；0.3 结果归档在 [历史报告](artifacts/baselines/0.3.0/optimized-runtime-comparison.md)。

## 0.4.0 计时与资源口径

PyTorch 使用原版 Ultralytics `model.fuse()` / `inference_mode()`，没有 `torch.compile`。ORT 开启全部图优化、顺序执行和默认线程自旋，CUDA 卷积自动选择算法；CUDA 对照关闭 TF32。ORT 通过官方 .NET Managed 绑定加载 GPU wheel 中同版本原生 DLL，在运行时不调用 Python；其全部输出复制为托管 `float[]`。本机 NuGet GPU 原生构建在 QuickGelu 存在 Pascal 内核兼容问题，使用的官方 DLL 哈希已保存到 JSON。该工具与主库和 NuGet 包分离。

进程 CPU% = 窗口累计 CPU 秒 / 窗口秒 / 12 × 100，CPU ms/帧 = 窗口 CPU 秒 / 计时帧数 × 1000；CPU% 不是单核心占用。RSS 为包含宿主、模型和缓存的进程工作集；内存%按物理 RAM 计算，Private MiB 为私有提交量。外部目标采样间隔 250 ms，ORT CPU 各窗口的平均实际间隔为 350–472 ms，部分约 417–472 ms。整机 CPU、RSS、GPU/显存均为采样平均/采样峰值，不按实际采样间隔加权；原始时间戳保留。进程 CPU 时间和每帧延迟不依赖该采样平均方式。

NVML GPU/显存是整张卡的数值，包含桌面和其他进程；GPU 窗口舍弃最初一秒的滚动采样，保留启动前快照和近似显存增量。它不是精确的单进程显存或计算单元利用率，CPU 推理期间的 GPU 活动也不能归因于该后端。完整报告按 Repeat 列保留每轮资源/延迟；小数排序、精确并列、缺项和重复窗口规则见该报告。

正式矩阵 32 个 Vulkan 窗口的 `PipelineCacheLoaded` 均为 true、恢复 310 个已有选核形状。`.pt` Vulkan 首次推理为 0.304–0.578 秒，ONNX Vulkan 为 0.311–0.606 秒；CPU 两种格式为 0.276–0.367 秒。这些不含模型构造/加载，`StartupMs` 另含加载及预热；已恢复缓存的首帧不能当作空缓存、新设备/驱动或新形状的冷启动。

```powershell
dotnet build -c Release
dotnet build tools/OrtBenchmark/OrtBenchmark.csproj -c Release
python tools/compare-runtimes.py --vision --threads 12 --warmup 30 --iterations 40 --duration 8 --repeats 2 --engines torch-cpu net-cpu net-onnx-cpu ort-net-cpu net-vulkan net-onnx-vulkan torch-cuda ort-net-cuda --input .cache/speed-comparison/bus-input.f32 --output artifacts/runtime-comparison-reproduction.json
```

## 0.4.0 已完成的验证

0.4 源码已冻结，Release 构建零警告、零错误，47 项默认算子/回归检查及 4 个真实模型的 shape/batch 检查共 51 项通过。[完整回归日志](artifacts/validation/0.4.0/tests-vulkan-models.log)。关闭硬件 intrinsics 后 CPU 回归运行记录 37 项通过，其中只适用于 intrinsics 的直接内核用例按 guard 跳过。[可移植 CPU 日志](artifacts/validation/0.4.0/tests-cpu-portable.log)。

冻结主 DLL SHA256 为 `ab4c2d0bd0f5a6ce29578317dc9c812973ca7fe2d61bca22f7e14f0328ab8d49`，与正式矩阵的 `SimdYoloDllSha256` 一致。验证汇总及记录：[0.4 manifest](artifacts/validation/0.4.0/manifest.json)。本页以下 0.3/0.2/0.1 数字均为历史结果，不作为 0.4 成绩。

主库继续保持纯托管 C#、无 PyTorch/ORT/BLAS/原生数学运行库依赖；GPU 仅使用系统 Vulkan / Metal API 和嵌入着色器。Metal 仍无 macOS 硬件验证。

0.4 生产配置启用固定权重 packed-A、Vulkan 驱动 pipeline / tile 选择磁盘缓存、固定形状全设备图的命令重放、逐点卷积与 Winograd 寄存器分块自动选核、F4 向量化读写、Vulkan ONNX ReduceMax / GatherElements，以及 `.pt` YOLOv10 专用 GPU TopK。回归覆盖缓存恢复/损坏/不可写目录、输入变化、shape 切换、公开输出所有权、数据依赖索引和层回调。

最终 CPU 的 8 模型 `.pt` / ONNX、640×640 共 20 个输出通过原版 PyTorch 对照。[CPU oracle](artifacts/validation/0.4.0/cpu-packed-oracle.json)。GPU 的 16 个模型/格式配置各运行 4 个不同输入帧，共 80 个输出全部通过，其中 15/16 配置观察到 2 帧命令重放，默认 v10 ONNX 因 CPU TopK 路径未重放；最大绝对误差约 0.003357。[多帧 GPU oracle](artifacts/validation/0.4.0/real-model-replay.json)。

静态/动态 ONNX 的 CPU / Vulkan 30 个输出全部通过，最大绝对误差为 0.003723，单输出最大平均绝对误差为 4.40e-5。[ONNX 验证](artifacts/validation/0.4.0/onnx-final.json)。bus 图像 5 模型共 24 个实例通过框、NMS、mask/关键点后处理对照，两种分割模型的 11 张二值 mask IoU 全部为 1.0；最大框误差 0.000158 px，最大关键点误差 0.000415 px。[图像验证](artifacts/validation/0.4.0/image-final.json)。原始张量对照采用 `atol=0.003, rtol=0.0003`；图像对照共享 C# letterbox 输入，不要求与所有版本的图片预处理逐位相同。

交付包的 DLL/文档一致性、无外部依赖/原生数学运行库资产审查及独立本地 NuGet 消费结果见 [0.4 包审查](artifacts/validation/0.4.0/package-audit.json)、[CPU 消费日志](artifacts/validation/0.4.0/consumer-cpu.log)和 [Vulkan 消费日志](artifacts/validation/0.4.0/consumer-vulkan.log)，具体状态以记录为准。消费项目仅引用 `SimdYolo 0.4.0`，覆盖 `.pt` 检测、动态 ONNX 检测、分割、姿态，每模型 4 个变化输入帧，检查所有公开输出及旧帧所有权；包哈希以审查记录为准，主 DLL 与正式矩阵绑定。

最终缓存专项使用冻结 DLL、v8n、640×640、单次预热及 2 秒窗口：空缓存首帧为 13.574 秒，恢复已知缓存为 0.310 秒；同结构 ONNX 首帧为 0.423 秒，恢复 32 个选核形状，native / tuning 缓存合计 7,741,291 B。专项稳态中位耗时依次为 11.56 / 11.57 / 11.54 ms，缓存主要降低启动开销。这只验证该已知设备/形状，不是完整 8 模型/8 配置/双重复矩阵或新形状的冷启动范围。[缓存专项](artifacts/validation/0.4.0/pipeline-cache-final.json)。

ONNX head 的开发正反序 A/B 中，v10n 中位耗时均值为 16.158→14.990 ms，每帧进程 CPU 时间为 4.396→3.096 ms；4 个原版 PyTorch 输出对照均通过，最大绝对误差约 0.000580。[A/B](.cache/optimization-04/onnx-head/ab.json)、[正确性记录](.cache/optimization-04/onnx-head/oracle.json)对应开发阶段 DLL，不能与 0.3 表格跨窗口拼接成 0.4 的整体成绩。packed-A 的 16 配置/32 开关窗口开发对照降耗时约 7.9–15.7%；命令重放、专用 TopK 等配对速度记录同样仅属开发验证，正式模型结论以本页 128 窗口矩阵为准。

缓存默认目录为系统临时目录下 `SimdYolo/VulkanCache`，可指定 `SIMDYOLO_VULKAN_CACHE_DIR`；缓存不兼容、损坏或 IO 失败时继续未缓存执行，新形状仍须编译/选核，会话释放时保存新增缓存。首帧测量必须分别说明空缓存、已恢复缓存和首次出现的形状，并记录 `PipelineCacheLoaded` / `CachedConvolutionShapes`，不能把已知缓存启动当成全新冷首帧。

重放只捕获适用的全设备图：首帧权重预热、非输入的动态上传、CPU 非常量数据读取或中途 Flush 都会禁用捕获；形状变化重新记录，每帧仍上传输入并回读全部输出。层回调和 GPU profiling 不使用重放。`.pt` v10 专用 GPU TopK 默认启用；通用 ONNX TopK 的 GPU 候选在配对测试中退步，默认关闭、仅显式 opt-in。wide-F4 与 stride-2 im2col 候选因不稳定或退步已从生产路径移除，不宣称它们带来正式收益。

## 0.3.0 已完成的检查（历史）

0.3.0 已验证源码的 Release 构建通过，零警告、零错误，37 项统一回归（33 项基础检查及 4 个真实模型）全部通过，使用默认生产配置。主 DLL 的 PE/CLR 标志为 ILOnly，程序集引用仅含 .NET framework；主项目无外部运行库 PackageReference，CPU 不依赖 PyTorch、ORT、BLAS/MKL/OpenBLAS 或其他原生数学运行库。GPU 仅通过系统 Vulkan / Metal API 执行嵌入着色器。ORT Managed 绑定和原生 DLL 只用于独立的 `tools/OrtBenchmark` 开发对照项目，不属于 SimdYolo 运行依赖或 NuGet 内容。

另在禁用硬件 intrinsics 的 CPU 回退配置下通过全部 30 项检查，验证非 AVX/FMA 执行路径。

已生成并检查 [SimdYolo.0.3.0.nupkg](artifacts/SimdYolo.0.3.0.nupkg)：包内共 7 个文件，仅含托管 `lib/net10.0/SimdYolo.dll`、README/VALIDATION 和 NuGet 元数据，没有外部 NuGet 依赖、Python 资产或原生数学运行库资产。主 DLL 哈希与正式基准一致；GPU 着色器嵌入托管 DLL，运行时使用系统 Vulkan / Metal API。检查记录：[包审查 JSON](artifacts/validation/0.3.0/package-audit.json)。

独立本地 NuGet 源消费项目仅引用 `SimdYolo 0.3.0`，已离线 restore/build；CPU / Vulkan 的 `.pt` 检测、动态 ONNX 检测、`.pt` 分割和 `.pt` 姿态 4 个模型均在 64×64 输入下连续两帧成功，并检查旧输出未被后续帧回收覆盖。执行过程仅使用 dotnet，未调用 Python / ORT。包 SHA256 以最终[包审查记录](artifacts/validation/0.3.0/package-audit.json)为准，重打包文档会改变包哈希，主 DLL 哈希保持与正式基准一致。

回归覆盖 SIMD GEMM 的尾部和变动输入，implicit convolution 的步长/边界/batch/可变权重，分组与 dilation 卷积，F2/F4 Winograd 的通道尾部、batch=2 和图块边缘，SIMD depthwise、max/average pool、转置卷积、广播/转置/稳定 Softmax，以及检测、分割和姿态解码。真实 Vulkan GEMM/卷积和模型连续推理也包含在统一验证中；分块 MatMul 的 37×41×23 尾部矩阵、batch=2 和广播输入对照独立的 double 标量计算。

ONNX 回归检查 Slice 的动态输入、别名、负边界和反向切片，shape 缓存对数据依赖 Reshape 和公开输出修改的处理，Conv/BN/SiLU 与 Sigmoid/Concat 的多使用方安全性，DFL 模式和非标准布局回退，以及 TopK/ReduceMax 的并列值、NaN 和正负零语义。公开 Shape 与存储不一致时须在 unsafe 内核前拒绝；张量池通过转移所有权发布输出，重复别名复制，后续帧不得覆盖已返回数组；CPU/GPU 嵌套会话须恢复外层作用域。

新增回归对照 Concat→1×1 Conv 的通道/计算块/K 尾部、重复输入别名、bias/SiLU 与独立标量计算，检查 batch、可变权重和步长卷积的回退，以及 ONNX 轴、多个使用方和动态 batch 的融合安全性。Winograd 共享输入检查设备存储偏移、不同形状别名和跨帧生命周期，防止错误复用变换结果。

0.3.0 主库的最终 640×640 ONNX 对照覆盖 8 个静态模型和 4 个动态模型的 CPU/Vulkan 输出，共 30 个输出全部通过。最大绝对误差为 0.003723，单输出的最大平均绝对误差为 4.40e-5，采用 `atol=0.003, rtol=0.0003`。原始记录：[0.3 ONNX 验证](artifacts/validation/0.3.0/onnx-final.json)。

0.3.0 主库的最终 bus 图像对照中，5 个 Vulkan 模型共 24 个实例通过独立 PyTorch 前向与后处理检查，结果如下；两种分割模型的 11 张二值 mask IoU 均为 1.0。该验证共享 C# letterbox 后的输入，覆盖框、NMS、mask 缩放和关键点，不声称与所有 Ultralytics 版本的图片预处理逐位相同。原始记录：[0.3 图像验证](artifacts/validation/0.3.0/image-final.json)。

| 模型 | 实例数 | 最大框坐标误差 px | 最大关键点误差 px | 二值 mask IoU |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 5 | 0.000125 | — | — |
| yolov8n-seg | 6 | 0.000124 | — | 1.0 |
| yolov8n-pose | 4 | 0.000158 | 0.000368 | — |
| yolo11n-seg | 5 | 0.000103 | — | 1.0 |
| yolo11n-pose | 4 | 0.000071 | 0.000415 | — |

```powershell
dotnet run --project tests/SimdYolo.Tests -c Release -- --vulkan --models .cache
```

## 0.3.0 性能记录（历史）

最终同机比较使用 8 个模型 × 8 个后端：本库 `.pt` / ONNX 的 CPU、Vulkan，原版 PyTorch CPU/CUDA 和 ORT CPU/CUDA。统一 FP32、batch=1、640×640、线程上限 12，预热 30 次后连续推理至少 8 秒、至少 40 次；每个配置独立进程、串行执行。计时包含网络前向、输出头解码，以及 GPU 输入上传和全部输出回读；排除加载、首次形状特化/内核选择、图片预处理、NMS 和最终 mask 后处理。

PyTorch 使用原版 Ultralytics 8.3.0 的 `model.fuse()` 和 `inference_mode()`，没有 `torch.compile`，版本为 CPU 2.11.0+cpu / CUDA 2.11.0+cu126。ORT 1.26.0 使用全部图优化、顺序执行和默认线程自旋，CUDA 卷积自动选择算法。CUDA 对照关闭 TF32。ORT 的全部输出复制为托管 `float[]`，与本库公开输出的主机可见口径一致。

新增优化的开发 A/B 分别运行 16 个独立窗口，并交换启用/禁用顺序。CPU 的 v8n/11n `.pt` 与 ONNX 中位耗时下降 4.7–7.3%；Vulkan 的 v8n-seg 两轮下降约 1.9%，v8n-pose 约 2.1%。这些是 4 秒窗口的局部优化验证，不是完整 8 模型/8 后端矩阵或 GPU 全面持平的证据。原始记录：[CPU 融合 A/B](artifacts/validation/0.3.0/cpu-fusion-ab.json)、[Vulkan 输入复用 A/B](artifacts/validation/0.3.0/vulkan-cache-ab.json)。

表中稳态速度排除 Vulkan 冷首帧的形状特化、shader pipeline 创建和内核选择。完整报告的 `FirstInferenceMs` 显示，本机新会话的 `.pt` Vulkan 首次推理为 12.907–21.161 秒，ONNX Vulkan 为 12.939–20.842 秒；CPU 两种格式为 0.238–0.323 秒。这些首帧数字不含模型构造/加载；`StartupMs` 另含加载和全部预热。不能把预热后的每帧耗时当作启动耗时，视频应用应复用会话并提前预热。

0.3.0 已验证源码的正式矩阵于 2026-10-14 完成 64/64 个独立窗口，共 23,793 次推理。以下是每个窗口的中位耗时，单位 ms：

每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗。有重复窗口时取各窗口中位耗时的中位数，按未四舍五入的数字排序。

名次只表示本轮数值顺序；小幅差距不代表稳定胜出，也不是跨窗口置信区间或统计显著性。显示值相同可能仍有名次差异；原始中位数完全相同则并列。缺项显示 N/A；四配置及全部重复窗口齐全后才排名和统计第一。

### CPU 速度排名

| 模型 | 原版 PyTorch CPU | 我们的 `.pt` CPU | 我们的 ONNX CPU | ORT CPU（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 47.50 ④ | 40.17 ② | 40.43 ③ | **25.76 ①** |
| yolov9t | 81.74 ④ | 53.44 ② | 59.71 ③ | **30.65 ①** |
| yolov10n | 62.53 ④ | 42.41 ② | 46.61 ③ | **23.84 ①** |
| yolo11n | 52.85 ④ | 41.22 ② | 42.17 ③ | **24.31 ①** |
| yolov8n-seg | 59.46 ④ | 51.64 ② | 52.40 ③ | **36.25 ①** |
| yolov8n-pose | 50.86 ④ | 45.25 ③ | 43.14 ② | **26.54 ①** |
| yolo11n-seg | 65.64 ④ | 53.29 ③ | 53.23 ② | **34.47 ①** |
| yolo11n-pose | 55.22 ④ | 47.49 ③ | 44.40 ② | **25.48 ①** |
| 第一次数（完整四配置 8/8 模型） | 0 | 0 | 0 | 8 |

### GPU 速度排名

| 模型 | 原版 PyTorch CUDA | 我们的 `.pt` Vulkan | 我们的 ONNX Vulkan | ORT CUDA（.NET） |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | **9.96 ①** | 12.90 ③ | 13.30 ④ | 12.58 ② |
| yolov9t | 21.97 ④ | **16.33 ①** | 18.23 ③ | 17.59 ② |
| yolov10n | **12.85 ①** | 14.12 ③ | 16.45 ④ | 13.30 ② |
| yolo11n | **11.01 ①** | 13.65 ③ | 14.07 ④ | 13.58 ② |
| yolov8n-seg | **13.04 ①** | 16.84 ③ | 17.26 ④ | 16.48 ② |
| yolov8n-pose | **10.33 ①** | 13.15 ③ | 13.63 ④ | 12.94 ② |
| yolo11n-seg | **13.80 ①** | 17.48 ② | 17.88 ④ | 17.49 ③ |
| yolo11n-pose | **10.96 ①** | 14.09 ③ | 14.59 ④ | 14.06 ② |
| 第一次数（完整四配置 8/8 模型） | 7 | 1 | 0 | 0 |

精确并列第一时，各实现都计一次第一，因此第一次数之和可能大于完整模型数。


本轮 8 个模型的两种 CPU 路径均快于原版 PyTorch CPU：`.pt` 中位耗时低 11.0–34.6%，ONNX 低 11.9–26.9%。两种 CPU 路径仍慢于 ORT CPU。Vulkan 的 v9t `.pt` / ONNX 中位耗时分别比 PyTorch CUDA 低 25.7% / 17.0%；其余 7 个模型的 `.pt` 仍慢 9.9–29.5%，ONNX 仍慢 27.8–33.6%，未达到全面 CUDA 持平。Vulkan `.pt` 的 11n-seg/11n-pose 与本轮 ORT CUDA 耗时接近，不外推为整体 GPU 持平。

Vulkan `.pt` 的平均进程 RSS 比 PyTorch CUDA 低 54.0–72.1%，每帧进程 CPU 时间低 79.8–89.4%；ONNX 对应低 57.7–72.6% 和 68.2–86.8%。相对 ORT CUDA，`.pt` 的 RSS / 每帧 CPU 时间低 40.5–62.3% / 80.3–90.0%，ONNX 低 45.2–62.4% / 69.0–89.2%。

RSS 包含宿主、模型和缓存，不能解释为显存或功耗节省。CPU% 按全部 12 个逻辑核心归一化；GPU/显存来自 NVML 整卡采样，含桌面及其他程序活动，不代表单进程显存或计算单元利用率。单次独立窗口的结果不构成跨机器或跨重复窗口的置信区间。

80 个输出张量记录均通过形状、有限值及 `atol=0.003, rtol=0.0003` 校验，其中 70 个为跨实现对照。最大绝对坐标误差约 0.0082 px，分割原型最大绝对误差约 4.02e-5；v10 TopK 输出先按行语义排序再比较，不要求不同实现的并列行顺序一致。

正式来源：[完整报告](artifacts/baselines/0.3.0/optimized-runtime-comparison.md)、[原始 JSON](artifacts/baselines/0.3.0/optimized-runtime-comparison.json)、[对比图](artifacts/baselines/0.3.0/optimized-runtime-comparison.png)、[SVG](artifacts/baselines/0.3.0/optimized-runtime-comparison.svg)。JSON 保存延迟样本、P95、资源时间序列、实际运行版本和模型/输入/主库/ORT DLL 哈希。本轮 `SimdYolo.dll` 的 SHA256 为 `399d77f7b8b77f7bd7dfc4ed835d11e022a367646d17c56d06c6b43693e3854f`，与报告的 `SimdYoloDllSha256` 一致。`artifacts/optimization-cpu-comparison.*` 是本轮开发中途的 CPU 快照，不含随后完成的全部优化，不作为最终 0.3 结果。下面 0.2/0.1 表格均为保留的历史基线。

```powershell
dotnet build -c Release
dotnet build tools/OrtBenchmark/OrtBenchmark.csproj -c Release
dotnet src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll preprocess .cache/image-validation/image.ppm .cache/speed-comparison/bus-input.f32 640
python tools/compare-runtimes.py --vision --threads 12 --warmup 30 --iterations 40 --duration 8 --engines torch-cpu net-cpu net-onnx-cpu ort-net-cpu net-vulkan net-onnx-vulkan torch-cuda ort-net-cuda --input .cache/speed-comparison/bus-input.f32 --output artifacts/runtime-comparison-reproduction.json
```

## 既有正确性对照记录

以下张量与真实图像对照在前版完成，保留用于追溯已覆盖的模型/任务；不冒充 0.3 最终版本的完整重新验证。

原版 PyTorch 输出对照共完成 120 组张量比较：

| 测试 | 模型/任务 | 输入 | 比较数量 |
| --- | --- | --- | --- |
| 静态模型 | v8n、v9t、v10n、11n 检测；v8n、11n 分割和姿态 | 128×128 | 40 |
| 静态模型 | 同上 | 640×640 | 40 |
| 动态 ONNX | v8n、11n 检测；v8n 分割、11n 姿态 | 128 导出、640 执行 | 20 |
| opset 12 兼容 | v8n、11n 检测；v8n 分割、11n 姿态 | 128×128 | 20 |

每组包含直接读取 `.pt` 或 `.onnx` 的 CPU / Vulkan 结果；分割同时比较输出头和 mask 原型。比较阈值为 `atol=0.003, rtol=0.0003`，不是逐位相等。640 测试的最大绝对误差约 0.0032，出现在以像素为单位的输出坐标；mask 原型误差约为 1e-5。

```powershell
python tools/validate-reference.py .cache/yolov8n.pt .cache/yolo11n.pt .cache/yolov9t.pt .cache/yolov10n.pt .cache/yolov8n-seg.pt .cache/yolov8n-pose.pt .cache/yolo11n-seg.pt .cache/yolo11n-pose.pt --size 640 --onnx
python tools/validate-reference.py .cache/yolov8n.pt .cache/yolo11n.pt .cache/yolov8n-seg.pt .cache/yolo11n-pose.pt --size 640 --export-size 128 --dynamic --onnx
```

官方 bus 示例图的 Vulkan 预测也通过了独立 PyTorch 前向、NMS、双线性 mask 缩放和关键点坐标对照：

| 模型 | 实例数 | 最大框坐标误差 | 最大关键点误差 | 二值 mask IoU |
| --- | --- | --- | --- | --- |
| yolov8n | 5 | 0.00013 px | — | — |
| yolov8n-seg | 6 | 0.00014 px | — | 1.0 |
| yolov8n-pose | 4 | 0.00015 px | 0.00018 px | — |
| yolo11n-seg | 5 | 0.00016 px | — | 1.0 |
| yolo11n-pose | 4 | 0.00012 px | 0.00019 px | — |

图像对照共享 C# letterbox 后的输入，独立验证模型执行和后处理。mask 对照使用实际整数 padding，避免旧版 `scale_masks` 将 `19.99999` 截断为 19 导致的位移。它不声称与所有 Ultralytics 版本的图片预处理逐位一致。

原始比较记录保存在 `artifacts/validation/`；开发缓存内有输入、ONNX 测试夹具和真实图像的预测 JSON/PGM。

## 0.2.0 与 PyTorch / ONNX Runtime 的同机实测（历史）

本节对应优化前的 0.2.0 二进制和报告，不能用于描述 0.3.0 当前速度或资源开销。

2026-10-13 使用 bus 图像预处理后的同一张量，640×640、batch=1、FP32，线程上限 12；每个配置单独进程，预热 15 次，随后连续推理至少 8 秒、至少 30 次。8 个模型 × 6 个后端共 48 个窗口、16,523 次推理。计时包含输出头解码和 GPU 输入上传/输出回读，排除加载、首帧编译、图片预处理、NMS 和 mask 后处理。

PyTorch 使用原版 Ultralytics 8.3.0 的 `fuse()`，CPU 2.11.0+cpu、CUDA 2.11.0+cu126。ORT 1.26.0 使用相同 .NET 宿主，开启全部图优化，输出复制为托管数组。GTX 1050 Ti 上 NuGet GPU 原生构建的 QuickGelu 出现内核兼容错误；正式对照使用官方 GPU wheel 中同版本、已执行校验的原生 DLL，通过 .NET Managed 绑定调用。未调用 Python 执行 ORT，未修改主库依赖。

以下单位为中位耗时 ms；本库列直接读取 `.pt`，ORT 读取同源的静态 ONNX。

| 模型 | 本库 CPU | PyTorch CPU | ORT CPU | 本库 Vulkan | PyTorch CUDA | ORT CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8n | 114.98 | 45.31 | 24.55 | 17.99 | 9.95 | 12.49 |
| yolov9t | 187.59 | 78.77 | 28.83 | 21.44 | 22.20 | 17.58 |
| yolov10n | 146.32 | 60.47 | 22.91 | 17.15 | 12.95 | 13.34 |
| yolo11n | 136.60 | 53.38 | 24.13 | 16.47 | 10.57 | 13.57 |
| yolov8n-seg | 147.47 | 58.81 | 34.37 | 26.41 | 13.03 | 16.52 |
| yolov8n-pose | 130.77 | 48.86 | 25.77 | 18.97 | 10.27 | 12.88 |
| yolo11n-seg | 167.02 | 63.80 | 33.40 | 24.90 | 13.62 | 17.48 |
| yolo11n-pose | 150.50 | 62.14 | 25.89 | 18.49 | 11.13 | 14.10 |

0.2.0 CPU 没有速度或进程内存优势：耗时是 PyTorch 的 2.38–2.68 倍、ORT 的 4.29–6.51 倍。该版 Vulkan `.pt` 的优势是进程内存和每帧 CPU 成本：RSS 比 PyTorch CUDA 低 72.9–77.8%、比 ORT CUDA 低 63.9–73.8%；每帧进程 CPU 时间分别低约 78.8–90.5% 和 80.8–89.6%。该版 Vulkan 在多数模型上仍慢于 CUDA。这些结论不适用于当前 0.3.0，也不能外推为 ONNX 路径、显存或功耗优势。

v9t 再次启动独立进程测了两个窗口：

| 窗口 | 本库 Vulkan `.pt` | PyTorch CUDA | ORT CUDA |
| --- | ---: | ---: | ---: |
| 复测 1 | 21.49 | 22.05 | 17.58 |
| 复测 2 | 21.63 | 22.28 | 17.50 |

本次复测比 PyTorch 快约 2.5–2.9%，仍比 ORT 慢约 22–24%。这是这台机器上该 tiny 检查点的小幅差距，不是所有 v9 模型或显卡的结论。

本库 ONNX 路径补测保持相同输入/精度/线程/预热/时长：

| 模型 | 本库 ONNX CPU ms | 本库 ONNX Vulkan ms |
| --- | ---: | ---: |
| yolov8n | 201.58 | 83.90 |
| yolov9t | 290.16 | 102.02 |
| yolov10n | 244.98 | 107.36 |
| yolo11n | 209.94 | 97.40 |
| yolov8n-seg | 235.24 | 96.05 |
| yolov8n-pose | 240.09 | 101.50 |
| yolo11n-seg | 256.09 | 94.03 |
| yolo11n-pose | 256.90 | 94.71 |

0.2.0 ONNX Vulkan 耗时约为对应 ORT CUDA 的 5.38–8.05 倍，GPU 整卡忙碌率约 25–36%，显著慢于该版 `.pt` 路径。当时的主要优化方向是 ONNX CPU 回退与同步、CPU 并行/缓存和 Vulkan 内核；0.3.0 已对此进行了更新，旧表不再代表当前执行路径。

CPU% 以全部 12 个逻辑核心为分母；内存为平均进程工作集 RSS，并记录物理内存占比、峰值、私有提交量。GPU/显存是 NVML 整卡采样，含桌面及其他程序；启动前快照并非精确进程显存基线，显存增量只供参考。GPU 忙碌率不是计算单元利用率。所有输出按 `atol=0.003, rtol=0.0003` 校验：60 个输出张量记录，其中 50 个跨实现比较通过；最大绝对坐标误差约 0.0082 px。

主矩阵加 ONNX 补测和 v9 重复窗口共完成 80 个窗口、21,089 次推理；98 个输出张量记录，其中 76 个跨实现对照全部通过。完整资源表、P95、每帧 CPU 时间与比例：[三方报告](artifacts/runtime-comparison.md)；原始延迟、资源时间序列、模型/输入/DLL 哈希：[JSON](artifacts/runtime-comparison.json)；[对比图](artifacts/runtime-comparison.png)；[ONNX 补测](artifacts/onnx-path-comparison.md)；[v9 复测](artifacts/v9-repeat-comparison.md)。

```powershell
dotnet build -c Release
dotnet build tools/OrtBenchmark/OrtBenchmark.csproj -c Release
python tools/compare-runtimes.py --vision --threads 12 --warmup 15 --duration 8 --input .cache/speed-comparison/bus-input.f32
```

## 0.1.0 性能基线（优化前，已过时）

以下是 `PredictRaw` 的 640×640、batch=1、FP32 测量，预热一次后执行五次，取中位数。包括当时 GPU 算子的传输和同步，不包括模型加载、图像解码、letterbox、NMS、mask 后处理和首次形状特化编译。机器负载和 JIT 会影响结果。

| 模型 | CPU SIMD | Vulkan |
| --- | ---: | ---: |
| YOLOv8n | 496.8 ms | 262.5 ms |
| YOLOv9t | 682.1 ms | 408.7 ms |
| YOLOv10n | 417.9 ms | 342.8 ms |
| YOLO11n | 422.5 ms | 318.9 ms |

这些历史数据只描述 0.1.0 优化前的实现；0.2.0 和 0.3.0 已采用设备张量缓存、批量提交和融合头解码等更新。

```powershell
.\tools\benchmark.ps1 -ModelDirectory .cache -Iterations 5
```

Metal 当前没有硬件验证结果，Linux Vulkan、其他 GPU、其他模型大小和 WongKinYiu 原始 YOLOv9 检查点也没有加入上述实测范围。macOS 可运行 README 中的 Metal 验证命令。
