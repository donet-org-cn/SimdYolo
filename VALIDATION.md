# 验证记录

## 0.8.0 YOLO26 与标签元数据

新增 YOLO26 检测、分割、姿态 `.pt`/ONNX 推理及独立标签读取接口。CPU/Vulkan、静态与动态、批量和真实图片的具体结果见 [YOLO26 验证](artifacts/validation/0.8.0/yolo26.md)。历史性能报告继续绑定 0.7.0 冻结程序集，本次兼容性验证没有生成或替换速度排名。

## 0.7.1 中文代码文档

0.7.1 补齐中文 XML 文档和关键算法说明，启用 `GenerateDocumentationFile`，将 `SimdYolo.xml` 随 NuGet 包分发。代码 token 对照、文档覆盖、编译与 CPU/Vulkan 回归结果见 [中文注释验证](artifacts/validation/0.7.1/chinese-documentation.md)。性能报告继续保留 0.7.0 冻结 DLL 的原始测量与哈希；版本号和中文注释使新程序集的文件哈希发生变化。

## 0.7.0 当前 CPU/GPU 比较

2026-10-14 14:31:43 +0800 同机正式比较完成 CPU/GPU 共 128/128 个独立窗口、57,628 次计时推理，160/160 个输出校验通过（20 个 PyTorch CPU 自对照、140 个跨实现对照）。硬件：Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz、NVIDIA GeForce GTX 1050 Ti。统一 FP32、batch=1、640×640、CPU 线程上限 12；各配置独立进程、串行运行两轮，预热 30 次后每窗口至少 8 秒、40 次。计时包含网络前向和输出头解码，GPU 包含上传及全部输出回读；排除图片预处理、NMS、最终 mask 后处理和冷首帧。

0.7.0 继续使用纯 C#、FP32 CPU 计算，新增实际地址对应的缓存行对齐、16/48 列点卷积调度、F4 卷积边界 SIMD、窄平面 depthwise padding SIMD、卷积尾部四/八元素 SIMD 激活、8×8 SIMD 转置、严格匹配的 ONNX 检测头融合及 2×2 平均池化专用循环。保持既有专属 CPU 执行器和 SIMD 指数路径。32 列内核、F6 和分块大小调参没有稳定收益，未设为发布路径。GPU 实现和着色器保持不变，本次重新测量 Vulkan、原版 PyTorch CUDA 和 ORT CUDA；Metal 尚无 macOS 硬件验证。

主 DLL SHA256：`51d21ba2bca561f3a21efdc3ab11be08814ae32c6510cdc910bd0338d0aaccfe`。

Release 构建零警告、零错误；Vulkan/CPU 统一回归 86 项通过；关闭硬件 intrinsics 的可移植回归 70 项通过（直接 SIMD 专项受硬件 guard 限制，部分跳过）。冻结 DLL 的 CPU `.pt`/ONNX 20 个输出、Vulkan 改输入和重放 80 个输出、静态/动态 ONNX CPU/Vulkan 30 个输出全部通过。真实图像 5 个配置、24 个实例通过，11 个 mask 最低 IoU=1.000000。

[构建](artifacts/validation/0.7.0/build.log)、[统一回归](artifacts/validation/0.7.0/test-vulkan.log)、[可移植回归](artifacts/validation/0.7.0/test-portable.log)、[CPU oracle](artifacts/validation/0.7.0/cpu-packed-oracle.json)、[GPU 重放](artifacts/validation/0.7.0/real-model-replay.json)、[ONNX oracle](artifacts/validation/0.7.0/onnx-oracle.json)、[图像 oracle](artifacts/validation/0.7.0/image-oracle.json)。

## CPU 速度排名

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

## GPU 速度排名

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

## 资源与口径

以下为逐模型比值的范围，不是把不同模型成本合并平均；大于 1 表示本库成本更高。CPU%按全机逻辑核心归一化，CPU ms/帧由进程累计 CPU 时间计算，RSS 为整个宿主进程的采样工作集。

| 路径 | CPU ms/帧 / PyTorch | CPU ms/帧 / ORT | RSS / PyTorch | RSS / ORT |
| --- | ---: | ---: | ---: | ---: |
| `.pt` CPU | 0.33–0.45× | 0.73–0.96× | 1.24–1.45× | 2.04–3.60× |
| ONNX CPU | 0.32–0.45× | 0.73–0.98× | 0.75–1.24× | 1.51–3.09× |

较高 CPU 占用不代表更高效率；较低 CPU 占用也不能单独证明效率优势。更快延迟可能伴随更多 CPU 时间，需要同时查看 CPU ms/帧和进程 RSS，不能据速度排名宣称资源成本全面领先。RSS 包含运行库、模型和缓存，不等于算子工作区大小。

GPU 使用率和显存是整卡采样，包含桌面及其他进程，不等同于单进程显存或功耗。CPU%按全机逻辑核心归一化，CPU ms/帧来自进程累计 CPU 时间；RSS 包含宿主、模型和缓存。速度名次与资源成本分别评估；两次窗口不足以证明小幅领先有跨环境统计显著性。PyTorch 使用原版 fuse()，ORT 开启全部图优化、默认自旋及自动 CUDA 算法；结果不能外推到 torch.compile、TensorRT、FP16/INT8 或 Metal。

[CPU 完整报告](artifacts/cpu-runtime-comparison.md)、[CPU/GPU 完整报告](artifacts/optimized-runtime-comparison.md)、[原始 JSON](artifacts/optimized-runtime-comparison.json)。每个重复窗口、资源采样、输入/模型/DLL 哈希和输出校验均保留。

## 与 0.6 的历史差值

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

## 模型与范围

八种 nano/tiny 配置的 `.pt` 完整模块检查点和对应 ONNX，覆盖检测、分割、姿态。其他尺寸、自定义模块、WongKinYiu 原仓库 YOLOv9 检查点、训练、TorchScript、独立 state_dict、OBB、分类、FP16/INT8 计算及 macOS Metal 硬件不在本次验证范围。主库只引用 .NET framework 程序集，GPU 使用系统 API 和内嵌着色器。

## 包和冻结版本

[0.7 NuGet 包](artifacts/SimdYolo.0.7.0.nupkg)、[包审查](artifacts/validation/0.7.0/package-audit.json)、[最终审查](artifacts/validation/0.7.0/final-audit.json)、[CPU 消费者](artifacts/validation/0.7.0/consumer-cpu.log)、[Vulkan 消费者](artifacts/validation/0.7.0/consumer-vulkan.log)。包内、CLI、正式基准与独立消费者绑定同一主 DLL；包哈希以包审查为准。文档重打包改变包哈希，保持主 DLL 不变。

## 历史记录

[0.6 CPU 报告](artifacts/baselines/0.6.0/cpu-runtime-comparison.md)、[0.6 验证](artifacts/baselines/0.6.0/VALIDATION.md)、[0.5 全后端报告](artifacts/baselines/0.5.0/optimized-runtime-comparison.md)。历史结果不代替本次测量。
