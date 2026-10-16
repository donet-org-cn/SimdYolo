# 验证记录

## 0.11.0 源码：验证、导出、标注与低精度

四项核心均为纯.NET实现。检测指标包含101点AP、独立拆分与内容隔离、最佳模型、早停及恢复状态；导出覆盖五代原模型和自定义类别训练模型，另测v8/26分割与姿态。本库回读与独立ONNX checker/ORT数值对照均记录原始结果。网页真实鼠标编辑、预标注审核、保存冲突、数据导出、训练验证、模型比较、ONNX导出、校准及低精度选择已经验证。[证据、精度/速度与复现](artifacts/validation/0.11.0/README.md)。

FP16为Half操作数/FP32累加；INT8包含实际AVX2整数点积及防溢出分段，不使用ORT。当前CPU低精度没有速度优势，GPU仍为FP32，不改写历史640尺寸排名。本轮COCO128的32张留出图用于实现诊断，预训练模型已见过该数据来源，不能作为未见数据的泛化结论。

## 0.10.1 训练速度、结果与发布

[中文发布说明](docs/releases/v0.10.1.md)列出0.10.1主库/训练库NuGet、33模型、网页及图文包。打包、独立NuGet消费和解压网页验证见[发布验证](artifacts/validation/0.10.1-release/README.md)。主库与训练库保留本轮计时和数值验证使用的DLL字节，历史推理排名不冒充本版重测。

SIMD激活/BN/SGD、直接转置点积反向、1×1零复制、卷积临时矩阵复用，以及Vulkan小矩阵CPU分派。保持纯C#、FP32和原有单步数值容差。[训练速度、保存模型精度及误差报告](artifacts/training-optimization-0.10.1.md)保留与原版的独立进程计时和资源采样。

真实COCO128图片按固定种子拆成96张微调/32张验证，五模型各CPU/Vulkan与PyTorch CPU/CUDA执行192步相同SGD；记录第1/16/96/192步参数、BN、动量与逐步损失。重新加载两边实际保存的模型执行完整网络，按相同官方指标比较P/R、mAP50和mAP50–95。数据属于预训练所用COCO train2017，结果用于实现一致性诊断，不作为泛化或训练配方有效性结论。[原始验证与复现](artifacts/validation/0.10.1/README.md)。

## 0.10.0 五代检测头训练

新增 YOLOv9t、YOLOv10n、YOLO11n、YOLO26n 检测头微调，与 v8n 一起覆盖 CPU/Vulkan。包含深度/分组卷积反向、完整双头、YOLO26 直接距离/L1、二次 TAL 匹配和可恢复的渐进损失调度。主库与训练库继续为纯托管 C#。

固定官方 Ultralytics 8.4.0 / PyTorch 2.11.0 CPU 参考，34 组、19,620 项对照覆盖前向、loss、logits/参数梯度、连接状态、BN、SGD 后权重与动量；性能所用128尺寸另有五组、2,712项对照。五模型各 CPU/Vulkan 真实100轮微调，并检查所有分支更新、取消恢复、动量载入、原生模型元数据与推理。训练回归16项、原推理回归117项及中文XML/纯包消费验证的原始证据见[本版验证目录](artifacts/validation/0.10.0/README.md)。

[训练评测与图表](artifacts/training-evaluation-0.10.0.md)包含40个串行独立窗口、20,480次更新，以及CPU、RSS、整卡GPU/显存。比较固定冻结特征后的检测头前向、损失、反向、SGD，输入128×128、2类、4线程；不包含骨干、预处理、保存，不是全参数训练或 mAP 结论。历史推理速度仍绑定0.9.0及更早冻结DLL，没有用训练数字改写推理排名。

## 0.9.0 历史推理性能与正式发布

0.9.0程序集以与0.8.1相同的固定实拍图张量重新比较11个模型、8个CPU/GPU配置、两轮共176个独立窗口；统一FP32、batch=1、640×640、12线程，预热60次，每窗口至少4秒/40帧。完成39,134次推理，224项输出校验通过。

当前[完整结果](artifacts/runtime-comparison-0.9.0.md)、[原始JSON](artifacts/runtime-comparison-0.9.0.json)、[CPU图](artifacts/performance-0.9.0-cpu.png)、[GPU图](artifacts/performance-0.9.0-gpu.png)包含加载、首帧、稳态和资源口径。额外随机输入下ORT CUDA的YOLO26分割校验超差单列保留，未混入正式矩阵或放宽容差。当前图文、中文Release修复、发布包摘要与复现输入见[0.9.0发布验证](artifacts/validation/0.9.0-release/README.md)。

## 0.9.0 训练与网页示例

训练模块通过 12 项单元/集成测试，包含真实微调、SGD 断点一致性、取消、重入拒绝、快照校验和 Vulkan 训练。CPU/Vulkan 既有推理回归通过 117 项。固定 Ultralytics 8.4.0 / PyTorch 2.11.0 CPU 独立参考，CPU、Vulkan、小目标和空标签四组共 640/640 检查通过；网页 API 和真实无头浏览器流程通过。

范围为冻结骨干/FPN 的 YOLOv8 检测头微调，没有全参数、分割/姿态、YOLO26 或 Metal 训练。单图损失下降是流程验证，不代表 mAP 或业务准确率；未测训练性能。下列历史推理性能数据保持原版本。复现命令、原始结果、截图及程序集绑定见 [0.9.0 验证目录](artifacts/validation/0.9.0/README.md)。

## 0.8.1 加载、首帧与推理优化

主库保持纯托管 C#、FP32 计算和仅 .NET 框架程序集依赖。优化包括连续 PT storage 的精确批量转换、ONNX protobuf 视图、活动分支的 BN 预融合、复用路由工作区、完整写入结果的未清零缓冲区、Winograd 权重批次缓存、单桶检测坐标 SIMD，以及注意力批次 GEMM 直接读写输入/输出偏移。YOLO26 ONNX 检测、分割、姿态输出头严格匹配后采用设备两阶段 TopK，保留全部实例附加字段并支持命令重放。

最终候选主库 SHA256：`74a09c65fd9b535746ca1e6fc56e62cedac6ef0c27d3f38421f7c98ab221ab91`；冻结 0.8.0 基线 SHA256：`f848caca154370e664c18d08d76f554bb852dd35c113bdf548bac958c9cc750a`。

- Release 构建零警告、零错误；[CPU/Vulkan 回归](artifacts/validation/0.8.1/core-vulkan.txt)共 117 项通过，含 4 个实际 PT 模型的持久权重与输入尺寸变化。
- [关闭 AVX/AVX2](artifacts/validation/0.8.1/core-portable.txt)与[关闭全部硬件 intrinsics](artifacts/validation/0.8.1/core-software.txt)各 94 项通过；受硬件 guard 限制的直接指令专项会跳过，不等同于已在 ARM 机器上实测。
- [640 静态对照](artifacts/validation/0.8.1/onnx-oracle640.json)70 项输出通过，包含 11 模型、CPU/Vulkan 的 PT 与 ONNX，以及 ORT CPU。独立的 PT JSON 是此报告的 28 项子集，不额外计数。
- [动态 batch=2 / 192 输入](artifacts/validation/0.8.1/dynamic-oracle192.json)45 项输出通过，覆盖 7 模型；ONNX 在 640 尺寸导出后以动态尺寸执行。
- 6 模型、PT/ONNX、CPU/Vulkan 的真实图片对照共 24 配置、108 实例通过：检测框、类别、分割掩码和关键点分别核对，结果在 `artifacts/validation/0.8.1/images-*.json`。
- [33 模型标签映射](artifacts/validation/0.8.1/model-labels.json)全部通过；[YOLO26 重放](artifacts/validation/0.8.1/yolo26-replay.json)6 配置在 5 次变更输入中各重放 3 帧，与关闭重放输出的最大绝对误差为 0，历史公开输出未被覆盖。
- [权重读取核验](artifacts/validation/0.8.1/reader-storage-verification.json)记录 11 PT、22 ONNX 的全部权重与冻结 0.8.0 的 FP32 位表示一致。另以全部 65,536 种 FP16 编码核验转换，并覆盖非连续 stride、storage offset、各种 dtype 和 protobuf 属性独立所有权。
- [生命周期首帧/末帧独立验证](artifacts/validation/0.8.1/lifecycle-outputs.json)确认 **352/352 窗口、896/896 输出校验通过**，同时检查输出文件形状、字节数与 SHA256。端到端检测、分割、姿态既检查 TopK 分数，又将整行与完整解码候选作唯一 anchor/class 配对；附加字段与分割原型均没有省略。

新的批次矩阵乘法回归覆盖独立标量结果、广播、矩阵尾部、K=257、输入别名和更新、正偏移保护区、越界拒绝、零 K、空批次、公开输出所有权和自定义后端约定。检测坐标专项对单桶及特殊浮点值要求位一致；多桶均匀分布允许现有指数归约的 FP32 舍入误差。未放宽模型输出对照的容差。

### 冻结 0.8.0 与最终 0.8.1 的生命周期对照

[优化报告](artifacts/optimization-0.8.1.md)及[完整生命周期 JSON](artifacts/lifecycle-comparison-0.8.1.json)完成 352 个独立窗口：11 模型 × PT/ONNX × CPU/Vulkan × 2 个 ABBA 区组，每个区组按旧版、新版、新版、旧版执行，各版本每配置 4 窗口。使用同一输入、同一生命周期宿主与冻结程序集，统一 FP32 / batch=1 / 640×640 / 12 线程；预热 60 帧，每窗口至少 4 秒、40 帧。下表为各窗口指标的中位数，旧版→新版，时间单位 ms、托管分配单位 MiB/帧。

| 模型 / 后端 | 模型加载 ms | 首次前向 ms | 程序内至首输出 ms | 稳态 ms | 托管分配 MiB/帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n.pt / CPU | 181.20→92.91 | 265.38→265.64 | 456.68→368.10 | 27.14→26.18 | 2.934→2.915 |
| yolo26n.pt / CPU | 172.62→101.87 | 261.14→258.90 | 442.81→370.86 | 29.47→28.00 | 6.416→0.337 |
| yolo26n-seg.onnx / CPU | 116.56→172.34 | 338.89→318.40 | 469.70→559.96 | 41.13→36.73 | 10.128→3.731 |
| yolo26n-seg.onnx / Vulkan | 531.59→505.22 | 397.47→370.95 | 975.31→890.57 | 17.56→14.52 | 3.906→3.169 |

PT 加载减少不代表首次前向全部改善：v8 PT 首次前向基本不变；26 分割 ONNX CPU 的加载和程序内至首输出的汇总中位数增加，稳态和分配量则下降。该配置旧版加载范围为 115.08–136.22 ms，新版为 102.46–267.00 ms；旧版至首输出为 468.39–490.05 ms，新版为 430.06–607.02 ms。慢窗口影响中位数，不是每窗都回退，也不能隐去其中 3/4 新版首输出窗口偏慢的结果；原因未确认。具体反例及全部 44 个模型/格式/后端组合保留在优化报告，不用代表行推断全面领先。

`ModelLoadMs` 包含会话构造、后端创建、模型解析和权重准备。`FirstInferenceMs` 只包围首个 `PredictRaw`；`ReadyToFirstOutputMs` 从首条托管入口语句至首次输出，包含输入读取但不包含 .NET 宿主启动及入口 JIT。`ProcessTotalMs` 包含整个生命周期进程、预热、稳态、输出写盘和释放，不能当首次输出延迟。托管分配从稳态窗口累计分配中扣除采样列表扩容，既不等于 RSS，也不等于显存。新进程没有清空系统或 GPU 驱动缓存；原始输出不随仓库上传，形状、字节数、哈希和复现工具保留在验证 JSON。

### 与原版 PyTorch、ONNX Runtime 的同机对照

[CPU/GPU 对照与排名](artifacts/runtime-comparison-0.8.1.md)及[原始 JSON](artifacts/runtime-comparison-0.8.1.json)完成 **176/176 窗口、36,460 次计时推理、224/224 输出校验通过**。11 模型、8 引擎、各 2 个独立进程窗口；60 帧预热后每窗口至少 4 秒、40 帧，包含前向与输出头解码，GPU 包含上传和全部输出回读。CPU 使用 CPU-only PyTorch；完整候选 oracle 在计时后生成，不计入性能或资源采样。

CPU ORT 存在明显两轮波动：v8 为 50.94/25.47 ms，11n 为 24.38/47.81 ms；26n、26n-seg、26n-pose 的 CPU 汇总仍是 ORT 最快。两窗口数值不能证明稳定全胜或统计显著性，结果也不能外推到其他 CPU、批量、torch.compile、FP16/INT8 或 TensorRT。CUDA 对照表示 CUDA EP 会话，不表示每个形状/索引节点都在 GPU；GPU 利用率及显存为整卡 NVML 采样。Metal 没有本机硬件实测。

跨引擎报告的 `ProcessStartupMs` 从外部启动进程到观测全部预热完成的 ready，含宿主、导入和预热及约 50 ms 轮询观测误差，不能当首次输出时间。其他启动拆分的边界在 JSON 中逐项定义。0.8.1 NuGet 包与最终 DLL 的一致性、依赖、模型许可证和中文 XML 检查见[本版包审查](artifacts/validation/0.8.1/package-audit.json)。[独立 NuGet 消费记录](artifacts/validation/0.8.1/package-consumer.json)记录只引用本地包的外部程序所加载程序集的哈希、33 份标签映射和检测/分割/姿态 PT 与动态 ONNX 在 CPU/Vulkan 的加载及输出所有权检查；它是交付烟雾验证，不重复计为原版数值对照。原始 JSON、工具、源码及交付文件的指纹汇总见[交付清单](artifacts/validation/0.8.1/delivery-manifest.json)。

## 0.8.0 YOLO26 与标签元数据

新增 YOLO26 检测、分割、姿态 `.pt`/ONNX 推理及独立标签读取接口。CPU/Vulkan、静态与动态、批量和真实图片的具体结果见 [YOLO26 验证](artifacts/validation/0.8.0/yolo26.md)。历史性能报告继续绑定 0.7.0 冻结程序集，本次兼容性验证没有生成或替换速度排名。

## 0.7.1 中文代码文档

0.7.1 补齐中文 XML 文档和关键算法说明，启用 `GenerateDocumentationFile`，将 `SimdYolo.xml` 随 NuGet 包分发。代码 token 对照、文档覆盖、编译与 CPU/Vulkan 回归结果见 [中文注释验证](artifacts/validation/0.7.1/chinese-documentation.md)。性能报告继续保留 0.7.0 冻结 DLL 的原始测量与哈希；版本号和中文注释使新程序集的文件哈希发生变化。

## 0.7.0 历史 CPU/GPU 比较

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
