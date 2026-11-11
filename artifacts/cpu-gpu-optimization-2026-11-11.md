# CPU / GPU 优化实测：2026-11-11

本轮在 i7-12700 / RTX 3060 上，测得 Vulkan 稳态推理耗时降低 **4.55%～10.70%**。CPU 在显式 HighQoS 下的三个 PT 模型中位耗时基本持平，每帧托管分配减少 **3.07%～4.04%**；在本机默认继承的后台启动环境中，两个 CPU 模型耗时降低 **42.84%～45.81%**。后者仅代表该调度条件，不能当作通常 CPU 性能的提升幅度。

English: Vulkan steady inference latency fell by 4.55–10.70% in the measured cases. CPU PT throughput was essentially unchanged with explicit HighQoS, while managed allocations fell by 3–4%. The larger CPU latency reduction applies only to this machine's inherited background scheduling environment. All results retain FP32 computation.

## 改动

- CPU 执行器只等待已实际进入作业的线程。任务被领取完后关闭新参与者，仍然等待正在执行的回调及局部状态清理完成；延迟到达的线程安全退出。异常队列改为按需创建，减少正常推理的分配。
- Vulkan Winograd 权重增加连续四通道布局，用 `vec4` 读取和共享内存复用；分别使用适合大空间和小空间的乘法内核。保持递增输入通道的 FP32 FMA 顺序，不使用 FP16 或量化。非适用形状走原路径，可设置 `SIMDYOLO_WINOGRAD_PACKED=0` 关闭新路径。
- CPU SIMD 尾部、矩阵分块及路由实验没有取得一致的整模型收益，未纳入本轮改动。
- 基准工具增加 `--windows-high-qos`，仅用于测试子进程。主库继续继承应用调度策略，不修改系统电源计划或 CPU 亲和性。

## 基线与测量口径

基线是**本轮开始时的工作区构建**，已包含前两轮尚未发布的池化、检测头和批次写入优化，不是 NuGet 0.11.3 原包。程序集版本号仍为 0.11.3，必须用 SHA256 区分：

| 构建 | SimdYolo.dll SHA256 |
| --- | --- |
| 基线 | `3df596db47e11e9b7a5efe717eeed49a5b5b9037b86406a38761ee606f7a0d9a` |
| 本轮结果 | `3feac5f61ef17bf5b367b65aed258629a7a07e9d492dd9ebde145424766e5fb0` |

Windows 11，Intel Core i7-12700（20 逻辑处理器，AVX2/FMA），RTX 3060 12 GB，驱动 610.88，.NET 10。固定真实图片的 FP32 NCHW 输入，640×640、batch 1、12 CPU 线程。输入 SHA256 为 `4312bccf98b9fcd92bbb73c7530f757d62bebb46da5a585bf1da1d642d8040af`。

共 72 个独立进程窗口、23,642 次稳态推理，串行执行 ABBA 对照。先取各窗口单帧耗时中位数，再取窗口间中位数。PT 主测每版本每配置四个窗口；ONNX 初测和继承策略组各两个；ONNX CPU 复测各四个。通常预热 80 帧、测量至少 3 秒 / 40 帧，ONNX CPU 复测为 100 帧、4 秒 / 40 帧。预热数包含首帧。

计时范围为 `PredictRaw`，GPU 包含输入上传和所有输出读回；不包含图片缩放、NMS、最终掩码生成。继承应用 Vulkan 缓存，不清除驱动或文件缓存，不代表硬件冷启动。本轮不据此声称启动性能、所有模型或其他 GPU 同样提升。

## 显式 HighQoS：稳态耗时

两版本的测试进程使用相同 HighQoS 请求，以减少当前 Windows 启动环境的调度影响。该选项使用微软公开的 [SetProcessInformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation) 执行速度控制；Windows 的 [QoS 文档](https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service) 说明了调度和处理器选择的关系。

| 模型 | 后端 | 基线 ms | 本轮 ms | 耗时变化 |
| --- | --- | ---: | ---: | ---: |
| YOLOv8n PT | CPU | 23.036 | 23.164 | +0.56% |
| YOLO11n PT | CPU | 21.928 | 21.844 | -0.38% |
| YOLO26n-seg PT | CPU | 30.705 | 30.544 | -0.53% |
| YOLOv8n PT | Vulkan | 4.204 | 3.768 | -10.38% |
| YOLO11n PT | Vulkan | 4.270 | 4.076 | -4.55% |
| YOLO26n-seg PT | Vulkan | 6.157 | 5.846 | -5.05% |
| YOLOv8n ONNX | Vulkan | 4.126 | 3.684 | -10.70% |
| YOLOv8n ONNX，初测 | CPU | 21.076 | 22.225 | +5.45% |
| YOLOv8n ONNX，加长复测 | CPU | 21.251 | 20.919 | -1.56% |

ONNX CPU 两次结果方向不同，保留两组证据，不能宣称稳定提速。三个 PT 的 CPU 中位数接近，但 YOLOv8n 和 YOLO26n-seg 的 P95 分别增加 3.43% 和 5.90%，因此也不宣称 CPU 尾延迟全面改善。

CPU PT 每帧托管分配分别由 3,055,028 → 2,961,157、3,121,347 → 3,005,710、3,723,779 → 3,573,434 字节。分配减少不等于驻留内存减少。新 GPU 路径保留额外主机权重布局；三个 PT 的进程工作集分别为 323.32 → 387.40、252.24 → 300.50、262.99 → 337.91 MiB，ONNX 为 299.84 → 361.87 MiB。这是整进程观测值，不能全部归因于权重缓存，也不代表显存用量；本轮未独立测量显存峰值。

## 继承当前启动环境的 CPU 调度策略

此组两版本都不请求 HighQoS，不改变系统设置；绝对耗时明显高于上一组。测试记录的是进程未主动请求 HighQoS，未直接追踪操作系统每一帧的有效调度等级。

| 模型 | 基线 ms | 本轮 ms | 耗时变化 |
| --- | ---: | ---: | ---: |
| YOLOv8n PT | 125.406 | 67.954 | -45.81% |
| YOLO26n-seg PT | 182.732 | 104.455 | -42.84% |

这组结果与减少等待迟到工作线程的设计一致，说明本机该启动条件下有收益。两组调度条件必须分别比较，不能混用基线或将这里的幅度推广到正常前台推理。

## 正确性与构建

- Release 全解决方案构建：0 警告、0 错误。
- CPU 测试 101 项、训练测试 11 项通过；CPU 包括 20,000 轮强制休眠唤醒、迟到参与者、嵌套作用域、异常及释放测试。
- 本轮相关 Vulkan 测试 20 项通过（`Vulkan` 过滤 19 项，另加卷积内核变体 1 项）。新增 Winograd 测试覆盖两个变换、批次、通道/空间尾部及输入变化后的命令重放，并与独立双精度标量卷积对照。
- 所有性能窗口的首帧和末帧原始输出通过形状、SHA256、有限数值及数值回归核对。GPU 对应模型所有输出逐字节一致；CPU 最大绝对差 0.002350，满足逐元素 `abs(a-b) <= 0.003 + 0.0003*abs(b)` 的既有容限。这里是与冻结基线的回归比较，不是新一轮数据集 mAP 评估。
- 两个新 shader 使用 glslang 16.6.0、Vulkan 1.0 重新编译，产物 SHA256 与工作区 `.spv` 一致。
- 上述是相关 Vulkan 测试范围，不等于全 GPU 套件通过。历史通用 GPU last-axis TopK 用例在基线也失败（256 与 28 不符），本轮未修改该路径；默认 ONNX 通用 TopK 走 CPU。

## 证据与复现

[PT 全记录](validation/cpu-gpu-2026-11-11/pt.json)、[ONNX 初测](validation/cpu-gpu-2026-11-11/onnx.json)、[ONNX CPU 复测](validation/cpu-gpu-2026-11-11/onnx-cpu-repeat.json)、[继承策略记录](validation/cpu-gpu-2026-11-11/inherited.json)、[数值核对](validation/cpu-gpu-2026-11-11/numerical.json)。JSON 原样保存了全部窗口、逐帧样本、加载/首帧/P95/分配/工作集数据和输出哈希。测试日志和当次数值检查脚本也保存在同目录。

冻结 DLL、模型、输入及原始 `.f32` 位于实验电脑的 `.cache` / `models`，未随证据提交；只下载 JSON 无法重查输出或恢复本轮基线。下列命令从仓库根目录运行，须先保留相同 SHA256 的基线及输入；新实验使用不同输出文件名，避免覆盖原记录。

```powershell
dotnet build SimdYolo.sln -c Release --artifacts-path .cache/deep-build -m:1 --no-restore
python tools/compare-lifecycle.py --baseline-dir .cache/deep-baseline --candidate-dir .cache/deep-build/bin/SimdYolo/release --input .cache/perf-bus.f32 --models yolov8n yolo11n yolo26n-seg --formats pt --backends cpu vulkan --repeats 2 --warmup 80 --duration 3 --min-frames 40 --size 640 --threads 12 --windows-high-qos --output .cache/recheck-pt.json
python tools/compare-lifecycle.py --baseline-dir .cache/deep-baseline --candidate-dir .cache/deep-build/bin/SimdYolo/release --input .cache/perf-bus.f32 --models yolov8n --formats onnx --backends cpu vulkan --repeats 1 --warmup 80 --duration 3 --min-frames 40 --size 640 --threads 12 --windows-high-qos --output .cache/recheck-onnx.json
python tools/compare-lifecycle.py --baseline-dir .cache/deep-baseline --candidate-dir .cache/deep-build/bin/SimdYolo/release --input .cache/perf-bus.f32 --models yolov8n --formats onnx --backends cpu --repeats 2 --warmup 100 --duration 4 --min-frames 40 --size 640 --threads 12 --windows-high-qos --output .cache/recheck-onnx-cpu.json
python tools/compare-lifecycle.py --baseline-dir .cache/deep-baseline --candidate-dir .cache/deep-build/bin/SimdYolo/release --input .cache/perf-bus.f32 --models yolov8n yolo26n-seg --formats pt --backends cpu --repeats 1 --warmup 80 --duration 3 --min-frames 40 --size 640 --threads 12 --output .cache/recheck-inherited.json
```

继承策略组还需确保外部环境未设置 `SIMDYOLO_BENCHMARK_HIGH_QOS=1`；各窗口记录 `WindowsHighQos` 和实际环境。数值检查脚本需要 NumPy，接收一个或多个上述 JSON 路径，结果写入 `.cache/deep-final-numerical.json`。本报告冻结时改动尚未发布；后续对应源码纳入 [0.11.4 发布说明](../docs/releases/v0.11.4.md)，上述 DLL 哈希继续标识版本号调整前的测量产物。
