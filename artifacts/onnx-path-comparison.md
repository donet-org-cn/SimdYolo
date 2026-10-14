# 三方推理性能与资源对比

日期：2026-10-13 21:55:08 +0800；CPU：Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz（12 逻辑核心）；内存：31.93 GiB；GPU：NVIDIA GeForce GTX 1050 Ti。

状态：complete；完成 24/24 个独立窗口。

统一 FP32 / batch=1 / 640×640，CPU 线程上限 12，每进程预热 15 次，然后连续推理至少 8.0 秒、至少 30 次。各配置独立进程、串行运行。

GPU 耗时全部包含输入上传、全部输出回读。计时包含网络前向与输出头解码，排除加载、首次编译、图片预处理、NMS、mask 后处理。PyTorch 使用原版 Ultralytics fuse()，没有 torch.compile；ORT 开启所有图优化，顺序执行、默认线程自旋，CUDA 卷积自动选择算法，关闭 TF32。

CPU% = 进程 CPU 秒 / 计时窗口秒 / 逻辑核心数 × 100；它不是单核心口径。CPU ms/帧用于判断同样工作量消耗的 CPU 时间，避免把低占用直接解释成效率高。内存是进程工作集 RSS，包括运行库、模型和缓存；内存%以物理 RAM 总量为分母。Private MiB 是 Windows 进程私有提交量。

外部采样间隔目标为 250 ms；CPU 满载时调度可能延迟。实际时间戳与采样次数在 JSON 中；进程平均 CPU%来自窗口内累计 CPU 时间，不依赖采样是否准点。内存峰值是采样峰值。

GPU% 和显存由独立控制进程通过 NVML 采样，为整张显卡，包含桌面及其他进程；不伪装成单个推理进程的使用率。舍弃 GPU 窗口第一秒，避免滚动采样混入启动阶段；记录每次启动前的 GPU/显存快照及显存增量。启动前快照不保证完全空闲，可能含前一个进程的滚动历史；显存增量也不是准确的进程显存。CPU 推理期间 GPU 数字是背景活动，不能归因于 CPU 后端。

主要 ORT 对照使用 .NET 宿主，与我们库共享相同 .NET 运行时，并把全部输出复制为托管 float[]。可选 Python ORT 行使用 Python 绑定，不导入 PyTorch。CUDA ORT 直接预加载现有 cuDNN/CUDA DLL，主库没有新增 ORT 依赖。内存对比是宿主进程总成本，包括不同运行库和缓存。

此机器的 NuGet ORT 1.26 GPU 原生构建在 QuickGelu 报 `cudaErrorNoKernelImageForDevice`；正式 .NET ORT 改用官方 Python GPU wheel 中的同版本原生 DLL，通过官方 .NET Managed 绑定调用。该构建已在 1050 Ti 上执行和校验，DLL 哈希写入 JSON；运行时没有调用 Python。

## 各模型速度

同一配置有重复窗口时，以下数字取各窗口中位耗时的中位数；它不代表跨窗口置信区间。单位 ms，越小越快。

| 模型 | .NET CPU | PyTorch CPU | ORT CPU | .NET Vulkan | PyTorch CUDA | ORT CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8n | 201.58 | 46.91 | N/A | 83.90 | N/A | N/A |
| yolov9t | 290.16 | 79.55 | N/A | 102.02 | N/A | N/A |
| yolov10n | 244.98 | 62.28 | N/A | 107.36 | N/A | N/A |
| yolo11n | 209.94 | 52.10 | N/A | 97.40 | N/A | N/A |
| yolov8n-seg | 235.24 | 59.68 | N/A | 96.05 | N/A | N/A |
| yolov8n-pose | 240.09 | 50.09 | N/A | 101.50 | N/A | N/A |
| yolo11n-seg | 256.09 | 64.78 | N/A | 94.03 | N/A | N/A |
| yolo11n-pose | 256.90 | 58.67 | N/A | 94.71 | N/A | N/A |

## 与基线的差距

耗时倍数 = 本库耗时 / 基线耗时，大于 1 表示本库更慢。内存节省按相同模型的进程平均 RSS 计算；CPU 节省按每帧进程 CPU 时间计算。

| 模型 | CPU / PyTorch | CPU / ORT | Vulkan / PyTorch CUDA | Vulkan / ORT CUDA | Vulkan RSS 节省 vs PyTorch | Vulkan RSS 节省 vs ORT | Vulkan CPU时间/帧 节省 vs PyTorch | Vulkan CPU时间/帧 节省 vs ORT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

低进程 CPU 占用与低 RSS 是资源成本优势；高 GPU 忙碌率只说明有内核在执行，不能证明算子效率、能效或计算单元占用率。显卡、驱动、宿主、线程策略和本次输入都会影响结果，不能外推到其他机器、模型规模、FP16/INT8、TensorRT、torch.compile 或 Metal。

| 模型 | 后端 | 中位 ms | P95 ms | 吞吐 FPS | 进程 CPU% | 整机 CPU% | CPU ms/帧 | RSS MiB | 内存% | GPU%（整卡） | 显存 MiB（整卡） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8n | Python/PyTorch CPU fused | 46.91 | 58.58 | 20.8 | 97.55 | 99.98 | 562.41 | 325.5 | 1.00 | 10.64 | 1289.93 |
| yolov8n | .NET SIMD CPU (.onnx) | 201.58 | 210.35 | 4.9 | 30.17 | 31.96 | 734.38 | 701.0 | 2.14 | 12.38 | 1286.68 |
| yolov8n | .NET Vulkan GPU (.onnx) | 83.90 | 86.41 | 11.9 | 5.90 | 9.38 | 59.70 | 230.2 | 0.70 | 29.29 | 1497.57 |
| yolov9t | Python/PyTorch CPU fused | 79.55 | 92.54 | 12.4 | 97.29 | 100.00 | 944.76 | 328.8 | 1.01 | 1.11 | 1289.69 |
| yolov9t | .NET Vulkan GPU (.onnx) | 102.02 | 104.78 | 9.8 | 5.51 | 9.64 | 67.64 | 235.6 | 0.72 | 31.10 | 1718.54 |
| yolov9t | .NET SIMD CPU (.onnx) | 290.16 | 301.19 | 3.4 | 25.55 | 27.03 | 893.75 | 670.5 | 2.05 | 10.23 | 1240.97 |
| yolov10n | Python/PyTorch CPU fused | 62.28 | 67.94 | 15.9 | 98.42 | 100.00 | 741.21 | 324.0 | 0.99 | 10.36 | 1241.53 |
| yolov10n | .NET SIMD CPU (.onnx) | 244.98 | 266.26 | 4.0 | 29.12 | 32.36 | 866.00 | 713.6 | 2.18 | 12.59 | 1232.09 |
| yolov10n | .NET Vulkan GPU (.onnx) | 107.36 | 128.51 | 9.0 | 5.65 | 8.02 | 75.13 | 280.1 | 0.86 | 28.34 | 1437.71 |
| yolo11n | Python/PyTorch CPU fused | 52.10 | 59.84 | 18.9 | 96.78 | 99.97 | 614.00 | 334.8 | 1.02 | 1.54 | 1250.41 |
| yolo11n | .NET SIMD CPU (.onnx) | 209.94 | 220.26 | 4.7 | 30.51 | 32.95 | 772.62 | 742.2 | 2.27 | 1.93 | 1254.86 |
| yolo11n | .NET Vulkan GPU (.onnx) | 97.40 | 114.72 | 10.2 | 6.04 | 7.60 | 71.07 | 260.7 | 0.80 | 29.90 | 1473.21 |
| yolov8n-seg | Python/PyTorch CPU fused | 59.68 | 82.23 | 15.7 | 93.88 | 99.98 | 719.74 | 342.3 | 1.05 | 2.46 | 1261.18 |
| yolov8n-seg | .NET Vulkan GPU (.onnx) | 96.05 | 99.20 | 10.4 | 5.52 | 8.69 | 64.01 | 248.6 | 0.76 | 35.75 | 1476.64 |
| yolov8n-seg | .NET SIMD CPU (.onnx) | 235.24 | 241.83 | 4.2 | 36.27 | 38.57 | 1028.49 | 882.5 | 2.70 | 1.66 | 1267.09 |
| yolov8n-pose | Python/PyTorch CPU fused | 50.09 | 61.71 | 19.6 | 97.68 | 99.98 | 597.71 | 326.1 | 1.00 | 16.14 | 1263.52 |
| yolov8n-pose | .NET SIMD CPU (.onnx) | 240.09 | 249.09 | 4.2 | 26.51 | 29.09 | 766.08 | 687.1 | 2.10 | 15.86 | 1265.91 |
| yolov8n-pose | .NET Vulkan GPU (.onnx) | 101.50 | 104.51 | 9.8 | 6.07 | 9.25 | 74.37 | 235.6 | 0.72 | 30.34 | 1487.53 |
| yolo11n-seg | Python/PyTorch CPU fused | 64.78 | 75.65 | 15.2 | 97.46 | 99.98 | 771.39 | 339.3 | 1.04 | 2.00 | 1265.59 |
| yolo11n-seg | .NET Vulkan GPU (.onnx) | 94.03 | 100.67 | 10.6 | 5.73 | 12.12 | 65.07 | 265.3 | 0.81 | 34.66 | 1506.38 |
| yolo11n-seg | .NET SIMD CPU (.onnx) | 256.09 | 291.56 | 3.9 | 33.91 | 44.59 | 1047.36 | 772.6 | 2.36 | 7.00 | 1320.01 |
| yolo11n-pose | Python/PyTorch CPU fused | 58.67 | 80.60 | 16.2 | 93.86 | 100.00 | 693.63 | 336.6 | 1.03 | 11.25 | 1310.33 |
| yolo11n-pose | .NET Vulkan GPU (.onnx) | 94.71 | 97.60 | 10.5 | 6.04 | 7.83 | 68.93 | 239.7 | 0.73 | 25.48 | 1500.75 |
| yolo11n-pose | .NET SIMD CPU (.onnx) | 256.90 | 261.36 | 3.9 | 27.24 | 28.20 | 841.80 | 768.0 | 2.35 | 0.41 | 1270.65 |

单进程采样、全部延迟、CPU 时间、RSS/私有提交量峰值、GPU/显存原始时间序列、输出正确性、版本、输入与模型 SHA256 均在同名 JSON。启动/首帧时间单独保留，不计入稳态速度。输出按 `atol=0.003, rtol=0.0003` 对照本次原版 PyTorch CPU；v10 TopK 行先排序。

参考：[ORT CUDA 配置](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、[ORT 线程与自旋](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)、[NVML GPU 忙碌率定义](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlUtilization__t.html)。
