# 三方推理性能与资源对比

日期：2026-10-13 23:04:49 +0800；CPU：Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz（12 逻辑核心）；内存：31.93 GiB；GPU：NVIDIA GeForce GTX 1050 Ti。

状态：complete；完成 32/32 个独立窗口。

统一 FP32 / batch=1 / 640×640，CPU 线程上限 12，每进程预热 30 次，然后连续推理至少 8.0 秒、至少 40 次。各配置独立进程、串行运行。

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
| yolov8n | 41.69 | 48.70 | 34.37 | N/A | N/A | N/A |
| yolov9t | 55.69 | 104.68 | 31.81 | N/A | N/A | N/A |
| yolov10n | 45.17 | 67.42 | 26.30 | N/A | N/A | N/A |
| yolo11n | 43.73 | 58.22 | 24.33 | N/A | N/A | N/A |
| yolov8n-seg | 52.88 | 59.37 | 36.05 | N/A | N/A | N/A |
| yolov8n-pose | 46.55 | 50.88 | 26.78 | N/A | N/A | N/A |
| yolo11n-seg | 54.78 | 67.49 | 33.75 | N/A | N/A | N/A |
| yolo11n-pose | 49.43 | 55.76 | 25.42 | N/A | N/A | N/A |

## 与基线的差距

耗时倍数 = 本库耗时 / 基线耗时，大于 1 表示本库更慢。内存节省按相同模型的进程平均 RSS 计算；CPU 节省按每帧进程 CPU 时间计算。

| 模型 | CPU / PyTorch | CPU / ORT | Vulkan / PyTorch CUDA | Vulkan / ORT CUDA | Vulkan RSS 节省 vs PyTorch | Vulkan RSS 节省 vs ORT | Vulkan CPU时间/帧 节省 vs PyTorch | Vulkan CPU时间/帧 节省 vs ORT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

低进程 CPU 占用与低 RSS 是资源成本优势；高 GPU 忙碌率只说明有内核在执行，不能证明算子效率、能效或计算单元占用率。显卡、驱动、宿主、线程策略和本次输入都会影响结果，不能外推到其他机器、模型规模、FP16/INT8、TensorRT、torch.compile 或 Metal。

| 模型 | 后端 | 中位 ms | P95 ms | 吞吐 FPS | 进程 CPU% | 整机 CPU% | CPU ms/帧 | RSS MiB | 内存% | GPU%（整卡） | 显存 MiB（整卡） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8n | Python/PyTorch CPU fused | 48.70 | 62.45 | 19.2 | 92.88 | 100.00 | 579.55 | 325.9 | 1.00 | 7.68 | 1608.27 |
| yolov8n | .NET SIMD CPU (.pt) | 41.69 | 46.74 | 23.6 | 64.78 | 68.29 | 330.03 | 436.7 | 1.34 | 1.00 | 1570.38 |
| yolov8n | .NET SIMD CPU (.onnx) | 44.00 | 54.03 | 21.6 | 61.39 | 68.80 | 340.43 | 469.3 | 1.44 | 1.36 | 1572.09 |
| yolov8n | ONNX Runtime CPU (.NET host) | 34.37 | 74.95 | 24.0 | 88.51 | 99.77 | 442.95 | 153.0 | 0.47 | 10.82 | 1608.49 |
| yolov9t | Python/PyTorch CPU fused | 104.68 | 140.94 | 9.3 | 83.06 | 100.00 | 1067.92 | 327.5 | 1.00 | 5.43 | 1595.42 |
| yolov9t | ONNX Runtime CPU (.NET host) | 31.81 | 73.85 | 25.5 | 93.60 | 99.96 | 439.79 | 170.4 | 0.52 | 0.00 | 1588.04 |
| yolov9t | .NET SIMD CPU (.onnx) | 63.55 | 82.22 | 14.9 | 53.14 | 61.15 | 427.47 | 339.1 | 1.04 | 0.72 | 1596.18 |
| yolov9t | .NET SIMD CPU (.pt) | 55.69 | 67.07 | 17.3 | 64.72 | 71.84 | 449.19 | 458.9 | 1.40 | 1.10 | 1620.39 |
| yolov10n | Python/PyTorch CPU fused | 67.42 | 102.83 | 13.0 | 86.43 | 100.00 | 796.88 | 324.1 | 0.99 | 10.28 | 1593.35 |
| yolov10n | ONNX Runtime CPU (.NET host) | 26.30 | 62.51 | 31.3 | 92.97 | 100.00 | 356.09 | 125.4 | 0.38 | 7.74 | 1604.41 |
| yolov10n | .NET SIMD CPU (.pt) | 45.17 | 52.38 | 21.5 | 61.86 | 67.02 | 345.20 | 445.3 | 1.36 | 4.21 | 1588.96 |
| yolov10n | .NET SIMD CPU (.onnx) | 69.49 | 73.17 | 14.3 | 40.15 | 42.12 | 336.68 | 285.9 | 0.87 | 5.31 | 1589.63 |
| yolo11n | Python/PyTorch CPU fused | 58.22 | 77.30 | 16.5 | 89.98 | 99.97 | 655.31 | 335.2 | 1.03 | 7.00 | 1654.51 |
| yolo11n | ONNX Runtime CPU (.NET host) | 24.33 | 46.87 | 36.0 | 95.44 | 99.98 | 318.47 | 177.6 | 0.54 | 1.90 | 1696.12 |
| yolo11n | .NET SIMD CPU (.pt) | 43.73 | 53.45 | 21.9 | 60.87 | 67.16 | 333.54 | 425.6 | 1.30 | 1.46 | 1727.52 |
| yolo11n | .NET SIMD CPU (.onnx) | 45.16 | 50.82 | 21.6 | 60.50 | 63.94 | 335.71 | 360.7 | 1.10 | 1.68 | 1680.37 |
| yolov8n-seg | Python/PyTorch CPU fused | 59.37 | 76.10 | 16.2 | 96.50 | 100.00 | 715.26 | 349.0 | 1.07 | 4.68 | 1682.44 |
| yolov8n-seg | .NET SIMD CPU (.onnx) | 54.58 | 62.20 | 17.9 | 64.01 | 68.75 | 429.25 | 626.9 | 1.92 | 1.97 | 1690.92 |
| yolov8n-seg | .NET SIMD CPU (.pt) | 52.88 | 60.67 | 18.5 | 66.74 | 71.01 | 433.49 | 557.0 | 1.70 | 2.07 | 1696.03 |
| yolov8n-seg | ONNX Runtime CPU (.NET host) | 36.05 | 52.28 | 26.1 | 97.56 | 100.00 | 449.39 | 200.5 | 0.61 | 2.00 | 1675.34 |
| yolov8n-pose | Python/PyTorch CPU fused | 50.88 | 77.34 | 18.1 | 92.02 | 100.00 | 610.88 | 324.7 | 0.99 | 7.00 | 1682.03 |
| yolov8n-pose | .NET SIMD CPU (.onnx) | 45.16 | 50.48 | 21.6 | 61.11 | 65.43 | 339.23 | 332.7 | 1.02 | 1.57 | 1683.38 |
| yolov8n-pose | .NET SIMD CPU (.pt) | 46.55 | 56.43 | 20.7 | 60.93 | 67.25 | 353.82 | 379.1 | 1.16 | 1.21 | 1681.61 |
| yolov8n-pose | ONNX Runtime CPU (.NET host) | 26.78 | 56.02 | 31.7 | 94.61 | 100.00 | 358.39 | 148.8 | 0.45 | 2.10 | 1680.27 |
| yolo11n-seg | Python/PyTorch CPU fused | 67.49 | 93.43 | 13.9 | 92.99 | 99.98 | 804.34 | 348.5 | 1.07 | 6.07 | 1684.78 |
| yolo11n-seg | .NET SIMD CPU (.onnx) | 58.54 | 73.66 | 16.3 | 58.28 | 67.14 | 429.63 | 611.4 | 1.87 | 2.14 | 1679.56 |
| yolo11n-seg | ONNX Runtime CPU (.NET host) | 33.75 | 48.09 | 27.3 | 96.88 | 99.79 | 425.59 | 232.1 | 0.71 | 0.60 | 1674.72 |
| yolo11n-seg | .NET SIMD CPU (.pt) | 54.78 | 69.97 | 17.3 | 62.28 | 68.63 | 431.09 | 533.2 | 1.63 | 0.50 | 1673.41 |
| yolo11n-pose | Python/PyTorch CPU fused | 55.76 | 69.29 | 17.1 | 96.10 | 99.98 | 672.78 | 336.6 | 1.03 | 3.62 | 1672.31 |
| yolo11n-pose | .NET SIMD CPU (.onnx) | 47.19 | 52.05 | 20.9 | 60.61 | 62.76 | 347.75 | 276.8 | 0.85 | 0.36 | 1669.98 |
| yolo11n-pose | .NET SIMD CPU (.pt) | 49.43 | 56.23 | 19.7 | 59.10 | 64.44 | 360.46 | 426.3 | 1.30 | 1.31 | 1673.60 |
| yolo11n-pose | ONNX Runtime CPU (.NET host) | 25.42 | 40.18 | 35.7 | 97.51 | 100.00 | 328.07 | 172.9 | 0.53 | 0.85 | 1673.66 |

单进程采样、全部延迟、CPU 时间、RSS/私有提交量峰值、GPU/显存原始时间序列、输出正确性、版本、输入与模型 SHA256 均在同名 JSON。启动/首帧时间单独保留，不计入稳态速度。输出按 `atol=0.003, rtol=0.0003` 对照本次原版 PyTorch CPU；v10 TopK 行先排序。

参考：[ORT CUDA 配置](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、[ORT 线程与自旋](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)、[NVML GPU 忙碌率定义](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlUtilization__t.html)。
