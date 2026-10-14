# 三方推理性能与资源对比

日期：2026-10-13 22:01:40 +0800；CPU：Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz（12 逻辑核心）；内存：31.93 GiB；GPU：NVIDIA GeForce GTX 1050 Ti。

状态：complete；完成 8/8 个独立窗口。

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
| yolov9t | N/A | 78.61 | N/A | 21.56 | 22.17 | 17.54 |

## 与基线的差距

耗时倍数 = 本库耗时 / 基线耗时，大于 1 表示本库更慢。内存节省按相同模型的进程平均 RSS 计算；CPU 节省按每帧进程 CPU 时间计算。

| 模型 | CPU / PyTorch | CPU / ORT | Vulkan / PyTorch CUDA | Vulkan / ORT CUDA | Vulkan RSS 节省 vs PyTorch | Vulkan RSS 节省 vs ORT | Vulkan CPU时间/帧 节省 vs PyTorch | Vulkan CPU时间/帧 节省 vs ORT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

低进程 CPU 占用与低 RSS 是资源成本优势；高 GPU 忙碌率只说明有内核在执行，不能证明算子效率、能效或计算单元占用率。显卡、驱动、宿主、线程策略和本次输入都会影响结果，不能外推到其他机器、模型规模、FP16/INT8、TensorRT、torch.compile 或 Metal。

| 模型 | 后端 | 中位 ms | P95 ms | 吞吐 FPS | 进程 CPU% | 整机 CPU% | CPU ms/帧 | RSS MiB | 内存% | GPU%（整卡） | 显存 MiB（整卡） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov9t | Python/PyTorch CPU fused | 78.09 | 100.01 | 12.3 | 96.67 | 100.00 | 943.18 | 328.7 | 1.01 | 1.69 | 1275.14 |
| yolov9t | .NET Vulkan GPU (.pt) | 21.49 | 22.52 | 46.3 | 0.70 | 1.55 | 1.81 | 192.6 | 0.59 | 92.11 | 1460.52 |
| yolov9t | Python/PyTorch CUDA fused | 22.05 | 22.81 | 45.3 | 8.32 | 14.83 | 22.04 | 869.8 | 2.66 | 62.46 | 1372.01 |
| yolov9t | ONNX Runtime CUDA (.NET host) | 17.58 | 17.99 | 56.6 | 8.46 | 16.76 | 17.94 | 738.0 | 2.26 | 91.93 | 1394.61 |
| yolov9t | Python/PyTorch CPU fused | 79.13 | 98.41 | 12.0 | 94.48 | 99.98 | 945.80 | 328.0 | 1.00 | 0.86 | 1318.01 |
| yolov9t | ONNX Runtime CUDA (.NET host) | 17.50 | 18.02 | 56.7 | 8.44 | 16.84 | 17.86 | 733.6 | 2.24 | 92.04 | 1455.30 |
| yolov9t | Python/PyTorch CUDA fused | 22.28 | 31.28 | 43.1 | 8.05 | 15.03 | 22.42 | 868.9 | 2.66 | 58.75 | 1429.55 |
| yolov9t | .NET Vulkan GPU (.pt) | 21.63 | 22.26 | 46.1 | 0.89 | 4.82 | 2.33 | 192.9 | 0.59 | 91.89 | 1514.12 |

单进程采样、全部延迟、CPU 时间、RSS/私有提交量峰值、GPU/显存原始时间序列、输出正确性、版本、输入与模型 SHA256 均在同名 JSON。启动/首帧时间单独保留，不计入稳态速度。输出按 `atol=0.003, rtol=0.0003` 对照本次原版 PyTorch CPU；v10 TopK 行先排序。

参考：[ORT CUDA 配置](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、[ORT 线程与自旋](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)、[NVML GPU 忙碌率定义](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlUtilization__t.html)。
