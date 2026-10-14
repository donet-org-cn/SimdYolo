# .NET 与 Python + PyTorch 速度对比

输入：640×640，batch=1，FP32；CPU 线程上限 12；预热 5 次，测量 20 次。

计时包含模型前向与输出头解码，不包含加载、图片预处理、NMS、分割 mask 后处理或姿态可视化。所有模型各配置串行执行，输入文件和权重相同。

CPU：Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz；PyTorch：2.11.0+cpu；Ultralytics：8.3.0；.NET SDK：10.0.401。

| 模型 | .NET CPU ms | Python CPU 原始 ms | Python CPU 融合 ms | .NET Vulkan ms | .NET CPU / Python 融合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n.pt | 530.66 | 53.49 | 48.58 | 308.44 | 10.92× |
| yolov9t.pt | 752.57 | 109.30 | 80.16 | 392.04 | 9.39× |
| yolov10n.pt | 464.48 | 80.22 | 64.56 | 359.03 | 7.19× |
| yolo11n.pt | 426.98 | 64.57 | 77.01 | 315.33 | 5.54× |
| yolov8n-seg.pt | 866.91 | 66.07 | 58.97 | 459.35 | 14.70× |
| yolov8n-pose.pt | 557.79 | 53.18 | 51.73 | 268.40 | 10.78× |
| yolo11n-seg.pt | 753.99 | 71.49 | 77.12 | 501.76 | 9.78× |
| yolo11n-pose.pt | 492.34 | 77.85 | 62.21 | 376.47 | 7.91× |

所有耗时均为中位数，数值越小越快。比值大于 1 表示当前 .NET CPU 比 Python/PyTorch CPU 慢。Vulkan 是 GPU 路径，不能据此把 CPU/GPU 的差异归因于语言。

Python 融合版调用原版 Ultralytics 的 fuse()，与 .NET 加载时的 Conv/BN 融合保持接近的优化配置；原版还会重参数化部分 RepConv。未使用 torch.compile、ONNX Runtime 或 TensorRT。

CUDA host-to-host 包含输入上传与结果回读；device-only 仅测已驻留设备的输入输出，二者在 JSON 中分别记录。

| 模型 | .NET Vulkan ms | PyTorch CUDA 含上传/回读 ms | PyTorch CUDA 设备内 ms | Vulkan / CUDA 含传输 |
| --- | ---: | ---: | ---: | ---: |
| yolov8n.pt | 308.44 | 9.99 | 11.09 | 30.87× |
| yolov9t.pt | 392.04 | 22.91 | 21.27 | 17.11× |
| yolov10n.pt | 359.03 | 13.43 | 12.80 | 26.73× |
| yolo11n.pt | 315.33 | 10.83 | 10.28 | 29.12× |
| yolov8n-seg.pt | 459.35 | 13.12 | 11.85 | 35.00× |
| yolov8n-pose.pt | 268.40 | 10.24 | 9.72 | 26.20× |
| yolo11n-seg.pt | 501.76 | 14.16 | 12.80 | 35.43× |
| yolo11n-pose.pt | 376.47 | 11.63 | 15.62 | 32.37× |

完整采样、P95、FPS、模型/input SHA256 和配置见同名 JSON；复跑命令见 README。

融合行为参考：[Ultralytics fuse](https://docs.ultralytics.com/reference/nn/tasks/#ultralytics.nn.tasks.BaseModel.fuse)。CUDA 计时使用 [torch.cuda.synchronize](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)。
