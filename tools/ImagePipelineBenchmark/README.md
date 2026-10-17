# 实际图片推理分段基准

仅依赖.NET和SimdYolo；开发编排脚本使用Python标准库。应用包不包含这些工具，也不依赖Python、PyTorch或ORT。

参数顺序为模型、RGB24文件、输出前缀、原图宽高、Cpu或Vulkan、CPU线程数。例如：

~~~powershell
dotnet run --project tools/ImagePipelineBenchmark -c Release -- models/pt/yolov8n-seg.pt bus.rgb .cache/image-bench/result 810 1080 Cpu 4
~~~

每个分段先预热12次，再测量至少2秒且40帧，记录中位数、P95、进程CPU核心百分比、每帧托管分配和原始采样。Predict包含预处理、网络前向、检测筛选及原图尺度的实例概率掩码，不包含HTTP、JSON和网页合成。8位掩码转换单独测量，不能把分段中位数相加冒充完整时间。加载及首个Predict另记，不清空OS/驱动缓存。

可通过SimdYoloLibraryPath指定冻结DLL，两个宿主必须使用相同工具源码。0.11.1的完整复现及原始输入摘要见[验证记录](../../artifacts/validation/0.11.1/README.md)。主要分割配置使用ABBA顺序各两个独立进程窗口，检测控制与大图配置各一对；小幅波动不代表稳定领先。不同进程的最终实例还会逐项核对框、概率及掩码几何，保留接近阈值的翻转数量。
