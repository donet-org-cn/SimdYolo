# SimdYolo 0.9.0 网页版运行包

这是 ASP.NET Core 框架依赖发布输出，包含推理与训练 DLL、网页和启动脚本。没有打包模型、Python、CUDA工具链或.NET运行时。

## Windows 启动

1. 安装 **ASP.NET Core Runtime 10（x64）**，安装 .NET 10 SDK 也可以。
2. 将 `SimdYolo-Web-v0.9.0.zip` 解压到一个目录。
3. 将 `SimdYolo-models-v0.9.0.zip` 解压到同一目录，目录中应出现 `models/pt/`、`models/onnx/`，与 `SimdYolo.Web.dll` 并列。
4. 双击 `start.cmd`，或在该目录执行 `dotnet SimdYolo.Web.dll --contentRoot . --SimdYolo:RepositoryRoot . --SimdYolo:DataRoot ./data --urls http://127.0.0.1:5188`。
5. 浏览器打开 **http://127.0.0.1:5188**。

默认使用CPU SIMD；Vulkan需要兼容显卡驱动和loader，不需要CUDA Toolkit/cuDNN。Metal需要macOS，本发布没有macOS实机验证。DLL以.NET10构建；其他平台安装相应运行时后可使用上述dotnet命令，仍需自行验证设备。

## 示例流程

推理页上传JPEG/PNG，选择PT/ONNX模型。检测显示框，分割叠加掩码，姿态显示关键点。浏览器完成图片解码，服务端使用C#计算。

训练页先用内置方块数据验证流程，再导入自己的图片与YOLO检测文本。每行标签为 `class cx cy w h`，类别从0开始、坐标相对原图归一化；缺少标签的图像按背景样本导入，请核对匹配情况。可观察真实损失曲线、取消、恢复和下载 `.sytrain`。

训练范围限于YOLOv8检测头微调，骨干/FPN冻结；Vulkan仅加速冻结网络和训练头GEMM，BN/损失/SGD在CPU。没有全参数、分割/姿态、YOLO26或Metal训练。梯度累加不等于联合批次BN。

上传、数据集和任务保存在启动目录 `data/`，重启后保留。取消/恢复以最近完整轮次检查点为准。下载的 `.sytrain` 可重新加载推理，也可恢复训练。

本示例仅供本机使用，默认绑定回环地址，没有账号/公网鉴权。详细说明、API和性能图见同版本文档ZIP或[GitHub](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.9.0)。模型保留AGPL-3.0及来源，源码暂未新增许可证。
