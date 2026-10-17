# SimdYolo 文档导航（源码0.11.3）

当前源码为 **0.11.3**，最新正式NuGet/Release为 **0.11.2**。[下载正式版](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.11.2)；历史Release保持原内容。0.11.3加载/首帧及稳定性改动使用源码构建，见[实测与图](../artifacts/startup-optimization-0.11.3.md)。

## 从这里开始

| 需要做什么 | 文档 |
| --- | --- |
| 本地运行网页、图片推理、人物/车辆分离和背景去除 | [网页示例](../examples/SimdYolo.Web/README.md) |
| 加载/首帧优化、ReadyToRun发布、会话稳定性与复现 | [启动说明](startup.md)、[0.11.3验证](../artifacts/validation/0.11.3/README.md) |
| C#加载PT/ONNX/SYTRAIN、读取标签、检测/分割/姿态 | [推理API](inference.md) |
| 五代YOLO检测头微调、数据格式、取消和恢复 | [训练说明](training.md) |
| 验证集、最佳模型、早停、标注、ONNX导出和实验低精度 | [模型工作流](model-workflows.md) |
| 0.11.2原版PyTorch/本库/ORT的CPU、GPU排名与资源 | [本版重新实测](../artifacts/runtime-comparison-0.11.2.md) |
| 0.11.1优化前后速度、表格和图 | [完整图片推理及网页性能](../artifacts/image-pipeline-optimization-0.11.1.md) |
| 测试、实际PNG、程序集摘要与复现 | [0.11.1验证](../artifacts/validation/0.11.1/README.md)、[全部历史验证](../VALIDATION.md) |
| 公众号与博客园图文 | [文章与配图](articles/README.md) |

## 0.11.2 下载与启动

| 内容 | 入口 |
| --- | --- |
| 当前源码 | [GitHub main](https://github.com/hnlyf168/SimdYolo/tree/main) |
| 主库0.11.2 | [SimdYolo.0.11.2.nupkg](../artifacts/SimdYolo.0.11.2.nupkg) |
| 训练库0.11.2 | [SimdYolo.Training.0.11.2.nupkg](../artifacts/SimdYolo.Training.0.11.2.nupkg)，依赖同版主库 |
| 33模型与原许可证 | [模型目录](../models/README.md)；权重通过Git LFS拉取，也可使用历史Release模型ZIP |
| 当前正式发布 | [v0.11.2 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.11.2)，含本版网页、模型与文档ZIP |

~~~powershell
git clone https://github.com/hnlyf168/SimdYolo.git
cd SimdYolo
git lfs pull
dotnet run --project examples/SimdYolo.Web -c Release
~~~

已有仓库先拉取main最新提交，再运行上述后两条命令。打开 **http://127.0.0.1:5188**；分离目标时使用分割模型，点击“人物 / 车辆示例”即可查看两张透明PNG。

## 功能与版本

| 源码版本 | 对应能力或测量 |
| --- | --- |
| 0.11.2 | 与PyTorch/ORT重新实测11模型、8配置，发布模型/网页/文档附件 |
| 0.11.1 | 原图掩码SIMD、坐标/行缓存、多实例并行；网页RGB24二进制、有界掩码缓存；逐次置信度复用会话 |
| 0.11.0起 | 训练验证、best.sytrain、早停、标准ONNX导出、网页检测框标注与预标注、CPU FP16混合/INT8实验模式 |
| 0.10.1 | 五代检测头微调速度及保存模型结果历史对照 |
| 0.9.0 | 冻结版本与PyTorch/ORT的11模型CPU/GPU前向排名，属于历史数据 |

0.11.1表格包含预处理、网络、筛选和原图掩码；0.9.0跨引擎排名排除图片预处理和最终掩码，两者不能直接混用。0.11.1的纯检测控制组基本持平；0.11.2跨引擎重测见[本版报告](../artifacts/runtime-comparison-0.11.2.md)。

核心为纯C#/.NET 10；浏览器承担图片解码、交互和RGBA/PNG合成。Vulkan通过系统图形API运行，无CUDA工具链依赖；本机验证Windows/GTX1050Ti，Metal未实机测试。训练仍是冻结骨干/FPN的检测头微调，分割/姿态只支持推理。低精度尚无速度优势，默认FP32。

文档修订不重写历史性能JSON、已提交NuGet包或发布附件；这些文件继续对应原来的测试与摘要。
