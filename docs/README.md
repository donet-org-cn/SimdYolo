# SimdYolo 文档导航（源码0.11.1）

当前源码和仓库内NuGet包为 **0.11.1**；GitHub最新正式Release仍为 **v0.10.1**。0.11.1的优化、标注、验证和导出功能请使用main源码或仓库内0.11.1包，历史Release附件保持原内容。

## 从这里开始

| 需要做什么 | 文档 |
| --- | --- |
| 本地运行网页、图片推理、人物/车辆分离和背景去除 | [网页示例](../examples/SimdYolo.Web/README.md) |
| C#加载PT/ONNX/SYTRAIN、读取标签、检测/分割/姿态 | [推理API](inference.md) |
| 五代YOLO检测头微调、数据格式、取消和恢复 | [训练说明](training.md) |
| 验证集、最佳模型、早停、标注、ONNX导出和实验低精度 | [模型工作流](model-workflows.md) |
| 0.11.1优化前后速度、表格和图 | [完整图片推理及网页性能](../artifacts/image-pipeline-optimization-0.11.1.md) |
| 测试、实际PNG、程序集摘要与复现 | [0.11.1验证](../artifacts/validation/0.11.1/README.md)、[全部历史验证](../VALIDATION.md) |
| 公众号与博客园图文 | [文章与配图](articles/README.md) |

## 0.11.1 下载与启动

| 内容 | 入口 |
| --- | --- |
| 当前源码 | [GitHub main](https://github.com/hnlyf168/SimdYolo/tree/main) |
| 主库0.11.1 | [SimdYolo.0.11.1.nupkg](../artifacts/SimdYolo.0.11.1.nupkg) |
| 训练库0.11.1 | [SimdYolo.Training.0.11.1.nupkg](../artifacts/SimdYolo.Training.0.11.1.nupkg)，依赖同版主库 |
| 33模型与原许可证 | [模型目录](../models/README.md)；权重通过Git LFS拉取，也可使用历史Release模型ZIP |
| 历史正式发布 | [v0.10.1 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1)，运行包及文档ZIP属于0.10.1 |

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
| 0.11.1 | 原图掩码SIMD、坐标/行缓存、多实例并行；网页RGB24二进制、有界掩码缓存；逐次置信度复用会话 |
| 0.11.0起 | 训练验证、best.sytrain、早停、标准ONNX导出、网页检测框标注与预标注、CPU FP16混合/INT8实验模式 |
| 0.10.1 | 五代检测头微调速度及保存模型结果对照；最新正式Release |
| 0.9.0 | 冻结版本与PyTorch/ORT的11模型CPU/GPU前向排名，属于历史数据 |

0.11.1表格包含预处理、网络、筛选和原图掩码；0.9.0跨引擎排名排除图片预处理和最终掩码，两者不能直接混用。本轮没有重测PyTorch/ORT，纯检测控制组基本持平。

核心为纯C#/.NET 10；浏览器承担图片解码、交互和RGBA/PNG合成。Vulkan通过系统图形API运行，无CUDA工具链依赖；本机验证Windows/GTX1050Ti，Metal未实机测试。训练仍是冻结骨干/FPN的检测头微调，分割/姿态只支持推理。低精度尚无速度优势，默认FP32。

文档修订不重写历史性能JSON、已提交NuGet包或发布附件；这些文件继续对应原来的测试与摘要。
