# 第三方模型和组件

## 模型权重

`models/pt/` 中的八个检查点来自 [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0)，保留原始文件和 SHA256。`models/onnx/` 中的十六个文件由相应检查点导出，许可证和归属随原模型保留。

0.8.0 新增的三个 YOLO26 检查点来自 [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0)，保留原始字节；另提供六个对应的静态/动态 ONNX。架构和开发对照采用 [Ultralytics v8.4.0 源代码](https://github.com/ultralytics/ultralytics/tree/v8.4.0)，同样保留上游 AGPL-3.0 许可与来源；仓库共 11 个 PT、22 个 ONNX。

这些模型使用上游 **AGPL-3.0** 许可证，完整文本见 [模型许可证](models/LICENSE-AGPL-3.0.txt)。模型、架构名称和上游代码的权利归各原作者及权利人所有。参见 [assets LICENSE](https://github.com/ultralytics/assets/blob/main/LICENSE)、[Ultralytics v8.3.0 源代码](https://github.com/ultralytics/ultralytics/tree/v8.3.0)和 [官方模型许可说明](https://www.ultralytics.com/license)。

模型文件、任务、输入尺寸、opset、来源和哈希见 [模型清单](models/manifest.json)。现有 ONNX 导出方法在 [validate-reference.py](tools/validate-reference.py) 中，使用原版 PyTorch/Ultralytics 作为开发时导出和校验工具。它们不是 SimdYolo 主库的运行依赖。

项目源码本次没有新增许可证；模型许可证适用于随仓库分发的上游模型，不表示上游作者创作了本仓库的 C# 实现。

## 运行和开发依赖

SimdYolo 主库使用 .NET framework 程序集和系统 Vulkan/Metal API，GPU 着色器随主库嵌入。GitHub 仓库不分发 Python/PyTorch、Ultralytics 安装环境、ONNX Runtime 原生 DLL、CUDA/cuDNN、Vulkan SDK 或 NuGet 缓存。

`tools/OrtBenchmark/` 和 Python 对照脚本是可选开发工具，依赖按各上游许可证安装。它们与无外部数学运行库依赖的主库分开。
