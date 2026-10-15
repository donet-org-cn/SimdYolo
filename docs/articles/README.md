# SimdYolo 0.10.1 图文发布包

公众号和博客园分别更新为0.10.1发布内容，覆盖五代检测头训练、SIMD优化、CPU/Vulkan速度与保存模型的真实图片对照。历史0.9.0推理表保留原数字，并在两篇文章中明确标注；不存在将旧推理图当作本版重测的替换。

[0.10.1 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1) · [训练报告](../../artifacts/training-optimization-0.10.1.md) · [中文文档ZIP](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-docs-v0.10.1.zip)

| 文件 | 用途 |
| --- | --- |
| [wechat.md](wechat.md) | 公众号：速度、模型结果、纯C#部署和网页上手 |
| [cnblogs.md](cnblogs.md) | 博客园：SIMD/工作区/设备分派、损失、数值和模型边界 |
| [manifest.json](manifest.json) | 当前发布/测量版本、源结果SHA256与文章/配图核验 |

## 当前训练配图

| PNG | 内容 |
| --- | --- |
| [CPU训练速度](../../artifacts/training-performance-0.10.1-cpu.png) | 原版、本版与0.10.0历史结果，细线表示两窗范围 |
| [GPU训练速度](../../artifacts/training-performance-0.10.1-gpu.png) | CUDA与Vulkan混合路径，标明计时边界 |
| [模型结果](../../artifacts/training-results-0.10.1.png) | 保存重载后32张微调留出图的mAP |
| [训练轨迹](../../artifacts/training-loss-parity-0.10.1.png) | 192步真实图片微调损失，保留分叉 |

上述图片均有同名SVG，中文转为轮廓便于导出；PNG用于上传。训练性能为128×128、2类、4线程的冻结特征后头更新；模型精度为256×256、80类、96/32拆分，各2轮。直接参数门槛9/10，指标10/10，固定CPU参数控制10/10；未通过项与精度下降照实列出。

## 历史图片与发布

[assets/0.9.0/](assets/0.9.0/)仍保留10张PNG、10份SVG和取数清单。正文引用的0.9.0推理CPU/GPU图和0.10.0网页截图均标注采集版本；0.7.x/0.8.x资料保持历史用途。文档ZIP保留所需图片及相对目录，未包含的历史文件链接到版本化仓库。

将Markdown导入平台编辑器后，上传其PNG并替换相对图链。公众号可把仓库设为阅读原文；博客园建议标签C#、.NET、YOLO、SIMD、Vulkan、ONNX。所有文字使用UTF-8，发布说明回读核验中文内容。
