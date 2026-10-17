# SimdYolo 0.11.1 图文文档

公众号和博客园已更新0.11.1标题、完整图片推理表格、优化图、人物/车辆分离截图及当前源码入口。0.10.1训练及0.9.0跨引擎推理作为历史章节保留，原数字与采集范围不变。当前图文使用仓库文件；GitHub正式Release及其中的文档ZIP仍为0.10.1。

[文档与0.11.1包](../README.md) · [0.11.1性能报告](../../artifacts/image-pipeline-optimization-0.11.1.md) · [历史0.10.1中文文档ZIP](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-docs-v0.10.1.zip)

| 文件 | 用途 |
| --- | --- |
| [wechat.md](wechat.md) | 公众号：0.11.1速度、透明PNG分离、部署和历史训练结果 |
| [cnblogs.md](cnblogs.md) | 博客园：0.11.1 SIMD/行缓存/网页管线、数值及兼容边界 |
| [manifest-0.11.1.json](manifest-0.11.1.json) | 当前文章、配图与取数源摘要 |
| [manifest.json](manifest.json) | 历史0.10.1发布图文快照清单，保持原内容 |

## 当前0.11.1配图

| PNG | 内容 |
| --- | --- |
| [完整图片推理](../../artifacts/image-pipeline-0.11.1.png) | 0.11.0→0.11.1，CPU/Vulkan与大小原图 |
| [车辆分离](../../artifacts/validation/0.11.1/web/cutout-vehicles.png) | 真实照片目标与背景双预览 |
| [人物分离](../../artifacts/validation/0.11.1/web/cutout-people.png) | 同一结果切换人物类别 |
| [目标PNG](../../artifacts/validation/0.11.1/web/cutout-foreground.png) | 实际下载的透明前景 |
| [背景PNG](../../artifacts/validation/0.11.1/web/cutout-background.png) | 实际下载的透明背景 |

## 0.10.1历史训练配图

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
