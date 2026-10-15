# SimdYolo 0.9.0 图文发布包

微信公众号和博客园分别写作，正文、表格、PNG/SVG与数据清单均更新为**0.9.0当前程序集实测**，不再沿用0.7.x图表。所有文字文件使用UTF-8。

[0.9.0 发布与33模型下载](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.9.0) · [完整性能报告](../../artifacts/runtime-comparison-0.9.0.md) · [文档/图文ZIP](https://github.com/hnlyf168/SimdYolo/releases/download/v0.9.0/SimdYolo-docs-v0.9.0.zip)

| 文件 | 用途 |
| --- | --- |
| [wechat.md](wechat.md) | 公众号：项目动机、当前实测、部署和网页版上手 |
| [cnblogs.md](cnblogs.md) | 博客园：模型解析、SIMD/Vulkan、完整排名、资源和训练边界 |
| [assets/0.9.0/](assets/0.9.0/) | 当前版本10张PNG、10份SVG及独立取数清单 |
| [manifest.json](manifest.json) | 当前文章、图片、版本、源报告SHA256及检查结果 |

## 当前配图

| 文件（均在 assets/0.9.0/） | 内容 |
| --- | --- |
| 01-wechat-cover.png | 公众号封面，0.9.0推理与检测头微调 |
| 02-cnblogs-cover.png | 博客园技术文章封面 |
| 03-architecture.png | 模型解析、推理、冻结特征、检测头训练与网页 |
| 04-cpu-detection.png | 5个检测模型的CPU对比 |
| 05-cpu-vision.png | 6个分割/姿态模型的CPU对比 |
| 06-gpu-detection.png | 5个检测模型的GPU对比 |
| 07-gpu-vision.png | 6个分割/姿态模型的GPU对比 |
| 08-cpu-optimization.png | 当前CPU路径结构示意，不表示单项收益 |
| 09-cpu-memory.png | 11模型进程RSS，包含宿主运行库 |
| 10-vulkan-deployment.png | Vulkan运行环境、CUDA依赖与验证范围 |

正文还引用当前版本的加载/首帧、CPU时间、GPU/显存图，以及真实网页截图；文档ZIP包含这些文件。SVG中文字转为轮廓，方便导出；PNG用于文章上传。

当前图表来自固定实拍图、FP32、batch=1、640×640、12线程、两轮独立窗口；每窗口预热60次，计至少4秒/40帧。窗口范围不是置信区间。额外随机输入的ORT CUDA数值超差单列诊断，正式图片基准不混入其他输入。

## 发布建议

公众号标题：**纯 C# 跑 YOLO：推理、检测头微调和网页版，这次一起交付**。

博客园标题：**SimdYolo 0.9.0：纯 C# YOLO 推理、检测头微调与 CPU/GPU 实测**。

将Markdown导入编辑器后，上传正文引用的PNG，把相对图片链接替换为平台托管地址，并在手机预览。公众号可将仓库地址设为“阅读原文”；博客园建议标签 `C#`、`.NET`、`YOLO`、`SIMD`、`Vulkan`、`ONNX`。

历史 `SimdYolo-articles-20261014.zip` 和 `SimdYolo-articles-0.8.1.zip` 保持原始内容，旧 assets 根目录图片也保留历史用途；当前正文只使用版本化目录中的新图。
