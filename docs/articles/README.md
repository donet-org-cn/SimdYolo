# SimdYolo 图文发布包

两篇文章分别写作，正文可直接编辑使用，全部文件使用 UTF-8 编码。性能引用 2026-10-14 的 0.7.0 冻结测量，下载版本为 0.7.1。

正文已补充 0.8.0 的 YOLO26 与标签读取能力，并明确其尚未进入 0.7.1 Release。当前仓库主库为 0.8.1，两篇文章开头增加了[本轮 11 模型 CPU/GPU 与首次使用对照](../../artifacts/optimization-0.8.1.md)。原正文的速度表和配图仍属于 0.7.0 历史测量，不能用作当前版本的速度结论。

更新后的图文文件可下载[0.8.1 图文压缩包](../../artifacts/SimdYolo-articles-0.8.1.zip)。历史 `SimdYolo-articles-20261014.zip` 保留原始内容。

| 文件 | 用途 |
| --- | --- |
| [wechat.md](wechat.md) | 微信公众号：项目动机、实测结果、图表与上手方式 |
| [cnblogs.md](cnblogs.md) | 博客园：兼容范围、架构、SIMD/GPU 实现、完整排名与复测命令 |
| [assets/](assets/) | 10 张 PNG 配图、同名 SVG 矢量原稿及图表数据 |

## 微信公众号建议

标题：**纯 C# 跑 YOLO：我把它和 PyTorch、ORT 放在同一台电脑上实测**

摘要：**纯 C# 直接加载 .pt 和 ONNX，GPU 采用无需 CUDA 依赖的 Vulkan 路径。附 CPU/GPU 实测、跨平台设计、模型下载和 .NET 上手代码。**

封面：`assets/01-wechat-cover.png`。封面和正文图片均为横向或纵向的独立图片，可在编辑器里按版式裁切、缩放。

导入 Markdown 到所用排版工具或微信编辑流程后，上传正文用到的 PNG，并确认图片已转为平台托管地址。导入工具是否自动上传本地图片取决于具体工具，发布前需在手机预览。源码链接可以设置为文末阅读原文。

## 博客园建议

标题：**SimdYolo：纯 C# YOLO 推理实现，CPU SIMD / Vulkan 架构与 PyTorch、ORT 实测**

摘要：**介绍 C# 模型解析、SIMD 内核，以及 Vulkan 无需 CUDA、跨平台与跨厂商的部署优势。附 8 模型 CPU/GPU 排名、内存成本与复测方法。**

封面：`assets/02-cnblogs-cover.png`。

建议标签：`C#`、`.NET`、`YOLO`、`SIMD`、`Vulkan`、`ONNX`、`性能优化`。

使用 Markdown 编辑模式粘贴正文，再把图片上传到文章编辑器，将 `assets/...png` 替换为平台提供的图片地址。正文表格、代码块和图片说明分别保留。

## 配图清单

| 文件 | 内容 | 文章使用 |
| --- | --- | --- |
| 01-wechat-cover.png | 公众号封面 | 微信公众号 |
| 02-cnblogs-cover.png | 技术文章封面 | 博客园 |
| 03-architecture.png | 模型加载、图融合和计算后端 | 两篇 |
| 04-cpu-detection.png | 检测模型 CPU 延迟 | 两篇 |
| 05-cpu-vision.png | 分割/姿态 CPU 延迟 | 两篇 |
| 06-gpu-detection.png | 检测模型 GPU 延迟 | 两篇 |
| 07-gpu-vision.png | 分割/姿态 GPU 延迟 | 博客园 |
| 08-cpu-optimization.png | 卷积分块、SIMD、缓存行对齐示意 | 博客园 |
| 09-cpu-memory.png | C# CPU RSS 相对 ORT 比值 | 博客园 |
| 10-vulkan-deployment.png | Vulkan 部署优势与各平台验证状态 | 两篇 |

PNG 用于上传，SVG 可用于后续排版与清晰导出。[benchmark-data.json](assets/benchmark-data.json)保存图表取数、聚合方式和原始报告 SHA-256。

图表中位数按两轮独立窗口聚合，细线为窗口中位数范围，未表示置信区间。原始基准包含 GPU 上传和全部回读，排除图片预处理、NMS、最终 mask 后处理和冷首帧。公众号与博客园使用相同数据，不混用历史版本。

## 发布前核对

- 上传图片并在手机或目标版式中预览，确认没有失效的本地路径。
- 保留 YOLOv9 CPU 仍落后于 ORT、YOLOv10 CPU 接近持平、Metal 未实测与 RSS 较高的说明。
- 保留 0.7.0 测量与 0.7.1 文档版本的区分；不同机器复测另行标注。
- 模型保留上游许可证；源码没有新增许可证，因此文章采用“公开仓库/源码”，没有把项目称为已按某许可证开源。

本目录提供文章与图片文件，尚未在微信公众号或博客园发布。
