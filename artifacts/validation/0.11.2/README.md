# 0.11.2 功能与程序集验证

本版包含0.11.0模型工作流及0.11.1掩码/网页优化，并以最终冻结DLL重新比较原版PyTorch和ORT。[176个性能窗口、39,264次推理、224项输出对照](../../runtime-comparison-0.11.2.json)完成；输出容差沿用既有规则。

| 检查 | 原始证据 |
| --- | --- |
| CPU/Vulkan推理116项 | [日志](inference-tests.log) |
| 训练集成19项 | [日志](training-tests.log) |
| 关闭硬件intrinsics的掩码专项2项 | [日志](masks-software.log) |
| 完整浏览器41条流程，含大图/PNG/标注/验证/导出/恢复 | [JSON](web/web-browser.json) |
| 纯.NET包、中文XML、33模型许可证 | [包审查](package-audit.json)、[训练元数据](training-metadata.json) |
| 离线NuGet CPU/Vulkan消费 | [CPU](package-consumer-cpu.json)、[Vulkan](package-consumer-vulkan.json) |

![本版目标分离](web/cutout-vehicles.png)

本轮前向排名排除图片预处理与最终掩码；[0.11.1原图管线对照](../0.11.1/README.md)保留原来的程序集与结果。训练性能与保存模型的官方对照仍为0.10.1历史结果，本版不声称重新执行该训练矩阵。

主库、训练库与正式计时的DLL摘要在[final-audit.json](final-audit.json)，发布附件及解压运行验证见[发布验证](../0.11.2-release/README.md)。
