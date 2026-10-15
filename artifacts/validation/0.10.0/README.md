# 0.10.0 五代检测头训练验证

本目录对应当前源码的检测头微调，不包含分割/姿态、全参数训练或 Metal 训练。[完整结果、排名和图表](../../training-evaluation-0.10.0.md)区分训练与历史推理数据。0.9.0 已发布包保持原字节，新增能力从0.10.0源码或本地构建包使用。

## 范围与证据

- 实测权重：`yolov8n.pt`、Ultralytics `yolov9t.pt`、`yolov10n.pt`、`yolo11n.pt`、`yolo26n.pt`。其他尺度未逐份验证，原 YOLOv9 DualDDetect/TripleDDetect 不在本轮支持范围。
- [最终数值矩阵](oracle-matrix.json)：34组、19,620项；每组 `oracle-模型-设备-场景.json` 保存逐项结果、容差、官方源码和实际 DLL SHA256，`fixture-*.json`保存张量形状/索引。FP32张量保存在`.cache/training-010`，可通过脚本从原始模型重新生成。
- [CPU100轮](integration-cpu.json)、[Vulkan100轮](integration-vulkan.json)：完整单/双头都有更新，固定口径评估损失下降，原生归档可推理，取消恢复后参数/BN和动量在既定数值容差内一致；载入的动量字节必须完全相同。
- [训练回归日志](training-tests.log)：16项，包括普通/分组/深度卷积与BN有限差分、SGD、空标签、结构能力探测、坏归档、旧格式、取消重入等。[推理回归](inference-tests.log)：117项，CPU/Vulkan及实际PT。
- [性能原始数据](performance/training-runtime-comparison.json)：5模型×4引擎×2轮=40窗口、20,480次计时更新；各窗有原始指标、资源采样与日志。`fixture-check-*.json`为128尺寸的五组、2,712项额外数值检查，使用与速度比较完全相同的初始数据。
- [主库XML与托管审查](core-xml-audit.json)、[训练库审查](training-xml-audit.json)、[本地NuGet包](packages.json)、[独立包消费](package-consumers.json)。不以ProjectReference代替包消费。
- [网页五代训练矩阵](web-matrix.json)、[网页完整API回归](web-api.json)、[真实浏览器](web-browser/web-browser.json)：实际训练、恢复、下载上传、推理和移动布局；截图与真实结果同时保留。
- [交付总审查与文件摘要](final-audit.json)核对39组共22,332项数值检查、最终DLL/包、33模型原始SHA256，以及UTF8文档和图链。

## 固定数值口径

官方参考为本地固定的 Ultralytics **8.4.0**、CPU-only PyTorch **2.11.0**，不从浮动main直接推断兼容。参考 `head.py/loss.py/tal.py/metrics.py` 和验证工具摘要在JSON内。v10使用官方兼容`E2EDetectLoss`，YOLO26使用`E2ELoss`及逐轮`update()`。

| 检查 | 固定绝对容差 + 相对容差×参考绝对值 |
| --- | --- |
| 头前向、输出/参数梯度、SGD动量 | 5e-4 + 2e-3 × \|参考\| |
| loss | 1e-4 + 3e-4 × \|参考\| |
| BN运行统计、SGD后权重 | 5e-5 + 3e-4 × \|参考\| |
| 标签分配数量、梯度连接状态、配置 | 完全一致 |
| 未更新参数、BN批次计数 | 完全一致 |

调度权重允许1e-7+1e-7×参考的float表示误差。恢复训练后的参数/BN使用2e-5绝对容差、动量使用上表梯度容差；这与**动量载入字节一致**是两个检查。CPU重新提取特征时存在FP32舍入，不能把数值一致说成跨会话逐位一致。

## 复现

先准备仓库33份实际权重和`.NET 10 SDK`。开发对照额外需要固定官方源码位于`.cache/yolo26-release-reference`、CPU-only PyTorch/NumPy及隔离的CUDA PyTorch位于`.cache/torch-cu126`；本机CUDA参考为2.11.0+cu126。这些不是库或网页的运行依赖。

```powershell
dotnet build SimdYolo.sln -c Release
dotnet tests/SimdYolo.Training.Tests/bin/Release/net10.0/SimdYolo.Training.Tests.dll test-matrix .cache/training-010/vulkan --epochs 100 --vulkan
python tools/benchmark-training.py
python tools/validate-training-build.py
```

性能脚本串行40窗口，固定初始权重/缓存特征、128×128、FP32、2类、4线程、预热10步、每窗512次更新。CPU/PyTorch-CUDA/.NET-Vulkan均记录真实计算，CUDA每步同步，Vulkan包含GEMM上传/回读；资源只在`measuring`至`measured`区间采样。GPU/显存为整卡，RSS为进程，不混用归属。

构建验证工具重新执行16项训练回归、117项推理回归、CPU100轮、最终34组数值矩阵，审查XML，并构建/独立消费0.10.0本地NuGet。全部结果绑定实际DLL；旧0.9.0包不覆盖。

网页另开一个终端，指定新的数据目录：

```powershell
dotnet run --project examples/SimdYolo.Web -c Release -- --urls http://127.0.0.1:5198 --SimdYolo:DataRoot F:/simdyolo/.cache/training-010/web
python tools/validate-web-training-matrix.py --url http://127.0.0.1:5198
python tools/validate-web.py --url http://127.0.0.1:5198 --image .cache/bus.jpg
node tools/validate-web-browser.mjs --url http://127.0.0.1:5198 --image .cache/bus.jpg --training-model yolo26n.pt --output artifacts/validation/0.10.0/web-browser
```

浏览器验证使用自有无头Edge实例，在finally中关闭；`--browser`可指定其他Chromium。随后执行`python tools/render-training-evaluation.py`生成报告与PNG/SVG。

## 评测边界

100轮样本是固定单张合成方块图，只证明真实梯度更新、可拟合性和生命周期；没有COCO/业务留出集mAP结果。YOLO26评估固定0.8/0.2权重，逐步曲线则保留实际变化的训练目标。速度比较使用同一预训练head仅替换分类末层、官方PyTorch模块/loss/SGD，不是完整Ultralytics train数据管线，也不是全参数反向。

Vulkan头部深度卷积、BN、激活、标签分配和SGD仍在CPU，因此不能声称全程GPU驻留或全面超越PyTorch。普通ORT未安装训练组件，不以其推理耗时充当训练基线。原始模型继续保留AGPL-3.0及来源，项目源码未新增许可证。
