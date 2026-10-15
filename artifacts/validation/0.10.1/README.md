# 0.10.1 训练优化与保存模型对照证据

[完整中文报告与图表](../../training-optimization-0.10.1.md)区分128尺寸的训练内核速度与256尺寸的真实图片训练结果。冻结骨干/FPN，仅更新五代检测头；没有全参数、分割/姿态或Metal训练。

- [总验证与文件摘要](validation-summary.json)：绑定最终主库与训练库DLL；[训练回归日志](training-tests.log)、[推理回归日志](inference-tests.log)。新增转置点积尾部、工作区复用/累加、非有限值检查。单图收敛流程从100步延长到200步，保留10%损失下降要求，不修改单步数值容差。
- [34组官方数值矩阵](oracle-matrix.json)：普通/小目标/全背景、CPU/Vulkan、五模型，YOLO26中间/末轮调度；19,620项原有严格数值检查。性能夹具另有五组、2,712项检查。
- [速度与资源原始数据](performance/training-runtime-comparison.json)：5模型×4后端×2独立串行窗口，40窗、20,480次计时更新。128×128、FP32、2类、4线程，预热10步，每窗512步；GPU包含上传回读、CUDA每步同步。排除骨干、预处理和保存。GPU/显存为整卡采样，RSS为进程。
- [真实图片结果总索引](outcomes/comparison.json)：10组训练结果。每份`outcomes/模型-设备.json`记录全部第1/16/96/192步参数/BN误差、动量误差、原始预测误差、逐步损失和保存模型指标；CPU对PyTorch CPU，Vulkan对PyTorch CUDA。两边保存的模型实际重新打开，执行完整骨干和检测头。
- [主库中文XML和托管审查](core-xml-audit.json)、[训练库审查](training-xml-audit.json)。主库与训练库仍是纯C#；系统GPU API通过P/Invoke访问，不依赖Python、PyTorch、TorchSharp、CUDA或BLAS运行库。

真实数据使用[官方COCO128](https://docs.ultralytics.com/datasets/detect/coco128/)，下载地址固定为`https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`，原ZIP SHA256为`61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`。固定种子2611划成96张微调、32张验证，各2轮、192次SGD、80类、256×256。数据manifest保留每张原图片/原标注SHA256，输入与冻结特征由C#导出，官方参考使用相同字节。原始图片、FP32中间张量和实际训练检查点留在`.cache/training-011`，可按脚本重建；未向仓库混入大体积数据集和临时模型。

普通单步容差仍为前向/梯度/动量5e-4+2e-3×|参考值|，损失1e-4+3e-4×|参考值|，BN/更新后参数5e-5+3e-4×|参考值|。多步参数/BN单独采用预先固定的0.002+0.01×|参考值|，P/R/mAP绝对差0.02，原始误差不会因通过门槛而省略。多步TAL离散匹配可造成轨迹分叉，不声称参数逐位一致。

多步结果明确区分完成与通过：`status=completed`仅表示实验完成；`directParameterComparisonPassed`记录CPU/CPU或Vulkan/CUDA的直接参数门槛，`metricComparisonPassed`记录保存模型指标门槛。v8 Vulkan/CUDA参数门槛未通过，保留false；新增`pytorchCpuVsCuda`对照确认原版自身同量级漂移，`cpuParameterControl`按原门槛检查C#两后端与固定PyTorch CPU参考，未扩大容差。总验证通过指实现控制、原有单步检查与指标检查，不代表所有直接参数比较通过。

实际模型保存在`.cache/training-011/outcomes/模型名/torch/trained.pt`、`torch-cuda/trained.pt`及`cpu/native/last.sytrain`、`vulkan/native/last.sytrain`，对应SHA256写入各报告。模型在本地可直接再次加载；这些小样本微调模型不替换仓库原始预训练权重。

32张只对本次微调留出；COCO128来自预训练使用的COCO train2017。因此这属于实现一致性诊断，不能证明未见数据泛化。固定低学习率的短程微调仍出现验证精度下降，报告同时保留训练前后值；没有选择最好轮次替代固定第192步，也没有据此宣称训练配方有效。

## 复现

准备.NET10、33份原模型、固定`.cache/yolo26-release-reference`官方Ultralytics8.4.0源码、CPU-only PyTorch2.11.0，以及隔离的`.cache/torch-cu126` CUDA2.11.0+cu126。NumPy/Pillow/psutil/Matplotlib仅为开发验证依赖。

```powershell
python tools/validate-training-optimization.py
python tools/render-training-optimization.py
```

数据集需先下载并安全解压到`.cache/training-011/coco128`。`tools/compare-training-outcomes.py`自动建立固定拆分和RGB缓存；`--resume`只复用匹配当前训练DLL的C#结果，并复用同一初始夹具的官方结果。完整验证脚本串行执行数值、性能和模型结果测试，不让其他ML测试干扰性能窗口。

历史0.10.0与0.9.0报告、NuGet包和公开Release均保留原字节；0.10.1源码与[发布包](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1)均包含本轮新增能力，附件验证见[发布记录](../0.10.1-release/README.md)。
