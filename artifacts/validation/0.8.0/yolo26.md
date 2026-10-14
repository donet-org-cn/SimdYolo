# 0.8.0 YOLO26 与标签读取验证

本次新增 YOLO26 检测、实例分割、姿态推理，以及独立 PT/ONNX 类别标签读取。主库继续只引用 .NET 框架程序集，Python、PyTorch、Ultralytics、ONNX Runtime 仅用于开发对照，不参与 C# 运行。

## 已验证范围

| 官方模型 | PT CPU / Vulkan | ONNX CPU / Vulkan | 标签 | 640 输入端到端输出 |
| --- | --- | --- | --- | --- |
| yolo26n | 通过 | 通过 | 80 类 | `[1,300,6]` |
| yolo26n-seg | 通过 | 通过 | 80 类 | `[1,300,38]` 与 `[1,32,160,160]` |
| yolo26n-pose | 通过 | 通过 | 1 类 person | `[1,300,57]`，17 个三维关键点 |

使用 assets v8.4.0 发布的原始权重，开发参考为 Ultralytics 8.4.0 / PyTorch 2.11.0 / ONNX 1.21.0 / ONNX Runtime 1.26.0。检查点自身记录的训练版本为 8.3.222，不等于本次参考包版本。权重和导出文件的 SHA256 见 [模型清单](../../../models/manifest.json)。GPU 为 GTX 1050 Ti，CPU 为 i7-8700K；Metal 没有本机 macOS 验证。其他 YOLO26 大小、分类和 OBB 不在本次实测范围。

主要适配包括：单桶直接回归边距、SPPF 的可配置池化次数和残差、Proto26 的多尺度融合、Pose26 的坐标投影和半像素锚点、one2one 分支，以及六字段后面的掩码系数或关键点。已知训练损失分支不参与 PT 推理；没有放宽任意 Python 类的执行白名单。PT 端到端标记兼容头属性、融合分支和旧版 YAML 配置。

## 数值与真实图片

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| 固定 128×128，PT/ONNX CPU/Vulkan 及 ORT | 20/20 个输出通过 | [static128](yolo26-static128.json) |
| 固定 640×640，同上 | 20/20 个输出通过 | [static640](yolo26-static640.json) |
| 动态 ONNX，128 导出、160 输入、batch=2 | 20/20 个输出通过 | [dynamic-batch2](yolo26-dynamic-batch2.json) |
| 旧 YOLOv8/v9/v10/11 的八个 PT，CPU/Vulkan | 20/20 个输出通过 | [legacy-models](legacy-models.json) |
| 真实图片，3 个任务 × PT/ONNX × CPU/Vulkan | 12/12 个配置、52 个实例通过 | [PT CPU](yolo26-image-pt-cpu.json)、[PT Vulkan](yolo26-image-pt-vulkan.json)、[ONNX CPU](yolo26-image-onnx-cpu.json)、[ONNX Vulkan](yolo26-image-onnx-vulkan.json) |
| 仓库所有 PT/ONNX 类别标签与来源 PT 交叉检查 | 33/33 个文件通过 | [model-labels](model-labels.json) |
| 连续改变输入，检查输出数组所有权和设备命令行为 | 6/6 个模型配置通过 | [replay](yolo26-replay.json) |
| CPU/Vulkan 单元及既有真实模型会话回归 | 101 项通过 | [test-vulkan](test-vulkan.log) |
| 关闭硬件 intrinsics 的可移植回归 | 78 项通过，硬件专项受 guard 限制 | [test-portable](test-portable.log) |
| 标签解析和不读取权重存储的专门回归 | 6 项通过 | [test-metadata](test-metadata.log) |

三个新 PT 模型均可在预热后重放 Vulkan 命令，五帧测试各重放三帧。当前新 ONNX 导出头包含需要主机处理的索引节点，自动使用普通执行路径，未捕获命令重放；五帧结果与关闭重放的会话一致，误差为零。该限制保留在记录中，不能把 PT 的重放状态推广到 ONNX。

60 个 YOLO26 原始输出检查中，48 个来自 C# CPU/Vulkan，12 个来自 ORT CPU。普通张量逐元素比较，`atol=0.003, rtol=0.0003`。端到端 TopK 验证分两部分：排序后的分数与参考 TopK 比较（`atol=2e-7, rtol=0.0003`），每个输出框及其类别、掩码系数/关键点再与参考的完整候选集合做唯一配对。配对禁止同一锚点/类别重复使用。这样同时检查候选数、得分和所有实例字段，允许 FP32 近似同分项在截断点或顺序上变化。

真实图片中的框最大误差约 0.000542 原图像素，关键点最大误差约 0.000728 像素；16 个掩码的最低二值 IoU 为 0.99908369。图像后处理以共同 C# letterbox 输入为基准，Python 独立计算框、关键点和掩码，不比较不同图片解码/resize 库产生的输入。

本次是兼容性和正确性验证。未重新测量性能排名，历史报告仍绑定 0.7.0 冻结程序集。

## 标签读取

`ModelInfo.Read(path)` 不创建计算后端。PT 只解释 `data.pkl` 和张量形状描述；测试将官方 Pose26 的所有张量存储条目移除后，仍能读取 `person`、任务、关键点和端到端标记。ONNX 流式跳过节点与权重，不读取 external data 文件。执行加载器与元数据接口共用标签解析规则。

支持 JSON/Python 字典、列表、元组，转义引号、反斜线、Unicode 及稀疏编号。缺少名称返回空映射；不会预设 COCO。负数、重复编号、非字符串名称和表达式均拒绝。冻结映射与形状副本防止调用者修改模型元数据。`ClassCount` 表示类别通道数，不能用稀疏标签的条目数替代。

```csharp
using SimdYolo;
ModelInfo info = ModelInfo.Read("models/pt/yolo26n.pt");
Console.WriteLine(info.ClassNames[0]); // person
Console.WriteLine(info.ClassCount);    // 80
```

```powershell
dotnet run --project src/SimdYolo.Cli -c Release -- labels models/onnx/yolo26n.onnx
dotnet run --project src/SimdYolo.Cli -c Release -- inspect models/pt/yolo26n-pose.pt
```

## 复现

官方架构依据：[YOLO26 文档](https://docs.ultralytics.com/models/yolo26/)、[v8.4.0 head.py](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/nn/modules/head.py)、[v8.4.0 block.py](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/nn/modules/block.py)。模型保留上游 AGPL-3.0；项目源码没有新增许可证。

```powershell
dotnet build -c Release
dotnet run --project tests/SimdYolo.Tests -c Release -- --vulkan --models models/pt

# 以下 Python 命令仅用于开发验证；先安装或选择 Ultralytics 8.4.0 的参考环境。
python tools/validate-reference.py models/pt/yolo26n.pt models/pt/yolo26n-seg.pt models/pt/yolo26n-pose.pt --onnx --ort --size 640
python tools/validate-reference.py models/pt/yolo26n.pt models/pt/yolo26n-seg.pt models/pt/yolo26n-pose.pt --onnx --ort --dynamic --size 160 --batch 2 --export-size 128
```

随仓库的动态 ONNX 来自该版本固定 TopK=300 的导出，候选锚点数必须至少 300。图像 `Predict/Detect` 使用 batch=1；原始 `PredictRaw` 与 CLI `raw ... [batch]` 可使用动态模型支持的批量。
