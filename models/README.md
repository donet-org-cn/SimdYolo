# 随仓库分发的 YOLO 模型

这里包含 **11 个原始 PyTorch `.pt` 检查点和 22 个 ONNX 导出**，覆盖检测、分割和姿态。模型通过 Git LFS 保存；所有原始文件保持不变，来源、输入尺寸、opset 和 SHA256 见 [manifest.json](manifest.json)。当前 [0.9.0 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.9.0) 提供全部33个模型的 ZIP，包含 YOLO26；另有 YOLOv8 检测头微调和网页示例。

| 模型 | 任务 | `.pt` | 640×640 ONNX / opset 17 | 动态 ONNX / opset 17 | 128×128 ONNX / opset 12 |
| --- | --- | --- | --- | --- | --- |
| yolov8n | 检测 | 有 | 有 | 有 | 有 |
| yolov9t | 检测 | 有 | 有 | — | — |
| yolov10n | 检测 | 有 | 有 | — | — |
| yolo11n | 检测 | 有 | 有 | 有 | 有 |
| yolov8n-seg | 分割 | 有 | 有 | 有 | 有 |
| yolov8n-pose | 姿态 | 有 | 有 | — | — |
| yolo11n-seg | 分割 | 有 | 有 | — | — |
| yolo11n-pose | 姿态 | 有 | 有 | 有 | 有 |
| yolo26n | 检测 | 有 | 有 | 有 | — |
| yolo26n-seg | 分割 | 有 | 有 | 有 | — |
| yolo26n-pose | 姿态 | 有 | 有 | 有 | — |

普通 `.onnx` 使用固定 `[1,3,640,640]` 输入，与正式速度对比一致。`*-dynamic.onnx` 的 batch、height 和 width 为动态维度。`*-opset12.onnx` 是固定 `[1,3,128,128]` 的兼容性验证导出，运行时应指定 128，不使用默认 640。

YOLO26 的 ONNX 由 Ultralytics 8.4.0 / PyTorch 2.11.0 导出，显式保存 `task`、`names`、`nc`、`end2end` 和姿态 `kpt_shape`。端到端输出前六字段为 xyxy、分数和类别；分割后接 32 个掩码系数，姿态后接关键点。该版本导出保留固定 TopK=300，动态 ONNX 要求候选锚点数至少 300；已验证 128×128、160×160（batch=2）。PT 在小尺寸下可自动缩减结果数量。

## 下载

安装 [Git LFS](https://git-lfs.com/)，然后执行：

```powershell
git lfs install
git clone https://github.com/hnlyf168/SimdYolo.git
cd SimdYolo
git lfs pull
```

也可以从 [v0.9.0 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.9.0) 下载 [SimdYolo-models-v0.9.0.zip](https://github.com/hnlyf168/SimdYolo/releases/download/v0.9.0/SimdYolo-models-v0.9.0.zip)，解压到仓库根目录。包含全部33个模型、原许可证及摘要清单，无需 Git LFS 即可使用。历史 v0.7.1 压缩包只有当时的24个模型，未包含 YOLO26。

## 使用

```powershell
dotnet build -c Release
dotnet run --project src/SimdYolo.Cli -c Release -- inspect models/pt/yolov8n.pt
dotnet run --project src/SimdYolo.Cli -c Release -- bench models/pt/yolov8n.pt cpu 640 30 12 30
dotnet run --project src/SimdYolo.Cli -c Release -- bench models/onnx/yolov8n.onnx cpu 640 30 12 30
dotnet run --project src/SimdYolo.Cli -c Release -- bench models/onnx/yolov8n-opset12.onnx cpu 128 30 12 30
```

将 `cpu` 改为 `vulkan` 可运行 Vulkan 后端，机器需要支持 Vulkan 的显卡和驱动。Metal 的构建实现随源码提供，本次基准在 Windows/Vulkan 上验证。

## 来源与许可证

原有 `.pt` 来自 [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0)，YOLO26 来自 [assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0)，ONNX 由对应检查点导出。上游模型使用 **AGPL-3.0**，完整文本见 [LICENSE-AGPL-3.0.txt](LICENSE-AGPL-3.0.txt)，归属和来源见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。项目源码本次没有新增许可证。

现有导出和对照方法在 [tools/validate-reference.py](../tools/validate-reference.py)。Python/PyTorch/Ultralytics 只用于开发时的导出和对照；C# 主库直接加载 `.pt` / ONNX，运行时不调用它们。
