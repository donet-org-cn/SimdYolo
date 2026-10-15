# 网页推理与训练示例

ASP.NET Core 后端直接调用本仓库的纯 C# 推理和训练库。前端为原生 HTML/CSS/JavaScript，不使用 CDN，也没有 npm 构建步骤。

当前发布版0.10.1优化CPU SIMD与Vulkan混合训练，五代检测模型入口保持一致。[训练速度及保存模型结果对比](../../artifacts/training-optimization-0.10.1.md)区分训练内核计时和真实图片验证指标；[0.10.1运行包](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-Web-v0.10.1.zip)已包含五代检测头训练，模型另下[33模型包](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-models-v0.10.1.zip)，按[启动说明](../../docs/releases/web-runtime.md)运行。

## 启动

在仓库根目录运行：

```powershell
git lfs pull
dotnet run --project examples/SimdYolo.Web -c Release
```

打开 **http://127.0.0.1:5188**。需要 .NET 10 SDK；Vulkan 需要显卡驱动支持。CPU 路径不需要 GPU、CUDA 或 Python。

![0.10.0 推理页面](../../artifacts/validation/0.10.0/web-browser/web-inference.png)

![0.10.0 YOLO26 实际训练页面](../../artifacts/validation/0.10.0/web-browser/web-training.png)

## 推理

1. 选择仓库中的 PT/ONNX 模型，也可上传 `.pt`、`.onnx` 或 `.sytrain`。
2. 上传 JPEG/PNG，选择 CPU SIMD 或可用 GPU 后端，点击运行。
3. 检测显示框和类别，分割叠加掩码，姿态显示关键点。训练生成的模型出现在同一列表。

浏览器将图片解码成 RGB24，后端执行真正的模型推理。界面展示预处理、前向和后处理的服务端耗时，不含 HTTP 传输；API 另外返回 `modelLoadMs` 和 `reusedSession`。最近两个模型会话按模型/设备/置信度复用；切换后端或第三个会话可能重新加载，首帧仍可能包含 JIT 和 GPU 编译成本。

## 训练自己的检测头

先选择内置的八张方块数据，使用 `yolov8n.pt`、CPU、128×128、20 轮和学习率 0.003，验证完整流程。内置数据用于演示，不代表真实业务精度。

导入自己的数据时填写类别列表，并选择包含图像和 YOLO `.txt` 的目录：

```text
my-data/
  images/
    001.jpg
    002.png
  labels/
    001.txt
    002.txt
```

标注格式为每行 `class cx cy w h`，类别从 0 开始，坐标归一化。支持 `images`/`labels` 对应相对路径和同名文件；缺少匹配标签的图像导入为空标签背景样本，请核对目录与页面导入数量。若同名文件存在歧义，页面会提示。分割多边形和姿态标签不能作为本阶段训练标注。

创建任务后显示实际 step、轮数和 Total/Box/Cls/DFL或L1损失。取消保留最近完整轮次；继续训练填写新的**总轮数**，从该检查点恢复。下载得到自包含 `.sytrain`，可重新上传、用 `YoloDetector` 推理，或继续微调。

0.10.1 源码与发布包支持 YOLOv8/v9/v10/YOLO11/YOLO26 检测头训练，训练下拉列表按真实 PT 结构识别能力，重命名文件不影响判断。v10、YOLO26 同时更新 one2many/one2one；骨干和 FPN 冻结。梯度累加不等于联合批次 BN，Vulkan 是普通卷积 GEMM 加速的混合路径；分割、姿态暂仅用于推理，Metal 不支持训练。旧0.9.0包保持原内容，仅支持v8训练。详细 API、支持矩阵与评测见[训练说明](../../docs/training.md)。

## 数据、任务与配置

默认输出 `.cache/web-example/`：`models/` 保存上传模型，`datasets/` 保存导入的 RGB/标注，`runs/任务编号/` 保存状态、历史和检查点。重启会载入记录；中断任务标记为已取消，可从完整检查点恢复。原仓库模型保持原文件。

| 配置键 | 用途 |
| --- | --- |
| `SimdYolo:RepositoryRoot` | 仓库路径，默认从程序目录向上查找 `SimdYolo.sln` |
| `SimdYolo:DataRoot` | 上传、数据集和任务保存位置，默认仓库 `.cache/web-example` |

例如发布到独立目录后，可通过环境变量 `SimdYolo__RepositoryRoot` 指向完整仓库。模型文件必须是 Git LFS 拉取后的实际权重。

示例限制：单模型上传 64 MiB、单图最大 2048×2048、单次数据集最多 200 张且 RGB 合计不超过 48 MiB；训练任务串行、最多四个等待任务，推理串行并缓存两个会话。较大的数据集应在自己的 C# 程序中调用训练 API。

这是绑定本机回环地址的示例，没有账号/鉴权或公网部署配置；不要直接将该训练服务暴露到公网。网页中的 CPU/Vulkan 状态由本机实际探测，Metal 推理仍需要 macOS；本机未验证 Metal。

## 验证

```powershell
dotnet run --project tests/SimdYolo.Training.Tests -c Release -- --integration --vulkan
# 启动服务后，开发环境可用 Python + Pillow 验证 API，Node 22+ 和 Edge/Chromium 验证界面：
python tools/validate-web.py --version 0.10.1 --output artifacts/validation/0.10.1/web-api.json --image .cache/bus.jpg
node tools/validate-web-browser.mjs --image .cache/bus.jpg
```

`--image` 应指向包含人的真实照片；现有验证使用 Ultralytics 官方 `bus.jpg`。浏览器工具默认 Windows Edge，可通过 `--browser` 指定 Chromium 路径。Python、Pillow、Node 与浏览器仅为验证工具，应用运行不依赖它们。
