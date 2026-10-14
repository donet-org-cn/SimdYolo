# 加载、首次推理与持续推理对照

此项目只用于开发验证，不进入 SimdYolo NuGet 包，也不引入主库运行依赖。控制脚本只使用 Python 标准库，计时程序只使用 C# 和 .NET 框架。

先构建当前 Release 版本，再把旧版 CLI 及其依赖冻结到单独目录。`compare-lifecycle.py` 为两个版本构建相同的计时程序，只替换 `SimdYolo.dll`，并记录程序集、模型和输入的 SHA256。

```powershell
dotnet build SimdYolo.sln -c Release
python tools/compare-lifecycle.py --baseline-dir <冻结旧版CLI目录> --candidate-dir src/SimdYolo.Cli/bin/Release/net10.0 --input <相同FP32输入文件> --repeats 2 --duration 4 --warmup 60 --size 640 --threads 12 --output artifacts/lifecycle-comparison-0.8.1.json
```

默认覆盖 11 个模型、`.pt` / ONNX、CPU / Vulkan。每轮按旧、新、新、旧的 ABBA 顺序启动独立进程，第二轮逆转模型配置顺序，共 352 个窗口。每个版本每配置有 4 个窗口。不要并行运行其他性能测试，修改被测程序集后应重新测量。

指标分别为模型与后端构造、首次 `PredictRaw`、托管 `Main` 首条语句至首次输出、预热后持续推理、每帧托管分配。准备至首次输出不包含 dotnet 宿主启动和进入 `Main` 前的入口 JIT。每帧分配不等于驻留内存；GPU 计时包含输入上传与全部输出回读。

默认继承 Vulkan 应用缓存。可以用 `--cache-mode isolated` 为每个窗口建立新的应用缓存目录，或用 `disabled` 禁用应用管线缓存；两者都不会清除显卡驱动缓存和操作系统文件缓存，不能称为硬件冷启动。

独立输出核验需要原版 CPU PyTorch、Ultralytics 8.4.0、NumPy 和仓库内模型。这些只用于对照，不是主库依赖：

```powershell
python tools/validate-lifecycle.py --report artifacts/lifecycle-comparison-0.8.1.json --ultralytics-dir <Ultralytics8.4.0包目录> --output artifacts/validation/0.8.1/lifecycle-outputs.json
python tools/render-optimization.py --runtime-report artifacts/runtime-comparison-0.8.1.json --lifecycle-report artifacts/lifecycle-comparison-0.8.1.json --validation-report artifacts/validation/0.8.1/lifecycle-outputs.json --output artifacts/optimization-0.8.1.md
```

控制脚本保留每个窗口的首帧和末帧原始输出，核验程序逐文件检查长度、形状和 SHA256。只有完整字节及参考模型、输入、形状、输出索引和校验策略均相同时，才复用数值比较结果。端到端检测头核验完整候选、类别及实例附加字段；分割原型输出也单独比较。

公开 JSON 保存测量结果、校验结果及哈希；原始 `.f32` 输出保存在实验电脑的本地 `.cache` 中，未随源码提交。其中绝对路径是当次实验记录，不代表其他电脑具有对应文件。仅下载 JSON 不能重新检查这些原始输出。

迁移已有证据时，保留原 JSON 的全部字节，不改其中的绝对路径。把输入文件和首帧、末帧的全部 `.f32` 文件复制到新电脑；原始输出文件名须保留且在 `--raw-root` 下唯一。模型按以下结构放置，并使用报告中相同 SHA256 的模型文件：

```text
models/
  pt/yolov8n.pt
  onnx/yolov8n.onnx
  ...
evidence/
  input.f32
  raw/原窗口文件名.first.0.f32
  raw/原窗口文件名.0.f32
  ...
```

```powershell
python tools/validate-lifecycle.py --report artifacts/lifecycle-comparison-0.8.1.json --model-root models --input evidence/input.f32 --raw-root evidence/raw --ultralytics-dir <Ultralytics8.4.0包目录> --output artifacts/validation/0.8.1/lifecycle-outputs-relocated.json
```

核验工具优先使用仍可读的记录路径。原 `ModelPath` 不可读时，才按 `--model-root` 下的 `pt` / `onnx` 子目录和原模型名定位，仍严格核对原报告的模型 SHA256，并拒绝路径穿越。ONNX 的官方参考 PT 默认从同一模型根目录的 `pt` 子目录读取；另有权重目录时可加 `--pt-dir`。输入和每个原始输出也必须与报告摘要、字节数和形状一致，缺失证据或摘要不符会失败。

重跑输出核验只生成新的独立校验 JSON，不改变原性能记录，也不重新计时。重新计时会产生新的原始输出和报告，不应覆盖历史实验。
