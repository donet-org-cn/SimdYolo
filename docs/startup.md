# 加载、首次推理与会话稳定性（源码 0.11.3）

0.11.3 优化 PT 元数据读取、BN 权重准备和 ONNX 执行工作区，并提供可选的快速启动发布配置。最新正式下载仍为 [v0.11.2](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.11.2)；本页描述后续源码，历史附件没有被替换。

[优化前后实测及图表](../artifacts/startup-optimization-0.11.3.md) · [验证证据](../artifacts/validation/0.11.3/README.md)。本轮分别比较旧版与新版普通 IL，以及同一新版源码的 IL 与 ReadyToRun。PyTorch/ORT 排名使用 [0.11.2 的独立实测](../artifacts/runtime-comparison-0.11.2.md)，不能把后续源码的耗时套入历史排名。

## 普通使用

NuGet 主库和普通构建继续提供托管 IL，CPU 使用 C# SIMD。PT 的操作码、整数和文本读取借用只读元数据视图，保存的字节对象仍复制到独立数组；BN 融合在一次 SIMD 扫描内复制并缩放，保留原权重。ONNX 在加载时编译整数槽位、节点参数和存活边界；相同输入形状只分析一次常量依赖，减少逐帧字典、LINQ 与数组分配。

ONNX 会话在正常返回及异常退出后清空工作区。常量公开输出保持独立副本；动态输入形状计划最多保留 8 个，超过上限的形状只用于当前调用。模型及权重不使用按路径全局缓存，更换文件后新建会话会重新读取。

Vulkan 保留进程唯一的 `VkInstance`，减少会话切换时的驱动加载成本，避免反复卸载驱动模块。各会话仍有独立设备、队列、命令和显存，`Dispose` 会等待计算完成并释放这些资源；一个小型进程实例及其驱动映射随进程退出回收。多会话交错释放和反复创建有专门的真实设备测试。

网页枚举模型时使用一次读取 API，同时获取标签和训练能力：

```csharp
bool trainable = YoloTrainer.CanTrain(path, out ModelInfo info);
Console.WriteLine($"{info.Task}: {info.ClassCount} classes; trainable={trainable}");
```

此重载位于 `SimdYolo.Training`。只需要标签时仍可调用主库 `ModelInfo.Read(path)`。能力识别依据实际检测头，文件名不参与判断；普通 ONNX、分割和姿态不声明检测头训练能力。

## 快速启动发布

```powershell
dotnet publish examples/SimdYolo.Web -c Release -r win-x64 -p:PublishProfile=FastStartup -o .cache/fast-startup-web
dotnet .cache/fast-startup-web/SimdYolo.Web.dll --contentRoot .cache/fast-startup-web --urls http://127.0.0.1:5188

dotnet publish src/SimdYolo.Cli -c Release -r win-x64 -p:PublishProfile=FastStartup -o .cache/fast-startup-cli
dotnet .cache/fast-startup-cli/SimdYolo.Cli.dll inspect models/pt/yolov8n.pt
```

这个可选配置使用 [.NET SDK 的 ReadyToRun](https://learn.microsoft.com/en-us/dotnet/core/deploying/ready-to-run)，提前编译部分托管代码以减少首次 JIT。应用仍由 C#/.NET 实现，依赖对应 .NET 10 运行时，网页还需要 ASP.NET Core 10；没有增加 Python、ORT 或原生数值库。**发布目录中的程序集包含 IL 和平台相关预编译代码，体积会增加；NuGet 主库保持普通托管 IL。** 如需其他平台，请按目标指定 RID 并在该平台验证；本轮实机覆盖 Windows x64、CPU 与 Vulkan，Metal 没有新增实机结论。

不设置 `-r` 时配置使用当前 SDK 的主机 RID。预编译不等于消除所有 JIT，SIMD、泛型及运行时特化仍可能产生首次开销；持续推理性能也需要实测。不要把平台专用发布目录中的 DLL 当作跨平台 NuGet 主库分发。

## 计时口径与复现

`ModelLoadMs` 包含 `YoloDetector` 构造、后端创建、模型解析和权重准备。`FirstInferenceMs` 是第一次 `PredictRaw`；`ReadyToFirstOutputMs` 从工作进程托管 `Main` 首条语句到首次原始输出，包含输入读取，不包含 .NET 宿主启动及进入 Main 之前的 JIT。GPU 包含输入上传和全部输出读回；这些指标不包含图片缩放、NMS、最终掩码或浏览器绘制。

每个窗口启动新进程，以 ABBA 顺序交错运行旧、新、新、旧，保存首帧/末帧、采样和 SHA256。默认继承 Vulkan 应用缓存；独立应用缓存测试也不会清除操作系统文件缓存和显卡驱动缓存，不能称为硬件冷启动。

先取得旧版 `SimdYolo.dll` 及同目录依赖，准备相同的真实图片 FP32 输入，然后运行：

```powershell
./tools/run-startup-comparison.ps1 -BaselineDirectory <旧版运行目录>
python tools/validate-lifecycle.py --report .cache/startup-0.11.3/source.json --threads 12 --ultralytics-dir <Ultralytics8.4.0目录> --output .cache/startup-0.11.3/source-outputs.json
python tools/validate-lifecycle.py --report .cache/startup-0.11.3/publish.json --threads 12 --ultralytics-dir <Ultralytics8.4.0目录> --output .cache/startup-0.11.3/publish-outputs.json
```

输入默认位于 `.cache/image-validation/input.f32`，是 640×640、RGB、NCHW、小端 FP32，需与报告中的 SHA256 一致。输入来源和预处理见历史性能报告及对应文档包。Python、NumPy、官方 PyTorch/Ultralytics 仅用于开发对照，不进入产品运行依赖。

参考线程数须与本轮12线程实测一致。另保留的4线程参考在旧版两个ONNX分割窗口出现严格候选匹配失败；新版窗口通过。改用12线程后全部通过，数值门槛没有变化，详见验证记录。

连续会话检查不参与速度排名，额外进行全量输出有限性/容差扫描、保留首帧、回收后内存比较及反复创建释放：

```powershell
dotnet run --project tools/SessionStability -c Release -- models/pt/yolo26n-seg.pt Cpu .cache/image-validation/input.f32 .cache/stability-seg.json 300
```

内存检查比较预热后、强制 GC 后的存活托管字节，门槛为增长不超过 32 MiB；工作集峰值另记。它能发现本轮场景中的持续保留问题，不代表任意数据、任意设备的长期内存上限。
