# 0.8.1 独立 NuGet 消费验证

此程序只通过本地 NuGet 源引用 `SimdYolo 0.8.1`，没有项目引用。模型需通过 Git LFS 完整下载，系统需有可用的 Vulkan 驱动。在仓库根目录执行：

```powershell
dotnet run --project tools/PackageConsumer/PackageConsumer.csproj -c Release
```

它要求实际加载的主库哈希与本轮正式测量程序集相同，核对 33 份标签映射，然后对 YOLO26 检测、分割、姿态的 PT / 动态 ONNX 在 CPU / Vulkan 各运行三个变更输入，共 12 配置。检查输出数量、形状、有限值和调用方持有的历史输出没有被覆盖。

结果写入 `artifacts/validation/0.8.1/package-consumer.json`，记录安装包与实际程序集的 SHA256。这是固定版本的交付烟雾验证，不计时，也不替代原版数值对照。本程序和其 NuGet 配置均不进入主库包。
