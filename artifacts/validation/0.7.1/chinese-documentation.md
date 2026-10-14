# 0.7.1 中文代码文档验证

0.7.1 补齐中文代码注释，并随 NuGet 包分发 `lib/net10.0/SimdYolo.xml`。安装包后，IDE 可显示中文类、方法、参数、返回值、属性与异常说明。

| 检查 | 结果 |
| --- | --- |
| 主库声明（类型、构造、方法、属性和枚举成员） | 625/625 有中文 XML 文档 |
| 测试辅助声明 | 144/144 有中文 XML 文档 |
| 参数、非 void 返回值与属性 value | 无缺项 |
| 编译生成的 XML 成员（含字段与 record 属性） | 1,064 个，均有中文说明 |
| 92 个 C# 源文件的语法 token 对比 | 无变化 |
| 0.7.0 与 0.7.1 的方法签名、IL、局部变量和异常区域 | 1,636 个全部一致 |
| Release 构建 | 0 警告、0 错误 |
| CPU/Vulkan 回归与真实模型测试 | 90 项通过 |

方法使用 `<summary>`、`<param>`、`<returns>`、`<exception>`；属性另有 `<value>`。异常描述实现实际抛出或重要调用链传播的条件；Vulkan 原生错误返回使用 VkResult 说明。局部函数使用普通中文注释，避免生成无效 XML。

算法说明覆盖 SIMD 向量和尾部、物理缓存行对齐、GC 固定、FP32 特殊值、线程发布/唤醒屏障、张量池与公开输出所有权、PT/Pickle/ONNX 读取、融合保护与回退，以及 GPU 字节偏移、缓冲区视图、同步、命令重放和资源释放。

源码注释与版本号会改变程序集文件哈希。0.7.0 冻结 DLL SHA-256 为 `51d21ba2bca561f3a21efdc3ab11be08814ae32c6510cdc910bd0338d0aaccfe`；0.7.1 DLL 为 `1ae5bc3eeaede9953c8f160ca73a060f9d755275d346acb6a46a42acf3887011`。速度排名与资源报告仍对应 0.7.0 的正式测量，不能把旧 DLL 哈希写成新包哈希。方法指令一致性用于核对本次文档改动，没有重新生成性能测量窗口。

[覆盖数据](chinese-documentation.json)、[代码 token 基线](code-token-hashes.json)、[方法 IL 审计](il-audit.json)、[构建日志](build.log)、[CPU/Vulkan 与真实模型测试](test-vulkan.log)。Metal 文档随源码提供，当前 Windows 环境没有 macOS/Metal 实机回归。

本地 NuGet 消费者在 CPU/Vulkan 各验证 4 个检测、动态 ONNX、分割、姿态模型，每模型连续 4 帧；输出有限值与公开结果存储独立性通过。[包审计](package-audit.json)、[CPU 消费者](consumer-cpu.log)、[Vulkan 消费者](consumer-vulkan.log)。
