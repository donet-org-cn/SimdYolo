using System.Reflection;
using SimdYolo;

/// <summary>验证 Vulkan 进程实例复用、独立设备、交错释放及反复创建不会破坏其他会话。</summary>
internal static class VulkanLifetimeTests
{
    /// <summary>注册真实设备生命周期检查；测试不共享设备、队列或显存。</summary>
    /// <param name="test">命名测试执行回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("Vulkan lifetime: process instance, independent devices and interleaved disposal", () =>
        {
            using var first = new VulkanBackend(); using var second = new VulkanBackend();
            nint instance = Handle(first, "instance");
            if (instance == 0 || instance != Handle(second, "instance") || Handle(first, "device") == Handle(second, "device")) throw new Exception("Vulkan instance/device ownership mismatch.");
            first.Dispose(); first.Dispose();
            Check(second, 2);
            for (int i = 0; i < 16; i++)
            {
                using var next = new VulkanBackend();
                if (Handle(next, "instance") != instance) throw new Exception("Vulkan process instance was recreated.");
                Check(next, i + 1); Check(second, i + 3);
            }
            second.Dispose();
            // 会话数量归零后再次加载，实例及驱动仍有效；每轮设备资源都完整释放。
            for (int i = 0; i < 32; i++)
            {
                using var next = new VulkanBackend();
                if (Handle(next, "instance") != instance) throw new Exception("Vulkan instance did not survive the final session disposal.");
                Check(next, i + 7);
            }
        });
    }

    /// <summary>读取测试用原生句柄以验证所有权，避免依赖计时或驱动日志。</summary>
    /// <param name="backend">受测计算后端。</param>
    /// <param name="field">实例或设备字段名。</param>
    /// <returns>对应的原生句柄值。</returns>
    private static nint Handle(VulkanBackend backend, string field) => (nint)typeof(VulkanBackend).GetField(field, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(backend)!;

    /// <summary>用独立标量结果核对当前设备，在其他设备释放后仍能正常计算。</summary>
    /// <param name="backend">当前仍有效的计算后端。</param>
    /// <param name="value">本次输入值，用于发现旧命令或跨设备污染。</param>
    /// <exception cref="Exception">设备结果不符合标量参考。</exception>
    private static void Check(VulkanBackend backend, float value)
    {
        var output = new float[1]; backend.MatMul([value], [3], output, 1, 1, 1);
        if (output[0] != value * 3) throw new Exception("Independent Vulkan device result changed after another session was disposed.");
    }
}
