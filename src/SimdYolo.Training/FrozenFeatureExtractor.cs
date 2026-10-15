namespace SimdYolo.Training;

/// <summary>在独立会话中运行冻结的 PT 骨干，并将特征安全地移交训练头。</summary>
internal sealed class FrozenFeatureExtractor : IDisposable
{
    internal readonly IComputeBackend Backend;
    internal readonly PtModel Model;
    private readonly CpuTensorPool pool = new();
    /// <summary>从原始 PT 字节创建独立冻结网络；构造失败时释放已创建的后端。</summary>
    /// <param name="source">不包含训练头覆盖参数的原始 PT 内容。</param><param name="options">设备和 CPU 线程配置。</param>
    /// <exception cref="InvalidDataException">PT 数据损坏。</exception><exception cref="NotSupportedException">PT 结构不受支持。</exception>
    internal FrozenFeatureExtractor(byte[] source, TrainingOptions options)
    {
        Backend = options.Device == ComputeDevice.Vulkan ? new VulkanBackend() : new CpuBackend(options.CpuThreads);
        if (Backend is CpuBackend cpu) pool.Parallel = cpu.Parallel;
        try { Model = new PtModel(new MemoryStream(source, false), new TensorOps(Backend), null, null); }
        catch { Backend.Dispose(); throw; }
    }

    /// <summary>执行冻结骨干/FPN，物化设备结果或脱离 CPU 池，随后恢复调用线程原有执行上下文。</summary>
    /// <param name="input">单张训练图像的 NCHW FP32 张量。</param><returns>可由训练头及特征缓存独立持有的三个尺度张量。</returns>
    internal Tensor[] Extract(Tensor input)
    {
        var previous = CpuTensorPool.Current; var device = DeviceScope.Current;
        try
        {
            if (Backend is IDeviceBackend gpu)
            {
                CpuTensorPool.Current = null; gpu.BeginFrame();
                try { var result = Model.ExtractDetectionFeatures(input); gpu.MaterializeOutputs(result); return result; }
                finally { gpu.EndFrame(); }
            }
            CpuTensorPool.Current = pool; DeviceScope.Current = null;
            return pool.DetachOutputs(Model.ExtractDetectionFeatures(input));
        }
        finally { pool.ReturnAll(); CpuTensorPool.Current = previous; DeviceScope.Current = device; }
    }
    /// <summary>释放当前特征池和独立计算后端。</summary>
    public void Dispose() { pool.Clear(); Backend.Dispose(); }
}
