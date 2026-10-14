namespace SimdYolo;

/// <summary>实现 Vulkan 帧最终输出的合并下载与主机数组物化。</summary>
public sealed unsafe partial class VulkanBackend
{
    /// <summary>把同一帧的设备输出合并到对齐下载区，仅提交一次并为各输出建立独立主机数组。</summary>
    /// <param name="outputs">当前帧的最终输出张量数组。</param>
    /// <exception cref="InvalidOperationException">输出属于其他后端或已结束的帧，或下载提交失败。</exception>
    /// <exception cref="OverflowException">合并下载区的字节数溢出。</exception>
    /// <remarks>每次推理的公开输出使用独立主机数组；后续帧复用上传/下载区不会改写已返回的数据。</remarks>
    void IDeviceBackend.MaterializeOutputs(Tensor[] outputs)
    {
        var pending = new List<(Tensor Tensor, DeviceStorage Storage, ulong Offset)>();
        ulong bytes = 0;
        foreach (var tensor in outputs)
        {
            if (tensor.Storage is not { } storage) continue;
            if (!frameActive || !ReferenceEquals(storage.Owner, this) || storage.FrameId != frameId) throw new InvalidOperationException("Device output belongs to another inference.");
            if (storage.Length == 0) { tensor.AcceptDownloadedData([]); continue; }
            bytes = checked((bytes + 255) & ~255UL);
            pending.Add((tensor, storage, bytes)); bytes = checked(bytes + (ulong)storage.Length * sizeof(float));
        }
        if (pending.Count == 0) return;
        Ensure(staging[2], bytes, true, 8);
        Barrier(4160, 2048, 6144, 4096);
        foreach (var item in pending)
        {
            BufferCopy copy = new() { SourceOffset = item.Storage.Offset, DestinationOffset = item.Offset, Size = (ulong)item.Storage.Length * sizeof(float) };
            Vk.CopyBuffer(command, ((GpuBuffer)item.Storage.Buffer).Buffer, staging[2].Buffer, 1, &copy); commandsRecorded++;
        }
        Barrier(4096, 8192, 4096, 16384);
        bool retainCommand = CanRetainReplay(outputs);
        FlushFrame(retainCommand);
        if (retainCommand) CompleteReplay(outputs, pending);
        foreach (var item in pending)
        {
            float[] values = GC.AllocateUninitializedArray<float>(item.Storage.Length);
            fixed (float* target = values) Buffer.MemoryCopy((byte*)staging[2].Mapped + item.Offset, target, (long)values.Length * sizeof(float), (long)values.Length * sizeof(float));
            item.Tensor.AcceptDownloadedData(values);
        }
    }
}
