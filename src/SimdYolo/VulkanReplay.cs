namespace SimdYolo;

/// <summary>捕获并重放固定形状的 Vulkan 推理命令，同时保持返回数组的独立所有权。</summary>
public sealed unsafe partial class VulkanBackend
{
    private readonly bool replayEnabled = Environment.GetEnvironmentVariable("SIMDYOLO_VULKAN_REPLAY") != "0";
    private bool replayCandidate, replayUploadReading, replayCommandRetained;
    private Tensor? replayCaptureInput;
    private GpuBuffer? replayInputHost;
    private ulong replayInputHostOffset;
    private ReplayPlan? replayPlan;
    /// <summary>记录重放输出的形状、下载区字节偏移和输出别名关系。</summary>
    /// <param name="Shape">重放输出的张量形状。</param>
    /// <param name="Length">区段中的 FP32 元素数。</param>
    /// <param name="Offset">缓冲区内的字节偏移。</param>
    /// <param name="Alias">同一结果数组中首先持有此张量的索引，用于恢复输出别名。</param>
    private sealed record ReplayOutput(int[] Shape, int Length, ulong Offset, int Alias);
    /// <summary>保存可重复提交的命令所引用的上传区和各输出布局；引用的 GPU 资源仍归后端持有。</summary>
    /// <param name="InputShape">录制命令时的输入形状。</param>
    /// <param name="InputLength">录制输入的 FP32 元素数。</param>
    /// <param name="InputHost">命令引用的主机可见上传缓冲区；所有权保留在后端。</param>
    /// <param name="InputOffset">输入在上传缓冲区中的字节偏移。</param>
    /// <param name="Outputs">输出的形状、下载位置和别名说明。</param>
    private sealed record ReplayPlan(int[] InputShape, int InputLength, GpuBuffer InputHost, ulong InputOffset, ReplayOutput[] Outputs);
    /// <summary>取得无需重新录制计算图而完成提交的推理帧数。</summary>
    /// <value>成功重放后递增的累计计数。</value>
    public long ReplayedFrames { get; private set; }

    /// <summary>为只依赖固定形状和单个主机输入的帧启动重放候选捕获。</summary>
    /// <param name="input">由主机数组支持的可变输入，形状须与录制命令匹配。</param>
    internal void BeginReplayCapture(Tensor input)
    {
        replayCandidate = replayEnabled && !gpuProfile && input.Storage is null && input.Length > 0 && !input.IsConstant;
        replayCaptureInput = input;
        replayInputHost = null;
    }

    /// <summary>观察张量主机数据读取；非常量值被 CPU 读取时取消当前帧重放候选。</summary>
    /// <param name="tensor">待上传或观察主机读取行为的张量。</param>
    void IDeviceBackend.ObserveHostData(Tensor tensor)
    {
        // 只依赖形状的 CPU 工作允许重放；读取非常量值可能改变后续切片、
        // 索引、上传或输出，因此这类帧必须重新录制。
        if (replayCandidate && !replayUploadReading && !tensor.IsConstant) replayCandidate = false;
    }

    /// <summary>记录唯一主机输入所在的上传缓冲区；遇到其他可变上传时取消重放候选。</summary>
    /// <param name="tensor">待上传或观察主机读取行为的张量。</param>
    /// <param name="host">录制上传命令所引用的主机可见缓冲区。</param>
    /// <param name="offset">输入在上传缓冲区中的起始字节偏移。</param>
    private void ObserveReplayUpload(Tensor tensor, GpuBuffer host, ulong offset)
    {
        if (!replayCandidate || tensor.IsConstant) return;
        if (!ReferenceEquals(tensor, replayCaptureInput) || replayInputHost is not null) { replayCandidate = false; return; }
        replayInputHost = host; replayInputHostOffset = offset;
    }

    /// <summary>判断本帧能否保留已录制命令，包括唯一上传输入和全部非空设备输出。</summary>
    /// <param name="outputs">当前帧的最终输出张量数组。</param>
    /// <returns>可以保留并重复提交当前命令时为 true。</returns>
    private bool CanRetainReplay(Tensor[] outputs)
        => replayCandidate && replayInputHost is not null && outputs.Length > 0 && outputs.All(t => t.Storage is { Length: > 0 });

    /// <summary>保存输入形状、上传位置和输出下载位置，并记录重复输出张量的别名。</summary>
    /// <param name="outputs">当前帧的最终输出张量数组。</param>
    /// <param name="pending">各输出对应的设备区段与合并下载区中的字节偏移。</param>
    private void CompleteReplay(Tensor[] outputs, List<(Tensor Tensor, DeviceStorage Storage, ulong Offset)> pending)
    {
        var descriptions = new ReplayOutput[outputs.Length];
        var aliases = new Dictionary<Tensor, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < outputs.Length; i++)
        {
            if (!aliases.TryGetValue(outputs[i], out int alias)) { alias = i; aliases.Add(outputs[i], i); }
            descriptions[i] = new((int[])outputs[i].Shape.Clone(), outputs[i].Length, pending[i].Offset, alias);
        }
        replayPlan = new((int[])replayCaptureInput!.Shape.Clone(), replayCaptureInput.Length, replayInputHost!, replayInputHostOffset, descriptions);
    }

    /// <summary>清除本帧捕获状态，不销毁已经建立且仍有效的重放计划。</summary>
    private void EndReplayCapture()
    {
        replayCandidate = false; replayCaptureInput = null; replayInputHost = null; replayUploadReading = false;
    }

    /// <summary>清除重放计划和保留标记，防止重置或改写命令后再次提交旧计划。</summary>
    private void InvalidateReplay()
    {
        replayPlan = null; replayCommandRetained = false;
    }

    /// <summary>按固定形状更新上传区并重新提交已录制命令；同步后为本次结果分配独立主机数组。</summary>
    /// <param name="input">由主机数组支持的可变输入，形状须与录制命令匹配。</param>
    /// <param name="outputs">成功重放时返回的主机结果数组；条件不匹配时为空数组。</param>
    /// <returns>成功提交重放并生成输出时为 true；条件不匹配时为 false。</returns>
    /// <exception cref="ObjectDisposedException">后端已经释放。</exception>
    /// <exception cref="InvalidOperationException">重放栅栏重置、命令提交或等待失败；失败会同时清除重放计划。</exception>
    /// <remarks>每次推理的公开输出使用独立主机数组；后续帧复用上传/下载区不会改写已返回的数据。</remarks>
    internal bool TryReplay(Tensor input, out Tensor[] outputs)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            outputs = [];
            if (replayPlan is not { } plan || frameActive || input.Storage is not null || input.IsConstant || input.Length != plan.InputLength || !input.Shape.AsSpan().SequenceEqual(plan.InputShape)) return false;
            try
            {
                fixed (float* source = input.Data)
                    Buffer.MemoryCopy(source, (byte*)plan.InputHost.Mapped + plan.InputOffset, (long)input.Length * sizeof(float), (long)input.Length * sizeof(float));
                ulong submittedFence = fence; Check(Vk.ResetFences(device, 1, &submittedFence), "ResetFences(replay)");
                nint submittedCommand = command;
                SubmitInfo submit = new() { Type = 4, CommandCount = 1, Commands = &submittedCommand };
                Check(Vk.Submit(queue, 1, &submit, fence), "QueueSubmit(replay)"); WaitForSubmission(submittedFence);
                outputs = new Tensor[plan.Outputs.Length];
                for (int i = 0; i < outputs.Length; i++)
                {
                    var description = plan.Outputs[i];
                    if (description.Alias != i) { outputs[i] = outputs[description.Alias]; continue; }
                    float[] values = GC.AllocateUninitializedArray<float>(description.Length);
                    fixed (float* target = values)
                        Buffer.MemoryCopy((byte*)staging[2].Mapped + description.Offset, target, (long)values.Length * sizeof(float), (long)values.Length * sizeof(float));
                    outputs[i] = new(values, description.Shape);
                }
                ReplayedFrames++;
                return true;
            }
            catch { InvalidateReplay(); throw; }
        }
    }
}
