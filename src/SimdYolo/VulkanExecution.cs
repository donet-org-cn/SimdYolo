using System.Runtime.InteropServices;
using System.Text;

namespace SimdYolo;

/// <summary>管理 Vulkan 帧内设备张量、命令录制、同步提交与通用算子调度。</summary>
public sealed unsafe partial class VulkanBackend
{
    private bool frameActive;
    private long frameId;
    private int commandsRecorded, arenaIndex, uploadIndex;
    private ulong arenaOffset, uploadOffset, frameDescriptorPool;
    private readonly List<GpuBuffer> arenas = [];
    private readonly List<GpuBuffer> uploads = [];
    private readonly List<GpuBuffer> pendingConstants = [];
    private readonly Dictionary<Tensor, DeviceStorage> frameInputs = [];
    private readonly Dictionary<int, ulong> tensorPipelines = [];
    private readonly Dictionary<string, ulong> matmul32Pipelines = [];
    private readonly bool matmul32 = Environment.GetEnvironmentVariable("SIMDYOLO_MATMUL32") != "0";
    private readonly Dictionary<int, Tensor> zeros = [];
    private readonly Dictionary<string, uint> convolutionTiles = [];
    private readonly Tensor scalarZero = new([0], []) { IsConstant = true };
    private readonly bool gpuProfile = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_PROFILE") == "1";
    private ulong timestampPool;
    private ulong tuningTimestampPool;
    private readonly List<ulong> timestampPipelines = [];
    private readonly Dictionary<ulong, ulong> timestampTotals = [];
    private readonly bool enableWinograd = Environment.GetEnvironmentVariable("SIMDYOLO_WINOGRAD") != "0";
    private readonly bool wideConvolutionTiles = Environment.GetEnvironmentVariable("SIMDYOLO_WIDE_TILES") != "0";
    private readonly bool lowChannelsConvolution = Environment.GetEnvironmentVariable("SIMDYOLO_LOW_CHANNELS") != "0";
    private readonly bool lowKConvolutionTiles = Environment.GetEnvironmentVariable("SIMDYOLO_LOW_K_TILES") != "0";

    /// <summary>取得当前后端是否仍在执行活动推理帧。</summary>
    /// <value>帧开始后为 true，结束或失败清理后为 false。</value>
    bool IDeviceBackend.FrameActive => frameActive;
    /// <summary>取得当前设备推理帧编号，用于校验设备区段有效期。</summary>
    /// <value>每次 BeginFrame 递增的帧标识。</value>
    long IDeviceBackend.FrameId => frameId;
    /// <summary>进入后端互斥锁并开始一帧命令录制，重置帧内分配器和张量缓存。</summary>
    /// <exception cref="ObjectDisposedException">后端已经释放。</exception>
    /// <exception cref="InvalidOperationException">当前已有帧在录制，或 Vulkan 描述符、查询池、命令操作失败。</exception>
    void IDeviceBackend.BeginFrame()
    {
        // 锁跨越整个推理帧，直到 EndFrame 的 finally 才释放；禁止并发复用帧分配器。
        Monitor.Enter(gate);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (frameActive) throw new InvalidOperationException("An inference is already recording.");
            if (frameDescriptorPool == 0)
            {
                PoolSize size = new() { Type = 7, Count = 8192 };
                DescriptorPoolInfo info = new() { Type = 33, MaxSets = 2048, SizeCount = 1, Sizes = &size }; ulong pool = 0;
                Check(Vk.CreateDescriptorPool(device, &info, null, &pool), "CreateDescriptorPool(frame)"); frameDescriptorPool = pool;
            }
            frameId++; frameActive = true; DeviceScope.Current = this;
            if (gpuProfile && timestampPool == 0)
            {
                QueryPoolInfo info = new() { Type = 11, QueryType = 2, Count = 8192 }; ulong pool = 0;
                Check(Vk.CreateQueryPool(device, &info, null, &pool), "CreateQueryPool"); timestampPool = pool;
            }
            timestampTotals.Clear();
            arenaIndex = uploadIndex = 0; arenaOffset = uploadOffset = 0; commandsRecorded = 0;
            frameInputs.Clear(); winogradInputs.Clear(); pendingConstants.Clear(); StartRecording();
        }
        catch { frameActive = false; DeviceScope.Current = null; Monitor.Exit(gate); throw; }
    }
    /// <summary>结束当前帧，提交剩余命令，并在 finally 中清除线程上下文、帧缓存和互斥锁。</summary>
    void IDeviceBackend.EndFrame()
    {
        try
        {
            if (!frameActive) return;
            if (!replayCommandRetained) { FlushFrame(); Check(Vk.EndCommand(command), "EndCommandBuffer(empty)"); }
            if (gpuProfile)
            {
                foreach (var item in timestampTotals.OrderByDescending(t => t.Value))
                {
                    string label = (matmul32Pipelines.FirstOrDefault(p => p.Value == item.Key).Key is { } mk ? "matmul32:" + mk : null) ?? winogradTransformPipelines.FirstOrDefault(p => p.Value == item.Key).Key ?? (winogradPipelines.FirstOrDefault(p => p.Value == item.Key).Key is { } wk ? "winograd-gemm:" + wk : convolutionPipelines.FirstOrDefault(p => p.Value == item.Key).Key ?? tensorPipelines.FirstOrDefault(p => p.Value == item.Key).Key.ToString());
                    Console.Error.WriteLine($"GPU_TICKS {item.Value} {label}");
                }
                Console.Error.WriteLine($"GPU_FRAME_TICKS {timestampTotals.Values.Aggregate(0UL, (sum, value) => sum + value)} {timestampTotals.Count}");
            }
        }
        finally { EndReplayCapture(); frameActive = false; DeviceScope.Current = null; frameInputs.Clear(); winogradInputs.Clear(); Monitor.Exit(gate); }
    }
    /// <summary>提交当前帧已录制的命令并等待完成，随后恢复可继续录制的状态。</summary>
    void IDeviceBackend.Flush() => FlushFrame();
    /// <summary>清除旧重放计划，重置命令和描述符池，并重新开始录制当前帧。</summary>
    private void StartRecording()
    {
        InvalidateReplay();
        Check(Vk.ResetCommand(command, 0), "ResetCommandBuffer(frame)");
        Check(Vk.ResetDescriptorPool(device, frameDescriptorPool, 0), "ResetDescriptorPool(frame)");
        CommandBeginInfo info = new() { Type = 42, Flags = replayEnabled && !gpuProfile ? 0u : 1u }; Check(Vk.BeginCommand(command, &info), "BeginCommandBuffer(frame)");
        if (gpuProfile) { Vk.ResetQueryPool(command, timestampPool, 0, 8192); timestampPipelines.Clear(); }
    }
    /// <summary>提交非空帧命令并等待栅栏，更新常量就绪状态；按需保留命令供后续重放。</summary>
    /// <param name="retainCommand">是否保留已经完成的命令缓冲区用于重放。</param>
    private void FlushFrame(bool retainCommand = false)
    {
        if (!frameActive || commandsRecorded == 0) return;
        if (!retainCommand) replayCandidate = false;
        Check(Vk.EndCommand(command), "EndCommandBuffer(frame)"); ulong f = fence; Check(Vk.ResetFences(device, 1, &f), "ResetFences(frame)"); nint cb = command;
        SubmitInfo submit = new() { Type = 4, CommandCount = 1, Commands = &cb };
        Check(Vk.Submit(queue, 1, &submit, fence), "QueueSubmit(frame)"); WaitForSubmission(f);
        if (gpuProfile && timestampPipelines.Count > 0)
        {
            ulong[] values = new ulong[timestampPipelines.Count * 2];
            fixed (ulong* ptr = values) Check(Vk.GetQueryPoolResults(device, timestampPool, 0, (uint)values.Length, (nuint)(values.Length * 8), ptr, 8, 3), "GetQueryPoolResults");
            for (int i = 0; i < timestampPipelines.Count; i++) { ulong p = timestampPipelines[i]; timestampTotals[p] = timestampTotals.GetValueOrDefault(p) + values[i * 2 + 1] - values[i * 2]; }
        }
        foreach (var buffer in pendingConstants) buffer.Ready = true;
        pendingConstants.Clear(); commandsRecorded = 0;
        if (retainCommand) replayCommandRetained = true; else StartRecording();
    }
    /// <summary>录制设备内存屏障，使前序传输或计算的写入对后续访问可见。</summary>
    /// <param name="source">前序写入操作的 Vulkan 访问掩码。</param>
    /// <param name="destination">后续操作的访问掩码。</param>
    /// <param name="sourceStage">产生前序写入的管线阶段掩码。</param>
    /// <param name="destinationStage">后续读取或写入的管线阶段掩码。</param>
    private void Barrier(uint source = 4160, uint destination = 2144, uint sourceStage = 6144, uint destinationStage = 6144)
    {
        MemoryBarrier b = new() { Type = 46, SourceAccess = source, DestinationAccess = destination };
        Vk.PipelineBarrier(command, sourceStage, destinationStage, 0, 1, &b, 0, null, 0, null);
    }
    /// <summary>从可复用缓冲块中分配按 256 字节对齐的区段，空间不足时转到下一个块。</summary>
    /// <param name="blocks">后端持有的可复用缓冲块集合。</param>
    /// <param name="index">当前缓冲块编号，空间不足时更新。</param>
    /// <param name="offset">当前块内的字节偏移，分配后更新到区段末尾。</param>
    /// <param name="bytes">所需区段字节数。</param>
    /// <param name="host">是否使用主机可见且一致的映射内存。</param>
    /// <returns>后端持有的缓冲块与本次分配的字节偏移。</returns>
    /// <remarks>缓冲块由后端持有并跨帧复用；返回区段只在当前帧有效。</remarks>
    private (GpuBuffer Buffer, ulong Offset) Allocate(List<GpuBuffer> blocks, ref int index, ref ulong offset, ulong bytes, bool host)
    {
        bytes = Math.Max(4, bytes);
        while (true)
        {
            if (index >= blocks.Count)
            {
                var block = new GpuBuffer();
                try { Ensure(block, Math.Max(host ? 16UL * 1024 * 1024 : 32UL * 1024 * 1024, bytes), host); blocks.Add(block); }
                catch { Free(block); throw; }
            }
            var current = blocks[index]; ulong aligned = (offset + 255) & ~255UL;
            if (aligned <= current.Capacity && bytes <= current.Capacity - aligned) { offset = aligned + bytes; return (current, aligned); }
            index++; offset = 0;
        }
    }
    /// <summary>在帧内设备分配区创建指定形状的 FP32 张量，返回不含主机数组的设备视图。</summary>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    private Tensor AllocateTensor(int[] shape)
    {
        int length = Tensor.Count(shape); var (buffer, offset) = Allocate(arenas, ref arenaIndex, ref arenaOffset, (ulong)length * 4, false);
        return new(new DeviceStorage(this, buffer, offset, length, frameId), shape);
    }
    /// <summary>取得本帧设备视图；复用已驻留张量或常量缓存，否则录制主机上传拷贝。</summary>
    /// <param name="tensor">待上传或观察主机读取行为的张量。</param>
    /// <returns>属于当前后端和帧编号的设备视图。</returns>
    private DeviceStorage Upload(Tensor tensor)
    {
        if (tensor.Storage is { } existing && ReferenceEquals(existing.Owner, this) && existing.FrameId == frameId) return existing;
        if (frameInputs.TryGetValue(tensor, out var same)) return same;
        DeviceStorage storage;
        if (tensor.IsConstant)
        {
            if (!constants.TryGetValue(tensor, out var buffer))
            {
                buffer = new GpuBuffer();
                try { Ensure(buffer, (ulong)tensor.Length * 4, false); constants.Add(tensor, buffer); }
                catch { Free(buffer); throw; }
            }
            storage = new(this, buffer, 0, tensor.Length, frameId);
            if (buffer.Ready || pendingConstants.Contains(buffer)) { frameInputs[tensor] = storage; return storage; }
            // 首次权重上传用于预热持久设备缓存。若保留本次命令，
            // 之后每一帧都会重复上传常量，因此取消此帧的重放候选。
            replayCandidate = false;
            pendingConstants.Add(buffer);
        }
        else storage = AllocateTensor(tensor.Shape).Storage!;
        if (tensor.Length > 0)
        {
            var host = Allocate(uploads, ref uploadIndex, ref uploadOffset, (ulong)tensor.Length * 4, true);
            ObserveReplayUpload(tensor, host.Buffer, host.Offset);
            bool previous = replayUploadReading; replayUploadReading = true;
            try { fixed (float* data = tensor.Data) Buffer.MemoryCopy(data, (byte*)host.Buffer.Mapped + host.Offset, (long)tensor.Length * 4, (long)tensor.Length * 4); }
            finally { replayUploadReading = previous; }
            BufferCopy copy = new() { SourceOffset = host.Offset, DestinationOffset = storage.Offset, Size = (ulong)tensor.Length * 4 };
            Vk.CopyBuffer(command, host.Buffer.Buffer, ((GpuBuffer)storage.Buffer).Buffer, 1, &copy); commandsRecorded++;
        }
        frameInputs[tensor] = storage; return storage;
    }
    /// <summary>校验设备张量所属帧，录制下载拷贝并同步返回独立主机数组。</summary>
    /// <param name="storage">待下载的设备区段与帧归属。</param>
    /// <returns>下载完成后的独立 FP32 数组，不借用 GPU 映射内存。</returns>
    /// <exception cref="InvalidOperationException">设备视图不属于当前后端活动帧，或 Vulkan 下载提交失败。</exception>
    float[] IDeviceBackend.Download(DeviceStorage storage)
    {
        if (!frameActive || !ReferenceEquals(storage.Owner, this) || storage.FrameId != frameId) throw new InvalidOperationException("Device tensor belongs to a completed inference.");
        var result = new float[storage.Length]; if (storage.Length == 0) return result;
        Ensure(staging[2], (ulong)storage.Length * 4, true, 8);
        Barrier(4160, 2048, 6144, 4096);
        BufferCopy copy = new() { SourceOffset = storage.Offset, Size = (ulong)storage.Length * 4 };
        Vk.CopyBuffer(command, ((GpuBuffer)storage.Buffer).Buffer, staging[2].Buffer, 1, &copy); commandsRecorded++;
        Barrier(4096, 8192, 4096, 16384); FlushFrame();
        fixed (float* target = result) Buffer.MemoryCopy((void*)staging[2].Mapped, target, (long)result.Length * 4, (long)result.Length * 4);
        return result;
    }
    /// <summary>绑定按 256 字节向下对齐的描述符区段，推送参数并录制计算调度。</summary>
    /// <param name="selectedPipeline">本次调度使用的计算管线句柄。</param>
    /// <param name="a">绑定 0 的设备区段，卷积时保存权重。</param>
    /// <param name="b">绑定 1 的设备区段，卷积时保存输入。</param>
    /// <param name="c">绑定 2 的设备区段，保存计算结果。</param>
    /// <param name="d">绑定 3 的设备区段，通常保存偏置或零张量。</param>
    /// <param name="parameters">按对应着色器参数布局排列的 uint 值，用于特化常量或推送常量。</param>
    /// <param name="gx">X 方向线程组数。</param>
    /// <param name="gy">Y 方向线程组数。</param>
    /// <param name="gz">Z 方向线程组数。</param>
    /// <exception cref="NotSupportedException">任一线程组数量超过可移植上限 65535。</exception>
    /// <exception cref="InvalidOperationException">Vulkan 描述符分配失败。</exception>
    /// <remarks>描述符起点向下对齐后产生的余量，由调用者编码为着色器元素偏移，不能丢弃。</remarks>
    private void Dispatch(ulong selectedPipeline, DeviceStorage a, DeviceStorage b, DeviceStorage c, DeviceStorage d, uint[] parameters, uint gx, uint gy = 1, uint gz = 1)
    {
        if (gx == 0 || gy == 0 || gz == 0) return;
        if (gx > 65535 || gy > 65535 || gz > 65535) throw new NotSupportedException("Vulkan dispatch exceeds portable limits.");
        Barrier();
        ulong layout = descriptorLayout, set = 0;
        if (gpuProfile) Vk.WriteTimestamp(command, 2048, timestampPool, (uint)timestampPipelines.Count * 2);
        DescriptorAllocateInfo allocate = new() { Type = 34, Pool = frameDescriptorPool, Count = 1, Layouts = &layout };
        Check(Vk.AllocateDescriptors(device, &allocate, &set), "AllocateDescriptorSets(frame)");
        DeviceStorage[] bindings = [a, b, c, d];
        DescriptorBufferInfo* infos = stackalloc DescriptorBufferInfo[4]; WriteDescriptor* writes = stackalloc WriteDescriptor[4];
        for (uint i = 0; i < 4; i++)
        {
            var s = bindings[i]; ulong aligned = s.Offset & ~255UL;
            // 描述符按字节对齐，参数中的余量按 FP32 元素计数，两者单位不可混用。
            infos[i] = new() { Buffer = ((GpuBuffer)s.Buffer).Buffer, Offset = aligned, Range = Math.Max(4, (ulong)s.Length * 4 + s.Offset - aligned) };
            writes[i] = new() { Type = 35, Set = set, Binding = i, Count = 1, DescriptorType = 7, Buffers = infos + i };
        }
        Vk.UpdateDescriptors(device, 4, writes, 0, null); Vk.BindPipeline(command, 1, selectedPipeline);
        Vk.BindDescriptorSets(command, 1, pipelineLayout, 0, 1, &set, 0, null);
        fixed (uint* p = parameters) Vk.PushConstants(command, pipelineLayout, 32, 0, (uint)parameters.Length * 4, p);
        Vk.Dispatch(command, gx, gy, gz); commandsRecorded++;
        if (gpuProfile) { Vk.WriteTimestamp(command, 2048, timestampPool, (uint)timestampPipelines.Count * 2 + 1); timestampPipelines.Add(selectedPipeline); }
    }
    /// <summary>按操作编号创建或复用设备管线，编码视图余量偏移并返回帧内设备结果张量。</summary>
    /// <param name="operation">与设备着色器特化常量约定一致的算子编号。</param>
    /// <param name="inputs">操作输入张量，最多使用三个数据输入。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="parameters">与着色器约定一致的 32 个 uint 推送常量槽位。</param>
    /// <param name="workItems">着色器需要覆盖的逻辑工作项数。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    Tensor IDeviceBackend.Compute(int operation, Tensor[] inputs, int[] shape, uint[] parameters, int workItems)
    {
        if (operation == 33 && inputs[1].IsConstant && (inputs.Length < 3 || inputs[2].IsConstant) && parameters[7] == 2 && parameters[8] == 2 && parameters[9] == 2 && parameters[10] == 2 && parameters[11] == 0 && parameters[12] == 0 && parameters[13] == 1 && parameters[14] == 1 && parameters[15] == 1 && shape[2] == inputs[0].Shape[2] * 2 && shape[3] == inputs[0].Shape[3] * 2)
            return TransposeConv2x2(inputs[0], inputs[1], inputs.Length > 2 ? inputs[2] : null, shape);
        var output = AllocateTensor(shape); var a = Upload(inputs[0]); var b = Upload(inputs.Length > 1 ? inputs[1] : scalarZero);
        var d = Upload(inputs.Length > 2 ? inputs[2] : scalarZero); var c = output.Storage!;
        parameters = (uint[])parameters.Clone(); parameters[27] = inputs.Length > 2 ? 1u : 0u;
        parameters[28] = (uint)(a.Offset % 256 / 4); parameters[29] = (uint)(b.Offset % 256 / 4);
        parameters[30] = (uint)(c.Offset % 256 / 4); parameters[31] = (uint)(d.Offset % 256 / 4);
        if (!tensorPipelines.TryGetValue(operation, out ulong opPipeline)) { opPipeline = CreateTensorPipeline((uint)operation); tensorPipelines.Add(operation, opPipeline); }
        Dispatch(opPipeline, a, b, c, d, parameters, (uint)(workItems + 255) / 256); return output;
    }
    /// <summary>按卷积形状选择深度卷积、低通道、Winograd、逐点或通用管线，并返回设备结果。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="stride">高、宽两个方向的卷积步长。</param>
    /// <param name="pads">上、左、下、右方向的填充量。</param>
    /// <param name="dilation">高、宽方向的卷积膨胀系数。</param>
    /// <param name="groups">卷积分组数，输入和输出通道须满足该分组布局。</param>
    /// <param name="silu">是否在偏置相加后融合 SiLU 激活。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <exception cref="InvalidOperationException">通用卷积的输入、权重或偏置未对齐，或 Vulkan 管线与设备调度失败。</exception>
    Tensor IDeviceBackend.DeviceConv(Tensor x, Tensor w, Tensor? bias, int[] shape, int[] stride, int[] pads, int[] dilation, int groups, bool silu)
    {
        if (groups == x.Shape[1] && w.Shape[0] == groups)
        {
            if (DepthwiseVectorEligible(x, w, shape, stride, pads, dilation, groups)) return DepthwiseVectorConv(x, w, bias, shape, stride, silu);
            var p = DeviceOps.Parameters((uint)Tensor.Count(shape), (uint)x.Shape[1], (uint)x.Shape[2], (uint)x.Shape[3], (uint)shape[1], (uint)shape[2], (uint)shape[3], (uint)w.Shape[2], (uint)w.Shape[3], (uint)stride[0], (uint)stride[1], (uint)pads[0], (uint)pads[1], (uint)dilation[0], (uint)dilation[1], (uint)groups, silu ? 1u : 0u);
            return ((IDeviceBackend)this).Compute(32, bias is null ? [x, w] : [x, w, bias], shape, p, Tensor.Count(shape));
        }
        if (lowChannelsConvolution && LowChannelsEligible(x, w, stride, pads, dilation, groups)) return LowChannelsConv(x, w, bias, shape, silu);
        if (enableWinograd && w.IsConstant && groups == 1 && w.Shape[2] == 3 && w.Shape[3] == 3 && stride[0] == 1 && stride[1] == 1 && pads.All(p => p == 1) && dilation.All(d => d == 1) && x.Shape[1] >= 16)
            return WinogradConv(x, w, bias, silu);
        if (RegisterPointwiseEligible(x, w, stride, pads, dilation, groups)) return RegisterPointwiseConv(x, w, bias, shape, silu);
        if (PointwisePackedEligible(x, w, shape, stride, pads, dilation, groups)) return PointwisePackedConv(x, w, bias, shape, silu);
        if (PointwisePacked32Eligible(x, w, shape, stride, pads, dilation, groups)) return PointwisePacked32Conv(x, w, bias, shape, silu);
        var output = AllocateTensor(shape);
        uint[] parameters = [silu ? 1u : 0u, (uint)x.Shape[1], (uint)x.Shape[2], (uint)x.Shape[3], (uint)shape[1], (uint)shape[2], (uint)shape[3], (uint)w.Shape[2], (uint)w.Shape[3], (uint)stride[0], (uint)stride[1], (uint)pads[0], (uint)pads[1], (uint)dilation[0], (uint)dilation[1], (uint)groups];
        string key = string.Join(',', parameters);
        if (bias is null)
        {
            if (!zeros.TryGetValue(shape[1], out bias)) { bias = new Tensor(shape[1]) { IsConstant = true }; zeros.Add(shape[1], bias); }
        }
        var a = Upload(w); var b = Upload(x); var d = Upload(bias);
        if ((a.Offset | b.Offset | d.Offset) % 256 != 0) throw new InvalidOperationException("Unaligned convolution tensor.");
        if (!convolutionTiles.TryGetValue(key, out uint tile))
        {
            uint fallback = ConvTile(parameters); uint[] candidates = fallback == 32 ? (wideConvolutionTiles ? [16u, 32u, 64u, 80u, 96u, 128u] : [16u, 32u, 64u]) : [16u];
            if (fallback == 32 && wideConvolutionTiles && w.IsConstant && RegisterPointwiseShape(parameters)) candidates = [.. candidates, 256u];
            foreach (uint candidate in candidates)
            {
                string variantKey = key + ":" + candidate;
                if (!convolutionPipelines.ContainsKey(variantKey)) convolutionPipelines[variantKey] = CreateConvPipeline(parameters, candidate);
            }
            tile = fallback;
            if (candidates.Length > 1)
            {
                FlushFrame(); ulong fastest = ulong.MaxValue;
                foreach (uint candidate in candidates)
                {
                    uint candidateRows = candidate == 256 ? 64 : candidate >= 32 ? 32 : candidate; uint candidateColumns = candidate is 80 or 96 or 256 ? 64 : candidate;
                    Dispatch(convolutionPipelines[key + ":" + candidate], a, b, output.Storage!, d, parameters, ((uint)(shape[2] * shape[3]) + candidateColumns - 1) / candidateColumns, ((uint)(shape[1] / groups) + candidateRows - 1) / candidateRows, (uint)(shape[0] * groups));
                }
                FlushFrame();
                for (int round = 0; round < 2; round++) foreach (uint candidate in candidates)
                {
                    uint candidateRows = candidate == 256 ? 64 : candidate >= 32 ? 32 : candidate; uint candidateColumns = candidate is 80 or 96 or 256 ? 64 : candidate;
                    ulong elapsed = MeasureConvolution(convolutionPipelines[key + ":" + candidate], a, b, output.Storage!, d, parameters, ((uint)(shape[2] * shape[3]) + candidateColumns - 1) / candidateColumns, ((uint)(shape[1] / groups) + candidateRows - 1) / candidateRows, (uint)(shape[0] * groups));
                    if (elapsed < fastest) { fastest = elapsed; tile = candidate; }
                }
            }
            convolutionTiles[key] = tile;
        }
        string selectedKey = key + ":" + tile;
        if (!convolutionPipelines.TryGetValue(selectedKey, out ulong convPipeline))
            convolutionPipelines[selectedKey] = convPipeline = CreateConvPipeline(parameters, tile);
        uint rows = tile == 256 ? 64 : tile >= 32 ? 32 : tile, columns = tile is 80 or 96 or 256 ? 64 : tile;
        Dispatch(convPipeline, a, b, output.Storage!, d, parameters, ((uint)(shape[2] * shape[3]) + columns - 1) / columns, ((uint)(shape[1] / groups) + rows - 1) / rows, (uint)(shape[0] * groups));
        return output;
    }
    /// <summary>重复调度十二次候选管线并同步，优先用设备时间戳测量，否则使用主机计时刻度。</summary>
    /// <param name="selectedPipeline">本次调度使用的计算管线句柄。</param>
    /// <param name="a">绑定 0 的设备区段，卷积时保存权重。</param>
    /// <param name="b">绑定 1 的设备区段，卷积时保存输入。</param>
    /// <param name="c">绑定 2 的设备区段，保存计算结果。</param>
    /// <param name="d">绑定 3 的设备区段，通常保存偏置或零张量。</param>
    /// <param name="parameters">按对应着色器参数布局排列的 uint 值，用于特化常量或推送常量。</param>
    /// <param name="gx">X 方向线程组数。</param>
    /// <param name="gy">Y 方向线程组数。</param>
    /// <param name="gz">Z 方向线程组数。</param>
    /// <returns>十二次调度的耗时刻度；设备时间戳不可用时为 Stopwatch 刻度。</returns>
    private ulong MeasureConvolution(ulong selectedPipeline, DeviceStorage a, DeviceStorage b, DeviceStorage c, DeviceStorage d, uint[] parameters, uint gx, uint gy, uint gz)
    {
        if (timestampBits == 0)
        {
            var timer = System.Diagnostics.Stopwatch.StartNew();
            for (int repeat = 0; repeat < 12; repeat++) Dispatch(selectedPipeline, a, b, c, d, parameters, gx, gy, gz);
            FlushFrame(); return (ulong)timer.ElapsedTicks;
        }
        if (tuningTimestampPool == 0)
        {
            QueryPoolInfo info = new() { Type = 11, QueryType = 2, Count = 2 }; ulong pool = 0;
            Check(Vk.CreateQueryPool(device, &info, null, &pool), "CreateQueryPool(tuning)"); tuningTimestampPool = pool;
        }
        Vk.ResetQueryPool(command, tuningTimestampPool, 0, 2);
        Vk.WriteTimestamp(command, 2048, tuningTimestampPool, 0);
        for (int repeat = 0; repeat < 12; repeat++) Dispatch(selectedPipeline, a, b, c, d, parameters, gx, gy, gz);
        Vk.WriteTimestamp(command, 2048, tuningTimestampPool, 1); FlushFrame();
        ulong* values = stackalloc ulong[2];
        Check(Vk.GetQueryPoolResults(device, tuningTimestampPool, 0, 2, 16, values, 8, 3), "GetQueryPoolResults(tuning)");
        ulong elapsed = values[1] - values[0];
        return timestampBits == 64 ? elapsed : elapsed & ((1UL << (int)timestampBits) - 1);
    }
    /// <summary>沿指定轴拼接设备张量；按外层块数选择张量着色器或设备缓冲区拷贝。</summary>
    /// <param name="tensors">沿同一轴按顺序拼接的输入张量集合。</param>
    /// <param name="axis">已经归一化的处理轴索引。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    Tensor IDeviceBackend.DeviceConcat(IReadOnlyList<Tensor> tensors, int axis, int[] shape)
    {
        var output = AllocateTensor(shape); int inner = Tensor.Count(shape[(axis + 1)..]), outer = Tensor.Count(shape[..axis]);
        var destination = output.Storage!; Barrier();
        if (outer > 4)
        {
            if (!tensorPipelines.TryGetValue(26, out ulong concatPipeline)) { concatPipeline = CreateTensorPipeline(26); tensorPipelines[26] = concatPipeline; }
            var zero = Upload(scalarZero); uint start = 0;
            foreach (var tensor in tensors)
            {
                var source = Upload(tensor);
                var p = DeviceOps.Parameters((uint)tensor.Length, (uint)tensor.Shape[axis], (uint)shape[axis], (uint)inner, start);
                p[28] = (uint)(source.Offset % 256 / 4); p[30] = (uint)(destination.Offset % 256 / 4);
                Dispatch(concatPipeline, source, zero, destination, zero, p, (uint)(tensor.Length + 255) / 256);
                start += (uint)tensor.Shape[axis];
            }
            return output;
        }
        for (int o = 0; o < outer; o++)
        {
            ulong target = destination.Offset + (ulong)(o * shape[axis] * inner) * 4;
            foreach (var tensor in tensors)
            {
                var source = Upload(tensor); ulong bytes = (ulong)(tensor.Shape[axis] * inner) * 4;
                if (bytes > 0)
                {
                    // 上传拷贝已在命令流前方录制；先建立屏障，让传输写入对本次拷贝可见。
                    Barrier(); BufferCopy copy = new() { SourceOffset = source.Offset + (ulong)o * bytes, DestinationOffset = target, Size = bytes };
                    Vk.CopyBuffer(command, ((GpuBuffer)source.Buffer).Buffer, ((GpuBuffer)destination.Buffer).Buffer, 1, &copy); commandsRecorded++;
                }
                target += bytes;
            }
        }
        return output;
    }
    /// <summary>切取指定轴的连续区段；可安全对齐时借用原设备区段，否则调度切片着色器。</summary>
    /// <param name="x">当前运算的输入张量，元素按连续行主序存储。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <param name="axis">已经归一化的处理轴索引。</param>
    /// <param name="start">切片沿指定轴的起点。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <remarks>借用视图共享底层缓冲区，不单独释放；字节偏移不等于 FP32 元素偏移。</remarks>
    Tensor IDeviceBackend.DeviceSlice(Tensor x, int[] shape, int axis, int start)
    {
        var source = Upload(x); int inner = Tensor.Count(x.Shape[(axis + 1)..]), outer = Tensor.Count(x.Shape[..axis]);
        ulong offset = source.Offset + (ulong)(start * inner) * 4;
        if (outer == 1 && offset % 256 == 0) return new(new DeviceStorage(this, source.Buffer, offset, Tensor.Count(shape), frameId), shape);
        return ((IDeviceBackend)this).Compute(24, [x], shape, DeviceOps.Parameters((uint)Tensor.Count(shape), (uint)x.Shape[axis], (uint)shape[axis], (uint)inner, (uint)start), Tensor.Count(shape));
    }
    /// <summary>执行含批次广播的设备矩阵乘法，按形状选择 32 或 16 的分块管线。</summary>
    /// <param name="a">左侧矩阵张量，最后两维为 m × k。</param>
    /// <param name="b">右侧矩阵张量，最后两维为 k × n。</param>
    /// <param name="shape">已经验证或计算好的结果张量形状。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    /// <exception cref="NotSupportedException">任一矩阵维度为 0，或调度超过可移植线程组上限。</exception>
    Tensor IDeviceBackend.DeviceMatMul(Tensor a, Tensor b, int[] shape)
    {
        var output = AllocateTensor(shape); int m = a.Shape[^2], k = a.Shape[^1], n = b.Shape[^1];
        if (m == 0 || n == 0 || k == 0) throw new NotSupportedException("Empty device MatMul.");
        var ab = Upload(a); var bb = Upload(b); var cb = output.Storage!; var zero = Upload(scalarZero); int[] batch = shape[..^2];
        // 判断批次是否仅在前导维度广播，以便单次调度按循环批次访问连续矩阵。
        bool TrailingBroadcast(int[] source)
        {
            var padded = Enumerable.Repeat(1, batch.Length - source.Length).Concat(source).ToArray(); bool nontrivial = false;
            for (int i = 0; i < batch.Length; i++) { if (!nontrivial && padded[i] == 1) continue; nontrivial = true; if (padded[i] != batch[i]) return false; }
            return true;
        }
        if (TrailingBroadcast(a.Shape[..^2]) && TrailingBroadcast(b.Shape[..^2]))
        {
            if (matmul32 && m >= 32 && n >= 32 && k >= 16 && ((ab.Offset | bb.Offset | cb.Offset) % 256) == 0)
            {
                uint[] dimensions = [0, (uint)k, 1, (uint)n, (uint)m, 1, (uint)n, 1, 1, 1, 1, 0, 0, 1, 1, 1, (uint)Tensor.Count(a.Shape[..^2]), (uint)Tensor.Count(b.Shape[..^2])];
                string key = string.Join(',', dimensions);
                if (!matmul32Pipelines.TryGetValue(key, out ulong matrixPipeline)) { matrixPipeline = CreateNamedPipeline("matmul32.spv", dimensions); matmul32Pipelines[key] = matrixPipeline; }
                Dispatch(matrixPipeline, ab, bb, cb, zero, dimensions, (uint)(n + 31) / 32, (uint)(m + 31) / 32, (uint)Tensor.Count(batch));
                return output;
            }
            uint[] p = [(uint)m, (uint)n, (uint)k, (uint)(ab.Offset % 256 / 4), (uint)(bb.Offset % 256 / 4), (uint)(cb.Offset % 256 / 4), (uint)Tensor.Count(a.Shape[..^2]), (uint)Tensor.Count(b.Shape[..^2])];
            Dispatch(pipeline, ab, bb, cb, zero, p, (uint)(n + 15) / 16, (uint)(m + 15) / 16, (uint)Tensor.Count(batch));
            return output;
        }
        // 将结果批次的扁平索引反解为坐标；输入中长度为 1 的轴始终取 0。
        int BatchIndex(int flat, int[] source)
        {
            int index = 0, stride = 1;
            for (int t = batch.Length - 1, s = source.Length - 1; t >= 0; t--, s--) { int coord = flat % batch[t]; flat /= batch[t]; if (s >= 0) { if (source[s] != 1) index += coord * stride; stride *= source[s]; } }
            return index;
        }
        for (int i = 0; i < Tensor.Count(batch); i++)
        {
            var aa = ab with { Offset = ab.Offset + (ulong)BatchIndex(i, a.Shape[..^2]) * (ulong)m * (ulong)k * 4, Length = m * k };
            var ba = bb with { Offset = bb.Offset + (ulong)BatchIndex(i, b.Shape[..^2]) * (ulong)k * (ulong)n * 4, Length = k * n };
            var ca = cb with { Offset = cb.Offset + (ulong)i * (ulong)m * (ulong)n * 4, Length = m * n };
            uint[] p = [(uint)m, (uint)n, (uint)k, (uint)(aa.Offset % 256 / 4), (uint)(ba.Offset % 256 / 4), (uint)(ca.Offset % 256 / 4), 1, 1];
            Dispatch(pipeline, aa, ba, ca, zero, p, (uint)(n + 15) / 16, (uint)(m + 15) / 16);
        }
        return output;
    }
    /// <summary>根据张量操作编号创建特化管线；创建结束后释放临时 SPIR-V 模块。</summary>
    /// <param name="operation">与设备着色器特化常量约定一致的算子编号。</param>
    /// <returns>由后端缓存并统一释放的 Vulkan 计算管线句柄。</returns>
    private ulong CreateTensorPipeline(uint operation)
    {
        var assembly = typeof(VulkanBackend).Assembly; string resource = assembly.GetManifestResourceNames().Single(n => n.EndsWith("tensor.spv", StringComparison.Ordinal));
        using var stream = assembly.GetManifestResourceStream(resource)!; using var memory = new MemoryStream(); stream.CopyTo(memory); byte[] code = memory.ToArray(); ulong module = 0, result = 0;
        fixed (byte* data = code) { ShaderInfo info = new() { Type = 16, Size = (nuint)code.Length, Code = (uint*)data }; Check(Vk.CreateShader(device, &info, null, &module), "CreateShaderModule(tensor)"); }
        try
        {
            SpecializationEntry entry = new() { Id = 0, Size = 4 }; SpecializationInfo spec = new() { Count = 1, Entries = &entry, Size = 4, Data = &operation };
            byte[] name = Encoding.UTF8.GetBytes("main\0"); fixed (byte* ptr = name) { ComputePipelineInfo info = new() { Type = 29, Stage = new() { Type = 18, Stage = 32, Module = module, Name = ptr, Specialization = (nint)(&spec) }, Layout = pipelineLayout, BaseIndex = -1 }; Check(CreateCachedComputePipeline(&info, &result), "CreateComputePipelines(tensor)"); }
            return result;
        }
        finally { Vk.DestroyShader(device, module, null); }
    }
    /// <summary>释放帧分配区、上传区、张量管线、描述符池和查询池，并清除专用算子缓存。</summary>
    private void FreeResidentResources()
    {
        FreeWinogradResources(); FreeLowChannelsResources();
        transposeWeights.Clear();
        foreach (var b in arenas.Concat(uploads)) Free(b); arenas.Clear(); uploads.Clear();
        foreach (ulong p in tensorPipelines.Values) Vk.DestroyPipeline(device, p, null); tensorPipelines.Clear();
        foreach (ulong p in matmul32Pipelines.Values) Vk.DestroyPipeline(device, p, null); matmul32Pipelines.Clear();
        if (frameDescriptorPool != 0) Vk.DestroyDescriptorPool(device, frameDescriptorPool, null); frameDescriptorPool = 0;
        if (timestampPool != 0) Vk.DestroyQueryPool(device, timestampPool, null); timestampPool = 0;
        if (tuningTimestampPool != 0) Vk.DestroyQueryPool(device, tuningTimestampPool, null); tuningTimestampPool = 0;
    }
    /// <summary>对应 VkQueryPoolCreateInfo，配置时间戳查询池类型和查询数量。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct QueryPoolInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。；查询类型，时间戳使用 VK_QUERY_TYPE_TIMESTAMP。；队列、描述符或查询的数量，依所属原生结构解释。；可选管线统计计数器掩码。</summary>
        public uint Flags, QueryType, Count, Statistics;
    }
    /// <summary>保存 Vulkan 原生函数指针；调用者负责传递有效句柄、缓冲区容量和指针生命周期。</summary>
    private static partial class Vk
    {
        /// <summary>重置描述符池并释放池内描述符集。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、描述符池句柄、重置标志。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, uint, int> ResetDescriptorPool = (delegate* unmanaged<nint, ulong, uint, int>)Load("ResetDescriptorPool");
        /// <summary>创建时间戳查询池。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、查询池描述、可选分配器、查询池输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, QueryPoolInfo*, void*, ulong*, int> CreateQueryPool = (delegate* unmanaged<nint, QueryPoolInfo*, void*, ulong*, int>)Load("CreateQueryPool");
        /// <summary>销毁时间戳查询池。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、查询池句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyQueryPool = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyQueryPool");
        /// <summary>录制查询池区间重置命令。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区、查询池句柄、首个查询索引、查询数量。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, uint, uint, void> ResetQueryPool = (delegate* unmanaged<nint, ulong, uint, uint, void>)Load("CmdResetQueryPool");
        /// <summary>录制指定管线阶段的设备时间戳写入。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区、阶段掩码、查询池句柄、查询索引。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, uint, ulong, uint, void> WriteTimestamp = (delegate* unmanaged<nint, uint, ulong, uint, void>)Load("CmdWriteTimestamp");
        /// <summary>取得时间戳查询结果。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、查询池、首个查询、查询数、输出字节容量、结果地址、结果步长字节数、读取标志。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, uint, uint, nuint, ulong*, ulong, uint, int> GetQueryPoolResults = (delegate* unmanaged<nint, ulong, uint, uint, nuint, ulong*, ulong, uint, int>)Load("GetQueryPoolResults");
    }
}
