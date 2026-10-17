using System.Runtime.InteropServices;
using System.Text;

namespace SimdYolo;

/// <summary>创建 Vulkan FP32 计算后端；设备内存、上传区和命令资源由本实例持有。</summary>
public sealed unsafe partial class VulkanBackend : IComputeBackend, IConvolutionBackend, IDeviceBackend
{
    /// <summary>进程内共享实例，使驱动在会话切换期间保持加载；设备、队列和显存仍由各后端独立拥有。</summary>
    private static readonly Lazy<nint> processInstance = new(CreateProcessInstance);
    private nint instance, physical, device, queue, command;
    private ulong descriptorLayout, pipelineLayout, shader, pipeline, descriptorPool, descriptorSet, commandPool, fence;
    private readonly GpuBuffer[] buffers = [new(), new(), new(), new()];
    private readonly GpuBuffer[] staging = [new(), new(), new(), new()];
    private readonly object gate = new();
    private readonly Dictionary<string, ulong> convolutionPipelines = [];
    private readonly Dictionary<Tensor, GpuBuffer> constants = [];
    private bool disposed;
    private uint timestampBits;
    /// <summary>取得包含设备名称的后端显示名称。</summary>
    /// <value>后端类型及初始化时查询到的设备名称。</value>
    public string Name { get; private set; } = "Vulkan";
    /// <summary>保存 Vulkan 缓冲区、绑定内存和可选主机映射地址；由后端统一分配与释放。</summary>
    private sealed class GpuBuffer
    {
        /// <summary>Vulkan 缓冲区句柄、绑定内存句柄和字节容量。</summary>
        public ulong Buffer, Memory, Capacity;
        /// <summary>主机映射地址；未映射时为 0。</summary>
        public nint Mapped;
        /// <summary>常量数据已成功提交并完成上传时为 true。</summary>
        public bool Ready;
    }
    /// <summary>创建 Vulkan FP32 计算后端；设备内存、上传区和命令资源由本实例持有。</summary>
    /// <exception cref="PlatformNotSupportedException">进程不是 64 位。</exception>
    /// <exception cref="InvalidOperationException">找不到计算设备、缺少着色器资源或 Vulkan 初始化失败。</exception>
    public VulkanBackend()
    {
        if (IntPtr.Size != 8) throw new PlatformNotSupportedException("Vulkan backend requires 64-bit .NET.");
        try { Initialize(); if (timestampBits == 0) gpuProfile = false; } catch { Dispose(); throw; }
    }
    /// <summary>按需创建进程唯一的 Vulkan 实例，避免最后会话释放时反复卸载驱动模块。</summary>
    /// <returns>在进程退出前保持有效的实例句柄；实例本身由操作系统随进程回收。</returns>
    /// <remarks>实例没有会话显存或命令资源。保留一个实例可以避免驱动卸载时的后台线程竞态，并减少后续模型加载开销。</remarks>
    /// <exception cref="InvalidOperationException">Vulkan 实例创建失败。</exception>
    private static nint CreateProcessInstance()
    {
        InstanceInfo info = new() { Type = 1 }; nint result = 0;
        Check(Vk.CreateInstance(&info, null, &result), "CreateInstance"); return result;
    }
    /// <summary>建立 Vulkan 实例、计算队列、基础管线、描述符、命令池和提交栅栏。</summary>
    /// <exception cref="InvalidOperationException">没有 Vulkan 设备或计算队列、缺少嵌入着色器，或 Vulkan 资源初始化失败。</exception>
    private void Initialize()
    {
        fixed (nint* devicePtr = &device, queuePtr = &queue, commandPtr = &command)
        fixed (ulong* descriptorLayoutPtr = &descriptorLayout, pipelineLayoutPtr = &pipelineLayout, shaderPtr = &shader, pipelinePtr = &pipeline, descriptorPoolPtr = &descriptorPool, descriptorSetPtr = &descriptorSet, commandPoolPtr = &commandPool, fencePtr = &fence)
        {
            instance = processInstance.Value;
            uint count = 0; Check(Vk.EnumeratePhysicalDevices(instance, &count, null), "EnumeratePhysicalDevices");
            if (count == 0) throw new InvalidOperationException("No Vulkan GPU found.");
            nint[] devices = new nint[count]; fixed (nint* ptr = devices) Check(Vk.EnumeratePhysicalDevices(instance, &count, ptr), "EnumeratePhysicalDevices");
            uint family = uint.MaxValue;
            // 优先选择独立计算设备，避免加载器先枚举集成显卡或软件适配器；
            // 同一设备类型内部保留原枚举次序。
            byte* candidateProperties = stackalloc byte[4096];
            var rankedDevices = new List<(nint Device, int Rank)>();
            foreach (nint gpu in devices)
            {
                Vk.GetProperties(gpu, candidateProperties);
                int rank = *(uint*)(candidateProperties + 16) switch { 2 => 0, 1 => 1, 3 => 2, 4 => 4, _ => 3 };
                rankedDevices.Add((gpu, rank));
            }
            foreach (var candidate in rankedDevices.OrderBy(candidate => candidate.Rank))
            {
                nint gpu = candidate.Device;
                uint qc = 0; Vk.GetQueueFamilies(gpu, &qc, null); QueueFamily[] qs = new QueueFamily[qc];
                fixed (QueueFamily* qp = qs) Vk.GetQueueFamilies(gpu, &qc, qp);
                for (uint i = 0; i < qc; i++) if ((qs[i].Flags & 2) != 0 && qs[i].Count > 0) { physical = gpu; family = i; timestampBits = qs[i].TimestampBits; break; }
                if (physical != 0) break;
            }
            if (physical == 0) throw new InvalidOperationException("No Vulkan compute queue found.");
            byte* properties = stackalloc byte[4096]; Vk.GetProperties(physical, properties);
            Name = "Vulkan (" + Marshal.PtrToStringUTF8((nint)(properties + 20)) + ")";
            float priority = 1; DeviceQueueInfo qi = new() { Type = 2, Family = family, Count = 1, Priorities = &priority };
            DeviceInfo di = new() { Type = 3, QueueCount = 1, Queues = &qi };
            Check(Vk.CreateDevice(physical, &di, null, devicePtr), "CreateDevice"); Vk.GetQueue(device, family, 0, queuePtr);
            InitializePipelineCache(properties);
            Binding* bindings = stackalloc Binding[4]; for (uint i = 0; i < 4; i++) bindings[i] = new() { Index = i, Type = 7, Count = 1, Stages = 32 };
            LayoutInfo li = new() { Type = 32, Count = 4, Bindings = bindings }; Check(Vk.CreateDescriptorLayout(device, &li, null, descriptorLayoutPtr), "CreateDescriptorSetLayout");
            PushRange range = new() { Stages = 32, Size = 128 }; ulong layout = descriptorLayout;
            PipelineLayoutInfo pi = new() { Type = 30, SetCount = 1, Sets = &layout, PushCount = 1, Pushes = &range }; Check(Vk.CreatePipelineLayout(device, &pi, null, pipelineLayoutPtr), "CreatePipelineLayout");
            var assembly = typeof(VulkanBackend).Assembly; string resource = assembly.GetManifestResourceNames().SingleOrDefault(n => n.EndsWith("gemm.spv", StringComparison.Ordinal)) ?? throw new InvalidOperationException("Embedded Vulkan SPIR-V missing. Compile shaders/gemm.comp with tools/build-shaders.ps1.");
            using var shaderStream = assembly.GetManifestResourceStream(resource)!; using var memory = new MemoryStream(); shaderStream.CopyTo(memory); byte[] code = memory.ToArray();
            fixed (byte* ptr = code) { ShaderInfo si = new() { Type = 16, Size = (nuint)code.Length, Code = (uint*)ptr }; Check(Vk.CreateShader(device, &si, null, shaderPtr), "CreateShaderModule"); }
            byte[] entry = Encoding.UTF8.GetBytes("main\0");
            fixed (byte* ptr = entry)
            {
                ComputePipelineInfo pc = new() { Type = 29, Stage = new() { Type = 18, Stage = 32, Module = shader, Name = ptr }, Layout = pipelineLayout, BaseIndex = -1 };
                Check(CreateCachedComputePipeline(&pc, pipelinePtr), "CreateComputePipelines");
            }
            PoolSize size = new() { Type = 7, Count = 4 }; DescriptorPoolInfo dpi = new() { Type = 33, MaxSets = 1, SizeCount = 1, Sizes = &size }; Check(Vk.CreateDescriptorPool(device, &dpi, null, descriptorPoolPtr), "CreateDescriptorPool");
            DescriptorAllocateInfo da = new() { Type = 34, Pool = descriptorPool, Count = 1, Layouts = &layout }; Check(Vk.AllocateDescriptors(device, &da, descriptorSetPtr), "AllocateDescriptorSets");
            CommandPoolInfo cp = new() { Type = 39, Flags = 2, Family = family }; Check(Vk.CreateCommandPool(device, &cp, null, commandPoolPtr), "CreateCommandPool");
            CommandAllocateInfo ca = new() { Type = 40, Pool = commandPool, Count = 1 }; Check(Vk.AllocateCommands(device, &ca, commandPtr), "AllocateCommandBuffers");
            FenceInfo fi = new() { Type = 8 }; Check(Vk.CreateFence(device, &fi, null, fencePtr), "CreateFence");
        }
    }
    /// <summary>计算行主序 FP32 矩阵乘法 C = A × B，并等待设备完成后写回 C。</summary>
    /// <param name="a">按行存储的左矩阵 A，至少包含 m × k 个元素。</param>
    /// <param name="b">按行存储的右矩阵 B，至少包含 k × n 个元素。</param>
    /// <param name="c">接收行主序结果 C 的调用方数组，至少包含 m × n 个元素。</param>
    /// <param name="m">左矩阵和输出矩阵的行数。</param>
    /// <param name="n">右矩阵和输出矩阵的列数。</param>
    /// <param name="k">矩阵乘法的归约深度。</param>
    /// <exception cref="ArgumentException">维度不是正数，或输入、输出数组长度不足。</exception>
    /// <exception cref="OverflowException">矩阵元素数超出 Int32 范围。</exception>
    /// <exception cref="ObjectDisposedException">后端已经释放。</exception>
    /// <exception cref="InvalidOperationException">设备资源分配、提交或计算失败。</exception>
    /// <exception cref="NotSupportedException">矩阵分块调度数量超过可移植线程组上限。</exception>
    public void MatMul(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        if (m <= 0 || n <= 0 || k <= 0 || a.Length < checked(m * k) || b.Length < checked(k * n) || c.Length < checked(m * n)) throw new ArgumentException("Invalid GPU GEMM dimensions.");
        if ((n + 15L) / 16 > 65535 || (m + 15L) / 16 > 65535) throw new NotSupportedException("GPU dispatch exceeds portable group limits.");
        Run(a, m * k, b, k * n, [0], c, m * n, [(uint)m, (uint)n, (uint)k, 0, 0, 0, 1, 1], pipeline, (uint)(n + 15) / 16, (uint)(m + 15) / 16, 1);
    }
    /// <summary>执行 NCHW FP32 卷积，可融合偏置和 SiLU，并把结果写入调用方输出张量。</summary>
    /// <param name="input">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="weights">NCHW 卷积权重。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="output">调用方提供的结果张量。</param>
    /// <param name="strides">高、宽方向卷积步长。</param>
    /// <param name="pads">上、左、下、右方向的填充量。</param>
    /// <param name="dilation">高、宽方向的卷积膨胀系数。</param>
    /// <param name="groups">卷积分组数，输入和输出通道须满足该分组布局。</param>
    /// <param name="silu">是否在偏置相加后融合 SiLU 激活。</param>
    /// <exception cref="ObjectDisposedException">后端已经释放。</exception>
    /// <exception cref="InvalidOperationException">设备管线创建、资源分配或计算提交失败。</exception>
    public void Conv(Tensor input, Tensor weights, Tensor? bias, Tensor output, int[] strides, int[] pads, int[] dilation, int groups, bool silu)
    {
        uint[] parameters = [silu ? 1u : 0u, (uint)input.Shape[1], (uint)input.Shape[2], (uint)input.Shape[3], (uint)output.Shape[1], (uint)output.Shape[2], (uint)output.Shape[3], (uint)weights.Shape[2], (uint)weights.Shape[3], (uint)strides[0], (uint)strides[1], (uint)pads[0], (uint)pads[1], (uint)dilation[0], (uint)dilation[1], (uint)groups];
        int spatial = output.Shape[2] * output.Shape[3];
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this); string key = string.Join(',', parameters);
            if (!convolutionPipelines.TryGetValue(key, out ulong specialized))
            {
                if (convolutionPipelines.Count >= 128) { foreach (ulong previous in convolutionPipelines.Values) Vk.DestroyPipeline(device, previous, null); convolutionPipelines.Clear(); convolutionTiles.Clear(); }
                specialized = CreateConvPipeline(parameters); convolutionPipelines[key] = specialized;
            }
            uint tile = ConvTile(parameters);
            uint rows = tile == 64 ? 32 : tile;
            Run(weights.Data, weights.Length, input.Data, input.Length, bias?.Data ?? new float[output.Shape[1]], output.Data, output.Length, parameters, specialized, ((uint)spatial + tile - 1) / tile, ((uint)(output.Shape[1] / groups) + rows - 1) / rows, (uint)(input.Shape[0] * groups), weights, bias);
        }
    }
    /// <summary>按卷积参数和分块规格创建专用计算管线；临时着色器模块在返回前释放。</summary>
    /// <param name="parameters">按对应着色器参数布局排列的 uint 值，用于特化常量或推送常量。</param>
    /// <param name="tile">卷积输出分块标识；0 表示按形状选择基础分块。</param>
    /// <returns>由后端缓存并统一释放的 Vulkan 计算管线句柄。</returns>
    /// <exception cref="InvalidOperationException">特定寄存器分块不适用于卷积形状，或 Vulkan 着色器、管线创建失败。</exception>
    private ulong CreateConvPipeline(uint[] parameters, uint tile = 0)
    {
        if (tile == 0) tile = ConvTile(parameters);
        if (tile == 256)
        {
            if (!RegisterPointwiseShape(parameters)) throw new InvalidOperationException("The register pointwise tile requires an eligible 1x1 convolution.");
            return CreateNamedPipeline("point64threads256.spv", parameters);
        }
        var assembly = typeof(VulkanBackend).Assembly; string suffix = tile switch { 80 => "conv64k16.spv", 96 => "conv64k32.spv", 128 => "conv128k16.spv", 64 => "conv64.spv", 32 => "conv.spv", _ => "conv16.spv" };
        if (wideConvolutionTiles && parameters[7] == 1 && parameters[8] == 1 && parameters[9] == 1 && parameters[10] == 1 && parameters[11] == 0 && parameters[12] == 0 && parameters[13] == 1 && parameters[14] == 1)
            suffix = tile == 32 ? "point32.spv" : suffix.Replace("conv", "point", StringComparison.Ordinal);
        string resource = assembly.GetManifestResourceNames().Single(n => n.EndsWith(suffix, StringComparison.Ordinal));
        using var stream = assembly.GetManifestResourceStream(resource)!; using var memory = new MemoryStream(); stream.CopyTo(memory); byte[] code = memory.ToArray(); ulong module = 0, result = 0;
        fixed (byte* ptr = code) { ShaderInfo si = new() { Type = 16, Size = (nuint)code.Length, Code = (uint*)ptr }; Check(Vk.CreateShader(device, &si, null, &module), "CreateShaderModule(conv)"); }
        try
        {
            SpecializationEntry* entries = stackalloc SpecializationEntry[16]; for (uint i = 0; i < 16; i++) entries[i] = new() { Id = i, Offset = i * 4, Size = 4 };
            byte[] entry = Encoding.UTF8.GetBytes("main\0");
            fixed (byte* ptr = entry) fixed (uint* data = parameters)
            {
                SpecializationInfo specialization = new() { Count = 16, Entries = entries, Size = 64, Data = data };
                ComputePipelineInfo pc = new() { Type = 29, Stage = new() { Type = 18, Stage = 32, Module = module, Name = ptr, Specialization = (nint)(&specialization) }, Layout = pipelineLayout, BaseIndex = -1 }; Check(CreateCachedComputePipeline(&pc, &result), "CreateComputePipelines(conv)");
            }
            return result;
        }
        finally { Vk.DestroyShader(device, module, null); }
    }
    /// <summary>根据每组输出通道和归约深度选择通用卷积的基础分块大小。</summary>
    /// <param name="parameters">按对应着色器参数布局排列的 uint 值，用于特化常量或推送常量。</param>
    /// <returns>以输出空间位置为单位的基础分块大小，值为 16 或 32。</returns>
    private uint ConvTile(uint[] parameters)
    {
        uint rows = parameters[4] / parameters[15], k = parameters[1] / parameters[15] * parameters[7] * parameters[8];
        return rows >= 32 && k >= (lowKConvolutionTiles ? 16 : 64) ? 32u : 16u;
    }
    /// <summary>上传输入、绑定计算资源、提交设备命令并同步下载结果；常量缓冲区按张量身份缓存。</summary>
    /// <param name="a">第一输入的主机 FP32 数据；卷积时为权重，矩阵乘法时为左矩阵。</param>
    /// <param name="acount">第一输入需要上传的 FP32 元素数。</param>
    /// <param name="b">第二输入的主机 FP32 数据；卷积时为特征图，矩阵乘法时为右矩阵。</param>
    /// <param name="bcount">第二输入需要上传的 FP32 元素数。</param>
    /// <param name="bias">每个输出通道的主机偏置数据；矩阵乘法使用零占位数据。</param>
    /// <param name="c">接收同步下载结果的调用方 FP32 数组。</param>
    /// <param name="ccount">需要下载到输出数组的 FP32 元素数。</param>
    /// <param name="parameters">按对应着色器参数布局排列的 uint 值，用于特化常量或推送常量。</param>
    /// <param name="selectedPipeline">本次调度使用的计算管线句柄。</param>
    /// <param name="gx">X 方向线程组数。</param>
    /// <param name="gy">Y 方向线程组数。</param>
    /// <param name="gz">Z 方向线程组数。</param>
    /// <param name="constantA">第一输入对应的常量张量，用于选择持久缓存；为空时使用临时缓冲区。</param>
    /// <param name="constantBias">偏置对应的常量张量，用于选择持久缓存；为空时使用临时缓冲区。</param>
    /// <exception cref="ObjectDisposedException">后端已经释放。</exception>
    /// <exception cref="InvalidOperationException">设备缓冲区分配、提交、等待或计算失败。</exception>
    /// <exception cref="NotSupportedException">任一调度线程组数量超过 65535。</exception>
    private void Run(float[] a, int acount, float[] b, int bcount, float[] bias, float[] c, int ccount, uint[] parameters, ulong selectedPipeline, uint gx, uint gy, uint gz, Tensor? constantA = null, Tensor? constantBias = null)
    {
        if (gx > 65535 || gy > 65535 || gz > 65535) throw new NotSupportedException("Vulkan dispatch exceeds portable limits.");
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            // 常量第一次使用前上传，后续使用已完成的设备副本；可变输入始终使用当前值上传。
            GpuBuffer Select(Tensor? tensor, GpuBuffer fallback, ulong capacity, out bool upload)
            {
                upload = true;
                if (tensor?.IsConstant != true) { Ensure(fallback, capacity, false); return fallback; }
                if (constants.TryGetValue(tensor, out var cached)) { upload = !cached.Ready; return cached; }
                var created = new GpuBuffer(); constants.Add(tensor, created); Ensure(created, capacity, false); return created;
            }
            var selectedA = Select(constantA, buffers[0], (ulong)acount * 4, out bool uploadA);
            var selectedBias = Select(constantBias, buffers[3], (ulong)bias.Length * 4, out bool uploadBias);
            Ensure(buffers[1], (ulong)bcount * 4, false); Ensure(buffers[2], (ulong)ccount * 4, false);
            Ensure(staging[0], (ulong)acount * 4, true); Ensure(staging[1], (ulong)bcount * 4, true); Ensure(staging[2], (ulong)ccount * 4, true, 8); Ensure(staging[3], (ulong)bias.Length * 4, true);
            if (uploadA) fixed (float* ap = a) Buffer.MemoryCopy(ap, (void*)staging[0].Mapped, (long)staging[0].Capacity, (long)acount * 4);
            fixed (float* bp = b) Buffer.MemoryCopy(bp, (void*)staging[1].Mapped, (long)staging[1].Capacity, (long)bcount * 4);
            if (uploadBias) fixed (float* dp = bias) Buffer.MemoryCopy(dp, (void*)staging[3].Mapped, (long)staging[3].Capacity, (long)bias.Length * 4);
            {
                DescriptorBufferInfo* infos = stackalloc DescriptorBufferInfo[4]; WriteDescriptor* writes = stackalloc WriteDescriptor[4];
                for (uint i = 0; i < 4; i++) { var selected = i == 0 ? selectedA : i == 3 ? selectedBias : buffers[i]; infos[i] = new() { Buffer = selected.Buffer, Range = selected.Capacity }; writes[i] = new() { Type = 35, Set = descriptorSet, Binding = i, Count = 1, DescriptorType = 7, Buffers = infos + i }; }
                Vk.UpdateDescriptors(device, 4, writes, 0, null);
            }
            InvalidateReplay();
            Check(Vk.ResetCommand(command, 0), "ResetCommandBuffer"); CommandBeginInfo begin = new() { Type = 42, Flags = 1 }; Check(Vk.BeginCommand(command, &begin), "BeginCommandBuffer");
            BufferCopy copyA = new() { Size = (ulong)acount * 4 }, copyB = new() { Size = (ulong)bcount * 4 }, copyBias = new() { Size = (ulong)bias.Length * 4 };
            if (uploadA) Vk.CopyBuffer(command, staging[0].Buffer, selectedA.Buffer, 1, &copyA); Vk.CopyBuffer(command, staging[1].Buffer, buffers[1].Buffer, 1, &copyB);
            if (uploadBias) Vk.CopyBuffer(command, staging[3].Buffer, selectedBias.Buffer, 1, &copyBias);
            MemoryBarrier upload = new() { Type = 46, SourceAccess = 4096, DestinationAccess = 32 }; Vk.PipelineBarrier(command, 4096, 2048, 0, 1, &upload, 0, null, 0, null);
            Vk.BindPipeline(command, 1, selectedPipeline); ulong set = descriptorSet; Vk.BindDescriptorSets(command, 1, pipelineLayout, 0, 1, &set, 0, null);
            fixed (uint* dims = parameters) Vk.PushConstants(command, pipelineLayout, 32, 0, (uint)parameters.Length * 4, dims);
            Vk.Dispatch(command, gx, gy, gz);
            MemoryBarrier download = new() { Type = 46, SourceAccess = 64, DestinationAccess = 2048 }; Vk.PipelineBarrier(command, 2048, 4096, 0, 1, &download, 0, null, 0, null);
            BufferCopy copyC = new() { Size = (ulong)ccount * 4 }; Vk.CopyBuffer(command, buffers[2].Buffer, staging[2].Buffer, 1, &copyC);
            MemoryBarrier barrier = new() { Type = 46, SourceAccess = 4096, DestinationAccess = 8192 }; Vk.PipelineBarrier(command, 4096, 16384, 0, 1, &barrier, 0, null, 0, null);
            Check(Vk.EndCommand(command), "EndCommandBuffer"); ulong f = fence; Check(Vk.ResetFences(device, 1, &f), "ResetFences"); nint cb = command;
            SubmitInfo submit = new() { Type = 4, CommandCount = 1, Commands = &cb }; Check(Vk.Submit(queue, 1, &submit, fence), "QueueSubmit"); WaitForSubmission(f);
            selectedA.Ready = selectedBias.Ready = true;
            fixed (float* cp = c) Buffer.MemoryCopy((void*)staging[2].Mapped, cp, (long)c.Length * 4, (long)ccount * 4);
        }
    }
    /// <summary>确保 Vulkan 缓冲区容量足够，必要时重建并按主机可见或设备本地要求分配内存。</summary>
    /// <param name="b">由当前后端持有的 Vulkan 缓冲区、内存和映射信息。</param>
    /// <param name="capacity">所需容量，单位为字节。</param>
    /// <param name="host">是否使用主机可见且一致的映射内存。</param>
    /// <param name="preferred">优先选择的附加 Vulkan 内存属性位。</param>
    /// <returns>发生资源扩容重建时为 true；原有容量足够时为 false。</returns>
    /// <exception cref="InvalidOperationException">设备没有兼容的内存类型，或缓冲区分配、绑定、映射失败。</exception>
    private bool Ensure(GpuBuffer b, ulong capacity, bool host, uint preferred = 0)
    {
        if (b.Capacity >= capacity) return false;
        Free(b); b.Capacity = Math.Max(4096, capacity);
        BufferInfo info = new() { Type = 12, Size = b.Capacity, Usage = host ? 3u : 35u }; ulong buffer = 0; Check(Vk.CreateBuffer(device, &info, null, &buffer), "CreateBuffer"); b.Buffer = buffer;
        MemoryRequirements req = default; Vk.GetMemoryRequirements(device, b.Buffer, &req);
        byte* properties = stackalloc byte[1024]; Vk.GetMemoryProperties(physical, properties); uint count = *(uint*)properties, index = uint.MaxValue;
        uint required = host ? 6u : 1u;
        for (uint i = 0; i < count; i++)
        {
            uint flags = *(uint*)(properties + 4 + i * 8);
            if ((req.TypeBits & (1u << (int)i)) == 0 || (flags & required) != required) continue;
            if (index == uint.MaxValue) index = i;
            if ((flags & preferred) == preferred) { index = i; break; }
        }
        if (index == uint.MaxValue) throw new InvalidOperationException("GPU has no compatible buffer memory.");
        MemoryAllocateInfo ai = new() { Type = 5, Size = req.Size, Index = index }; ulong allocation = 0; Check(Vk.AllocateMemory(device, &ai, null, &allocation), "AllocateMemory"); b.Memory = allocation;
        Check(Vk.BindBufferMemory(device, b.Buffer, b.Memory, 0), "BindBufferMemory");
        if (host) { void* mapped = null; Check(Vk.MapMemory(device, b.Memory, 0, b.Capacity, 0, &mapped), "MapMemory"); b.Mapped = (nint)mapped; }
        return true;
    }
    /// <summary>解除主机映射并释放 Vulkan 缓冲区及绑定内存，随后清空资源句柄。</summary>
    /// <param name="b">由当前后端持有的 Vulkan 缓冲区、内存和映射信息。</param>
    private void Free(GpuBuffer b)
    {
        if (b.Mapped != 0) Vk.UnmapMemory(device, b.Memory);
        if (b.Buffer != 0) Vk.DestroyBuffer(device, b.Buffer, null);
        if (b.Memory != 0) Vk.FreeMemory(device, b.Memory, null);
        b.Buffer = b.Memory = b.Capacity = 0; b.Mapped = 0;
    }
    /// <summary>等待设备工作完成并释放此后端持有的缓冲区、管线、命令和设备资源；可重复调用。</summary>
    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return; disposed = true;
            if (device != 0)
            {
                Vk.WaitIdle(device); SavePipelineCache(); foreach (var b in buffers.Concat(staging)) Free(b);
                FreeResidentResources();
                foreach (var b in constants.Values) Free(b); constants.Clear();
                if (fence != 0) Vk.DestroyFence(device, fence, null);
                if (commandPool != 0) Vk.DestroyCommandPool(device, commandPool, null);
                if (descriptorPool != 0) Vk.DestroyDescriptorPool(device, descriptorPool, null);
                if (pipeline != 0) Vk.DestroyPipeline(device, pipeline, null);
                foreach (ulong previous in convolutionPipelines.Values) Vk.DestroyPipeline(device, previous, null); convolutionPipelines.Clear();
                if (shader != 0) Vk.DestroyShader(device, shader, null);
                if (pipelineLayout != 0) Vk.DestroyPipelineLayout(device, pipelineLayout, null);
                if (descriptorLayout != 0) Vk.DestroyDescriptorLayout(device, descriptorLayout, null);
                if (pipelineCache != 0) Vk.DestroyPipelineCache(device, pipelineCache, null); pipelineCache = 0;
                Vk.DestroyDevice(device, null); device = 0;
            }
            // VkInstance 属于进程，不能在单个会话释放时销毁；否则其他会话的物理设备句柄会失效。
            instance = 0;
        }
    }
    /// <summary>检查 Vulkan 返回码，把非成功状态转换成包含操作名和 VkResult 的托管异常。</summary>
    /// <param name="result">原生 VkResult；0 表示成功。</param>
    /// <param name="operation">失败消息中使用的 Vulkan 原生操作名称。</param>
    /// <exception cref="InvalidOperationException">原生返回码不为 0，即 VkResult 不表示成功。</exception>
    private static void Check(int result, string operation) { if (result != 0) throw new InvalidOperationException($"Vulkan {operation} failed: VkResult {result}."); }
    /// <summary>等待提交栅栏完成，确保设备读写已结束后再访问映射内存或复用资源。</summary>
    /// <param name="submittedFence">等待完成的 Vulkan 栅栏句柄。</param>
    private void WaitForSubmission(ulong submittedFence)
    {
        Check(Vk.WaitFences(device, 1, &submittedFence, 1, ulong.MaxValue), "WaitForFences");
    }
    /// <summary>对应 VkInstanceCreateInfo，保持 Vulkan 原生字段顺序和平台对齐。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct InstanceInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。</summary>
        public uint Flags;
        /// <summary>可选应用信息结构指针。</summary>
        public nint App;
        /// <summary>启用层的数量。</summary>
        public uint LayerCount;
        /// <summary>层名称数组指针。</summary>
        public nint Layers;
        /// <summary>启用扩展数量。</summary>
        public uint ExtensionCount;
        /// <summary>扩展名称数组指针。</summary>
        public nint Extensions;
    }
    /// <summary>对应 VkQueueFamilyProperties，包含队列能力、数量、时间戳有效位和传输粒度。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct QueueFamily
    {
        /// <summary>对应 Vulkan 结构或资源的标志位。；队列、描述符或查询的数量，依所属原生结构解释。；队列时间戳有效位数；0 表示不支持时间戳。；传输粒度的 X 分量。；传输粒度的 Y 分量。；传输粒度的 Z 分量。</summary>
        public uint Flags, Count, TimestampBits, GranularityX, GranularityY, GranularityZ;
    }
    /// <summary>对应 VkDeviceQueueCreateInfo，指定队列族、队列数量和优先级指针。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct DeviceQueueInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。；队列族索引。；队列、描述符或查询的数量，依所属原生结构解释。</summary>
        public uint Flags, Family, Count;
        /// <summary>队列优先级数组的原生指针。</summary>
        public float* Priorities;
    }
    /// <summary>对应 VkDeviceCreateInfo，列出队列、扩展与功能启用信息。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct DeviceInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。；队列创建信息数或共享队列族索引数。</summary>
        public uint Flags, QueueCount;
        /// <summary>队列创建信息数组指针。</summary>
        public DeviceQueueInfo* Queues;
        /// <summary>启用层的数量。</summary>
        public uint LayerCount;
        /// <summary>层名称数组指针。</summary>
        public nint Layers;
        /// <summary>启用扩展数量。</summary>
        public uint ExtensionCount;
        /// <summary>扩展名称数组指针。；可选设备功能结构指针。</summary>
        public nint Extensions, Features;
    }
    /// <summary>对应 VkDescriptorSetLayoutBinding，描述存储缓冲区绑定的位置与计算阶段可见性。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct Binding
    {
        /// <summary>绑定位置或内存类型索引，依所属原生结构解释。；Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。；队列、描述符或查询的数量，依所属原生结构解释。；可见的着色器阶段掩码。</summary>
        public uint Index, Type, Count, Stages;
        /// <summary>可选不可变采样器数组指针。</summary>
        public nint Samplers;
    }
    /// <summary>对应 VkDescriptorSetLayoutCreateInfo，引用描述符绑定数组。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct LayoutInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。；队列、描述符或查询的数量，依所属原生结构解释。</summary>
        public uint Flags, Count;
        /// <summary>描述符绑定数组指针。</summary>
        public Binding* Bindings;
    }
    /// <summary>对应 VkPushConstantRange，描述推送常量可见阶段及字节范围。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct PushRange
    {
        /// <summary>可见的着色器阶段掩码。；数据区段的字节偏移。；结构引用的数据或资源大小，单位为字节。</summary>
        public uint Stages, Offset, Size;
    }
    /// <summary>对应 VkPipelineLayoutCreateInfo，引用描述符布局和推送常量范围。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct PipelineLayoutInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。；描述符集布局数量。</summary>
        public uint Flags, SetCount;
        /// <summary>描述符集布局句柄数组指针。</summary>
        public ulong* Sets;
        /// <summary>推送常量范围数量。</summary>
        public uint PushCount;
        /// <summary>推送常量范围数组指针。</summary>
        public PushRange* Pushes;
    }
    /// <summary>对应 VkShaderModuleCreateInfo，引用 SPIR-V 字节数与按 uint 对齐的代码。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct ShaderInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。</summary>
        public uint Flags;
        /// <summary>结构引用的数据或资源大小，单位为字节。</summary>
        public nuint Size;
        /// <summary>SPIR-V 代码的 uint 指针。</summary>
        public uint* Code;
    }
    /// <summary>对应 VkPipelineShaderStageCreateInfo，指定着色器模块、入口名及特化信息。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct ShaderStage
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。；计算着色器阶段或内嵌的阶段描述。</summary>
        public uint Flags, Stage;
        /// <summary>着色器模块句柄。</summary>
        public ulong Module;
        /// <summary>以零字节结尾的 UTF-8 着色器入口名指针。</summary>
        public byte* Name;
        /// <summary>可选特化常量结构指针。</summary>
        public nint Specialization;
    }
    /// <summary>对应 VkSpecializationMapEntry，把一个常量编号映射到特化数据的字节区段。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct SpecializationEntry
    {
        /// <summary>特化常量编号。；数据区段的字节偏移。</summary>
        public uint Id, Offset;
        /// <summary>结构引用的数据或资源大小，单位为字节。</summary>
        public nuint Size;
    }
    /// <summary>对应 VkSpecializationInfo，引用特化映射表及原始数据。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct SpecializationInfo
    {
        /// <summary>队列、描述符或查询的数量，依所属原生结构解释。</summary>
        public uint Count;
        /// <summary>特化常量映射数组指针。</summary>
        public SpecializationEntry* Entries;
        /// <summary>结构引用的数据或资源大小，单位为字节。</summary>
        public nuint Size;
        /// <summary>初始缓存或特化常量原始数据指针。</summary>
        public void* Data;
    }
    /// <summary>对应 VkComputePipelineCreateInfo，描述计算阶段、管线布局与派生管线。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct ComputePipelineInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。</summary>
        public uint Flags;
        /// <summary>计算着色器阶段或内嵌的阶段描述。</summary>
        public ShaderStage Stage;
        /// <summary>计算管线布局句柄。；可选派生管线的基础句柄。</summary>
        public ulong Layout, Base;
        /// <summary>基础管线在创建数组中的索引；-1 表示未指定。</summary>
        public int BaseIndex;
    }
    /// <summary>对应 VkDescriptorPoolSize，指定某类描述符的总数量。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct PoolSize
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。；队列、描述符或查询的数量，依所属原生结构解释。</summary>
        public uint Type, Count;
    }
    /// <summary>对应 VkDescriptorPoolCreateInfo，指定描述符池容量及每类描述符数量。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct DescriptorPoolInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。；池允许分配的最大描述符集数。；描述符容量种类数。</summary>
        public uint Flags, MaxSets, SizeCount;
        /// <summary>描述符类型与容量数组指针。</summary>
        public PoolSize* Sizes;
    }
    /// <summary>对应 VkDescriptorSetAllocateInfo，指定分配池和描述符布局。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct DescriptorAllocateInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>描述符池或命令池句柄。</summary>
        public ulong Pool;
        /// <summary>队列、描述符或查询的数量，依所属原生结构解释。</summary>
        public uint Count;
        /// <summary>待分配描述符集的布局句柄数组指针。</summary>
        public ulong* Layouts;
    }
    /// <summary>对应 VkCommandPoolCreateInfo，指定所属队列族和命令池标志。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct CommandPoolInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。；队列族索引。</summary>
        public uint Flags, Family;
    }
    /// <summary>对应 VkCommandBufferAllocateInfo，指定池、命令级别和分配数量。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct CommandAllocateInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>描述符池或命令池句柄。</summary>
        public ulong Pool;
        /// <summary>命令缓冲区级别。；队列、描述符或查询的数量，依所属原生结构解释。</summary>
        public uint Level, Count;
    }
    /// <summary>对应 VkCommandBufferBeginInfo，指定命令录制标志与继承信息。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct CommandBeginInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。</summary>
        public uint Flags;
        /// <summary>可选次级命令缓冲区继承信息指针。</summary>
        public nint Inheritance;
    }
    /// <summary>对应 VkFenceCreateInfo，指定栅栏初始状态标志。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct FenceInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。</summary>
        public uint Flags;
    }
    /// <summary>对应 VkBufferCreateInfo，指定字节容量、使用类型和队列共享模式。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct BufferInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。</summary>
        public uint Flags;
        /// <summary>结构引用的数据或资源大小，单位为字节。</summary>
        public ulong Size;
        /// <summary>缓冲区允许的使用方式掩码。；队列共享模式。；队列创建信息数或共享队列族索引数。</summary>
        public uint Usage, Sharing, QueueCount;
        /// <summary>共享队列族索引数组指针。</summary>
        public nint QueueFamilies;
    }
    /// <summary>对应 VkMemoryRequirements，返回容量、对齐要求和允许的内存类型位。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryRequirements
    {
        /// <summary>结构引用的数据或资源大小，单位为字节。；绑定内存要求的字节对齐。</summary>
        public ulong Size, Alignment;
        /// <summary>允许使用的内存类型位掩码。</summary>
        public uint TypeBits;
    }
    /// <summary>对应 VkMemoryAllocateInfo，指定分配字节数和内存类型索引。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryAllocateInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>结构引用的数据或资源大小，单位为字节。</summary>
        public ulong Size;
        /// <summary>绑定位置或内存类型索引，依所属原生结构解释。</summary>
        public uint Index;
    }
    /// <summary>对应 VkDescriptorBufferInfo，描述缓冲区句柄和可访问的字节范围。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct DescriptorBufferInfo
    {
        /// <summary>缓冲区句柄。；数据区段的字节偏移。；描述符可访问的字节范围。</summary>
        public ulong Buffer, Offset, Range;
    }
    /// <summary>对应 VkWriteDescriptorSet，写入目标绑定及关联缓冲区描述。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct WriteDescriptor
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>目标描述符集句柄。</summary>
        public ulong Set;
        /// <summary>目标绑定编号。；绑定数组中的起始元素索引。；队列、描述符或查询的数量，依所属原生结构解释。；写入的描述符类型。</summary>
        public uint Binding, Element, Count, DescriptorType;
        /// <summary>可选图像描述数组指针。</summary>
        public nint Images;
        /// <summary>缓冲区描述数组指针。</summary>
        public DescriptorBufferInfo* Buffers;
        /// <summary>可选纹素缓冲视图数组指针。</summary>
        public nint Texels;
    }
    /// <summary>对应 VkSubmitInfo，引用本次提交的命令缓冲区与可选同步信号量。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct SubmitInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>需要等待的信号量数量。</summary>
        public uint WaitCount;
        /// <summary>等待信号量数组指针。；各等待信号量对应的阶段掩码数组指针。</summary>
        public nint Waits, WaitStages;
        /// <summary>提交的命令缓冲区数。</summary>
        public uint CommandCount;
        /// <summary>命令缓冲区句柄数组指针。</summary>
        public nint* Commands;
        /// <summary>提交完成后发出的信号量数量。</summary>
        public uint SignalCount;
        /// <summary>发出信号量数组指针。</summary>
        public nint Signals;
    }
    /// <summary>对应 VkMemoryBarrier，指定前序与后续访问的内存可见性关系。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryBarrier
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>前序操作访问掩码。；后续操作访问掩码。</summary>
        public uint SourceAccess, DestinationAccess;
    }
    /// <summary>对应 VkBufferCopy，记录源、目标字节偏移和拷贝字节数。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct BufferCopy
    {
        /// <summary>源缓冲区的字节偏移。；目标缓冲区的字节偏移。；结构引用的数据或资源大小，单位为字节。</summary>
        public ulong SourceOffset, DestinationOffset, Size;
    }
    /// <summary>保存 Vulkan 原生函数指针；调用者负责传递有效句柄、缓冲区容量和指针生命周期。</summary>
    private static partial class Vk
    {
        private static readonly nint Library = NativeLibrary.Load(OperatingSystem.IsWindows() ? "vulkan-1.dll" : "libvulkan.so.1");
        /// <summary>从已加载的 Vulkan 库取得指定 vk 入口的原生函数地址。</summary>
        /// <param name="name">Vulkan 导出入口的名称后缀，实际查找时添加 vk 前缀。</param>
        /// <returns>对应 Vulkan 原生入口的函数指针。</returns>
        /// <exception cref="EntryPointNotFoundException">原生库没有导出所请求的 Vulkan 入口。</exception>
        private static void* Load(string name) => (void*)NativeLibrary.GetExport(Library, "vk" + name);
        /// <summary>创建 Vulkan 实例。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：创建信息指针、可选分配器指针、实例输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<InstanceInfo*, void*, nint*, int> CreateInstance = (delegate* unmanaged<InstanceInfo*, void*, nint*, int>)Load("CreateInstance");
        /// <summary>销毁 Vulkan 实例。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：实例句柄、可选分配器指针。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, void*, void> DestroyInstance = (delegate* unmanaged<nint, void*, void>)Load("DestroyInstance");
        /// <summary>枚举实例可见的物理设备。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：实例句柄、输入容量/输出数量指针、可选设备输出数组。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, uint*, nint*, int> EnumeratePhysicalDevices = (delegate* unmanaged<nint, uint*, nint*, int>)Load("EnumeratePhysicalDevices");
        /// <summary>查询物理设备队列族属性。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：物理设备句柄、输入容量/输出数量指针、可选队列属性数组。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, uint*, QueueFamily*, void> GetQueueFamilies = (delegate* unmanaged<nint, uint*, QueueFamily*, void>)Load("GetPhysicalDeviceQueueFamilyProperties");
        /// <summary>查询物理设备标识、名称和能力属性。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：物理设备句柄、属性结构输出地址。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, byte*, void> GetProperties = (delegate* unmanaged<nint, byte*, void>)Load("GetPhysicalDeviceProperties");
        /// <summary>查询设备内存类型和堆属性。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：物理设备句柄、内存属性结构输出地址。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, byte*, void> GetMemoryProperties = (delegate* unmanaged<nint, byte*, void>)Load("GetPhysicalDeviceMemoryProperties");
        /// <summary>创建逻辑设备。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：物理设备句柄、设备创建信息、可选分配器、设备输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, DeviceInfo*, void*, nint*, int> CreateDevice = (delegate* unmanaged<nint, DeviceInfo*, void*, nint*, int>)Load("CreateDevice");
        /// <summary>销毁逻辑设备。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：逻辑设备句柄、可选分配器指针。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, void*, void> DestroyDevice = (delegate* unmanaged<nint, void*, void>)Load("DestroyDevice");
        /// <summary>取得已创建的计算队列。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、队列族索引、族内队列索引、队列输出指针。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, uint, uint, nint*, void> GetQueue = (delegate* unmanaged<nint, uint, uint, nint*, void>)Load("GetDeviceQueue");
        /// <summary>创建描述符集布局。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、布局描述、可选分配器、布局输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, LayoutInfo*, void*, ulong*, int> CreateDescriptorLayout = (delegate* unmanaged<nint, LayoutInfo*, void*, ulong*, int>)Load("CreateDescriptorSetLayout");
        /// <summary>创建计算管线布局。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、布局描述、可选分配器、布局输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, PipelineLayoutInfo*, void*, ulong*, int> CreatePipelineLayout = (delegate* unmanaged<nint, PipelineLayoutInfo*, void*, ulong*, int>)Load("CreatePipelineLayout");
        /// <summary>创建 SPIR-V 着色器模块。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、代码描述、可选分配器、模块输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, ShaderInfo*, void*, ulong*, int> CreateShader = (delegate* unmanaged<nint, ShaderInfo*, void*, ulong*, int>)Load("CreateShaderModule");
        /// <summary>创建一个或多个计算管线。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、管线缓存、创建数量、创建描述数组、可选分配器、管线输出数组。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, uint, ComputePipelineInfo*, void*, ulong*, int> CreateComputePipelines = (delegate* unmanaged<nint, ulong, uint, ComputePipelineInfo*, void*, ulong*, int>)Load("CreateComputePipelines");
        /// <summary>创建描述符池。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、池容量描述、可选分配器、池输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, DescriptorPoolInfo*, void*, ulong*, int> CreateDescriptorPool = (delegate* unmanaged<nint, DescriptorPoolInfo*, void*, ulong*, int>)Load("CreateDescriptorPool");
        /// <summary>从描述符池分配描述符集。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、分配描述、描述符集输出数组。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, DescriptorAllocateInfo*, ulong*, int> AllocateDescriptors = (delegate* unmanaged<nint, DescriptorAllocateInfo*, ulong*, int>)Load("AllocateDescriptorSets");
        /// <summary>更新或复制描述符集绑定。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、写入数量、写入描述数组、复制数量、复制描述数组。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, uint, WriteDescriptor*, uint, void*, void> UpdateDescriptors = (delegate* unmanaged<nint, uint, WriteDescriptor*, uint, void*, void>)Load("UpdateDescriptorSets");
        /// <summary>创建命令池。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、命令池描述、可选分配器、池输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, CommandPoolInfo*, void*, ulong*, int> CreateCommandPool = (delegate* unmanaged<nint, CommandPoolInfo*, void*, ulong*, int>)Load("CreateCommandPool");
        /// <summary>从命令池分配命令缓冲区。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、分配描述、命令缓冲区输出数组。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, CommandAllocateInfo*, nint*, int> AllocateCommands = (delegate* unmanaged<nint, CommandAllocateInfo*, nint*, int>)Load("AllocateCommandBuffers");
        /// <summary>创建提交完成栅栏。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、栅栏描述、可选分配器、栅栏输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, FenceInfo*, void*, ulong*, int> CreateFence = (delegate* unmanaged<nint, FenceInfo*, void*, ulong*, int>)Load("CreateFence");
        /// <summary>创建设备缓冲区。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、缓冲区描述、可选分配器、缓冲区输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, BufferInfo*, void*, ulong*, int> CreateBuffer = (delegate* unmanaged<nint, BufferInfo*, void*, ulong*, int>)Load("CreateBuffer");
        /// <summary>查询缓冲区的内存大小、对齐和允许的内存类型。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、缓冲区句柄、需求结构输出指针。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, MemoryRequirements*, void> GetMemoryRequirements = (delegate* unmanaged<nint, ulong, MemoryRequirements*, void>)Load("GetBufferMemoryRequirements");
        /// <summary>分配 Vulkan 设备内存。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、内存分配描述、可选分配器、内存输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, MemoryAllocateInfo*, void*, ulong*, int> AllocateMemory = (delegate* unmanaged<nint, MemoryAllocateInfo*, void*, ulong*, int>)Load("AllocateMemory");
        /// <summary>将内存区段绑定到缓冲区。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、缓冲区句柄、内存句柄、内存字节偏移。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, ulong, ulong, int> BindBufferMemory = (delegate* unmanaged<nint, ulong, ulong, ulong, int>)Load("BindBufferMemory");
        /// <summary>把主机可见内存映射到进程地址空间。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、内存句柄、起始字节偏移、映射字节数、标志、映射地址输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, ulong, ulong, uint, void**, int> MapMemory = (delegate* unmanaged<nint, ulong, ulong, ulong, uint, void**, int>)Load("MapMemory");
        /// <summary>解除主机内存映射。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、内存句柄。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void> UnmapMemory = (delegate* unmanaged<nint, ulong, void>)Load("UnmapMemory");
        /// <summary>重置已完成的命令缓冲区。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区句柄、重置标志。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, uint, int> ResetCommand = (delegate* unmanaged<nint, uint, int>)Load("ResetCommandBuffer");
        /// <summary>开始命令缓冲区录制。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区句柄、开始录制描述。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, CommandBeginInfo*, int> BeginCommand = (delegate* unmanaged<nint, CommandBeginInfo*, int>)Load("BeginCommandBuffer");
        /// <summary>结束命令缓冲区录制。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区句柄。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, int> EndCommand = (delegate* unmanaged<nint, int>)Load("EndCommandBuffer");
        /// <summary>录制计算管线绑定命令。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区句柄、管线绑定点、计算管线句柄。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, uint, ulong, void> BindPipeline = (delegate* unmanaged<nint, uint, ulong, void>)Load("CmdBindPipeline");
        /// <summary>录制描述符集与可选动态偏移绑定命令。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区、绑定点、管线布局、首个集索引、集数量、集句柄数组、动态偏移数量、动态偏移数组。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, uint, ulong, uint, uint, ulong*, uint, uint*, void> BindDescriptorSets = (delegate* unmanaged<nint, uint, ulong, uint, uint, ulong*, uint, uint*, void>)Load("CmdBindDescriptorSets");
        /// <summary>录制推送常量更新命令。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区、管线布局、着色器阶段掩码、字节偏移、字节大小、数据指针。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, uint, uint, uint, void*, void> PushConstants = (delegate* unmanaged<nint, ulong, uint, uint, uint, void*, void>)Load("CmdPushConstants");
        /// <summary>录制三维线程组计算命令。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区句柄、X 线程组数、Y 线程组数、Z 线程组数。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, uint, uint, uint, void> Dispatch = (delegate* unmanaged<nint, uint, uint, uint, void>)Load("CmdDispatch");
        /// <summary>录制设备缓冲区之间的区域拷贝命令。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区、源缓冲区、目标缓冲区、区域数量、拷贝区域描述数组。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, ulong, uint, BufferCopy*, void> CopyBuffer = (delegate* unmanaged<nint, ulong, ulong, uint, BufferCopy*, void>)Load("CmdCopyBuffer");
        /// <summary>录制访问可见性与执行阶段屏障命令。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：命令缓冲区、源阶段、目标阶段、依赖标志、内存屏障数量及数组、缓冲屏障数量及数组、图像屏障数量及数组。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, uint, uint, uint, uint, MemoryBarrier*, uint, void*, uint, void*, void> PipelineBarrier = (delegate* unmanaged<nint, uint, uint, uint, uint, MemoryBarrier*, uint, void*, uint, void*, void>)Load("CmdPipelineBarrier");
        /// <summary>向队列提交命令并可关联完成栅栏。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：队列句柄、提交描述数量、提交描述数组、完成栅栏句柄。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, uint, SubmitInfo*, ulong, int> Submit = (delegate* unmanaged<nint, uint, SubmitInfo*, ulong, int>)Load("QueueSubmit");
        /// <summary>重置栅栏到未触发状态。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、栅栏数量、栅栏句柄数组。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, uint, ulong*, int> ResetFences = (delegate* unmanaged<nint, uint, ulong*, int>)Load("ResetFences");
        /// <summary>等待一个或多个栅栏完成。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、栅栏数量、栅栏数组、是否等待全部、超时纳秒数。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, uint, ulong*, uint, ulong, int> WaitFences = (delegate* unmanaged<nint, uint, ulong*, uint, ulong, int>)Load("WaitForFences");
        /// <summary>等待设备上所有已提交队列工作完成。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, int> WaitIdle = (delegate* unmanaged<nint, int>)Load("DeviceWaitIdle");
        /// <summary>销毁缓冲区，不同时释放其绑定内存。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、缓冲区句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyBuffer = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyBuffer");
        /// <summary>释放已解除映射且不再被设备使用的内存。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、内存句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> FreeMemory = (delegate* unmanaged<nint, ulong, void*, void>)Load("FreeMemory");
        /// <summary>销毁完成栅栏。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、栅栏句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyFence = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyFence");
        /// <summary>销毁命令池及其分配的命令缓冲区。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、命令池句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyCommandPool = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyCommandPool");
        /// <summary>销毁描述符池及其分配的描述符集。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、描述符池句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyDescriptorPool = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyDescriptorPool");
        /// <summary>销毁计算管线。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、管线句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyPipeline = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyPipeline");
        /// <summary>销毁着色器模块。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、着色器模块句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyShader = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyShaderModule");
        /// <summary>销毁管线布局。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、管线布局句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyPipelineLayout = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyPipelineLayout");
        /// <summary>销毁描述符集布局。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、描述符集布局句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyDescriptorLayout = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyDescriptorSetLayout");
    }
}
