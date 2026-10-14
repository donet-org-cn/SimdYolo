using System.Runtime.InteropServices;

namespace SimdYolo;

/// <summary>通过 Objective-C 运行时调用 Metal 的 FP32 计算后端，支持 64 位 macOS。</summary>
public sealed unsafe class MetalBackend : IComputeBackend, IConvolutionBackend
{
    private nint device, queue, pipeline, convPipeline;
    private readonly nint[] buffers = new nint[4];
    private readonly int[] capacities = new int[4];
    private readonly object gate = new();
    private readonly Dictionary<Tensor, nint> constants = [];
    private bool disposed;
    /// <summary>取得包含设备名称的后端显示名称。</summary>
    /// <value>后端类型及初始化时查询到的设备名称。</value>
    public string Name { get; private set; } = "Metal";
    /// <summary>通过 Objective-C 运行时调用 Metal 的 FP32 计算后端，支持 64 位 macOS。</summary>
    /// <exception cref="PlatformNotSupportedException">当前系统不是 64 位 macOS。</exception>
    /// <exception cref="InvalidOperationException">Metal 设备、队列、函数、计算管线或线程组能力不可用。</exception>
    public MetalBackend()
    {
        if (!OperatingSystem.IsMacOS() || IntPtr.Size != 8) throw new PlatformNotSupportedException("Metal requires 64-bit macOS.");
        nint pool = Native.PoolPush();
        try
        {
            NativeLibrary.Load("/System/Library/Frameworks/Foundation.framework/Foundation");
            device = Native.CreateDevice(); if (device == 0) throw new InvalidOperationException("No Metal device found.");
            Name = "Metal (" + String(Native.Send(device, Sel("name"))) + ")";
            queue = Native.Send(device, Sel("newCommandQueue")); if (queue == 0) throw new InvalidOperationException("Metal command queue unavailable.");
            var assembly = typeof(MetalBackend).Assembly; string resource = assembly.GetManifestResourceNames().Single(n => n.EndsWith("gemm.metal", StringComparison.Ordinal));
            using var reader = new StreamReader(assembly.GetManifestResourceStream(resource)!); nint source = NSString(reader.ReadToEnd()), error = 0;
            nint library = Native.Compile(device, Sel("newLibraryWithSource:options:error:"), source, 0, &error);
            if (library == 0) throw new InvalidOperationException("Metal shader compile failed: " + Error(error));
            nint function = 0;
            try
            {
                function = Native.SendObject(library, Sel("newFunctionWithName:"), NSString("gemm")); if (function == 0) throw new InvalidOperationException("Metal gemm function missing.");
                pipeline = Native.CreatePipeline(device, Sel("newComputePipelineStateWithFunction:error:"), function, &error);
                if (pipeline == 0) throw new InvalidOperationException("Metal pipeline failed: " + Error(error));
                if ((nuint)Native.Send(pipeline, Sel("maxTotalThreadsPerThreadgroup")) < 256) throw new InvalidOperationException("Metal device cannot execute 256-thread tiles.");
                Release(function); function = Native.SendObject(library, Sel("newFunctionWithName:"), NSString("convolution"));
                if (function == 0) throw new InvalidOperationException("Metal convolution function missing.");
                convPipeline = Native.CreatePipeline(device, Sel("newComputePipelineStateWithFunction:error:"), function, &error);
                if (convPipeline == 0) throw new InvalidOperationException("Metal convolution pipeline failed: " + Error(error));
                if ((nuint)Native.Send(convPipeline, Sel("maxTotalThreadsPerThreadgroup")) < 256) throw new InvalidOperationException("Metal convolution requires 256-thread tiles.");
            }
            finally { Release(function); Release(library); }
        }
        catch { Dispose(); throw; }
        finally { Native.PoolPop(pool); }
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
    public void MatMul(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        if (m <= 0 || n <= 0 || k <= 0 || a.Length < checked(m * k) || b.Length < checked(k * n) || c.Length < checked(m * n)) throw new ArgumentException("Invalid Metal GEMM dimensions.");
        Run(a, m * k, b, k * n, [0], c, m * n, [(uint)m, (uint)n, (uint)k, 0], false, new((nuint)(n + 15) / 16, (nuint)(m + 15) / 16, 1));
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
        Run(weights.Data, weights.Length, input.Data, input.Length, bias?.Data ?? new float[output.Shape[1]], output.Data, output.Length, parameters, true, new((nuint)(spatial + 15) / 16, (nuint)(output.Shape[1] / groups + 15) / 16, (nuint)(input.Shape[0] * groups)), weights, bias);
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
    /// <param name="convolution">是否使用卷积管线和卷积参数布局。</param>
    /// <param name="groups">三个方向的线程组数量。</param>
    /// <param name="constantA">第一输入对应的常量张量，用于选择持久缓存；为空时使用临时缓冲区。</param>
    /// <param name="constantBias">偏置对应的常量张量，用于选择持久缓存；为空时使用临时缓冲区。</param>
    /// <exception cref="ObjectDisposedException">后端已经释放。</exception>
    /// <exception cref="InvalidOperationException">设备缓冲区分配、提交、等待或计算失败。</exception>
    private void Run(float[] a, int acount, float[] b, int bcount, float[] bias, float[] c, int ccount, uint[] parameters, bool convolution, Size groups, Tensor? constantA = null, Tensor? constantBias = null)
    {
        lock (gate)
        {
            // 一个实例只允许一个提交使用共享缓冲区；此锁同时与 Dispose 互斥。
            ObjectDisposedException.ThrowIf(disposed, this); nint pool = Native.PoolPush();
            try
            {
                Ensure(0, checked(acount * 4)); Ensure(1, checked(bcount * 4)); Ensure(2, checked(ccount * 4)); Ensure(3, checked(bias.Length * 4));
                // 选择按常量张量身份缓存的资源，或复用可变数据共享缓冲区；返回对象均由后端释放。
                nint Select(Tensor? tensor, nint fallback, float[] values, int count)
                {
                    // 仅不可变张量可按对象身份缓存；可变输入每次拷贝，以反映调用方更新的数据。
                    bool immutable = tensor?.IsConstant == true;
                    if (immutable && constants.TryGetValue(tensor!, out var cached)) return cached;
                    nint selected = immutable ? Native.NewBuffer(device, Sel("newBufferWithLength:options:"), (nuint)count * 4, 0) : fallback;
                    if (selected == 0) throw new InvalidOperationException("Metal constant buffer allocation failed.");
                    fixed (float* data = values) Buffer.MemoryCopy(data, (void*)Native.Send(selected, Sel("contents")), (long)count * 4, (long)count * 4);
                    if (immutable) constants[tensor!] = selected; return selected;
                }
                nint selectedA = Select(constantA, buffers[0], a, acount), selectedBias = Select(constantBias, buffers[3], bias, bias.Length);
                fixed (float* bp = b) Buffer.MemoryCopy(bp, (void*)Native.Send(buffers[1], Sel("contents")), capacities[1], (long)bcount * 4);
                nint command = Native.Send(queue, Sel("commandBuffer")); if (command == 0) throw new InvalidOperationException("Metal command buffer unavailable.");
                nint encoder = Native.Send(command, Sel("computeCommandEncoder")); if (encoder == 0) throw new InvalidOperationException("Metal compute encoder unavailable.");
                Native.VoidObject(encoder, Sel("setComputePipelineState:"), convolution ? convPipeline : pipeline);
                for (nuint i = 0; i < 3; i++) Native.SetBuffer(encoder, Sel("setBuffer:offset:atIndex:"), i == 0 ? selectedA : buffers[i], 0, i);
                // Metal uint3 的大小和对齐均为 16 字节；参数区遵守着色器规定的布局。
                if (convolution) Native.SetBuffer(encoder, Sel("setBuffer:offset:atIndex:"), selectedBias, 0, 3);
                fixed (uint* dims = parameters) Native.SetBytes(encoder, Sel("setBytes:length:atIndex:"), dims, (nuint)parameters.Length * 4, convolution ? 4u : 3u);
                Native.Dispatch(encoder, Sel("dispatchThreadgroups:threadsPerThreadgroup:"), groups, new(16, 16, 1));
                // Shared 内存的结果只有在命令完成后才可读取；等待同时保证下一次复用不覆盖在途输入。
                Native.Void(encoder, Sel("endEncoding")); Native.Void(command, Sel("commit")); Native.Void(command, Sel("waitUntilCompleted"));
                if ((nuint)Native.Send(command, Sel("status")) != 4) throw new InvalidOperationException("Metal compute failed: " + Error(Native.Send(command, Sel("error"))));
                fixed (float* cp = c) Buffer.MemoryCopy((void*)Native.Send(buffers[2], Sel("contents")), cp, (long)c.Length * 4, (long)ccount * 4);
            }
            finally { Native.PoolPop(pool); }
        }
    }
    /// <summary>确保指定共享缓冲区容量足够，扩容前释放旧对象。</summary>
    /// <param name="i">需要扩容的共享缓冲区槽位索引。</param>
    /// <param name="bytes">共享缓冲区需要容纳的字节数。</param>
    /// <exception cref="InvalidOperationException">Metal 共享缓冲区分配失败。</exception>
    private void Ensure(int i, int bytes)
    {
        if (capacities[i] >= bytes) return; Release(buffers[i]); buffers[i] = 0; capacities[i] = 0;
        nint buffer = Native.NewBuffer(device, Sel("newBufferWithLength:options:"), (nuint)bytes, 0); // MTLStorageModeShared：CPU 与 GPU 共享可映射内存。
        if (buffer == 0) throw new InvalidOperationException("Metal shared buffer allocation failed."); buffers[i] = buffer; capacities[i] = bytes;
    }
    /// <summary>在互斥锁保护下释放共享缓冲区、常量缓存、Metal 管线、队列和设备引用；可重复调用。</summary>
    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return; disposed = true;
            foreach (nint b in buffers) Release(b); Release(pipeline); Release(convPipeline); Release(queue); Release(device); convPipeline = pipeline = queue = device = 0;
            foreach (nint b in constants.Values) Release(b); constants.Clear();
        }
    }
    /// <summary>向 Objective-C 运行时注册消息选择器名称。</summary>
    /// <param name="s">要注册的 Objective-C 选择器名称。</param>
    /// <returns>运行时注册的选择器地址。</returns>
    private static nint Sel(string s) => Native.Selector(s);
    /// <summary>把托管 UTF-8 文本转换为自动释放池管理的 NSString 对象。</summary>
    /// <param name="s">转换成 NSString 的托管文本。</param>
    /// <returns>自动释放的 NSString 句柄，由当前自动释放池管理。</returns>
    private static nint NSString(string s) => Native.String(Native.Class("NSString"), Sel("stringWithUTF8String:"), s);
    /// <summary>读取 NSString 的 UTF-8 内容；空原生句柄或空文本指针返回空字符串。</summary>
    /// <param name="s">NSString 对象句柄，允许为 0。</param>
    /// <returns>转换后的托管文本。</returns>
    private static string String(nint s) => s == 0 ? "" : Marshal.PtrToStringUTF8(Native.Send(s, Sel("UTF8String"))) ?? "";
    /// <summary>读取 NSError 的本地化说明；空错误句柄返回默认说明。</summary>
    /// <param name="error">要读取说明的 NSError 句柄，允许为 0。</param>
    /// <returns>适用于错误消息的托管文本。</returns>
    private static string Error(nint error) => error == 0 ? "unknown error" : String(Native.Send(error, Sel("localizedDescription")));
    /// <summary>对非空 Objective-C 对象发送 release，释放当前持有的原生引用。</summary>
    /// <param name="obj">Objective-C 消息接收对象句柄。</param>
    private static void Release(nint obj) { if (obj != 0) Native.Void(obj, Sel("release")); }
    /// <summary>按 Objective-C ABI 顺序传递 Metal 三维线程组尺寸。</summary>
    /// <param name="Width">X 方向尺寸。</param>
    /// <param name="Height">Y 方向尺寸。</param>
    /// <param name="Depth">Z 方向尺寸。</param>
    [StructLayout(LayoutKind.Sequential)] private readonly record struct Size(nuint Width, nuint Height, nuint Depth);
    /// <summary>声明 Objective-C 与 Metal 原生入口，使用与对应消息匹配的 ABI 参数类型。</summary>
    private static class Native
    {
        private const string ObjC = "/usr/lib/libobjc.A.dylib";
        /// <summary>取得系统默认 Metal 设备。</summary>
        /// <returns>Metal 设备句柄；没有可用设备时为 0，成功后由后端 release。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport("/System/Library/Frameworks/Metal.framework/Metal", EntryPoint = "MTLCreateSystemDefaultDevice")] public static extern nint CreateDevice();
        /// <summary>压入当前线程的 Objective-C 自动释放池。</summary>
        /// <returns>由 PoolPop 使用的池标记。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_autoreleasePoolPush")] public static extern nint PoolPush();
        /// <summary>弹出指定自动释放池，释放池内自动释放对象。</summary>
        /// <param name="pool">由 PoolPush 返回的自动释放池标记。</param>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_autoreleasePoolPop")] public static extern void PoolPop(nint pool);
        /// <summary>注册或查找 Objective-C 消息选择器。</summary>
        /// <param name="name">要注册或查询的 Objective-C 选择器名称，按 UTF-8 传入。</param>
        /// <returns>注册后的选择器地址。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "sel_registerName")] public static extern nint Selector([MarshalAs(UnmanagedType.LPUTF8Str)] string name);
        /// <summary>按 UTF-8 名称查找 Objective-C 类。</summary>
        /// <param name="name">需要查找的 Objective-C 类名，按 UTF-8 传入。</param>
        /// <returns>类对象地址；找不到类时为 0。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_getClass")] public static extern nint Class([MarshalAs(UnmanagedType.LPUTF8Str)] string name);
        /// <summary>向对象发送无参数且返回指针或机器字大小数值的消息。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <returns>消息返回的对象地址、指针或机器字大小数值，由选择器决定。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern nint Send(nint obj, nint selector);
        /// <summary>向对象发送含一个对象参数并返回对象的消息。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="arg">消息携带的原生对象句柄。</param>
        /// <returns>消息返回的原生对象句柄。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern nint SendObject(nint obj, nint selector, nint arg);
        /// <summary>发送含 UTF-8 字符串参数并返回对象的消息。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="arg">传给消息的托管字符串，互操作层按 UTF-8 编码。</param>
        /// <returns>消息返回的 NSString 或其他原生对象句柄。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern nint String(nint obj, nint selector, [MarshalAs(UnmanagedType.LPUTF8Str)] string arg);
        /// <summary>向对象发送无参数且无返回值的消息。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern void Void(nint obj, nint selector);
        /// <summary>向对象发送含一个对象参数且无返回值的消息。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="arg">消息携带的原生对象句柄。</param>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern void VoidObject(nint obj, nint selector, nint arg);
        /// <summary>调用 Metal 源码库编译消息，并输出可选 NSError。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="source">NSString 着色器源码对象句柄。</param>
        /// <param name="options">可选 Metal 编译选项对象句柄；本实现使用 0。</param>
        /// <param name="error">接收 NSError 对象句柄的原生输出指针。</param>
        /// <returns>编译后的库对象；成功引用需 release，失败时为 0 并可通过 error 取得错误对象。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern nint Compile(nint obj, nint selector, nint source, nint options, nint* error);
        /// <summary>调用 Metal 计算管线创建消息，并输出可选 NSError。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="function">Metal 计算函数对象句柄。</param>
        /// <param name="error">接收 NSError 对象句柄的原生输出指针。</param>
        /// <returns>新的计算管线对象，成功引用需 release；失败时为 0。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern nint CreatePipeline(nint obj, nint selector, nint function, nint* error);
        /// <summary>按字节长度与资源选项创建 Metal 缓冲区。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="length">缓冲区或常量数据长度，单位为字节。</param>
        /// <param name="options">Metal 缓冲区资源选项位；0 对应共享存储模式。</param>
        /// <returns>新缓冲区对象；失败时为 0，成功对象需由调用方 release。</returns>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern nint NewBuffer(nint obj, nint selector, nuint length, nuint options);
        /// <summary>为计算编码器绑定缓冲区及字节偏移。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="buffer">需要绑定的 Metal 缓冲区句柄。</param>
        /// <param name="offset">绑定缓冲区的起始字节偏移。</param>
        /// <param name="index">编码器中的缓冲区或常量绑定编号。</param>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern void SetBuffer(nint obj, nint selector, nint buffer, nuint offset, nuint index);
        /// <summary>把调用时提供的常量字节复制到指定编码器绑定位置。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="bytes">常量字节地址；Metal 在调用时复制内容，无需在提交期间保持该指针。</param>
        /// <param name="length">缓冲区或常量数据长度，单位为字节。</param>
        /// <param name="index">编码器中的缓冲区或常量绑定编号。</param>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern void SetBytes(nint obj, nint selector, void* bytes, nuint length, nuint index);
        /// <summary>以三维线程组数量和组内线程数编码计算调度。</summary>
        /// <param name="obj">Objective-C 消息接收对象句柄。</param>
        /// <param name="selector">与参数签名匹配的 Objective-C 选择器。</param>
        /// <param name="groups">三个方向的线程组数量。</param>
        /// <param name="threads">每个线程组的三维线程数。</param>
        /// <remarks>原生入口不将失败状态转换成托管异常；调用方检查返回句柄、NSError 或命令状态。</remarks>
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] public static extern void Dispatch(nint obj, nint selector, Size groups, Size threads);
    }
}
