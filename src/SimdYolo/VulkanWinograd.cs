using System.Text;

namespace SimdYolo;

/// <summary>实现 Vulkan Winograd 卷积的输入/权重变换、中间乘法、输出变换与调优。</summary>
public sealed unsafe partial class VulkanBackend
{
    private readonly Dictionary<(Tensor Weight, int Tile), Tensor> winogradWeights = [];
    private readonly Dictionary<string, ulong> winogradPipelines = [];
    private readonly Dictionary<string, uint> winogradMiddleTiles = [];
    private readonly Dictionary<string, ulong> winogradTransformPipelines = [];
    private readonly Dictionary<WinogradInputKey, Tensor> winogradInputs = [];
    private readonly bool cacheWinogradInputs = Environment.GetEnvironmentVariable("SIMDYOLO_WINOGRAD_INPUT_CACHE") != "0";
    /// <summary>按缓冲区身份、字节偏移、输入形状和输出块大小区分帧内 Winograd 输入变换。</summary>
    /// <param name="Buffer">后端持有的设备缓冲区对象；此视图不取得独立所有权。</param>
    /// <param name="Offset">缓冲区内的字节偏移。</param>
    /// <param name="Batch">输入批次数。</param>
    /// <param name="Channels">输入通道数。</param>
    /// <param name="Height">输入特征图高度。</param>
    /// <param name="Width">输入特征图宽度。</param>
    /// <param name="Tile">Winograd 每块的输出边长。</param>
    private readonly record struct WinogradInputKey(GpuBuffer Buffer, ulong Offset, int Batch, int Channels, int Height, int Width, int Tile);
    private readonly bool enableWinograd4 = Environment.GetEnvironmentVariable("SIMDYOLO_WINOGRAD4") != "0";
    private readonly string? winogradRegisterTile = Environment.GetEnvironmentVariable("SIMDYOLO_WINOGRAD_REGISTER_TILE");
    private readonly bool winogradVectorIo = Environment.GetEnvironmentVariable("SIMDYOLO_WINOGRAD_VECTOR_IO") != "0";
    private readonly string? winogradColumnMode = Environment.GetEnvironmentVariable("SIMDYOLO_WINOGRAD_COLUMNS64");
    /// <summary>取得是否允许 Winograd 中间矩阵使用 64 列分块。</summary>
    /// <value>对应环境变量不为 0 时为 true。</value>
    private bool winogradColumns64 => winogradColumnMode != "0";
    private const int Winograd4Minimum = 20;

    /// <summary>执行 F(2,3) 或 F(4,3) 卷积，缓存权重变换并在同一帧复用可识别的输入变换。</summary>
    /// <param name="x">按 [批次, 输入通道, 高, 宽] 排列的卷积输入。</param>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="bias">每个输出通道的偏置；允许为空时由实现提供零偏置。</param>
    /// <param name="silu">是否在偏置相加后融合 SiLU 激活。</param>
    /// <returns>属于当前后端活动帧的设备结果张量；需要在帧结束前物化主机数据。</returns>
    private Tensor WinogradConv(Tensor x, Tensor w, Tensor? bias, bool silu)
    {
        int batch = x.Shape[0], ic = x.Shape[1], h = x.Shape[2], width = x.Shape[3], oc = w.Shape[0];
        int tile = enableWinograd4 && h >= Winograd4Minimum && width >= Winograd4Minimum ? 4 : 2, planes = tile == 4 ? 36 : 16;
        int th = (h + tile - 1) / tile, tw = (width + tile - 1) / tile, tiles = th * tw;
        bool vectorIo = winogradVectorIo && tile == 4 && h % 4 == 0 && width % 4 == 0;
        if (!winogradWeights.TryGetValue((w, tile), out var weight))
        {
            weight = new Tensor(planes, oc, ic) { IsConstant = true };
            if (tile == 4) PackWinograd4Weights(w, weight, oc, ic);
            else
            {
            Span<float> temp = stackalloc float[12];
            for (int co = 0; co < oc; co++) for (int ci = 0; ci < ic; ci++)
            {
                int offset = (co * ic + ci) * 9;
                for (int c = 0; c < 3; c++)
                {
                    float a = w.Data[offset + c], b = w.Data[offset + 3 + c], d = w.Data[offset + 6 + c];
                    temp[c] = a; temp[3 + c] = (a + b + d) * .5f; temp[6 + c] = (a - b + d) * .5f; temp[9 + c] = d;
                }
                for (int r = 0; r < 4; r++)
                {
                    float a = temp[r * 3], b = temp[r * 3 + 1], c = temp[r * 3 + 2]; int index = co * ic + ci;
                    weight.Data[(r * 4) * oc * ic + index] = a; weight.Data[(r * 4 + 1) * oc * ic + index] = (a + b + c) * .5f;
                    weight.Data[(r * 4 + 2) * oc * ic + index] = (a - b + c) * .5f; weight.Data[(r * 4 + 3) * oc * ic + index] = c;
                }
            }
            }
            if (w.IsConstant) winogradWeights[(w, tile)] = weight;
        }
        if (bias is null)
        {
            if (!zeros.TryGetValue(oc, out bias)) { bias = new Tensor(oc) { IsConstant = true }; zeros.Add(oc, bias); }
        }
        var input = Upload(x); var weights = Upload(weight); var bv = Upload(bias); var zero = Upload(scalarZero);
        bool cacheInput = cacheWinogradInputs && x.Storage is { } source && ReferenceEquals(source.Owner, this) && source.FrameId == frameId;
        var cacheKey = new WinogradInputKey((GpuBuffer)input.Buffer, input.Offset, batch, ic, h, width, tile);
        if (!cacheInput || !winogradInputs.TryGetValue(cacheKey, out var transformed))
        {
            transformed = AllocateTensor([batch * planes, ic, 1, tiles]);
            uint[] ip = DeviceOps.Parameters((uint)(batch * ic * tiles), (uint)ic, (uint)h, (uint)width, (uint)tw, (uint)th);
            string inputKey = planes + ":input:" + string.Join(',', ip.Take(7)) + (vectorIo ? ":vec" : "");
            if (!winogradTransformPipelines.TryGetValue(inputKey, out ulong inputTransform)) { inputTransform = CreateNamedPipeline(vectorIo ? "winograd4_input_vec.spv" : tile == 4 ? "winograd4_input.spv" : "winograd_input.spv", ip); winogradTransformPipelines[inputKey] = inputTransform; }
            Dispatch(inputTransform, input, zero, transformed.Storage!, zero, ip, (uint)(batch * ic * tiles + 127) / 128);
            if (cacheInput) winogradInputs[cacheKey] = transformed;
        }
        var products = AllocateTensor([batch * planes, oc, 1, tiles]); var output = AllocateTensor([batch, oc, h, width]);
        uint[] mp = [0, (uint)ic, 1, (uint)tiles, (uint)oc, 1, (uint)tiles, 1, 1, 1, 1, 0, 0, 1, 1, 1, (uint)planes];
        if (!zeros.TryGetValue(oc, out var middleBias)) { middleBias = new Tensor(oc) { IsConstant = true }; zeros.Add(oc, middleBias); }
        uint columns = oc <= 16 ? 64u : 32u, rows = oc <= 16 ? 16u : 32u;
        var middleBiasStorage = Upload(middleBias);
        bool wideColumnsEligible = oc >= 64 && oc % 64 == 0 && tiles >= 64 && batch > 0;
        if (winogradColumnMode == "1" && wideColumnsEligible)
        {
            rows = columns = 64;
        }
        else if (oc >= 64 && oc % 64 == 0 && tiles > 0 && batch > 0 && winogradRegisterTile != "0")
        {
            if (winogradRegisterTile == "256") rows = 64;
            else
            {
                string dimensions = string.Join(',', mp);
                if (!winogradMiddleTiles.TryGetValue(dimensions, out uint choice) || choice is not (32 or 64 or 128)
                    || choice == 128 && (!wideColumnsEligible || !winogradColumns64))
                {
                    uint[] candidates = wideColumnsEligible && winogradColumns64 ? [32u, 64u, 128u] : [32u, 64u];
                    ulong[] pipelines = candidates.Select(candidate => WinogradMiddlePipeline(tile, planes, mp,
                        candidate == 32 ? 32u : 64u, candidate == 128 ? 64u : 32u)).ToArray();
                    // 候选计时前先完成输入变换和常量上传。帧内变换缓冲区在这些
                    // 同步提交之间保持有效，计时只比较中间乘法候选。
                    FlushFrame();
                    for (int index = 0; index < candidates.Length; index++)
                    {
                        uint candidateRows = candidates[index] == 32 ? 32u : 64u, candidateColumns = candidates[index] == 128 ? 64u : 32u;
                        Dispatch(pipelines[index], weights, transformed.Storage!, products.Storage!, middleBiasStorage, mp,
                            ((uint)tiles + candidateColumns - 1) / candidateColumns, ((uint)oc + candidateRows - 1) / candidateRows, (uint)(batch * planes));
                    }
                    FlushFrame();
                    ulong fastest = ulong.MaxValue; choice = 32;
                    for (int round = 0; round < 2; round++) for (int index = 0; index < candidates.Length; index++)
                    {
                        uint candidateRows = candidates[index] == 32 ? 32u : 64u, candidateColumns = candidates[index] == 128 ? 64u : 32u;
                        ulong elapsed = MeasureConvolution(pipelines[index], weights, transformed.Storage!, products.Storage!, middleBiasStorage, mp,
                            ((uint)tiles + candidateColumns - 1) / candidateColumns, ((uint)oc + candidateRows - 1) / candidateRows, (uint)(batch * planes));
                        if (elapsed < fastest) { fastest = elapsed; choice = candidates[index]; }
                    }
                    winogradMiddleTiles[dimensions] = choice;
                }
                rows = choice == 128 ? 64u : choice;
                columns = choice == 128 ? 64u : 32u;
            }
        }
        ulong middle = WinogradMiddlePipeline(tile, planes, mp, rows, columns);
        Dispatch(middle, weights, transformed.Storage!, products.Storage!, middleBiasStorage, mp, ((uint)tiles + columns - 1) / columns, ((uint)oc + rows - 1) / rows, (uint)(batch * planes));
        uint[] op = DeviceOps.Parameters((uint)(batch * oc * tiles), (uint)oc, (uint)h, (uint)width, (uint)tw, (uint)th, silu ? 1u : 0u);
        string outputKey = planes + ":output:" + string.Join(',', op.Take(7)) + (vectorIo ? ":vec" : "");
        if (!winogradTransformPipelines.TryGetValue(outputKey, out ulong outputTransform)) { outputTransform = CreateNamedPipeline(vectorIo ? "winograd4_output_vec.spv" : tile == 4 ? "winograd4_output.spv" : "winograd_output.spv", op); winogradTransformPipelines[outputKey] = outputTransform; }
        Dispatch(outputTransform, products.Storage!, bv, output.Storage!, zero, op, (uint)(batch * oc * tiles + 127) / 128);
        return output;
    }
    /// <summary>按 Winograd 变换平面、输出通道行块和空间列块创建或复用中间矩阵乘管线。</summary>
    /// <param name="tile">Winograd 每块输出边长，值为 2 或 4。</param>
    /// <param name="planes">Winograd 变换平面数，F(2,3) 为 16，F(4,3) 为 36。</param>
    /// <param name="parameters">按对应着色器参数布局排列的 uint 值，用于特化常量或推送常量。</param>
    /// <param name="rows">每个工作组处理的输出通道行数。</param>
    /// <param name="columns">每个工作组处理的空间块列数。</param>
    /// <returns>由后端缓存并统一释放的 Vulkan 计算管线句柄。</returns>
    private ulong WinogradMiddlePipeline(int tile, int planes, uint[] parameters, uint rows, uint columns = 32)
    {
        string kernel = rows == 16 ? "16x64" : rows == 64 && columns == 64 ? "64x64threads256" : rows == 64 ? "64threads256" : "32";
        string key = planes + ":" + string.Join(',', parameters) + ":" + kernel;
        if (!winogradPipelines.TryGetValue(key, out ulong pipeline))
        {
            string prefix = rows == 64 ? "winograd_middle" : tile == 4 ? "winograd4_middle" : "winograd_middle";
            winogradPipelines[key] = pipeline = CreateNamedPipeline(prefix + kernel + ".spv", parameters);
        }
        return pipeline;
    }
    /// <summary>把 3×3 常量权重变换为 F(4,3) 的 36 个平面，并写入平面优先的权重张量。</summary>
    /// <param name="w">卷积权重张量。</param>
    /// <param name="weight">接收 F(4,3) 变换结果的权重张量。</param>
    /// <param name="oc">输出通道数。</param>
    /// <param name="ic">输入通道数。</param>
    private static void PackWinograd4Weights(Tensor w, Tensor weight, int oc, int ic)
    {
        Span<float> temp = stackalloc float[18];
        for (int co = 0; co < oc; co++) for (int ci = 0; ci < ic; ci++)
        {
            int offset = (co * ic + ci) * 9;
            for (int c = 0; c < 3; c++)
            {
                float a = w.Data[offset + c], b = w.Data[offset + 3 + c], d = w.Data[offset + 6 + c];
                temp[c] = a * .25f; temp[3 + c] = -(a + b + d) / 6; temp[6 + c] = -(a - b + d) / 6;
                temp[9 + c] = a / 24 + b / 12 + d / 6; temp[12 + c] = a / 24 - b / 12 + d / 6; temp[15 + c] = d;
            }
            for (int r = 0; r < 6; r++)
            {
                float a = temp[r * 3], b = temp[r * 3 + 1], d = temp[r * 3 + 2]; int index = co * ic + ci;
                weight.Data[(r * 6) * oc * ic + index] = a * .25f;
                weight.Data[(r * 6 + 1) * oc * ic + index] = -(a + b + d) / 6;
                weight.Data[(r * 6 + 2) * oc * ic + index] = -(a - b + d) / 6;
                weight.Data[(r * 6 + 3) * oc * ic + index] = a / 24 + b / 12 + d / 6;
                weight.Data[(r * 6 + 4) * oc * ic + index] = a / 24 - b / 12 + d / 6;
                weight.Data[(r * 6 + 5) * oc * ic + index] = d;
            }
        }
    }
    /// <summary>从嵌入式 SPIR-V 资源创建具名计算管线，并按需绑定 uint 特化常量。</summary>
    /// <param name="suffix">嵌入式 SPIR-V 资源名称后缀。</param>
    /// <param name="parameters">按对应着色器参数布局排列的 uint 值，用于特化常量或推送常量。</param>
    /// <returns>由后端缓存并统一释放的 Vulkan 计算管线句柄。</returns>
    /// <exception cref="InvalidOperationException">嵌入资源无法唯一匹配，或 Vulkan 着色器、管线创建失败。</exception>
    private ulong CreateNamedPipeline(string suffix, uint[]? parameters = null)
    {
        var assembly = typeof(VulkanBackend).Assembly; string resource = assembly.GetManifestResourceNames().Single(n => n.EndsWith(suffix, StringComparison.Ordinal));
        using var stream = assembly.GetManifestResourceStream(resource)!; using var memory = new MemoryStream(); stream.CopyTo(memory); byte[] code = memory.ToArray(); ulong module = 0, result = 0;
        fixed (byte* data = code) { ShaderInfo info = new() { Type = 16, Size = (nuint)code.Length, Code = (uint*)data }; Check(Vk.CreateShader(device, &info, null, &module), "CreateShaderModule(winograd)"); }
        try
        {
            uint count = (uint)(parameters?.Length ?? 0);
            SpecializationEntry* entries = stackalloc SpecializationEntry[(int)count]; for (uint i = 0; i < count; i++) entries[i] = new() { Id = i, Offset = i * 4, Size = 4 };
            byte[] name = Encoding.UTF8.GetBytes("main\0");
            fixed (byte* ptr = name) fixed (uint* p = parameters)
            {
                SpecializationInfo spec = new() { Count = count, Entries = entries, Size = count * 4, Data = p };
                ComputePipelineInfo info = new() { Type = 29, Stage = new() { Type = 18, Stage = 32, Module = module, Name = ptr, Specialization = parameters is null ? 0 : (nint)(&spec) }, Layout = pipelineLayout, BaseIndex = -1 };
                Check(CreateCachedComputePipeline(&info, &result), "CreateComputePipelines(winograd)");
            }
            return result;
        }
        finally { Vk.DestroyShader(device, module, null); }
    }
    /// <summary>释放 Winograd 中间与变换管线，并清除权重、形状调优、输入变换和逐点权重缓存。</summary>
    private void FreeWinogradResources()
    {
        foreach (ulong p in winogradPipelines.Values) Vk.DestroyPipeline(device, p, null); winogradPipelines.Clear(); winogradWeights.Clear();
        winogradMiddleTiles.Clear();
        foreach (ulong p in winogradTransformPipelines.Values) Vk.DestroyPipeline(device, p, null); winogradTransformPipelines.Clear();
        winogradInputs.Clear();
        pointwisePackedWeights.Clear();
    }
}

