namespace SimdYolo;

/// <summary>实现 Vulkan 检测头两阶段选择和 ONNX 最后一轴 TopK。</summary>
public sealed unsafe partial class VulkanBackend
{
    private readonly bool gpuTopK = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_TOPK") != "0";
    private readonly bool gpuOnnxTopK = Environment.GetEnvironmentVariable("SIMDYOLO_GPU_TOPK") == "1"
        || Environment.GetEnvironmentVariable("SIMDYOLO_GPU_ONNX_TOPK") == "1";

    /// <summary>尝试在设备端执行 PyTorch 检测头的两阶段 TopK；不满足条件时由调用方回退。</summary>
    /// <param name="input">按 [批次, 4 + 类别数, 锚点数] 排列的当前帧设备检测头。</param>
    /// <param name="maximum">最多保留的检测数。</param>
    /// <param name="classes">检测头的类别总数。</param>
    /// <returns>当前帧的设备结果；条件不适用时为 null，由调用方选择回退路径。</returns>
    internal Tensor? TryEndToEndTopK(Tensor input, int maximum, int classes)
        => TryEndToEndTopKCore(input, maximum, classes, false, false);

    /// <summary>尝试执行 ONNX 检测头 TopK，使用 NaN 优先和第一阶段选择次序的平分规则。</summary>
    /// <param name="input">按 [批次, 4 + 类别数, 锚点数] 排列的当前帧设备检测头。</param>
    /// <param name="maximum">最多保留的检测数。</param>
    /// <param name="classes">检测头的类别总数。</param>
    /// <returns>当前帧的设备结果；条件不适用时为 null，由调用方选择回退路径。</returns>
    internal Tensor? TryEndToEndTopKOnnx(Tensor input, int maximum, int classes)
        => TryEndToEndTopKCore(input, maximum, classes, true, true);

    /// <summary>检查设备驻留、索引精度及工作区限制，并执行两阶段检测头选择与结果收集。</summary>
    /// <param name="input">按 [批次, 4 + 类别数, 锚点数] 排列的当前帧设备检测头。</param>
    /// <param name="maximum">最多保留的检测数。</param>
    /// <param name="classes">检测头的类别总数。</param>
    /// <param name="nanFirst">是否将 NaN 排在普通浮点值之前。</param>
    /// <param name="tieBySelection">是否以第一阶段选中位置排序作为分数相同时的次级键。</param>
    /// <returns>当前帧的设备结果；条件不适用时为 null，由调用方选择回退路径。</returns>
    private Tensor? TryEndToEndTopKCore(Tensor input, int maximum, int classes, bool nanFirst, bool tieBySelection)
    {
        if (!gpuTopK || !frameActive || input.Rank != 3 || input.Shape[1] != classes + 4 || classes <= 0 || maximum <= 0 || Tensor.Count(input.Shape) != input.Length
            || input.Storage is not { } source || !ReferenceEquals(source.Owner, this) || source.FrameId != frameId) return null;
        int batch = input.Shape[0], anchors = input.Shape[2], k = Math.Min(maximum, anchors);
        if ((long)input.Length * sizeof(float) + (long)(source.Offset % 256) > 128L * 1024 * 1024) return null;
        if (anchors is < 1 or > 65536 || (long)k * classes > 65536 || (long)anchors * classes > 16_777_216 || batch is < 1 or > 65535
            || (long)batch * anchors > 65535L * 128 || (long)batch * k * classes > 65535L * 128 || (long)batch * k * 6 > 65535L * 128
            || (long)batch * ((anchors + 255) / 256) * 512 > int.MaxValue
            || (long)batch * ((k * classes + 255) / 256) * 512 > int.MaxValue) return null;
        var zero = Upload(scalarZero); var maxima = AllocateTensor([batch, anchors]);
        uint[] reduce = [(uint)(batch * anchors), (uint)anchors, (uint)classes, (uint)(source.Offset % 256 / 4)];
        Dispatch(TopKPipeline("topk_head_max.spv", reduce), source, zero, maxima.Storage!, zero, reduce, (uint)(maxima.Length + 127) / 128);
        var selected = TopKLastAxis(maxima, k, true, nanFirst, null);
        var candidates = AllocateTensor([batch, k * classes]); var keys = AllocateTensor([batch, k * classes]);
        uint[] prepare = [(uint)candidates.Length, (uint)anchors, (uint)classes, (uint)k, (uint)(source.Offset % 256 / 4), 0, tieBySelection ? 1u : 0u];
        Dispatch(TopKPipeline("topk_head_prepare.spv", prepare), source, selected[1].Storage!, candidates.Storage!, keys.Storage!, prepare, (uint)(candidates.Length + 127) / 128);
        var best = TopKLastAxis(candidates, k, true, nanFirst, keys);
        var output = AllocateTensor([batch, k, 6]);
        uint[] gather = [(uint)output.Length, (uint)anchors, (uint)classes, (uint)k, (uint)(source.Offset % 256 / 4)];
        Dispatch(TopKPipeline(tieBySelection ? "topk_head_gather_selection.spv" : "topk_head_gather.spv", gather), source, best[1].Storage!, output.Storage!,
            tieBySelection ? selected[1].Storage! : best[0].Storage!, gather, (uint)(output.Length + 127) / 128);
        return output;
    }

    /// <summary>尝试对张量最后一轴执行 ONNX TopK，受可移植工作区与设备驻留条件限制。</summary>
    /// <param name="input">最后一轴待执行 TopK 的当前帧设备张量。</param>
    /// <param name="k">每行需要选出的元素数。</param>
    /// <param name="largest">是否按从大到小选择；false 表示选择最小值。</param>
    /// <returns>当前帧的设备结果；条件不适用时为 null，由调用方选择回退路径。</returns>
    internal Tensor[]? TryTopKLastAxis(Tensor input, int k, bool largest)
    {
        if (!gpuOnnxTopK || !frameActive || input.Rank < 1 || input.Shape[^1] is < 1 or > 65536 || k < 0 || k > input.Shape[^1] || Tensor.Count(input.Shape) != input.Length
            || input.Storage is not { } source || !ReferenceEquals(source.Owner, this) || source.FrameId != frameId) return null;
        int size = input.Shape[^1], rows = input.Length / size;
        if ((long)input.Length * sizeof(float) + (long)(source.Offset % 256) > 128L * 1024 * 1024) return null;
        // 排序工作区含补齐后的 256 元素子段，整体必须落在 Vulkan
        // 可移植存储缓冲区 128 MiB 范围内。
        if (rows is < 1 or > 65535 || (long)rows * ((size + 255) / 256) * 512 > 128L * 1024 * 1024 / sizeof(float)) return null;
        return TopKLastAxis(input, k, largest, true, null);
    }

    /// <summary>先排序 256 元素子块再合并 TopK，返回值张量及以 FP32 表示的索引张量。</summary>
    /// <param name="input">最后一轴待执行 TopK 的当前帧设备张量。</param>
    /// <param name="k">每行需要选出的元素数。</param>
    /// <param name="largest">是否按从大到小选择；false 表示选择最小值。</param>
    /// <param name="nanFirst">是否将 NaN 排在普通浮点值之前。</param>
    /// <param name="keys">用于平分排序的可选键张量；为空时使用原位置索引。</param>
    /// <returns>依次包含 TopK 数值和索引的两个设备张量。</returns>
    private Tensor[] TopKLastAxis(Tensor input, int k, bool largest, bool nanFirst, Tensor? keys)
    {
        int size = input.Shape[^1], rows = input.Length / size, blocks = (size + 255) / 256;
        var shape = (int[])input.Shape.Clone(); shape[^1] = k;
        var values = AllocateTensor(shape); var indices = AllocateTensor(shape);
        if (k == 0) return [values, indices];
        var sorted = AllocateTensor([rows, blocks, 256, 2]);
        var a = Upload(input); var b = Upload(keys ?? scalarZero);
        uint[] parameters = [(uint)rows, (uint)size, (uint)k, largest ? 1u : 0u, nanFirst ? 1u : 0u,
            (uint)(a.Offset % 256 / 4), keys is not null ? 1u : 0u, (uint)(b.Offset % 256 / 4)];
        Dispatch(TopKPipeline("topk_block.spv", parameters), a, b, sorted.Storage!, Upload(scalarZero), parameters, (uint)blocks, (uint)rows);
        Dispatch(TopKPipeline("topk_merge.spv", parameters), sorted.Storage!, Upload(scalarZero), values.Storage!, indices.Storage!, parameters, (uint)rows);
        return [values, indices];
    }

    /// <summary>按着色器名和特化参数缓存 TopK 计算管线。</summary>
    /// <param name="shader">TopK 着色器资源名称。</param>
    /// <param name="parameters">按对应着色器参数布局排列的 uint 值，用于特化常量或推送常量。</param>
    /// <returns>由后端缓存并统一释放的 Vulkan 计算管线句柄。</returns>
    private ulong TopKPipeline(string shader, uint[] parameters)
    {
        string key = "topk:" + shader + ":" + string.Join(',', parameters);
        if (!convolutionPipelines.TryGetValue(key, out ulong pipeline)) convolutionPipelines[key] = pipeline = CreateNamedPipeline(shader, parameters);
        return pipeline;
    }
}
