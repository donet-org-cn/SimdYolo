namespace SimdYolo;

// ONNX 检测头操作把大型特征和分数保留在设备中。
// 索引转换与边界检查在分派前执行，使负数、小数和越界 FP32 索引
// 遵循当前托管算子的转换及错误规则。
/// <summary>在设备上执行检测头的归约与索引选取，避免下载大型特征和分数张量。</summary>
internal static class OnnxDeviceOperations
{
    // 环境开关用于与托管回退路径进行可重复的 A/B 验证。
    /// <summary>是否启用 ONNX 设备检测头快路径；设置 SIMDYOLO_GPU_ONNX_HEAD=0 可关闭。</summary>
    /// <value>是否启用 ONNX 设备检测头快路径；设置 SIMDYOLO_GPU_ONNX_HEAD=0 可关闭。</value>
    internal static bool Enabled => Environment.GetEnvironmentVariable("SIMDYOLO_GPU_ONNX_HEAD") != "0";
    /// <summary>当前着色器分派覆盖的最大工作元素数量。</summary>
    private const int MaximumWorkItems = 65535 * 256;

    /// <summary>检查最后一维归约行数、行长度与总元素数是否满足设备内核限制。</summary>
    /// <param name="input">当前节点或模块的输入。</param>
    /// <param name="rows">按最后一维分组后的行数量。</param>
    /// <param name="size">每行参与归约的元素数量。</param>
    /// <returns>几何匹配且不超过分派限制时返回 true。</returns>
    internal static bool CanReduce(Tensor input, int rows, int size)
        => rows > 0 && rows <= MaximumWorkItems && size is >= 0 and <= 1024 && (long)rows * size == input.Length;

    /// <summary>检查张量维度、逻辑存储长度与索引工作量是否适合设备逐元素选取。</summary>
    /// <param name="input">当前节点或模块的输入。</param>
    /// <param name="indices">使用 FP32 保存索引值的张量，支持负索引。</param>
    /// <returns>满足设备内核尺寸限制时返回 true。</returns>
    internal static bool CanGather(Tensor input, Tensor indices)
        => input.Rank <= 4 && indices.Length <= MaximumWorkItems && Tensor.Count(input.Shape) == input.Length && Tensor.Count(indices.Shape) == indices.Length;

    /// <summary>提交设备最后一维最大值归约内核。</summary>
    /// <param name="owner">持有当前输入存储并提交计算的设备后端。</param>
    /// <param name="input">当前节点或模块的输入。</param>
    /// <param name="shape">按行优先顺序声明的张量维度。</param>
    /// <param name="rows">按最后一维分组后的行数量。</param>
    /// <param name="size">每行参与归约的元素数量。</param>
    /// <returns>指定输出形状的设备张量。</returns>
    internal static Tensor ReduceMaxLast(IDeviceBackend owner, Tensor input, int[] shape, int rows, int size)
        => owner.Compute(36, [input], shape, DeviceOps.Parameters((uint)rows, (uint)size), rows);

    /// <summary>在主机上先验证索引与非归约维度，再把逐元素索引选取提交给设备。</summary>
    /// <param name="owner">持有当前输入存储并提交计算的设备后端。</param>
    /// <param name="input">当前节点或模块的输入。</param>
    /// <param name="indices">使用 FP32 保存索引值的张量，支持负索引。</param>
    /// <param name="axis">操作轴；负数表示从最后一维反向计数。</param>
    /// <returns>与 indices 形状相同的设备输出张量。</returns>
    /// <exception cref="ArgumentException">负索引规范后越界，或非所选轴的索引维度超过输入维度。</exception>
    /// <exception cref="OverflowException">FP32 索引不能转换为 Int32。</exception>
    internal static Tensor GatherElements(IDeviceBackend owner, Tensor input, Tensor indices, int axis)
    {
        if (indices.Length != 0)
        {
            for (int d = 0; d < input.Rank; d++)
                if (d != axis && indices.Shape[d] > input.Shape[d]) throw new ArgumentException("GatherElements index out of range.");
            float[] indexData = indices.Data;
            for (int i = 0; i < indexData.Length; i++)
            {
                int index = checked((int)indexData[i]);
                if (index < 0) index += input.Shape[axis];
                if ((uint)index >= input.Shape[axis]) throw new ArgumentException("GatherElements index out of range.");
            }
        }
        var parameters = DeviceOps.Parameters((uint)indices.Length);
        int lead = 4 - input.Rank; int[] strides = Tensor.Strides(input.Shape);
        for (int dimension = 0; dimension < 4; dimension++)
        {
            parameters[1 + dimension] = dimension < lead ? 1u : (uint)indices.Shape[dimension - lead];
            parameters[5 + dimension] = dimension < lead ? 0u : (uint)strides[dimension - lead];
        }
        parameters[9] = (uint)(axis + lead); parameters[10] = (uint)input.Shape[axis];
        return owner.Compute(37, [input, indices], indices.Shape, parameters, indices.Length);
    }
}
