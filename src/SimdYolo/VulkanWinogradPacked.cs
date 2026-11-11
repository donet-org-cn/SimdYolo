namespace SimdYolo;

public sealed unsafe partial class VulkanBackend
{
    // 按 [变换坐标, 输入通道, 输出通道] 缓存只读权重，让一次 vec4 读取服务四个输出通道。
    private readonly bool winogradPacked = Environment.GetEnvironmentVariable("SIMDYOLO_WINOGRAD_PACKED") != "0";
    private readonly Dictionary<Tensor, Tensor> winogradPackedWeights = [];

    private bool TryPackedWinogradMiddle(Tensor weight, Tensor transformed, Tensor products, DeviceStorage zero,
                                        int batch, int planes, int ic, int oc, int tiles)
    {
        bool small = tiles < 64;
        if (!winogradPacked || !weight.IsConstant || oc < 16 || oc % 4 != 0 || (!small && tiles % 4 != 0))
            return false;
        if (!winogradPackedWeights.TryGetValue(weight, out var packed))
        {
            packed = new Tensor(planes, ic, oc) { IsConstant = true };
            for (int coordinate = 0; coordinate < planes; coordinate++)
                for (int ci = 0; ci < ic; ci++)
                    for (int co = 0; co < oc; co++)
                        packed.Data[(coordinate * ic + ci) * oc + co] = weight.Data[(coordinate * oc + co) * ic + ci];
            winogradPackedWeights.Add(weight, packed);
        }
        uint[] parameters = [(uint)ic, (uint)oc, (uint)tiles, (uint)planes];
        string key = "packed:" + string.Join(',', parameters);
        if (!winogradPipelines.TryGetValue(key, out ulong pipeline))
            winogradPipelines[key] = pipeline = CreateNamedPipeline(small ? "winograd_packed_small.spv" : "winograd_packed32.spv", parameters);
        // 小空间使用 16×32 结果块，大空间使用 32×64；均按递增输入通道执行 FP32 FMA。
        Dispatch(pipeline, Upload(packed), transformed.Storage!, products.Storage!, zero, parameters,
                 ((uint)tiles + (small ? 31u : 63u)) / (small ? 32u : 64u),
                 ((uint)oc + (small ? 15u : 31u)) / (small ? 16u : 32u), (uint)(batch * planes));
        return true;
    }
}
