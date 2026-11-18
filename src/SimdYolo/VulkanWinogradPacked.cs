namespace SimdYolo;

public sealed unsafe partial class VulkanBackend
{
    private readonly bool winogradPacked = Environment.GetEnvironmentVariable("SIMDYOLO_WINOGRAD_PACKED") != "0";

    private bool CanPackWinogradMiddle(int outputChannels, int tiles)
        => winogradPacked && outputChannels >= 16 && outputChannels % 4 == 0 && (tiles < 64 || tiles % 4 == 0);

    // 权重直接变换为 [变换坐标, 输入通道, 输出通道]，供一个 vec4 读取四个输出通道。
    private void DispatchPackedWinogradMiddle(Tensor packed, DeviceStorage transformed, DeviceStorage products, DeviceStorage zero,
                                               int batch, int planes, int ic, int oc, int tiles)
    {
        bool small = tiles < 64;
        uint[] parameters = [(uint)ic, (uint)oc, (uint)tiles, (uint)planes];
        string key = "packed:" + string.Join(',', parameters);
        if (!winogradPipelines.TryGetValue(key, out ulong pipeline))
            winogradPipelines[key] = pipeline = CreateNamedPipeline(small ? "winograd_packed_small.spv" : "winograd_packed32.spv", parameters);
        // 小空间使用 16×32 结果块，大空间使用 32×64；均按递增输入通道执行 FP32 FMA。
        Dispatch(pipeline, Upload(packed), transformed, products, zero, parameters,
                 ((uint)tiles + (small ? 31u : 63u)) / (small ? 32u : 64u),
                 ((uint)oc + (small ? 15u : 31u)) / (small ? 16u : 32u), (uint)(batch * planes));
    }
}
