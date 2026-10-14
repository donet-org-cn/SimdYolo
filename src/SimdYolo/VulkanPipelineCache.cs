using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace SimdYolo;

/// <summary>管理与设备、驱动及着色器版本匹配的 Vulkan 管线和分块持久缓存。</summary>
public sealed unsafe partial class VulkanBackend
{
    private const int MaximumCacheBytes = 64 * 1024 * 1024;
    private static readonly byte[] CacheMagic = Encoding.ASCII.GetBytes("SYVK004\0");
    private static readonly Lazy<string> ShaderRevision = new(() =>
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var assembly = typeof(VulkanBackend).Assembly;
        foreach (string name in assembly.GetManifestResourceNames().Where(name => name.EndsWith(".spv", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
        {
            hash.AppendData(Encoding.UTF8.GetBytes(name));
            using var stream = assembly.GetManifestResourceStream(name)!;
            using var memory = new MemoryStream(); stream.CopyTo(memory); hash.AppendData(memory.GetBuffer().AsSpan(0, checked((int)memory.Length)));
        }
        return Convert.ToHexString(hash.GetHashAndReset());
    });
    private ulong pipelineCache;
    private bool pipelineCacheDirty;
    private string? pipelineCacheFile, tuningCacheFile;
    private uint cacheVendor, cacheDevice;
    private byte[] cacheUuid = [];
    private readonly bool persistentPipelineCache = Environment.GetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE") != "0";
    /// <summary>取得是否从可选本地缓存恢复兼容的已编译管线。</summary>
    /// <value>成功装载兼容驱动缓存时为 true。</value>
    public bool PipelineCacheLoaded { get; private set; }
    /// <summary>取得从本地缓存恢复的卷积及 Winograd 分块选择数量。</summary>
    /// <value>通过格式、设备与功能条件筛选的形状条目数。</value>
    public int CachedConvolutionShapes { get; private set; }

    /// <summary>按设备与驱动身份初始化可选持久管线缓存，不兼容或不可读时使用空缓存。</summary>
    /// <param name="properties">当前物理设备 VkPhysicalDeviceProperties 结构的字节地址。</param>
    /// <remarks>文件缓存仅用于减少启动与调优开销；常见路径、权限错误或驱动拒绝缓存时回退空缓存。</remarks>
    private void InitializePipelineCache(byte* properties)
    {
        if (!persistentPipelineCache) return;
        cacheVendor = *(uint*)(properties + 8); cacheDevice = *(uint*)(properties + 12);
        cacheUuid = new ReadOnlySpan<byte>(properties + 276, 16).ToArray();
        try
        {
            string directory = Environment.GetEnvironmentVariable("SIMDYOLO_VULKAN_CACHE_DIR") ?? Path.Combine(Path.GetTempPath(), "SimdYolo", "VulkanCache");
            string identity = $"v1-{cacheVendor:x8}-{cacheDevice:x8}-{*(uint*)(properties + 4):x8}-{Convert.ToHexString(cacheUuid)}";
            pipelineCacheFile = Path.Combine(Path.GetFullPath(directory), identity + ".bin");
            tuningCacheFile = Path.Combine(Path.GetFullPath(directory), identity + $"-{ShaderRevision.Value}-routes3-{(wideConvolutionTiles ? 1 : 0)}{(lowKConvolutionTiles ? 1 : 0)}{(pointRegisterTile != "0" ? 1 : 0)}{(winogradRegisterTile != "0" ? 1 : 0)}{(winogradColumns64 ? 1 : 0)}.tiles.json");
        }
        catch (Exception error) when (CacheIoFailure(error)) { pipelineCacheFile = tuningCacheFile = null; }
        byte[] initial = ReadPipelineCache();
        bool restored = initial.Length != 0;
        fixed (byte* bytes = initial)
        {
            PipelineCacheInfo info = new() { Type = 17, Size = (nuint)initial.Length, Data = bytes };
            ulong created = 0;
            int result = Vk.CreatePipelineCache(device, &info, null, &created);
            if (result != 0 && initial.Length != 0)
            {
                info.Size = 0; info.Data = null;
                restored = false;
                result = Vk.CreatePipelineCache(device, &info, null, &created);
            }
            if (result == 0) { pipelineCache = created; PipelineCacheLoaded = restored; }
        }
        ReadTuningCache();
    }

    /// <summary>读取有长度与 SHA-256 校验的管线缓存，仅接受与当前设备兼容的驱动头。</summary>
    /// <returns>验证后的驱动缓存字节；缺失、无效或不可读取时为空数组。</returns>
    private byte[] ReadPipelineCache()
    {
        if (pipelineCacheFile is null) return [];
        try
        {
            using var stream = new FileStream(pipelineCacheFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            if (stream.Length is < 76 or > MaximumCacheBytes + 44L) return [];
            Span<byte> header = stackalloc byte[44]; stream.ReadExactly(header);
            int length = BinaryPrimitives.ReadInt32LittleEndian(header[8..]);
            if (!header[..8].SequenceEqual(CacheMagic) || length < 32 || length > MaximumCacheBytes || length != stream.Length - 44) return [];
            byte[] data = GC.AllocateUninitializedArray<byte>(length); stream.ReadExactly(data);
            if (!CryptographicOperations.FixedTimeEquals(header[12..], SHA256.HashData(data)) || !CompatibleCacheHeader(data)) return [];
            return data;
        }
        catch (Exception error) when (CacheIoFailure(error)) { return []; }
    }

    /// <summary>验证 Vulkan 缓存头的版本、供应商、设备标识和管线缓存 UUID。</summary>
    /// <param name="data">需要验证的驱动缓存字节。</param>
    /// <returns>缓存头与当前设备兼容时为 true。</returns>
    private bool CompatibleCacheHeader(ReadOnlySpan<byte> data)
        => data.Length >= 32 && BinaryPrimitives.ReadUInt32LittleEndian(data) == 32 && BinaryPrimitives.ReadUInt32LittleEndian(data[4..]) == 1
        && BinaryPrimitives.ReadUInt32LittleEndian(data[8..]) == cacheVendor && BinaryPrimitives.ReadUInt32LittleEndian(data[12..]) == cacheDevice
        && data.Slice(16, 16).SequenceEqual(cacheUuid);

    /// <summary>读取并筛选形状分块选择；拒绝不支持的分块、格式和超大缓存条目。</summary>
    private void ReadTuningCache()
    {
        if (tuningCacheFile is null) return;
        try
        {
            using var stream = new FileStream(tuningCacheFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            if (stream.Length > 1024 * 1024) return;
            var entries = JsonSerializer.Deserialize<Dictionary<string, uint>>(stream);
            if (entries is null || entries.Count > 4096) return;
            foreach (var entry in entries)
            {
                if (entry.Key.StartsWith("wino:", StringComparison.Ordinal))
                {
                    string key = entry.Key[5..]; var values = key.Split(',');
                    if (key.Length > 512 || entry.Value is not (32 or 64 or 128) || values.Length != 17 || values.Any(part => !uint.TryParse(part, out _))) continue;
                    if (entry.Value is 64 or 128 && winogradRegisterTile == "0") continue;
                    if (entry.Value == 128 && (!winogradColumns64 || uint.Parse(values[4]) < 64 || uint.Parse(values[4]) % 64 != 0 || uint.Parse(values[3]) < 64)) continue;
                    winogradMiddleTiles[key] = entry.Value;
                    continue;
                }
                if (entry.Value is not (16 or 32 or 64 or 80 or 96 or 128 or 256) || entry.Key.Length > 512) continue;
                var parts = entry.Key.Split(',');
                if (parts.Length != 16 || parts.Any(part => !uint.TryParse(part, out _))) continue;
                if (!wideConvolutionTiles && entry.Value is 80 or 96 or 128 or 256) continue;
                if (entry.Value == 256 && !RegisterPointwiseShape(parts.Select(uint.Parse).ToArray())) continue;
                convolutionTiles[entry.Key] = entry.Value;
            }
            CachedConvolutionShapes = convolutionTiles.Count + winogradMiddleTiles.Count;
        }
        catch (Exception error) when (CacheIoFailure(error) || error is JsonException) { }
    }

    /// <summary>使用当前驱动缓存创建单个计算管线，成功时标记缓存需要保存。</summary>
    /// <param name="info">待创建的 Vulkan 计算管线描述。</param>
    /// <param name="result">新计算管线句柄的原生输出地址。</param>
    /// <returns>原生 VkResult；0 表示成功，非零由调用者决定是否转换为异常。</returns>
    private int CreateCachedComputePipeline(ComputePipelineInfo* info, ulong* result)
    {
        int status = Vk.CreateComputePipelines(device, pipelineCache, 1, info, null, result);
        if (status == 0) pipelineCacheDirty = true;
        return status;
    }

    /// <summary>保存已更新的驱动缓存和卷积分块选择；可选缓存的常见文件错误不影响后端释放。</summary>
    private void SavePipelineCache()
    {
        if (pipelineCache == 0 || !pipelineCacheDirty) return;
        try
        {
            nuint size = 0;
            if (pipelineCacheFile is not null && Vk.GetPipelineCacheData(device, pipelineCache, &size, null) == 0 && size is >= 32 and <= MaximumCacheBytes)
            {
                byte[] bytes = GC.AllocateUninitializedArray<byte>(checked((int)size));
                fixed (byte* data = bytes)
                {
                    int result = Vk.GetPipelineCacheData(device, pipelineCache, &size, data);
                    if (result is 0 or 5 && size >= 32 && size <= (nuint)bytes.Length && CompatibleCacheHeader(bytes))
                    {
                        var payload = bytes.AsSpan(0, (int)size); byte[] envelope = GC.AllocateUninitializedArray<byte>(checked(payload.Length + 44));
                        CacheMagic.CopyTo(envelope, 0); BinaryPrimitives.WriteInt32LittleEndian(envelope.AsSpan(8), payload.Length);
                        SHA256.HashData(payload, envelope.AsSpan(12, 32)); payload.CopyTo(envelope.AsSpan(44));
                        WriteCacheFile(pipelineCacheFile, envelope);
                    }
                }
            }
            if (tuningCacheFile is not null && (convolutionTiles.Count != 0 || winogradMiddleTiles.Count != 0))
            {
                var tiles = new Dictionary<string, uint>(convolutionTiles);
                foreach (var entry in winogradMiddleTiles) tiles["wino:" + entry.Key] = entry.Value;
                WriteCacheFile(tuningCacheFile, JsonSerializer.SerializeToUtf8Bytes(tiles));
            }
        }
        catch (Exception error) when (CacheIoFailure(error)) { }
    }

    /// <summary>先写同目录临时文件再替换缓存文件，并在 finally 中清理临时文件。</summary>
    /// <param name="path">要更新的缓存文件绝对路径。</param>
    /// <param name="contents">带校验包头或 JSON 格式的缓存内容。</param>
    /// <exception cref="IOException">临时写入或文件替换失败。</exception>
    /// <exception cref="UnauthorizedAccessException">缓存目录或文件没有写权限。</exception>
    private static void WriteCacheFile(string path, byte[] contents)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try { File.WriteAllBytes(temporary, contents); File.Move(temporary, path, overwrite: true); }
        finally { try { File.Delete(temporary); } catch (Exception error) when (CacheIoFailure(error)) { } }
    }
    /// <summary>识别可选磁盘缓存允许忽略的文件、权限、路径与平台异常。</summary>
    /// <param name="error">需要分类的可选缓存文件访问异常。</param>
    /// <returns>属于允许忽略的缓存 IO 异常时为 true。</returns>
    private static bool CacheIoFailure(Exception error) => error is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException or System.Security.SecurityException;

    /// <summary>对应 VkPipelineCacheCreateInfo，引用可选驱动缓存初始数据。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct PipelineCacheInfo
    {
        /// <summary>Vulkan 结构类型标记；在描述符或池容量结构中表示描述符类型。</summary>
        public uint Type;
        /// <summary>可选扩展结构链指针，默认空指针。</summary>
        public nint Next;
        /// <summary>对应 Vulkan 结构或资源的标志位。</summary>
        public uint Flags;
        /// <summary>结构引用的数据或资源大小，单位为字节。</summary>
        public nuint Size;
        /// <summary>初始缓存或特化常量原始数据指针。</summary>
        public void* Data;
    }
    /// <summary>保存 Vulkan 原生函数指针；调用者负责传递有效句柄、缓冲区容量和指针生命周期。</summary>
    private static partial class Vk
    {
        /// <summary>创建可带初始驱动数据的管线缓存。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、缓存创建描述、可选分配器、缓存输出指针。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, PipelineCacheInfo*, void*, ulong*, int> CreatePipelineCache = (delegate* unmanaged<nint, PipelineCacheInfo*, void*, ulong*, int>)Load("CreatePipelineCache");
        /// <summary>查询或读取驱动管线缓存字节。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、缓存句柄、输入容量/输出字节数指针、可选数据地址。返回 VkResult，0 表示成功；此函数指针不负责抛出托管异常。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, nuint*, void*, int> GetPipelineCacheData = (delegate* unmanaged<nint, ulong, nuint*, void*, int>)Load("GetPipelineCacheData");
        /// <summary>销毁管线缓存。</summary>
        /// <value>从 Vulkan 动态库加载的非托管函数入口。</value>
        /// <remarks>原生参数顺序：设备句柄、缓存句柄、可选分配器。原生调用无返回值，句柄有效性与资源生命周期由调用方负责。</remarks>
        public static readonly delegate* unmanaged<nint, ulong, void*, void> DestroyPipelineCache = (delegate* unmanaged<nint, ulong, void*, void>)Load("DestroyPipelineCache");
    }
}
