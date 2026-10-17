using System.IO.Compression;
using System.Reflection;
using System.Text;
using SimdYolo;

/// <summary>验证冷启动解析的所有权，以及 ONNX 路由复用在动态形状、异常和跨帧场景中的稳定性。</summary>
internal static class StartupReuseTests
{
    /// <summary>注册解析器边界与各设备执行计划的恢复测试。</summary>
    /// <param name="test">命名测试执行回调。</param>
    /// <param name="devices">本轮需要验证的计算设备。</param>
    internal static void Run(Action<string, Action> test, IEnumerable<ComputeDevice> devices)
    {
        test("Startup pickle spans preserve owned bytes and reject truncation", PickleBytes);
        foreach (var device in devices)
            test($"ONNX routing reuse: aliases, optional inputs, dynamic shapes, exceptions and ownership ({device})", () => Routing(device));
    }

    /// <summary>检查字节对象仍拥有独立数组，所有借用读取路径仍拒绝截断元数据。</summary>
    /// <exception cref="Exception">输出所有权或异常行为改变。</exception>
    private static void PickleBytes()
    {
        foreach (byte opcode in new byte[] { (byte)'C', (byte)'B', 0x8e, 0x96 })
        {
            using var data = new MemoryStream();
            data.WriteByte(opcode);
            int lengthBytes = opcode == (byte)'C' ? 1 : opcode == (byte)'B' ? 4 : 8;
            for (int i = 0; i < lengthBytes; i++) data.WriteByte(i == 0 ? (byte)3 : (byte)0);
            data.Write([17, 29, 41, (byte)'.']);
            byte[] pickle = data.ToArray(); var result = (byte[])ReadPickle(pickle)!;
            Array.Fill(pickle, (byte)255);
            if (!result.SequenceEqual(new byte[] { 17, 29, 41 })) throw new Exception("Pickle byte object borrowed mutable metadata.");
        }
        foreach (byte[] bad in new byte[][] { [(byte)'J', 1, 2, 3], [(byte)'X', 2, 0, 0, 0, 65], [(byte)'C', 3, 1], [0x8e, 1, 0], [(byte)'G', 0] })
        {
            try { ReadPickle(bad); throw new Exception("Truncated pickle accepted."); }
            catch (TargetInvocationException error) when (error.InnerException is InvalidDataException) { }
        }
    }

    /// <summary>以无权重 ZIP 调用只解释数据的内部 Pickle 解析器。</summary>
    /// <param name="pickle">独立编码的 Pickle 消息。</param>
    /// <returns>解析后的根数据对象。</returns>
    private static object? ReadPickle(byte[] pickle)
    {
        using var bytes = new MemoryStream();
        using (var writer = new ZipArchive(bytes, ZipArchiveMode.Create, true)) writer.CreateEntry("empty");
        bytes.Position = 0; using var zip = new ZipArchive(bytes, ZipArchiveMode.Read);
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.PickleReader")!;
        object reader = Activator.CreateInstance(type, [zip, "", false])!;
        return type.GetMethod("Load")!.Invoke(reader, [pickle]);
    }

    /// <summary>对独立构造的小图执行失败恢复、九种形状切换、输出保留与调用方改写检查。</summary>
    /// <param name="device">本轮测试使用的计算后端。</param>
    /// <exception cref="Exception">路由、缓存、生命周期或数值不符合独立参考。</exception>
    private static void Routing(ComputeDevice device)
    {
        string path = OnnxFixtures.Write("startup-routing-" + device, [1, 1, -1, -1],
        [
            new("Shape", ["images"], ["dims"]),
            new("Reshape", ["images", "dims"], ["alias"]),
            new("Add", ["alias", "alias"], ["sum"]),
            new("Neg", ["sum"], ["negative"]),
            new("Slice", ["negative", "starts", "ends", "", ""], ["slice"])
        ],
        [new("starts", [0], [1], 7), new("ends", [1], [1], 7), new("constant", [17, 29], [2])],
        ("sum", [1, 1, -1, -1], 1), ("slice", [1, 1, -1, -1], 1), ("dims", [4], 7), ("constant", [2], 1));
        bool fail = true, reenter = false; int callbacks = 0; YoloDetector? session = null;
        using var detector = new YoloDetector(path, new() { Device = device, CpuThreads = 2, LayerCompleted = _ =>
        {
            if (fail && ++callbacks == 3) { fail = false; throw new InvalidOperationException("startup-reuse-callback"); }
            if (!reenter) return;
            reenter = false;
            try { session!.PredictRaw(new Tensor(1, 1, 2, 3)); }
            catch (InvalidOperationException) { return; }
            throw new Exception("Reentrant inference accepted.");
        } });
        session = detector;
        try { detector.PredictRaw(new Tensor(1, 1, 2, 3)); throw new Exception("Callback exception swallowed."); }
        catch (InvalidOperationException error) when (error.Message == "startup-reuse-callback") { }
        var retained = new List<(Tensor Tensor, float[] Copy)>();
        for (int frame = 0; frame < 36; frame++)
        {
            int height = 2 + frame % 9, width = 3 + frame % 4;
            var input = new Tensor(Enumerable.Range(0, height * width).Select(i => MathF.Sin(i + frame * .3f)).ToArray(), 1, 1, height, width);
            reenter = frame == 1; Tensor[] actual = detector.PredictRaw(input);
            for (int i = 0; i < input.Length; i++)
                if (actual[0].Data[i] != input.Data[i] * 2 || actual[1].Data[i] != -input.Data[i] * 2) throw new Exception("Live alias or optional-input routing changed values.");
            if (!actual[2].Data.SequenceEqual(new float[] { 1, 1, height, width }) || !actual[3].Data.SequenceEqual(new float[] { 17, 29 })) throw new Exception("Shape/constant cache was corrupted.");
            foreach (var item in retained) if (!item.Tensor.Data.SequenceEqual(item.Copy)) throw new Exception("A later frame overwrote a retained output.");
            retained.Add((actual[0], (float[])actual[0].Data.Clone()));
            actual[2].Data[0] = -99; actual[3].Data[0] = -99;
            // 当前帧工作区不能在退出后持有张量引用；异常路径和设备路径也遵循同一规则。
            object model = typeof(YoloDetector).GetField("model", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(detector)!;
            object execution = model.GetType().GetField("execution", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model)!;
            foreach (string property in new[] { "Values", "Roots" })
                if (((Array)execution.GetType().GetProperty(property, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(execution)!).Cast<object?>().Any(value => value is not null)) throw new Exception("ONNX workspace retained a frame.");
        }
    }
}
