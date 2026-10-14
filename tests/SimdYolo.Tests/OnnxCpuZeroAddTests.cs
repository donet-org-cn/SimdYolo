using System.Reflection;
using SimdYolo;

/// <summary>提供ONNX 正零加法折叠、SiLU 融合和公开中间结果保护的回归验证。</summary>
internal static class OnnxCpuZeroAddTests
{
    /// <summary>向测试执行器注册ONNX 正零加法折叠、SiLU 融合和公开中间结果保护的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <param name="devices">需要验证的计算后端集合。</param>
    internal static void Run(Action<string, Action> test, IEnumerable<ComputeDevice> devices)
    {
        test("CPU ONNX positive-zero Add unlocks SiLU fusion; tracing and public intermediates retain nodes", Cpu);
        string v9 = Path.Combine(".cache", "validation", "yolov9t.onnx");
        if (File.Exists(v9))
            test("CPU ONNX YOLOv9 fixture folds 42 positive-zero SiLU diamonds without inference", () => V9(v9));
        foreach (var device in devices.Where(device => device != ComputeDevice.Cpu))
            test($"CPU ONNX positive-zero Add fusion leaves device graph unchanged ({device})", () => Gpu(device));
    }

    /// <summary>加载真实 YOLOv9 测试图，核对正零加法与 SiLU 融合数量。</summary>
    /// <param name="path">已生成的模型或测试夹具文件路径。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void V9(string path)
    {
        string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD");
        try
        {
            using var backend = new CpuBackend();
            var ops = new TensorOps(backend);
            Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD", "1");
            int folded = Count(path, ops, null), traced = Count(path, ops, _ => { });
            Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD", "0");
            int disabled = Count(path, ops, null);
            if (folded != 42 || traced != 0 || disabled != 0)
                throw new Exception($"YOLOv9 zero-Add fold count: enabled={folded}, trace={traced}, disabled={disabled}.");
            Console.WriteLine($"YOLOv9 zero-Add fold count: {folded}; trace={traced}; disabled={disabled}; no inference executed.");
        }
        finally { Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD", previous); }
    }

    /// <summary>在 CPU 上验证正零加法折叠、追踪回调和公开中间结果的保护条件。</summary>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Cpu()
    {
        string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD");
        Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD", "1");
        try
        {
            foreach (var (name, data, shape, intermediate, extraUse, expectedFusions) in new[]
            {
                ("scalar", new[] { 0f }, Array.Empty<int>(), "", false, 1),
                ("channel", new[] { 0f, 0f }, new[] { 1, 2, 1, 1 }, "", false, 1),
                ("negative-zero", new[] { BitConverter.Int32BitsToSingle(int.MinValue) }, Array.Empty<int>(), "", false, 0),
                ("nonzero", new[] { 1f }, Array.Empty<int>(), "", false, 0),
                ("width", new float[7], new[] { 1, 1, 1, 7 }, "", false, 0),
                ("public-conv", new[] { 0f }, Array.Empty<int>(), "raw", false, 0),
                ("public-add", new[] { 0f }, Array.Empty<int>(), "plus", false, 0),
                ("public-sigmoid", new[] { 0f }, Array.Empty<int>(), "sigmoid", false, 0),
                ("extra-conv-use", new[] { 0f }, Array.Empty<int>(), "", true, 0)
            })
            {
                string path = Fixture("cpu-zero-add-" + name, data, shape, intermediate, extraUse);
                using var backend = new CpuBackend();
                if (Count(path, new TensorOps(backend), null) != expectedFusions) throw new Exception("Zero-Add fusion eligibility: " + name);
                if (Count(path, new TensorOps(backend), _ => { }) != 0) throw new Exception("Tracing fused away a zero-Add node.");
                using var reference = new YoloDetector(path, new() { Device = ComputeDevice.Cpu, LayerCompleted = _ => { } });
                using var optimized = new YoloDetector(path, new() { Device = ComputeDevice.Cpu });
                for (int frame = 0; frame < 3; frame++)
                {
                    var input = new Tensor(1, 2, 3, 7);
                    for (int i = 0; i < input.Length; i++) input.Data[i] = MathF.Sin(i * .171f + frame) * 5;
                    if (frame == 2)
                    {
                        input.Data[0] = float.NaN; input.Data[1] = float.PositiveInfinity; input.Data[2] = float.NegativeInfinity;
                        input.Data[3] = 0f; input.Data[4] = -0f; input.Data[5] = float.Epsilon; input.Data[6] = -float.Epsilon;
                    }
                    Tensor[] expected = reference.PredictRaw(input), actual = optimized.PredictRaw(input);
                    Equal(actual, expected);
                    // 原有 Conv+SiLU 融合采用 FP32 数值容差。
                    // 加正零可能把负零规范化，因此此融合不要求输出的零符号位一致。
                }
            }
        }
        finally { Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD", previous); }
    }

    /// <summary>验证 CPU 正零优化不会改变设备后端的图与结果。</summary>
    /// <param name="device">本次验证使用的设备类型。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Gpu(ComputeDevice device)
    {
        string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD");
        Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD", "1");
        try
        {
            string path = Fixture("cpu-zero-add-" + device, [0f], [], "", false);
            using IComputeBackend backend = device == ComputeDevice.Vulkan ? new VulkanBackend() : new MetalBackend();
            if (Count(path, new TensorOps(backend), null) != 0) throw new Exception("CPU fusion changed a GPU graph.");
        }
        finally { Environment.SetEnvironmentVariable("SIMDYOLO_CPU_ONNX_ZERO_ADD", previous); }
    }

    /// <summary>在给定后端和追踪设置下统计测试图中保留的零加法节点。</summary>
    /// <param name="path">已生成的模型或测试夹具文件路径。</param>
    /// <param name="ops">使用指定后端执行测试图的张量算子。</param>
    /// <param name="trace">可选的逐层计时回调，用于核对追踪条件下的融合行为。</param>
    /// <returns>保留的目标节点数量。</returns>
    private static int Count(string path, TensorOps ops, Action<LayerTiming>? trace)
    {
        Type type = typeof(Tensor).Assembly.GetType("SimdYolo.OnnxModel")!;
        object model = Activator.CreateInstance(type, [path, ops, trace])!;
        return (int)type.GetField("cpuZeroAddCount", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model)!;
    }

    /// <summary>构造指定数据流、常量和消费者关系的最小 ONNX 测试模型。</summary>
    /// <param name="name">测试文件、张量或 ONNX 属性的名称。</param>
    /// <param name="zero">用于正零加法折叠的常量数据。</param>
    /// <param name="zeroShape">正零常量的逻辑维度。</param>
    /// <param name="intermediate">需要公开或保护的中间节点名称。</param>
    /// <param name="extraUse">是否给中间结果添加额外消费者。</param>
    /// <returns>写入的 ONNX 测试文件路径。</returns>
    private static string Fixture(string name, float[] zero, int[] zeroShape, string intermediate, bool extraUse)
    {
        var nodes = new List<OnnxFixtures.Node>
        {
            new("Conv", ["images", "weight", "bias"], ["raw"], [OnnxFixtures.Integer("group", 2)]),
            new("Add", ["zero", "raw"], ["plus"]),
            new("Sigmoid", ["plus"], ["sigmoid"]),
            new("Mul", ["plus", "sigmoid"], ["output"])
        };
        var outputs = new List<(string, int[], int)> { ("output", [1, 2, 3, 7], 1) };
        if (intermediate != "") outputs.Add((intermediate, [1, 2, 3, 7], 1));
        if (extraUse) { nodes.Add(new("Identity", ["raw"], ["extra"])); outputs.Add(("extra", [1, 2, 3, 7], 1)); }
        return OnnxFixtures.Write(name, [1, 2, 3, 7], nodes.ToArray(),
            [new("weight", [1f, -1f], [2, 1, 1, 1]), new("bias", [-0f, 0f], [2]), new("zero", zero, zeroShape)], outputs.ToArray());
    }

    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Equal(Tensor[] actual, Tensor[] expected)
    {
        if (actual.Length != expected.Length) throw new Exception("Fusion output count mismatch.");
        for (int output = 0; output < actual.Length; output++)
        {
            if (!actual[output].Shape.SequenceEqual(expected[output].Shape)) throw new Exception("Fusion output shape mismatch.");
            for (int i = 0; i < actual[output].Length; i++)
            {
                float x = actual[output].Data[i], y = expected[output].Data[i];
                if (float.IsNaN(y) ? !float.IsNaN(x) : float.IsInfinity(y) ? x != y : !float.IsFinite(x) || MathF.Abs(x - y) > 2e-6f * MathF.Max(1, MathF.Abs(y)))
                    throw new Exception($"Fusion output {output}, index {i}: {x} vs {y}.");
            }
        }
    }
}
