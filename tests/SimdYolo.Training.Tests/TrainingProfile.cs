using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using SimdYolo;
using SimdYolo.Training;

/// <summary>按阶段剖析真实检测头；仅用于开发诊断，不进入库的运行路径。</summary>
internal static class TrainingProfile
{
    internal static void Run(string[] args)
    {
        string fixture = args[1]; using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixture, "fixture.json"))); var f = doc.RootElement;
        Tensor Read(JsonElement d) => new(MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(fixture, d.GetProperty("path").GetString()!))).ToArray(), d.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
        using var extractor = new FrozenFeatureExtractor(File.ReadAllBytes(f.GetProperty("sourceModel").GetString()!), new() { ImageSize = 128, CpuThreads = 4, Device = args.Contains("--vulkan") ? ComputeDevice.Vulkan : ComputeDevice.Cpu });
        var head = new DetectionHead(extractor.Model.DetectionHead, ["object-a", "object-b"], extractor.Model.Names, 26);
        head.Restore(f.GetProperty("parameters").EnumerateArray().ToDictionary(d => d.GetProperty("name").GetString()!, Read));
        var features = f.GetProperty("features").EnumerateArray().Select(Read).ToArray();
        var truth = JsonSerializer.Deserialize<TrainingBox[]>(f.GetProperty("truth"), new JsonSerializerOptions(JsonSerializerDefaults.Web))!;
        var profile = new TimedBackend(extractor.Backend); var ops = new HeadOperations(args.Contains("--gemm") ? profile : extractor.Backend);
        var options = new TrainingOptions { CpuThreads = 4, LearningRate = .0001f }; double[] phases = new double[4];
        void Step(bool measured)
        {
            var tape = new GradientTape(); long start = Stopwatch.GetTimestamp();
            var many = head.Forward(features, ops, tape, true);
            var one = head.Schema.EndToEnd ? head.Forward(features, ops, tape, true, true) : default;
            double forward = Stopwatch.GetElapsedTime(start).TotalMilliseconds; start = Stopwatch.GetTimestamp();
            float mw = head.Schema.Progressive ? .8f : 1;
            _ = DetectionLoss.Compute(many.Boxes, many.Classes, head.Strides, truth, 2, true, head.Schema.RegMax, 10, 10, mw);
            if (head.Schema.EndToEnd) _ = DetectionLoss.Compute(one.Boxes, one.Classes, head.Strides, truth, 2, true, head.Schema.RegMax, head.Schema.Progressive ? 7 : 1, 1, head.Schema.Progressive ? .2f : 1);
            double loss = Stopwatch.GetElapsedTime(start).TotalMilliseconds; start = Stopwatch.GetTimestamp(); tape.Backward();
            double backward = Stopwatch.GetElapsedTime(start).TotalMilliseconds; start = Stopwatch.GetTimestamp();
            foreach (var p in head.Parameters) p.Update(options, 1);
            double update = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            if (measured) { phases[0] += forward; phases[1] += loss; phases[2] += backward; phases[3] += update; }
        }
        for (int i = 0; i < 10; i++) Step(false);
        profile.Rows.Clear(); long alloc = GC.GetTotalAllocatedBytes();
        for (int i = 0; i < 64; i++) Step(true);
        Console.WriteLine(JsonSerializer.Serialize(new { forwardMs = phases[0] / 64, lossMs = phases[1] / 64, backwardMs = phases[2] / 64, sgdMs = phases[3] / 64,
            allocatedMiBPerStep = (GC.GetTotalAllocatedBytes() - alloc) / 64d / 1048576, gemm = profile.Rows.OrderByDescending(p => p.Value.Ms).Select(p => new { shape = p.Key, msPerStep = p.Value.Ms / 64, callsPerStep = p.Value.Calls / 64d }) }, new JsonSerializerOptions { WriteIndented = true }));
    }

    private sealed class TimedBackend(IComputeBackend backend) : IComputeBackend
    {
        internal readonly Dictionary<string, (double Ms, int Calls)> Rows = [];
        public string Name => backend.Name;
        public void Dispose() { }
        public void MatMul(float[] a, float[] b, float[] c, int m, int n, int k)
        {
            long start = Stopwatch.GetTimestamp(); backend.MatMul(a, b, c, m, n, k);
            string key = $"{m}x{n}x{k}"; var old = Rows.GetValueOrDefault(key); Rows[key] = (old.Ms + Stopwatch.GetElapsedTime(start).TotalMilliseconds, old.Calls + 1);
        }
    }
}
