using System.Text.Json;
using System.Runtime.InteropServices;
using SimdYolo;
using SimdYolo.Training;

/// <summary>新功能的纯.NET数值、归档和真实模型验证入口。</summary>
internal static class FeatureTests
{
    internal static void Run(string[] args)
    {
        string output = args.Length > 1 ? args[1] : ".cache/features-011"; Directory.CreateDirectory(output);
        var checks = new List<object>();
        void Check(string name, Action action) { action(); checks.Add(new { name, passed = true }); Console.WriteLine("PASS " + name); }
        Check("AP101: perfect, duplicate, false positive, wrong class and no ground truth", () =>
        {
            var truth = new[] { new TrainingBox(0, 0, 0, 10, 10) }; var hit = new Detection(0, .9f, 0, 0, 10, 10);
            var perfect = DetectionEvaluator.Calculate(["a"], [new("a", truth, [hit])]); Near(perfect.Map50To95, 1, 1e-12);
            var duplicate = DetectionEvaluator.Calculate(["a"], [new("a", truth, [hit, hit with { Confidence = .8f }])]); Near(duplicate.Precision, .5, 1e-12); Near(duplicate.Map50, 1, 1e-12);
            var falseFirst = DetectionEvaluator.Calculate(["a"], [new("a", truth, [hit with { X = 20, Confidence = .99f }, hit])]); Near(falseFirst.Map50, .5, 1e-12);
            var wrong = DetectionEvaluator.Calculate(["a", "b"], [new("a", truth, [hit with { ClassId = 1 }])]); Near(wrong.Map50, 0, 0); if (wrong.ConfusionMatrix[0][1] != 1) throw new Exception("Confusion matrix lost wrong class");
            Near(DetectionEvaluator.Calculate(["a"], [new("a", [], [hit])]).Map50, 0, 0);
        });
        Check("Deterministic holdout split and content leakage rejection", () =>
        {
            var dataset = TrainingDataset.CreateDemo(8); var a = dataset.Split(.25, 26); var b = dataset.Split(.25, 26);
            if (a.Training.Images.Count != 6 || a.Validation.Images.Count != 2 || a.Validation.Sha256 != b.Validation.Sha256) throw new Exception("Split mismatch");
            a.Training.ValidateHoldout(a.Validation);
            MustThrow<ArgumentException>(() => a.Training.ValidateHoldout(a.Training));
        });
        Check("INT8 exact dot: SIMD tails and INT32 overflow segmentation", () =>
        {
            foreach (int count in new[] { 1, 15, 16, 17, 63, 4097, 200000 })
            {
                sbyte[] a = Enumerable.Range(0, count + 3).Select(i => (sbyte)(i % 255 - 127)).ToArray(), b = Enumerable.Repeat((sbyte)127, count + 5).ToArray();
                long expected = 0; for (int i = 0; i < count; i++) expected += (int)a[i + 3] * b[i + 5];
                if (PrecisionCpuBackend.Dot(a, 3, b, 5, count) != expected) throw new Exception("INT8 dot differs from exact integer reference");
            }
        });
        Check("INT8 convolution grouped/depthwise/batch/asymmetric pads/dilation/SiLU and GEMM reference", IntegerKernels);
        var dataset = new TrainingDataset(["object-a", "object-b"], TrainingDataset.CreateDemo(2).Images);
        var random = new Random(42); var input = new Tensor(Enumerable.Range(0, 3 * 64 * 64).Select(_ => (float)random.NextDouble()).ToArray(), 1, 3, 64, 64);
        foreach (string name in new[] { "yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo26n" })
        {
            string pt = "models/pt/" + name + ".pt", onnx = Path.Combine(output, name + ".onnx");
            Check(name + " full PT standard ONNX export numerical parity", () =>
            {
                ModelExporter.ExportOnnx(pt, onnx, 64); Compare(pt, onnx, input, .004f);
            });
            Check(name + " trained custom-class SYTRAIN to ONNX parity", () =>
            {
                string directory = Path.Combine(output, name + "-trained");
                using (var trainer = new YoloTrainer(pt, dataset, new() { ImageSize = 64, Epochs = 1, CpuThreads = 2 })) trainer.Train(directory);
                string trained = Path.Combine(directory, "last.sytrain"), exported = Path.Combine(directory, "trained.onnx");
                ModelExporter.ExportOnnx(trained, exported); Compare(trained, exported, input, .004f);
                if (ModelInfo.Read(exported).ClassNames.Count != 2) throw new Exception("Custom classes not exported");
            });
        }
        string model = "models/pt/yolov8n.pt";
        foreach (string name in new[] { "yolov10n", "yolo26n" })
            Check(name + " ONNX two-stage TopK with more than 300 anchors", () =>
            {
                var tensor = new Tensor(Enumerable.Range(0, 3 * 320 * 320).Select(i => (i % 251) / 251f).ToArray(), 1, 3, 320, 320);
                string onnx = Path.Combine(output, name + "-320.onnx"); ModelExporter.ExportOnnx("models/pt/" + name + ".pt", onnx, 320); Compare("models/pt/" + name + ".pt", onnx, tensor, .004f);
            });
        foreach (string name in new[] { "yolov8n-seg", "yolov8n-pose", "yolo26n-seg", "yolo26n-pose" })
            Check(name + " full PT ONNX numerical parity", () => { string onnx = Path.Combine(output, name + ".onnx"); ModelExporter.ExportOnnx("models/pt/" + name + ".pt", onnx, 64); Compare("models/pt/" + name + ".pt", onnx, input, .004f); });
        Check("FP16 mixed operands and INT8 actual model execution are finite", () =>
        {
            foreach (string path in new[] { model, Path.Combine(output, "yolov8n.onnx"), Path.Combine(output, "yolov8n-trained/last.sytrain") })
            foreach (var precision in new[] { InferencePrecision.Float16, InferencePrecision.Int8 })
            {
                using var detector = new YoloDetector(path, new() { Width = 64, Height = 64, CpuThreads = 2, Precision = precision });
                var result = detector.PredictRaw(input); if (result.Any(t => t.Data.Any(v => !float.IsFinite(v)))) throw new Exception("Nonfinite low precision output");
                if (!detector.BackendName.Contains(precision == InferencePrecision.Int8 ? "INT8" : "FP16")) throw new Exception("Precision not actually selected");
            }
            MustThrow<NotSupportedException>(() => new YoloDetector(model, new() { Precision = InferencePrecision.Int8, Device = ComputeDevice.Vulkan }));
        });
        Check("Static INT8 calibration binds model and input", () =>
        {
            string calibration = Path.Combine(output, "calibration.json");
            var profile = ModelCalibration.Calibrate(model, dataset.Images.Select(i => new CalibrationImage(i.Pixels, i.Width, i.Height)), calibration, 64, 2);
            if (profile.Images != 2 || profile.ActivationMaxima.Count < 10) throw new Exception("Calibration omitted convolution activations");
            using var detector = new YoloDetector(model, new() { Precision = InferencePrecision.Int8, CalibrationPath = calibration, Width = 64, Height = 64, CpuThreads = 2 });
            _ = detector.PredictRaw(input); MustThrow<ArgumentException>(() => detector.PredictRaw(new Tensor(1, 3, 96, 96)));
            MustThrow<InvalidDataException>(() => new YoloDetector("models/pt/yolo11n.pt", new() { Precision = InferencePrecision.Int8, CalibrationPath = calibration }));
            MustThrow<ArgumentException>(() => ModelCalibration.Calibrate(model, [], model));
            int[] shape = profile.InputShape; profile.InputShape = null!; string malformed = Path.Combine(output, "malformed-calibration.json");
            File.WriteAllText(malformed, JsonSerializer.Serialize(profile));
            MustThrow<InvalidDataException>(() => new YoloDetector(model, new() { Precision = InferencePrecision.Int8, CalibrationPath = malformed }));
            profile.InputShape = shape; profile.ActivationMaxima.Remove(profile.ActivationMaxima.Keys.First()); File.WriteAllText(malformed, JsonSerializer.Serialize(profile));
            using var missing = new YoloDetector(model, new() { Precision = InferencePrecision.Int8, CalibrationPath = malformed, CpuThreads = 2 });
            MustThrow<InvalidDataException>(() => missing.PredictRaw(input));
        });
        Check("Validation, best checkpoint, early stop and resume holdout binding", () =>
        {
            var split = TrainingDataset.CreateDemo(4).Split(.5, 32); string directory = Path.Combine(output, "validated"); var history = new List<EpochValidation>();
            using (var trainer = new YoloTrainer(model, split.Training, new() { ValidationDataset = split.Validation, ImageSize = 64, Epochs = 5, CpuThreads = 2, EarlyStoppingPatience = 1, MinimumImprovement = 1 }))
            {
                trainer.Train(directory, validationProgress: history.Add);
                if (!trainer.StoppedEarly || trainer.CompletedEpochs != 1 || history.Count != 2) throw new Exception("Early stop failed");
            }
            string last = Path.Combine(directory, "last.sytrain"); if (!File.Exists(Path.Combine(directory, "best.sytrain"))) throw new Exception("Best model not saved");
            MustThrow<InvalidDataException>(() => YoloTrainer.Resume(last, split.Training, 2));
            using var resumed = YoloTrainer.Resume(last, split.Training, 2, cpuThreads: 2, validationDataset: split.Validation);
            resumed.Train(Path.Combine(output, "validated-resumed"));
            if (resumed.CompletedEpochs != 2) throw new Exception("Validated resume failed");
        });
        File.WriteAllText(Path.Combine(output, "features.json"), JsonSerializer.Serialize(new { status = "passed", coreAssemblySha256 = ModelCalibration.Hash(typeof(YoloDetector).Assembly.Location), trainingAssemblySha256 = ModelCalibration.Hash(typeof(YoloTrainer).Assembly.Location), checks }, new JsonSerializerOptions { WriteIndented = true }));
    }
    private static void IntegerKernels()
    {
        // 动态ONNX权重可复用同一个张量存储，低精度缓存不得把它当作只读常量。
        foreach (var precision in new[] { InferencePrecision.Float16, InferencePrecision.Int8 })
        {
            using var dynamicBackend = new PrecisionCpuBackend(precision, 2, null, null); var ops = new TensorOps(dynamicBackend);
            var input = new Tensor([2f], 1, 1, 1, 1); var weight = new Tensor([.5f], 1, 1, 1, 1);
            Near(ops.Conv(input, weight).Data[0], 1, .00001); weight.Data[0] = 1.5f;
            Near(ops.Conv(input, weight).Data[0], 3, .00001);
        }
        float Scale(float[] values) => Math.Max(values.Max(MathF.Abs) / 127, 1e-30f);
        int Q(float value, float scale) => (int)Math.Clamp(MathF.Round(value / scale), -127, 127);
        foreach (var config in new[] { (Ic: 3, Oc: 5, G: 1, Kh: 3), (Ic: 4, Oc: 6, G: 2, Kh: 3), (Ic: 3, Oc: 3, G: 3, Kh: 3), (Ic: 2, Oc: 4, G: 2, Kh: 1) })
        {
            var x = new Tensor(Enumerable.Range(0, 2 * config.Ic * 7 * 9).Select(i => MathF.Sin(i * .17f)).ToArray(), 2, config.Ic, 7, 9);
            var w = new Tensor(Enumerable.Range(0, config.Oc * config.Ic / config.G * config.Kh * 3).Select(i => MathF.Cos(i * .31f)).ToArray(), config.Oc, config.Ic / config.G, config.Kh, 3);
            var bias = new Tensor(Enumerable.Range(0, config.Oc).Select(i => i * .03f).ToArray(), config.Oc);
            int[] strides = [2, 1], pads = [1, 0, 2, 1], dilation = [2, 1];
            using var backend = new PrecisionCpuBackend(InferencePrecision.Int8, 2, null, null); backend.Begin(x.Shape);
            var y = new TensorOps(backend).Conv(x, w, bias, strides, pads, dilation, config.G, true);
            float xs = Scale(x.Data); int k = w.Length / config.Oc;
            for (int n = 0; n < 2; n++) for (int co = 0; co < config.Oc; co++) for (int oy = 0; oy < y.Shape[2]; oy++) for (int ox = 0; ox < y.Shape[3]; ox++)
            {
                float ws = Scale(w.Data.AsSpan(co * k, k).ToArray()); long sum = 0; int group = co / (config.Oc / config.G);
                for (int ci = 0; ci < config.Ic / config.G; ci++) for (int ky = 0; ky < config.Kh; ky++) for (int kx = 0; kx < 3; kx++)
                {
                    int iy = oy * 2 - 1 + ky * 2, ix = ox + kx;
                    if ((uint)iy < 7 && (uint)ix < 9) sum += Q(x.Data[((n * config.Ic + group * (config.Ic / config.G) + ci) * 7 + iy) * 9 + ix], xs) * Q(w.Data[co * k + (ci * config.Kh + ky) * 3 + kx], ws);
                }
                float value = sum * xs * ws + bias.Data[co]; value /= 1 + MathF.Exp(-value);
                Near(y.Data[((n * config.Oc + co) * y.Shape[2] + oy) * y.Shape[3] + ox], value, .00001);
            }
        }
        using var gemm = new PrecisionCpuBackend(InferencePrecision.Int8, 2, null, null);
        const int m = 3, ncol = 7, reduce = 67;
        float[] a = Enumerable.Range(0, m * reduce).Select(i => MathF.Sin(i)).ToArray(), b = Enumerable.Range(0, reduce * ncol).Select(i => MathF.Cos(i)).ToArray(), c = new float[m * ncol];
        gemm.MatMul(a, b, c, m, ncol, reduce); float sa = Scale(a);
        for (int row = 0; row < m; row++) for (int col = 0; col < ncol; col++) { float sb = Scale(Enumerable.Range(0, reduce).Select(i => b[i * ncol + col]).ToArray()); long sum = 0; for (int i = 0; i < reduce; i++) sum += Q(a[row * reduce + i], sa) * Q(b[i * ncol + col], sb); Near(c[row * ncol + col], sum * sa * sb, .00001); }
    }
    private static void Compare(string a, string b, Tensor input, float tolerance)
    {
        using var source = new YoloDetector(a, new() { CpuThreads = 2 }); using var exported = new YoloDetector(b, new() { CpuThreads = 2 });
        var x = source.PredictRaw(input); var y = exported.PredictRaw(input);
        File.WriteAllBytes(b + ".input.f32", MemoryMarshal.AsBytes(input.Data.AsSpan()).ToArray());
        for (int t = 0; t < x.Length; t++) File.WriteAllBytes(b + $".output-{t}.f32", MemoryMarshal.AsBytes(x[t].Data.AsSpan()).ToArray());
        for (int t = 0; t < y.Length; t++) File.WriteAllBytes(b + $".managed-output-{t}.f32", MemoryMarshal.AsBytes(y[t].Data.AsSpan()).ToArray());
        File.WriteAllText(b + ".reference.json", JsonSerializer.Serialize(new { inputShape = input.Shape, outputShapes = x.Select(t => t.Shape).ToArray(), tolerance }));
        if (x.Length != y.Length) throw new Exception("Output count differs");
        for (int t = 0; t < x.Length; t++)
        {
            if (!x[t].Shape.SequenceEqual(y[t].Shape)) throw new Exception("Output shape differs");
            float[] actual = t == 0 && source.Info.EndToEnd && x[t].Rank == 3 ? AlignNearTies(x[t], y[t], tolerance) : y[t].Data;
            for (int i = 0; i < x[t].Length; i++) Near(actual[i], x[t].Data[i], tolerance * (1 + Math.Abs(x[t].Data[i])));
        }
    }
    // 只接受近同分排序交换：同一类别、全部坐标/附加字段必须一对一匹配，不能遗漏或补造候选。
    private static float[] AlignNearTies(Tensor expected, Tensor actual, float tolerance)
    {
        int count = expected.Shape[1], fields = expected.Shape[2]; var result = new float[expected.Length]; var used = new bool[count];
        for (int row = 0; row < count; row++)
        {
            int chosen = -1; float best = float.PositiveInfinity;
            for (int candidate = 0; candidate < count; candidate++)
            {
                if (used[candidate] || actual.Data[candidate * fields + 5] != expected.Data[row * fields + 5] ||
                    MathF.Abs(actual.Data[candidate * fields + 4] - expected.Data[row * fields + 4]) > 2e-6f ||
                    MathF.Abs(expected.Data[candidate * fields + 4] - expected.Data[row * fields + 4]) > 2e-6f) continue;
                float error = 0; for (int field = 0; field < fields; field++) error = MathF.Max(error, MathF.Abs(actual.Data[candidate * fields + field] - expected.Data[row * fields + field]) / (1 + MathF.Abs(expected.Data[row * fields + field])));
                if (error < best) { chosen = candidate; best = error; }
            }
            if (chosen < 0 || best > tolerance) throw new Exception("TopK candidates cannot be matched one-to-one at row " + row);
            used[chosen] = true; actual.Data.AsSpan(chosen * fields, fields).CopyTo(result.AsSpan(row * fields, fields));
        }
        return result;
    }
    private static void Near(double value, double expected, double tolerance) { if (!double.IsFinite(value) || Math.Abs(value - expected) > tolerance) throw new Exception($"Expected {expected}, actual {value}, tolerance {tolerance}"); }
    private static void MustThrow<T>(Action action) where T : Exception { try { action(); } catch (T) { return; } throw new Exception("Expected " + typeof(T).Name); }
}
