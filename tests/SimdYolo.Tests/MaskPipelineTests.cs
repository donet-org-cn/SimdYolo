using SimdYolo;

/// <summary>用独立标量公式验证掩码坐标、插值、概率、裁剪与输出所有权。</summary>
internal static class MaskPipelineTests
{
    internal static void Run(Action<string, Action> test)
    {
        test("Mask SIMD decoder vs scalar: ROI, fractional boxes, odd rows, tiny prototypes and output ownership", () =>
        {
            var random = new Random(173);
            foreach (var cfg in new[] { (W: 257, H: 193, P: 13, Q: 11, C: 17), (W: 19, H: 307, P: 9, Q: 7, C: 32), (W: 17, H: 19, P: 1, Q: 1, C: 3), (W: 101, H: 83, P: 17, Q: 13, C: 1) })
            {
                var (_, tr) = ImagePreprocessor.Letterbox(new byte[cfg.W * cfg.H * 3], cfg.W, cfg.H, 640, 640);
                var proto = new Tensor(Enumerable.Range(0, cfg.C * cfg.P * cfg.Q).Select(_ => (float)random.NextDouble() * 4 - 2).ToArray(), 1, cfg.C, cfg.Q, cfg.P);
                int fields = 6 + cfg.C; var head = new Tensor(1, 3, fields);
                var boxes = new[] { new Detection(0, .9f, 0, 0, cfg.W, cfg.H), new Detection(0, .8f, .2f, 1.4f, cfg.W * .73f, cfg.H * .61f), new Detection(0, .7f, cfg.W - .7f, cfg.H - .6f, .5f, .4f) };
                for (int i = 0; i < boxes.Length; i++)
                {
                    var box = boxes[i]; int offset = i * fields;
                    head.Data[offset] = box.X * tr.ScaleX + tr.PadX; head.Data[offset + 1] = box.Y * tr.ScaleY + tr.PadY;
                    head.Data[offset + 2] = (box.X + box.Width) * tr.ScaleX + tr.PadX; head.Data[offset + 3] = (box.Y + box.Height) * tr.ScaleY + tr.PadY;
                    head.Data[offset + 4] = box.Confidence; head.Data[offset + 5] = 0;
                    for (int c = 0; c < cfg.C; c++) head.Data[offset + 6 + c] = (float)random.NextDouble() - .5f;
                }
                var options = new DetectorOptions { Task = YoloTask.Segmentation, ClassCount = 1, OutputLayout = OutputLayout.EndToEnd };
                var result = VisionDecoder.Decode([head, proto], tr, options);
                if (result.Count != 3) throw new Exception("Lost a valid detection");
                var before = result[0].Mask!.Probabilities.ToArray();
                for (int i = 0; i < result.Count; i++)
                {
                    var mask = result[i].Mask!; var expected = Reference(result[i].Box, proto, head.Data.AsSpan(i * fields + 6, cfg.C).ToArray(), tr);
                    if ((mask.X, mask.Y, mask.Width, mask.Height) != (expected.X, expected.Y, expected.Width, expected.Height)) throw new Exception("Mask crop changed");
                    for (int j = 0; j < mask.Probabilities.Length; j++)
                        if (!float.IsFinite(mask.Probabilities[j]) || Math.Abs(mask.Probabilities[j] - expected.Probabilities[j]) > 3e-6) throw new Exception("Mask probability differs from scalar");
                }
                Array.Fill(proto.Data, .7f); _ = VisionDecoder.Decode([head, proto], tr, options);
                if (!before.SequenceEqual(result[0].Mask!.Probabilities)) throw new Exception("Returned mask aliases pooled temporary storage");
                if (VisionDecoder.Decode([head, proto], tr, options, decodeMasks: false).Any(value => value.Mask is not null)) throw new Exception("Boxes-only path created masks");
            }
        });
        test("Mask SIMD bytes preserve nearest-even quantization, tails, clamping and nonfinite behavior", () =>
        {
            foreach (int length in new[] { 0, 1, 7, 31, 32, 33, 257, 65537 })
            {
                var values = Enumerable.Range(0, length).Select(i => (i % 1021 - 127) / 767f).ToArray();
                if (length > 5) { values[0] = float.NaN; values[1] = float.PositiveInfinity; values[2] = float.NegativeInfinity; values[3] = .5f; values[4] = 1; }
                var actual = MaskConverter.ToBytes(values);
                for (int i = 0; i < values.Length; i++) { byte expected = float.IsFinite(values[i]) ? (byte)MathF.Round(Math.Clamp(values[i], 0, 1) * 255) : (byte)0; if (actual[i] != expected) throw new Exception("Quantized byte differs at " + i); }
            }
        });
        test("Per-call prediction confidence reuses a session without changing its default", () =>
        {
            using var detector = new YoloDetector("models/pt/yolov8n.pt", new() { Width = 64, Height = 64, CpuThreads = 2, Confidence = 1 });
            var pixels = new byte[64 * 64 * 3];
            if (detector.Predict(pixels, 64, 64).Count != 0 || detector.Predict(pixels, 64, 64, 0f).Count == 0 || detector.Predict(pixels, 64, 64).Count != 0) throw new Exception("Per-call confidence changed default or ignored override");
            bool rejected = false; try { detector.Predict(pixels, 64, 64, float.NaN); } catch (ArgumentException) { rejected = true; }
            if (!rejected) throw new Exception("Invalid per-call confidence accepted");
        });
    }
    private static InstanceMask Reference(Detection box, Tensor prototype, float[] coefficients, Letterbox t)
    {
        int mw = prototype.Shape[3], mh = prototype.Shape[2], plane = mw * mh;
        var low = new float[plane];
        for (int c = 0; c < coefficients.Length; c++) for (int i = 0; i < plane; i++) low[i] += coefficients[c] * prototype.Data[c * plane + i];
        int left = (int)MathF.Floor(box.X), top = (int)MathF.Floor(box.Y), right = (int)MathF.Ceiling(box.X + box.Width), bottom = (int)MathF.Ceiling(box.Y + box.Height), width = right - left;
        int roiLeft = Math.Clamp((int)MathF.Round((float)t.PadX * mw / t.TargetWidth), 0, mw - 1), roiTop = Math.Clamp((int)MathF.Round((float)t.PadY * mh / t.TargetHeight), 0, mh - 1);
        int roiRight = Math.Clamp((int)MathF.Round((t.PadX + t.SourceWidth * t.ScaleX) * mw / t.TargetWidth), roiLeft + 1, mw), roiBottom = Math.Clamp((int)MathF.Round((t.PadY + t.SourceHeight * t.ScaleY) * mh / t.TargetHeight), roiTop + 1, mh);
        var values = new float[width * (bottom - top)];
        for (int y = top; y < bottom; y++) for (int x = left; x < right; x++)
        {
            float px = Math.Clamp(roiLeft + (x + .5f) * (roiRight - roiLeft) / t.SourceWidth - .5f, roiLeft, roiRight - 1), py = Math.Clamp(roiTop + (y + .5f) * (roiBottom - roiTop) / t.SourceHeight - .5f, roiTop, roiBottom - 1);
            int x0 = (int)px, y0 = (int)py, x1 = Math.Min(x0 + 1, roiRight - 1), y1 = Math.Min(y0 + 1, roiBottom - 1); float fx = px - x0, fy = py - y0;
            float value = (low[y0 * mw + x0] * (1 - fx) + low[y0 * mw + x1] * fx) * (1 - fy) + (low[y1 * mw + x0] * (1 - fx) + low[y1 * mw + x1] * fx) * fy;
            values[(y - top) * width + x - left] = x >= box.X && y >= box.Y && x < box.X + box.Width && y < box.Y + box.Height ? 1 / (1 + MathF.Exp(-value)) : 0;
        }
        return new(left, top, width, bottom - top, values);
    }
}
