using System.IO.Compression;

namespace SimdYolo;

public static partial class ModelExporter
{
    private const int PlateModelLimit = 16 * 1024 * 1024;

    /// <summary>
    /// 导入 chinese-plate-recognition 的 PlateNet 字符权重，生成供车牌 OCR 使用的 ONNX。
    /// 输入为 [1,3,48,168] 的 BGR 张量，范围 [-1,1]；输出为 [1,时间步,字符数+1] 的 CTC 概率。
    /// </summary>
    /// <param name="checkpoint">从文件开头读取的 torch.save ZIP 流，包含 cfg 和 state_dict。调用方保有流的所有权。</param>
    /// <returns>含归一化、字符网络和 CTC 概率输出的 ONNX 数据；最后一类用于 Paddle OCR 的保留空格。</returns>
    /// <exception cref="InvalidDataException">权重、结构或归档大小不符合 PlateNet 字符模型约定。</exception>
    /// <exception cref="NotSupportedException">Torch 归档格式或存储类型不受支持。</exception>
    /// <exception cref="IOException">无法读取检查点。</exception>
    public static byte[] ExportPlateRecognizerOnnx(Stream checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        using var buffered = checkpoint.CanSeek ? null : CopyPlateCheckpoint(checkpoint);
        Stream source = buffered ?? checkpoint;
        if (source.Length > PlateModelLimit) throw new InvalidDataException("PlateNet checkpoint exceeds 16 MiB.");
        using var archive = new ZipArchive(source, ZipArchiveMode.Read, leaveOpen: true);
        if (archive.Entries.Count > 1024 || archive.Entries.Sum(entry => entry.Length) > PlateModelLimit)
            throw new InvalidDataException("PlateNet checkpoint exceeds the archive limits.");
        var pickle = archive.Entries.SingleOrDefault(entry => entry.FullName.EndsWith("/data.pkl", StringComparison.Ordinal))
            ?? throw new NotSupportedException("PlateNet requires a torch.save ZIP checkpoint.");
        string prefix = pickle.FullName[..^"data.pkl".Length];
        if (archive.GetEntry(prefix + "byteorder") is { } byteOrder)
        {
            using var order = new StreamReader(byteOrder.Open());
            if (order.ReadToEnd().Trim() != "little") throw new NotSupportedException("Big-endian PlateNet storage is unsupported.");
        }
        var metadata = new byte[checked((int)pickle.Length)];
        using (var data = pickle.Open()) data.ReadExactly(metadata);
        if (new PickleReader(archive, prefix).Load(metadata) is not Dictionary<object, object?> root ||
            root.GetValueOrDefault("cfg") is not List<object?> configuration || configuration.Count is < 1 or > 32 ||
            root.GetValueOrDefault("state_dict") is not Dictionary<object, object?> state)
            throw new InvalidDataException("PlateNet requires cfg and state_dict.");

        using var graph = new OnnxTraceBackend();
        var ops = new TensorOps(graph);
        var previousDevice = DeviceScope.Current;
        var previousPool = CpuTensorPool.Current;
        try
        {
            DeviceScope.Current = graph;
            CpuTensorPool.Current = null;
            var input = graph.Input([1, 3, 48, 168]);
            Tensor features = TensorOps.Arithmetic(
                TensorOps.Arithmetic(TensorOps.Scale(input, .5f), Scalar(-.088f), 10), Scalar(.193f), 13);
            int layer = 0, inChannels = 3;
            for (int index = 0; index < configuration.Count; index++)
            {
                if (configuration[index] is "M" && index != 0)
                {
                    // Lower the 3x3/stride-2 ceil windows to explicit end padding for mobile OCR runtimes.
                    int endHeight = features.Shape[2] % 2 == 0 ? 1 : 0;
                    int endWidth = features.Shape[3] % 2 == 0 ? 1 : 0;
                    features = TensorOps.Pool(features, [3, 3], [2, 2], [0, 0, endHeight, endWidth]);
                    layer++;
                    continue;
                }
                if (configuration[index] is not (int or long) ||
                    Convert.ToInt64(configuration[index]) is < 1 or > 512)
                    throw new InvalidDataException("PlateNet cfg entries must be channel counts or M.");
                int outChannels = Convert.ToInt32(configuration[index]);
                int kernel = index == 0 ? 5 : 3, pad = index == 0 ? 0 : 1;
                string conv = $"feature.{layer}";
                string bn = $"feature.{layer + 1}";
                var weight = Parameter(state, conv + ".weight", [outChannels, inChannels, kernel, kernel]);
                var bias = Parameter(state, conv + ".bias", [outChannels]);
                var gamma = Parameter(state, bn + ".weight", [outChannels]);
                var beta = Parameter(state, bn + ".bias", [outChannels]);
                var mean = Parameter(state, bn + ".running_mean", [outChannels]);
                var variance = Parameter(state, bn + ".running_var", [outChannels]);
                var fusedWeight = new Tensor(weight.Shape) { IsConstant = true };
                var fusedBias = new Tensor(outChannels) { IsConstant = true };
                int weightStride = weight.Length / outChannels;
                for (int c = 0; c < outChannels; c++)
                {
                    if (variance.Data[c] < 0) throw new InvalidDataException("Negative PlateNet batch normalization variance.");
                    float scale = gamma.Data[c] / MathF.Sqrt(variance.Data[c] + 1e-5f);
                    for (int j = c * weightStride; j < (c + 1) * weightStride; j++)
                        fusedWeight.Data[j] = weight.Data[j] * scale;
                    fusedBias.Data[c] = beta.Data[c] + (bias.Data[c] - mean.Data[c]) * scale;
                }
                features = ops.Conv(features, fusedWeight, fusedBias, pads: [pad, pad, pad, pad]);
                features = graph.Compute(3, [features], features.Shape, [], features.Length);
                inChannels = outChannels;
                layer += 3;
            }

            if (features.Shape[2] != 5 || state.GetValueOrDefault("newCnn.weight") is not Tensor classifier ||
                classifier.Rank != 4 || classifier.Shape[0] is < 2 or > 1024)
                throw new InvalidDataException("Unexpected PlateNet character classifier or feature height.");
            int classes = classifier.Shape[0];
            var projection = Parameter(state, "newCnn.weight", [classes, inChannels, 1, 1]);
            var projectionBias = Parameter(state, "newCnn.bias", [classes]);
            features = TensorOps.Pool(features, [5, 2], [1, 1], [0, 1, 0, 1]);
            var logits = ops.Conv(features, projection, projectionBias);
            int steps = logits.Shape[3];
            var sequence = graph.Transpose(graph.Squeeze(logits, 2, [1, classes, steps]), [0, 2, 1], [1, steps, classes]);
            var reservedSpace = new Tensor(1, steps, 1) { IsConstant = true };
            Array.Fill(reservedSpace.Data, -100f);
            var probabilities = TensorOps.Softmax(TensorOps.Concat([sequence, reservedSpace], 2), 2);
            return graph.Serialize([probabilities], "PlateNet CTC", new Dictionary<string, string>
            {
                ["task"] = "ocr",
                ["simdyolo.export"] = "PlateNet/state_dict/CTC"
            });
        }
        finally
        {
            DeviceScope.Current = previousDevice;
            CpuTensorPool.Current = previousPool;
        }
    }

    private static Tensor Scalar(float value) => new([value], []) { IsConstant = true };

    private static Tensor Parameter(Dictionary<object, object?> state, string name, int[] shape)
    {
        if (state.GetValueOrDefault(name) is not Tensor tensor ||
            !tensor.Shape.AsSpan().SequenceEqual(shape) || tensor.Data.Any(value => !float.IsFinite(value)))
            throw new InvalidDataException($"Invalid PlateNet parameter: {name}.");
        return tensor;
    }

    private static MemoryStream CopyPlateCheckpoint(Stream checkpoint)
    {
        var result = new MemoryStream();
        try
        {
            Span<byte> buffer = stackalloc byte[8192];
            int count;
            while ((count = checkpoint.Read(buffer)) != 0)
            {
                if (result.Length + count > PlateModelLimit)
                    throw new InvalidDataException("PlateNet checkpoint exceeds 16 MiB.");
                result.Write(buffer[..count]);
            }
            result.Position = 0;
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }
}
