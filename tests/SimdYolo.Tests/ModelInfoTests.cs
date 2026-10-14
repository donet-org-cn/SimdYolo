using System.IO.Compression;
using System.Text;
using SimdYolo;

/// <summary>验证 PT/ONNX 标签解析、缺失标签、稀疏编号及元数据独立读取。</summary>
internal static class ModelInfoTests
{
    /// <summary>注册模型元数据回归用例。</summary>
    /// <param name="test">执行命名断言的测试回调。</param>
    internal static void Run(Action<string, Action> test)
    {
        test("Metadata labels: Python/JSON dicts and lists preserve Unicode and escapes", () =>
        {
            foreach (var (text, expected) in new (string, string[])[]
            {
                ("{0: '人', 1: \"driver's seat\", 2: '反斜线\\\\与\\u732b',}", ["人", "driver's seat", "反斜线\\与猫"]),
                ("[\"猫\", \"quote\\\"inside\", \"line\\nend\"]", ["猫", "quote\"inside", "line\nend"]),
                ("('猫', 'dog\\'s', '\\U0001f431',)", ["猫", "dog's", "🐱"]),
                ("{\"0\": \"猫\", \"1\": \"狗\"}", ["猫", "狗"])
            })
            {
                string path = Graph("labels-" + Guid.NewGuid().ToString("N"), text);
                var info = ModelInfo.Read(path);
                if (info.ClassCount != expected.Length || !info.HasClassNames || !expected.SequenceEqual(Enumerable.Range(0, expected.Length).Select(id => info.ClassNames[id]))) throw new Exception("Labels were changed.");
                using var detector = new YoloDetector(path);
                if (!detector.ClassNames.OrderBy(p => p.Key).Select(p => p.Value).SequenceEqual(expected)) throw new Exception("Metadata and execution loaders disagree.");
                int[] shape = info.InputShape; shape[0] = 999;
                if (info.InputShape[0] == 999 || detector.Info.ClassCount != expected.Length) throw new Exception("Metadata snapshot is mutable.");
            }
        });
        test("Metadata labels: sparse IDs retain class channels, absent names stay empty", () =>
        {
            var sparse = ModelInfo.Read(Graph("sparse-labels", "{0: '人', 7: '猫'}"));
            if (sparse.ClassNames.Count != 2 || sparse.ClassCount != 8 || sparse.ClassNames[7] != "猫") throw new Exception("Sparse IDs were renumbered.");
            var missing = ModelInfo.Read(OnnxFixtures.Write("missing-labels", [1, 3, 32, 32], [new("Identity", ["images"], ["output"])], [], ("output", [1, 84, 21], 1)));
            if (missing.HasClassNames || missing.ClassNames.Count != 0) throw new Exception("Missing labels were invented.");
        });
        test("Metadata labels: invalid labels and expressions are rejected", () =>
        {
            foreach (string labels in new[] { "{-1: 'bad'}", "{0: 'a', 0: 'b'}", "{0:'a', '0':'b'}", "{2147483647:'bad'}", "{0: None}", "['a', 2]", "{0:'unterminated}", "{0:__import__('os')}", "{0:'ok'} + {'x':'y'}" })
            {
                try { ModelInfo.Read(Graph("bad-labels", labels)); }
                catch (InvalidDataException) { continue; }
                throw new Exception("Invalid label metadata was accepted: " + labels);
            }
        });
        test("Metadata reader skips unsupported weight data and never creates an execution backend", () =>
        {
            string path = OnnxFixtures.Metadata(OnnxFixtures.Write("metadata-no-weights", [1, 3, 32, 32], [], [new("unused", [0], [1], 99)], ("output", [1, 300, 57], 1)),
                ("names", "['person']"), ("task", "pose"), ("kpt_shape", "[17,3]"), ("args", "{'end2end': True, 'confidence': 0.25}"));
            var info = ModelInfo.Read(path);
            if (!info.EndToEnd || info.Task != YoloTask.Pose || !info.KeypointShape.SequenceEqual([17, 3]) || info.ClassNames[0] != "person") throw new Exception("Pose metadata was not read.");
            try { using var detector = new YoloDetector(path); }
            catch (NotSupportedException) { return; }
            throw new Exception("Fixture did not contain an unsupported weight.");
        });
        test("PT metadata reads dictionary/list labels without a model architecture", () =>
        {
            string path = Path.GetFullPath(".cache/tests/metadata-only.pt"); Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            using (var stream = File.Create(path)) using (var zip = new ZipArchive(stream, ZipArchiveMode.Create))
            {
                using var data = zip.CreateEntry("fixture/data.pkl").Open(); using var writer = new BinaryWriter(data, Encoding.UTF8);
                writer.Write(new byte[] { 0x80, 2, (byte)'}' }); PickleText(writer, "names");
                writer.Write((byte)'('); PickleText(writer, "人"); PickleText(writer, "猫"); writer.Write(new byte[] { (byte)'l', (byte)'s', (byte)'.' });
            }
            var info = ModelInfo.Read(path);
            if (info.ClassCount != 2 || info.ClassNames[0] != "人" || info.ClassNames[1] != "猫") throw new Exception("PT list labels not read.");
        });
        test("Official YOLO26 PT metadata reads labels after all tensor storage entries are removed", () =>
        {
            string original = "models/pt/yolo26n-pose.pt";
            if (!File.Exists(original)) { Console.WriteLine("SKIP optional official YOLO26 metadata fixture."); return; }
            string stripped = Path.GetFullPath(".cache/tests/yolo26-metadata-without-storage.pt");
            using (var input = ZipFile.OpenRead(original)) using (var output = ZipFile.Open(stripped, ZipArchiveMode.Create))
            {
                var entry = input.Entries.Single(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal));
                using var source = entry.Open(); using var target = output.CreateEntry(entry.FullName).Open(); source.CopyTo(target);
            }
            var info = ModelInfo.Read(stripped);
            if (info.ClassNames[0] != "person" || info.ClassCount != 1 || !info.EndToEnd || info.Task != YoloTask.Pose) throw new Exception("Metadata reader accessed missing weight storage.");
            File.Delete(stripped);
        });
    }

    /// <summary>建立带标签的最小 ONNX 图，供元数据和执行加载器交叉验证。</summary>
    /// <param name="name">测试文件名。</param>
    /// <param name="labels">names 字面量文本。</param>
    /// <returns>测试 ONNX 路径。</returns>
    private static string Graph(string name, string labels) => OnnxFixtures.Metadata(OnnxFixtures.Write(name, [1, 3, 32, 32], [new("Identity", ["images"], ["output"])], [], ("output", [1, 84, 21], 1)), ("names", labels));

    /// <summary>用独立编码器写入 Pickle BINUNICODE 字段。</summary>
    /// <param name="writer">测试归档的二进制写入器。</param>
    /// <param name="text">要保存的 Unicode 标签。</param>
    private static void PickleText(BinaryWriter writer, string text) { byte[] data = Encoding.UTF8.GetBytes(text); writer.Write((byte)'X'); writer.Write(data.Length); writer.Write(data); }
}
