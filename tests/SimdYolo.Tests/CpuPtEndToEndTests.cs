using System.Numerics;
using System.Reflection;
using SimdYolo;

/// <summary>提供PyTorch 检测头两级堆选择的 NaN、符号零、同分、批次和空输出的回归验证。</summary>
internal static class CpuPtEndToEndTests
{
    /// <summary>向测试执行器注册PyTorch 检测头两级堆选择的 NaN、符号零、同分、批次和空输出的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU Pt end-to-end cached heaps preserve NaNs, signed zeros, ties and clamped batch shapes", () =>
        {
            var select = typeof(Tensor).Assembly.GetType("SimdYolo.CpuEndToEnd")!.GetMethod("SelectPt", BindingFlags.Static | BindingFlags.NonPublic)!;
            foreach (int batch in new[] { 1, 2 })
                foreach (int classes in new[] { 1, 3, 7 })
                    foreach (int anchors in new[] { 1, 7, 8, 9, 37 })
                        foreach (int maximum in new[] { 1, 8, 49 })
                            for (int frame = 0; frame < 4; frame++)
                            {
                                Tensor input = Input(batch, classes, anchors, frame);
                                Tensor expected = Reference(input, maximum, classes);
                                Tensor actual = (Tensor)select.Invoke(null, [input, maximum, classes])!;
                                Equal(actual, expected);
                            }
        });
        test("CPU Pt end-to-end dispatch preserves zero-k and changing input behavior", () =>
        {
            var select = typeof(Tensor).Assembly.GetType("SimdYolo.PtModel")!.GetMethod("EndToEndTopK", BindingFlags.Static | BindingFlags.NonPublic)!;
            const int classes = 5, anchors = 61;
            Tensor? previous = null;
            float[]? saved = null;
            for (int frame = 0; frame < 4; frame++)
            {
                Tensor input = Input(2, classes, anchors, frame);
                Tensor actual = (Tensor)select.Invoke(null, [input, 17, classes])!;
                Equal(actual, Reference(input, 17, classes));
                if (previous is not null)
                    for (int i = 0; i < previous.Length; i++)
                        if (BitConverter.SingleToInt32Bits(previous.Data[i]) != BitConverter.SingleToInt32Bits(saved![i])) throw new Exception("Pt top-k reused an earlier output.");
                previous = actual;
                saved = (float[])actual.Data.Clone();
            }
            Tensor empty = (Tensor)select.Invoke(null, [Input(2, classes, anchors, 0), 0, classes])!;
            if (!empty.Shape.SequenceEqual([2, 0, 6]) || empty.Length != 0) throw new Exception("Pt top-k zero-k contract changed.");
            Tensor zeroBatch = (Tensor)select.Invoke(null, [new Tensor(0, classes + 4, 100_000_000), 17, classes])!;
            if (!zeroBatch.Shape.SequenceEqual([0, 17, 6]) || zeroBatch.Length != 0) throw new Exception("Pt top-k zero-batch contract changed.");
        });
    }

    /// <summary>生成包含特殊分数、同分值和可变帧坐标的检测头测试输入。</summary>
    /// <param name="batch">测试批次数量或当前批次索引。</param>
    /// <param name="classes">检测类别数量。</param>
    /// <param name="anchors">候选框数量。</param>
    /// <param name="frame">用于改变输入和覆盖多帧行为的帧索引。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Input(int batch, int classes, int anchors, int frame)
    {
        var result = new Tensor(batch, classes + 4, anchors);
        for (int b = 0; b < batch; b++)
        {
            int offset = b * (classes + 4) * anchors;
            for (int c = 0; c < 4; c++)
                for (int a = 0; a < anchors; a++) result.Data[offset + c * anchors + a] = b * 10000 + c * 1000 + a + frame * 100000;
            for (int c = 0; c < classes; c++)
                for (int a = 0; a < anchors; a++)
                {
                    float score = ((a * 7 + c * 3 + frame) % 17 - 8) / 10f;
                    score = ((a + c * 3 + frame) % 13) switch
                    {
                        0 => BitConverter.Int32BitsToSingle(0x7fc00001 + a + c),
                        1 => float.PositiveInfinity,
                        2 => float.NegativeInfinity,
                        3 => 0,
                        4 => BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)),
                        5 => float.Epsilon,
                        6 => -float.Epsilon,
                        _ => score
                    };
                    if (frame == 3) score = c == 0 ? a % 7 * .1f : .25f;
                    result.Data[offset + (c + 4) * anchors + a] = score;
                }
        }
        return result;
    }

    /// <summary>使用独立的完整排序生成检测头的两级选择参考结果。</summary>
    /// <param name="input">待验证的输入数据。</param>
    /// <param name="maximum">两级检测头允许输出的最大候选数量。</param>
    /// <param name="classes">检测类别数量。</param>
    /// <returns>构造完成的测试张量或独立参考结果。</returns>
    private static Tensor Reference(Tensor input, int maximum, int classes)
    {
        int anchors = input.Shape[2], count = Math.Min(maximum, anchors);
        var output = new Tensor(input.Shape[0], count, 6);
        for (int b = 0; b < input.Shape[0]; b++)
        {
            int offset = b * (classes + 4) * anchors;
            var maxima = Enumerable.Repeat(float.NegativeInfinity, anchors).ToArray();
            // 保留既有向量通道最大值规则；两级选择分别完整排序，作为独立于堆内核的参考。
            for (int c = 0; c < classes; c++)
            {
                int src = offset + (c + 4) * anchors, a = 0;
                for (; a <= anchors - Vector<float>.Count; a += Vector<float>.Count)
                    Vector.Max(new Vector<float>(maxima, a), new Vector<float>(input.Data, src + a)).CopyTo(maxima, a);
                for (; a < anchors; a++) maxima[a] = MathF.Max(maxima[a], input.Data[src + a]);
            }
            int[] selectedAnchors = Enumerable.Range(0, anchors).OrderByDescending(a => maxima[a]).ThenBy(a => a).Take(count).ToArray();
            var selected = selectedAnchors.SelectMany(a => Enumerable.Range(0, classes).Select(c => (Anchor: a, Class: c, Score: input.Data[offset + (c + 4) * anchors + a])))
                .OrderByDescending(value => value.Score).ThenBy(value => value.Anchor).ThenBy(value => value.Class).Take(count).ToArray();
            for (int i = 0; i < selected.Length; i++)
            {
                int dst = (b * count + i) * 6;
                for (int coordinate = 0; coordinate < 4; coordinate++) output.Data[dst + coordinate] = input.Data[offset + coordinate * anchors + selected[i].Anchor];
                output.Data[dst + 4] = selected[i].Score;
                output.Data[dst + 5] = selected[i].Class;
            }
        }
        return output;
    }

    /// <summary>比较实际结果与独立参考结果，核对形状、元素和本用例的误差约束。</summary>
    /// <param name="actual">被测实现产生的实际结果。</param>
    /// <param name="expected">独立参考实现产生的预期结果。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    private static void Equal(Tensor actual, Tensor expected)
    {
        if (!actual.Shape.SequenceEqual(expected.Shape)) throw new Exception("Pt top-k output shape changed.");
        for (int i = 0; i < actual.Length; i++)
            if (BitConverter.SingleToInt32Bits(actual.Data[i]) != BitConverter.SingleToInt32Bits(expected.Data[i])) throw new Exception($"Pt top-k changed bits at {i}: {actual.Data[i]} vs {expected.Data[i]}.");
    }
}
