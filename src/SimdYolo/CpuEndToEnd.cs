using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SimdYolo;

// 导出检测头先对候选框排序筛选，再对“入选候选排序位置、类别”组合排序筛选。
// 保留两阶段的索引顺序、NaN 和有符号零的并列处理。
// 省去转置张量、索引复制与 Gather 中间缓冲区。
/// <summary>以两个有界堆融合检测头的两阶段 TopK，分别保留 PT 与 ONNX 的并列排序和 NaN 语义。</summary>
internal static class CpuEndToEnd
{
    /// <summary>保存堆中候选项的得分和稳定排序索引。</summary>
    /// <param name="Score">候选项的得分。</param>
    /// <param name="Index">稳定排序使用的扁平索引。</param>
    /// <remarks>各构造参数对应同名自动生成属性，描述本记录保存的数据。</remarks>
    private readonly record struct Entry(float Score, int Index);
    /// <summary>是否启用 PT 检测头的两阶段 TopK 融合。</summary>
    internal static readonly bool PtEnabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_PT_E2E") != "0";

    /// <summary>按 PT 检测头语义执行两阶段 TopK，返回边框、得分和类别；并列项使用原始候选与类别索引。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="maximum">最多保留的检测数量。</param>
    /// <param name="classes">类别总数。</param>
    /// <returns>形状为 [批次,检测数量,6] 的张量，每项依次为四个边框值、得分和类别编号。</returns>
    internal static Tensor SelectPt(Tensor input, int maximum, int classes)
    {
        int anchors = input.Shape[2], count = Math.Min(maximum, anchors);
        float[] source = input.Data;
        var result = Tensor.TemporaryUninitialized(input.Shape[0], count, 6);
        if (count == 0 || input.Shape[0] == 0) return result;
        float[] target = result.Data;
        var anchorHeap = new Entry[count];
        var selected = new Entry[count];
        float[] maxima = ArrayPool<float>.Shared.Rent(anchors);
        try
        {
            for (int b = 0; b < input.Shape[0]; b++)
            {
                int offset = b * (classes + 4) * anchors;
                maxima.AsSpan(0, anchors).Fill(float.NegativeInfinity);
                for (int c = 0; c < classes; c++)
                {
                    int src = offset + (c + 4) * anchors, a = 0;
                    // 保留 PT 原有的向量最大值和标量尾部语义。
                    // PT 的最大值处理与 ONNX 的 NaN 优先规则不同。
                    for (; a <= anchors - Vector<float>.Count; a += Vector<float>.Count)
                        Vector.Max(new Vector<float>(maxima, a), new Vector<float>(source, src + a)).CopyTo(maxima, a);
                    for (; a < anchors; a++) maxima[a] = MathF.Max(maxima[a], source[src + a]);
                }
                int used = 0;
                for (int a = 0; a < anchors; a++) AddPt(anchorHeap, ref used, new(maxima[a], a));
                used = 0;
                for (int c = 0; c < classes; c++)
                    for (int rank = 0; rank < count; rank++)
                    {
                        int a = anchorHeap[rank].Index;
                        AddPt(selected, ref used, new(source[offset + (c + 4) * anchors + a], a * classes + c));
                    }
                Array.Sort(selected, ComparePt);
                for (int i = 0; i < count; i++)
                {
                    int index = selected[i].Index, anchor = index / classes, dst = (b * count + i) * 6;
                    for (int coordinate = 0; coordinate < 4; coordinate++) target[dst + coordinate] = source[offset + coordinate * anchors + anchor];
                    target[dst + 4] = selected[i].Score;
                    target[dst + 5] = index % classes;
                }
            }
            return result;
        }
        finally { ArrayPool<float>.Shared.Return(maxima); }
    }

    /// <summary>按 PT 规则比较候选：得分降序，NaN 位于末尾，并列时原始索引升序。</summary>
    /// <param name="left">参与比较的左侧候选。</param>
    /// <param name="right">参与比较的右侧候选。</param>
    /// <returns>负数表示左项优先，零表示等价，正数表示右项优先。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ComparePt(Entry left, Entry right)
    {
        // PT 的元组优先级与 OrderByDescending 使用 Single.CompareTo。
        // 此处 NaN 排在最后；并列项使用原始候选/类别索引。
        int order = -left.Score.CompareTo(right.Score);
        return order != 0 ? order : left.Index.CompareTo(right.Index);
    }

    /// <summary>将候选纳入 PT 有界堆，堆顶始终保存当前最差的入选项。</summary>
    /// <param name="heap">保留当前最优候选的固定容量堆。</param>
    /// <param name="count">当前有效候选数；引用参数会随插入更新。</param>
    /// <param name="candidate">准备加入候选堆的新条目。</param>
    private static void AddPt(Entry[] heap, ref int count, Entry candidate)
    {
        if (count < heap.Length)
        {
            int slot = count++;
            while (slot > 0)
            {
                int parent = (slot - 1) / 2;
                if (ComparePt(candidate, heap[parent]) <= 0) break;
                heap[slot] = heap[parent];
                slot = parent;
            }
            heap[slot] = candidate;
            return;
        }
        if (ComparePt(candidate, heap[0]) >= 0) return;
        int root = 0;
        while (true)
        {
            int child = root * 2 + 1;
            if (child >= count) break;
            if (child + 1 < count && ComparePt(heap[child + 1], heap[child]) > 0) child++;
            if (ComparePt(candidate, heap[child]) >= 0) break;
            heap[root] = heap[child];
            root = child;
        }
        heap[root] = candidate;
    }

    /// <summary>融合 ONNX 检测头的两阶段 TopK，第二阶段并列项使用第一阶段选中候选的排序位置。</summary>
    /// <param name="input">输入张量或连续输入缓冲区。</param>
    /// <param name="maximum">最多保留的检测数量。</param>
    /// <param name="classes">类别总数。</param>
    /// <returns>形状为 [批次,检测数量,6] 的张量，每项依次为四个边框值、得分和类别编号。</returns>
    internal static Tensor SelectOnnx(Tensor input, int maximum, int classes)
    {
        int anchors = input.Shape[2];
        float[] source = input.Data;
        float[] maxima = ArrayPool<float>.Shared.Rent(anchors);
        try
        {
            maxima.AsSpan(0, anchors).Fill(float.NegativeInfinity);
            for (int c = 0; c < classes; c++)
            {
                int src = (c + 4) * anchors, a = 0;
                for (; a <= anchors - Vector<float>.Count; a += Vector<float>.Count)
                {
                    var previous = new Vector<float>(maxima, a);
                    var next = new Vector<float>(source, src + a);
                    var finite = Vector.BitwiseAnd(Vector.Equals(previous, previous), Vector.Equals(next, next));
                    Vector.ConditionalSelect(finite, Vector.Max(previous, next), new Vector<float>(float.NaN)).CopyTo(maxima, a);
                }
                for (; a < anchors; a++) maxima[a] = MathF.Max(maxima[a], source[src + a]);
            }
            var anchorHeap = new Entry[maximum];
            int count = 0;
            for (int a = 0; a < anchors; a++) Add(anchorHeap, ref count, new(maxima[a], a));
            Array.Sort(anchorHeap, Compare);
            var selected = new Entry[maximum];
            count = 0;
            // 按类别依次遍历连续的输入平面。
            // 堆中缓存得分，比较时无需重复索引除法或稀疏读取。
            // 并列项仍使用导出图中的扁平排序位置。
            for (int c = 0; c < classes; c++)
                for (int rank = 0; rank < maximum; rank++)
                    Add(selected, ref count, new(source[(c + 4) * anchors + anchorHeap[rank].Index], rank * classes + c));
            Array.Sort(selected, Compare);
            var result = Tensor.TemporaryUninitialized(1, maximum, 6);
            float[] target = result.Data;
            for (int i = 0; i < maximum; i++)
            {
                int index = selected[i].Index, anchor = anchorHeap[index / classes].Index, c = index % classes, dst = i * 6;
                for (int coordinate = 0; coordinate < 4; coordinate++) target[dst + coordinate] = source[coordinate * anchors + anchor];
                target[dst + 4] = selected[i].Score;
                target[dst + 5] = c;
            }
            return result;
        }
        finally { ArrayPool<float>.Shared.Return(maxima); }
    }

    /// <summary>按 ONNX 规则比较候选：NaN 优先、得分降序，并列时导出图的扁平排序索引升序。</summary>
    /// <param name="left">参与比较的左侧候选。</param>
    /// <param name="right">参与比较的右侧候选。</param>
    /// <returns>负数表示左项优先，零表示等价，正数表示右项优先。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Compare(Entry left, Entry right)
    {
        float a = left.Score, b = right.Score;
        int order = float.IsNaN(a) ? float.IsNaN(b) ? 0 : -1 : float.IsNaN(b) ? 1 : -a.CompareTo(b);
        return order != 0 ? order : left.Index.CompareTo(right.Index);
    }

    /// <summary>将候选纳入 ONNX 有界堆，以稳定比较规则保留得分最优的指定数量项。</summary>
    /// <param name="heap">保留当前最优候选的固定容量堆。</param>
    /// <param name="count">当前有效候选数；引用参数会随插入更新。</param>
    /// <param name="candidate">准备加入候选堆的新条目。</param>
    private static void Add(Entry[] heap, ref int count, Entry candidate)
    {
        if (count < heap.Length)
        {
            int slot = count++;
            while (slot > 0)
            {
                int parent = (slot - 1) / 2;
                if (Compare(candidate, heap[parent]) <= 0) break;
                heap[slot] = heap[parent];
                slot = parent;
            }
            heap[slot] = candidate;
            return;
        }
        if (Compare(candidate, heap[0]) >= 0) return;
        int root = 0;
        while (true)
        {
            int child = root * 2 + 1;
            if (child >= count) break;
            if (child + 1 < count && Compare(heap[child + 1], heap[child]) > 0) child++;
            if (Compare(candidate, heap[child]) >= 0) break;
            heap[root] = heap[child];
            root = child;
        }
        heap[root] = candidate;
    }
}
