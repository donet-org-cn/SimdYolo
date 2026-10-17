namespace SimdYolo;

/// <summary>在加载时把 ONNX 名称路由转换为整数槽位，复用逐帧工作区并预计算张量存活边界。</summary>
internal sealed class OnnxExecutionPlan
{
    /// <summary>每帧初始化的只读权重引用；数组本身不参加执行。</summary>
    private readonly Tensor?[] initialValues;
    /// <summary>本次执行的张量槽位，帧结束或异常后全部清空。</summary>
    internal Tensor?[] Values { get; }
    /// <summary>图像输入对应的槽位。</summary>
    internal int Input { get; }
    /// <summary>按公开输出顺序排列的槽位。</summary>
    internal int[] Outputs { get; }
    /// <summary>逐节点的输入、输出、存活槽位与可复用参数数组。</summary>
    internal Node[] Nodes { get; }
    /// <summary>提交张量池的紧凑根集合，不扫描全部模型权重。</summary>
    internal object?[] Roots { get; }

    /// <summary>解析优化后的拓扑，拒绝未定义输入和重复定义，生成不依赖实际图像尺寸的路由。</summary>
    /// <param name="graph">已完成融合和死节点消除的图。</param>
    /// <exception cref="InvalidDataException">图像输入数量、名称拓扑或输出引用不合法。</exception>
    internal OnnxExecutionPlan(OnnxGraph graph)
    {
        if (graph.Inputs.Length != 1) throw new InvalidDataException("ONNX inference requires exactly one image input.");
        var slots = new Dictionary<string, int>(StringComparer.Ordinal);
        var initial = new List<Tensor?>();
        int Define(string name, Tensor? value)
        {
            if (string.IsNullOrEmpty(name) || !slots.TryAdd(name, slots.Count)) throw new InvalidDataException($"ONNX value '{name}' has multiple definitions or an empty name.");
            initial.Add(value); return initial.Count - 1;
        }
        int Resolve(string name) => slots.TryGetValue(name, out int slot) ? slot : throw new InvalidDataException($"ONNX value '{name}' is not defined before use.");
        foreach (var weight in graph.Weights) Define(weight.Key, weight.Value);
        Input = Define(graph.Inputs[0], null);
        Nodes = new Node[graph.Nodes.Count];
        for (int i = 0; i < Nodes.Length; i++)
        {
            var node = graph.Nodes[i]; var inputs = new int[node.Inputs.Length]; var outputs = new int[node.Outputs.Length];
            for (int j = 0; j < inputs.Length; j++) inputs[j] = node.Inputs[j] == "" ? -1 : Resolve(node.Inputs[j]);
            for (int j = 0; j < outputs.Length; j++) outputs[j] = node.Outputs[j] == "" ? -1 : Define(node.Outputs[j], null);
            Nodes[i] = new(inputs, outputs);
        }
        Outputs = graph.Outputs.Select(Resolve).ToArray();
        initialValues = initial.ToArray(); Values = new Tensor?[initial.Count];
        var remaining = new int[initial.Count];
        foreach (var node in Nodes) foreach (int slot in node.Inputs) if (slot >= 0) remaining[slot]++;
        foreach (int slot in Outputs) remaining[slot]++;
        var live = new HashSet<int> { Input }; int maximumLive = 0;
        foreach (var node in Nodes)
        {
            foreach (int slot in node.Outputs) if (slot >= 0 && remaining[slot] > 0) live.Add(slot);
            foreach (int slot in node.Inputs) if (slot >= 0 && --remaining[slot] == 0) live.Remove(slot);
            node.Live = live.ToArray(); maximumLive = Math.Max(maximumLive, live.Count);
        }
        Roots = new object?[maximumLive];
    }

    /// <summary>恢复当前帧的权重与输入引用；调用方必须保证同一会话串行执行。</summary>
    /// <param name="input">当前帧图像张量。</param>
    internal void Begin(Tensor input) { Array.Copy(initialValues, Values, Values.Length); Values[Input] = input; }

    /// <summary>清理当前帧引用，保证异常后可重用且不持有已归还张量池的数组。</summary>
    internal void Clear()
    {
        Array.Clear(Values); Array.Clear(Roots);
        foreach (var node in Nodes) Array.Clear(node.Arguments);
    }

    /// <summary>保存单个节点的整数路由及会话独占参数工作区。</summary>
    /// <param name="inputs">输入槽位；-1 表示省略的可选输入。</param>
    /// <param name="outputs">输出槽位；-1 表示省略的输出。</param>
    internal sealed class Node(int[] inputs, int[] outputs)
    {
        /// <summary>按节点声明排列的输入槽位。</summary>
        internal int[] Inputs { get; } = inputs;
        /// <summary>按节点声明排列的输出槽位。</summary>
        internal int[] Outputs { get; } = outputs;
        /// <summary>节点完成后仍被后续节点或公开输出使用的非权重槽位。</summary>
        internal int[] Live { get; set; } = [];
        /// <summary>按输入顺序排列、在每帧结束时清空的参数引用。</summary>
        internal Tensor?[] Arguments { get; } = new Tensor?[inputs.Length];
    }
}
