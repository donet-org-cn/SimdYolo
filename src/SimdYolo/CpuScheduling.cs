using System.Runtime.CompilerServices;

namespace SimdYolo;

/// <summary>在专用执行器与 Parallel.For 之间选择调度路径，并将嵌套 CPU 调度降为串行以避免线程超额。</summary>
internal static class CpuScheduling
{
    /// <summary>未关联会话时使用的逻辑处理器线程预算。</summary>
    private static readonly ParallelOptions Default = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };
    /// <summary>并行选项与执行器之间的弱引用关联表。</summary>
    private static readonly ConditionalWeakTable<ParallelOptions, CpuExecutor> Executors = new();
    /// <summary>执行器共用的并行选项。</summary>
    /// <value>执行器共用的并行选项。</value>
    internal static ParallelOptions Options => CpuExecutor.Participant?.Options ?? CpuTensorPool.Current?.Parallel ?? Default;

    /// <summary>将并行配置与后端专用执行器关联，供后续算子调度复用。</summary>
    /// <param name="options">当前后端使用的并行配置。</param>
    /// <param name="executor">需要与并行配置关联的 CPU 专用执行器。</param>
    internal static void Register(ParallelOptions options, CpuExecutor executor) => Executors.Add(options, executor);
    /// <summary>解除并行配置与专用执行器的关联。</summary>
    /// <param name="options">当前后端使用的并行配置。</param>
    internal static void Unregister(ParallelOptions options) => Executors.Remove(options);

    /// <summary>执行半开区间循环；嵌套内核串行运行，可用的后端执行器优先于公共线程池。</summary>
    /// <param name="fromInclusive">循环起点，包含此索引。</param>
    /// <param name="toExclusive">循环终点，不包含此索引。</param>
    /// <param name="options">当前后端使用的并行配置。</param>
    /// <param name="body">对当前索引执行的回调，泛型版本返回更新后的线程局部状态。</param>
    /// <exception cref="AggregateException">参与线程、循环回调或局部状态清理出现一个或多个错误。</exception>
    /// <exception cref="OperationCanceledException">当前并行选项或作业的取消令牌已请求取消。</exception>
    internal static void For(int fromInclusive, int toExclusive, ParallelOptions options, Action<int> body)
    {
        if (CpuExecutor.Participant is not null)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            for (int index = fromInclusive; index < toExclusive; index++) body(index);
        }
        else if (!options.CancellationToken.CanBeCanceled && (options.TaskScheduler is null || options.TaskScheduler == TaskScheduler.Default)
                 && Executors.TryGetValue(options, out var executor)) executor.For(fromInclusive, toExclusive, body);
        else Parallel.For(fromInclusive, toExclusive, options, body);
    }

    /// <summary>执行半开区间循环；嵌套内核串行运行，可用的后端执行器优先于公共线程池。</summary>
    /// <typeparam name="TLocal">每个参与线程独享的局部状态类型。</typeparam>
    /// <param name="fromInclusive">循环起点，包含此索引。</param>
    /// <param name="toExclusive">循环终点，不包含此索引。</param>
    /// <param name="options">当前后端使用的并行配置。</param>
    /// <param name="localInit">为参与线程创建局部状态的回调。</param>
    /// <param name="body">对当前索引执行的回调，泛型版本返回更新后的线程局部状态。</param>
    /// <param name="localFinally">线程结束时释放其局部状态的回调。</param>
    /// <exception cref="AggregateException">参与线程、循环回调或局部状态清理出现一个或多个错误。</exception>
    /// <exception cref="OperationCanceledException">当前并行选项或作业的取消令牌已请求取消。</exception>
    internal static void For<TLocal>(int fromInclusive, int toExclusive, ParallelOptions options,
                                    Func<TLocal> localInit, Func<int, ParallelLoopState?, TLocal, TLocal> body,
                                    Action<TLocal> localFinally)
    {
        if (CpuExecutor.Participant is not null)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            if (toExclusive <= fromInclusive) return;
            var local = localInit();
            try { for (int index = fromInclusive; index < toExclusive; index++) local = body(index, null, local); }
            finally { localFinally(local); }
        }
        else if (!options.CancellationToken.CanBeCanceled && (options.TaskScheduler is null || options.TaskScheduler == TaskScheduler.Default)
                 && Executors.TryGetValue(options, out var executor)) executor.For(fromInclusive, toExclusive, localInit, body, localFinally);
        else Parallel.For(fromInclusive, toExclusive, options, localInit, (index, state, local) => body(index, state, local), localFinally);
    }
}
