using System.Collections.Concurrent;
using System.Reflection;
using SimdYolo;

/// <summary>提供专属 CPU 执行器的强制休眠交接、线程预算、异常清理和并发释放的回归验证。</summary>
internal static class CpuExecutorTests
{
    private static readonly Assembly Library = typeof(CpuBackend).Assembly;
    private static readonly Type Scheduling = Library.GetType("SimdYolo.CpuScheduling")!;
    private static readonly MethodInfo ActionFor = Scheduling.GetMethods(BindingFlags.Static | BindingFlags.NonPublic)
        .Single(method => method.Name == "For" && !method.IsGenericMethod);
    private static readonly MethodInfo LocalFor = Scheduling.GetMethods(BindingFlags.Static | BindingFlags.NonPublic)
        .Single(method => method.Name == "For" && method.IsGenericMethod);
    private static readonly PropertyInfo Options = typeof(CpuBackend).GetProperty("Parallel", BindingFlags.Instance | BindingFlags.NonPublic)!;

    /// <summary>向测试执行器注册专属 CPU 执行器的强制休眠交接、线程预算、异常清理和并发释放的验证用例。</summary>
    /// <param name="test">注册并执行命名回归用例的回调。</param>
    /// <exception cref="Exception">测试断言不满足预期条件时。</exception>
    /// <exception cref="InvalidOperationException">测试所需的状态或作用域约束不满足时。</exception>
    internal static void Run(Action<string, Action> test)
    {
        test("CPU executor: 20k forced-park epochs with changing participants and Task caller", () => Enabled(() =>
        {
            string? previousSpins = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR_SPINS");
            Environment.SetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR_SPINS", "0");
            try
            {
                var task = Task.Run(() =>
                {
                    using var backend = new CpuBackend(4);
                    int[] counts = [1, 2, 3, 4, 13];
                    for (int epoch = 0; epoch < 20_000; epoch++)
                    {
                        int count = counts[epoch % counts.Length];
                        var hits = new int[count];
                        For(backend, 0, count, index => Interlocked.Increment(ref hits[index]));
                        if (hits.Any(hit => hit != 1)) throw new Exception($"Lost or duplicated work at forced-park epoch {epoch}.");
                    }
                });
                // 后端由任务持有；看门狗超时时，避免在等待线程上同步释放可能卡在屏障的执行器。
                if (!task.Wait(TimeSpan.FromSeconds(30))) throw new Exception("CPU executor lost a forced-park wakeup within 20k epochs.");
                task.GetAwaiter().GetResult();
            }
            finally { Environment.SetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR_SPINS", previousSpins); }
        }));

        test("CPU executor: bounded worker budget, nested backend calls and thread scopes", () => Enabled(() =>
        {
            using var backend = new CpuBackend(4);
            using var innerBackend = new CpuBackend(4);
            using var entered = new CountdownEvent(4);
            var threads = new ConcurrentDictionary<int, byte>();
            var hits = new int[257];
            int caller = Environment.CurrentManagedThreadId, maximum = 0, active = 0;
            Type poolType = Library.GetType("SimdYolo.CpuTensorPool")!;
            var pool = Activator.CreateInstance(poolType, true)!;
            FieldInfo currentPool = poolType.GetField("Current", BindingFlags.Static | BindingFlags.NonPublic)!;
            var prior = currentPool.GetValue(null);
            currentPool.SetValue(null, pool);
            try
            {
                For(backend, -7, 250, index =>
                {
                    int id = Environment.CurrentManagedThreadId;
                    if (threads.TryAdd(id, 0))
                    {
                        int now = Interlocked.Increment(ref active);
                        int before;
                        do { before = Volatile.Read(ref maximum); } while (now > before && Interlocked.CompareExchange(ref maximum, now, before) != before);
                        entered.Signal();
                        if (!entered.Wait(TimeSpan.FromSeconds(10))) throw new Exception("The caller and all three workers did not participate.");
                        if (id != caller && currentPool.GetValue(null) is not null) throw new Exception("A worker inherited the inference caller's tensor pool.");
                        // 内核中嵌套另一后端时仍由当前参与线程执行，避免额外启动四个工作线程。
                        For(innerBackend, 0, 3, _ =>
                        {
                            if (Environment.CurrentManagedThreadId != id) throw new Exception("Nested scheduling exceeded the CPU thread budget.");
                        });
                        Interlocked.Decrement(ref active);
                    }
                    Interlocked.Increment(ref hits[index + 7]);
                });
                if (maximum != 4 || threads.Count != 4 || hits.Any(hit => hit != 1)) throw new Exception("The executor lost work or exceeded its participant budget.");
                if (!ReferenceEquals(currentPool.GetValue(null), pool)) throw new Exception("The caller's tensor scope was not restored.");
                var participant = Library.GetType("SimdYolo.CpuExecutor")!.GetField("Participant", BindingFlags.Static | BindingFlags.NonPublic)!;
                if (participant.GetValue(null) is not null) throw new Exception("A nested executor leaked thread state onto the caller.");
                for (int frame = 0; frame < 200; frame++)
                {
                    var next = new int[19];
                    For(backend, 0, next.Length, index => next[index] = frame + index);
                    if (next.Where((value, index) => value != frame + index).Any()) throw new Exception("Kernel publication missed a wakeup.");
                }
            }
            finally { currentPool.SetValue(null, prior); }
        }));

        test("CPU executor: local scratch ownership, exceptions, cleanup and recovery", () => Enabled(() =>
        {
            using var backend = new CpuBackend(4);
            int initialized = 0, finalized = 0;
            var hits = new int[113];
            void Invoke(bool fail)
            {
                Func<Local> init = () => { Interlocked.Increment(ref initialized); return new Local(Environment.CurrentManagedThreadId); };
                Func<int, ParallelLoopState?, Local, Local> body = (index, _, local) =>
                {
                    if (local.Thread != Environment.CurrentManagedThreadId) throw new Exception("Local scratch was shared between workers.");
                    Interlocked.Increment(ref hits[index]);
                    if (fail && index == 17) throw new InvalidOperationException("Expected body failure");
                    return local;
                };
                Action<Local> finallyAction = local =>
                {
                    if (local.Thread != Environment.CurrentManagedThreadId) throw new Exception("Local scratch cleanup changed threads.");
                    Interlocked.Increment(ref finalized);
                };
                LocalFor.MakeGenericMethod(typeof(Local)).Invoke(null, [0, hits.Length, Options.GetValue(backend), init, body, finallyAction]);
            }
            bool rejected = false;
            try { Invoke(true); }
            catch (TargetInvocationException error) when (error.InnerException is AggregateException aggregate && aggregate.InnerExceptions.Any(item => item.Message == "Expected body failure")) { rejected = true; }
            if (!rejected || initialized != finalized) throw new Exception("An exceptional kernel skipped its completion barrier or scratch cleanup.");
            Array.Clear(hits); initialized = finalized = 0;
            Invoke(false);
            if (initialized != finalized || initialized is < 1 or > 4 || hits.Any(hit => hit != 1)) throw new Exception("The executor did not recover after an exception.");
            int singleThread = 0;
            using var single = new CpuBackend(1);
            For(single, 0, 29, _ =>
            {
                if (singleThread == 0) singleThread = Environment.CurrentManagedThreadId;
                else if (singleThread != Environment.CurrentManagedThreadId) throw new Exception("CpuThreads=1 started a worker.");
            });
            if (singleThread != Environment.CurrentManagedThreadId) throw new Exception("The serial executor did not use its caller.");
        }));

        test("CPU executor: concurrent dispose drains active work and unregisters workers", () => Enabled(() =>
        {
            var backend = new CpuBackend(3);
            using var started = new CountdownEvent(3);
            using var release = new ManualResetEventSlim();
            using var disposeStarted = new ManualResetEventSlim();
            using var disposeFinished = new ManualResetEventSlim();
            Exception? failure = null;
            var call = new Thread(() =>
            {
                try
                {
                    var first = new ConcurrentDictionary<int, byte>();
                    For(backend, 0, 80, _ =>
                    {
                        if (first.TryAdd(Environment.CurrentManagedThreadId, 0)) started.Signal();
                        if (!release.Wait(TimeSpan.FromSeconds(10))) throw new Exception("Timed out draining the active kernel.");
                    });
                }
                catch (Exception error) { failure = error; }
            });
            call.Start();
            if (!started.Wait(TimeSpan.FromSeconds(10))) { release.Set(); call.Join(); backend.Dispose(); throw new Exception("Workers did not start."); }
            var disposer = new Thread(() =>
            {
                disposeStarted.Set();
                try { backend.Dispose(); }
                catch (Exception error) { failure = error; }
                finally { disposeFinished.Set(); }
            });
            disposer.Start(); disposeStarted.Wait();
            bool endedEarly = disposeFinished.Wait(TimeSpan.FromMilliseconds(50));
            release.Set(); call.Join(); disposer.Join();
            backend.Dispose();
            if (endedEarly || failure is not null || !disposeFinished.IsSet) throw new Exception("Dispose released workers before the active kernel drained.", failure);
        }));
    }

    /// <summary>描述测试线程本地状态。</summary>
    /// <param name="Thread">持有本地工作区的测试线程标识。</param>
    private sealed record Local(int Thread);
    /// <summary>通过反射调用 CPU 调度器，在指定整数区间验证参与线程行为。</summary>
    /// <param name="backend">本次验证使用的计算后端。</param>
    /// <param name="from">迭代区间的起始索引，包含该索引。</param>
    /// <param name="to">迭代区间的结束索引，不包含该索引。</param>
    /// <param name="body">针对每个领取索引执行的测试回调。</param>
    private static void For(CpuBackend backend, int from, int to, Action<int> body)
        => ActionFor.Invoke(null, [from, to, Options.GetValue(backend), body]);

    /// <summary>暂时启用专项优化开关，执行验证后恢复原有环境设置。</summary>
    /// <param name="action">在临时测试设置或设备帧作用域内执行的操作。</param>
    private static void Enabled(Action action)
    {
        string? previous = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR");
        string? previousGrain = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR_GRAIN");
        Environment.SetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR", "1");
        Environment.SetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR_GRAIN", "1");
        try { action(); }
        finally
        {
            Environment.SetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR", previous);
            Environment.SetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR_GRAIN", previousGrain);
        }
    }
}
