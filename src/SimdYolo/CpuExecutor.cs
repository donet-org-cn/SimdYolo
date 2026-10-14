using System.Collections.Concurrent;

namespace SimdYolo;

// 每个后端独占自己的工作线程；推理调用线程也计入参与数。
// 线程预算由后端控制，不依赖进程级 ThreadPool 的配置。
/// <summary>由 CPU 后端拥有的专用执行器；调用线程也参与计算，工作线程总数不超过配置预算。</summary>
internal sealed class CpuExecutor : IDisposable
{
    /// <summary>当前线程正在参与的执行器；用于禁止嵌套线程池扩张和内核内释放。</summary>
    [ThreadStatic] internal static CpuExecutor? Participant;
    /// <summary>执行器共用的并行选项。</summary>
    /// <value>执行器共用的并行选项。</value>
    internal ParallelOptions Options { get; }
    /// <summary>串行化作业发布与释放的后端互斥锁。</summary>
    private readonly object gate = new();
    /// <summary>包含调用线程在内的最大参与线程数。</summary>
    private readonly int degree;
    /// <summary>显式配置的每次任务领取量；零表示按工作量自动计算。</summary>
    private readonly int configuredGrain;
    /// <summary>工作线程休眠前的短时自旋次数上限。</summary>
    private readonly int spinAttempts;
    /// <summary>按需创建的后台工作线程集合。</summary>
    private Worker[]? workers;
    /// <summary>当前已发布且尚未撤销的作业。</summary>
    private Job? published;
    /// <summary>执行器是否已经释放。</summary>
    private bool disposed;
    /// <summary>通过易失读写通知所有工作线程停止的标记。</summary>
    private int stopping;
    /// <summary>每次作业发布递增的代际编号。</summary>
    private long epoch;

    /// <summary>从并行配置创建执行器，读取并限制任务粒度与短时自旋次数。</summary>
    /// <param name="options">当前后端使用的并行配置。</param>
    internal CpuExecutor(ParallelOptions options)
    {
        Options = options;
        degree = options.MaxDegreeOfParallelism > 0 ? options.MaxDegreeOfParallelism : Environment.ProcessorCount;
        int.TryParse(Environment.GetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR_GRAIN"), out configuredGrain);
        configuredGrain = Math.Clamp(configuredGrain, 0, 1024);
        spinAttempts = int.TryParse(Environment.GetEnvironmentVariable("SIMDYOLO_CPU_EXECUTOR_SPINS"), out int attempts)
            ? Math.Clamp(attempts, 0, 4096) : 64;
    }

    /// <summary>在半开索引区间内执行回调，并等待所有参与线程完成；泛型版本每个线程独享局部状态。</summary>
    /// <param name="from">循环起点，包含此索引。</param>
    /// <param name="to">循环终点，不包含此索引。</param>
    /// <param name="body">对当前索引执行的回调，泛型版本返回更新后的线程局部状态。</param>
    /// <exception cref="AggregateException">参与线程、循环回调或局部状态清理出现一个或多个错误。</exception>
    /// <exception cref="ObjectDisposedException">执行器已经释放，不能再提交作业。</exception>
    /// <exception cref="OperationCanceledException">当前并行选项或作业的取消令牌已请求取消。</exception>
    internal void For(int from, int to, Action<int> body)
        => Run(from, to, (participants, grain, token) => new ActionJob(from, to, participants, grain, token, body));

    /// <summary>在半开索引区间内执行回调，并等待所有参与线程完成；泛型版本每个线程独享局部状态。</summary>
    /// <typeparam name="TLocal">每个参与线程独享的局部状态类型。</typeparam>
    /// <param name="from">循环起点，包含此索引。</param>
    /// <param name="to">循环终点，不包含此索引。</param>
    /// <param name="init">为参与线程创建局部状态的回调。</param>
    /// <param name="body">对当前索引执行的回调，泛型版本返回更新后的线程局部状态。</param>
    /// <param name="finallyAction">线程结束时释放其局部状态的回调。</param>
    /// <exception cref="AggregateException">参与线程、循环回调或局部状态清理出现一个或多个错误。</exception>
    /// <exception cref="ObjectDisposedException">执行器已经释放，不能再提交作业。</exception>
    /// <exception cref="OperationCanceledException">当前并行选项或作业的取消令牌已请求取消。</exception>
    internal void For<TLocal>(int from, int to, Func<TLocal> init,
                             Func<int, ParallelLoopState?, TLocal, TLocal> body, Action<TLocal> finallyAction)
        => Run(from, to, (participants, grain, token) => new LocalJob<TLocal>(from, to, participants, grain, token, init, body, finallyAction));

    /// <summary>在后端互斥锁内发布作业，由调用线程参与计算，等待完整完成屏障后传播错误。</summary>
    /// <param name="from">循环起点，包含此索引。</param>
    /// <param name="to">循环终点，不包含此索引。</param>
    /// <param name="create">按参与数、任务粒度和取消令牌构造作业的工厂。</param>
    /// <exception cref="AggregateException">参与线程、循环回调或局部状态清理出现一个或多个错误。</exception>
    /// <exception cref="ObjectDisposedException">执行器已经释放，不能再提交作业。</exception>
    /// <exception cref="OperationCanceledException">当前并行选项或作业的取消令牌已请求取消。</exception>
    private void Run(int from, int to, Func<int, int, CancellationToken, Job> create)
    {
        // 同一后端的作业提交串行化；Dispose 也使用这个互斥锁。
        // 已发布内核运行期间，释放过程不能先销毁休眠工作线程。
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Options.CancellationToken.ThrowIfCancellationRequested();
            long count = (long)to - from;
            if (count <= 0) return;
            int participants = (int)Math.Min(degree, count);
            int grain = configuredGrain > 0 ? configuredGrain : (int)Math.Max(1, Math.Min(8, count / (participants * 16L)));
            using var job = create(participants, grain, Options.CancellationToken);
            if (participants > 1)
            {
                EnsureWorkers();
                job.Epoch = ++epoch;
                // 作业发布与休眠检查构成一次存储/读取握手。
                // 发布方和休眠方都必须有完整屏障；仅 release/acquire 仍可能让双方同时读到旧值。
                // 否则唯一一次唤醒信号可能丢失。
                Interlocked.Exchange(ref published, job);
                for (int i = 0; i < participants - 1; i++) workers![i].WakeIfParked();
            }
            var previous = Participant;
            Participant = this;
            try
            {
                job.Participate();
                job.Wait();
            }
            finally
            {
                Participant = previous;
                Volatile.Write(ref published, null);
            }
            job.ThrowErrors();
        }
    }

    /// <summary>按需创建并启动线程预算内的后台工作线程。</summary>
    private void EnsureWorkers()
    {
        if (workers is not null) return;
        workers = new Worker[degree - 1];
        for (int i = 0; i < workers.Length; i++) workers[i] = new Worker(this, i);
        for (int i = 0; i < workers.Length; i++) workers[i].Thread.Start();
    }

    /// <summary>等待当前作业结束，唤醒并加入所有工作线程，再释放唤醒事件；重复释放无副作用。</summary>
    /// <exception cref="InvalidOperationException">试图在本后端自己的内核回调中释放它，会导致线程自等待。</exception>
    public void Dispose()
    {
        if (ReferenceEquals(Participant, this)) throw new InvalidOperationException("A CPU backend cannot be disposed from its own kernel callback.");
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            Volatile.Write(ref stopping, 1);
            if (workers is null) return;
            foreach (var worker in workers) worker.Wake.Set();
            foreach (var worker in workers) worker.Thread.Join();
            foreach (var worker in workers) worker.Wake.Dispose();
            workers = null;
        }
    }

    /// <summary>执行器拥有的后台工作线程，以完整内存屏障协调作业发布和休眠唤醒。</summary>
    private sealed class Worker
    {
        /// <summary>拥有该后台工作线程的执行器。</summary>
        private readonly CpuExecutor owner;
        /// <summary>该工作线程在执行器中的零基索引。</summary>
        private readonly int index;
        /// <summary>用于执行 CPU 内核的后台线程。</summary>
        /// <value>用于执行 CPU 内核的后台线程。</value>
        internal Thread Thread { get; }
        /// <summary>唤醒已休眠工作线程的自动复位事件。</summary>
        /// <value>唤醒已休眠工作线程的自动复位事件。</value>
        internal AutoResetEvent Wake { get; } = new(false);
        /// <summary>以完整屏障发布的工作线程休眠状态。</summary>
        private int parked;

        /// <summary>创建尚未启动的后台线程，并记录其作业参与索引。</summary>
        /// <param name="owner">拥有该工作线程的 CPU 执行器。</param>
        /// <param name="index">工作线程、任务或输出元素的索引。</param>
        internal Worker(CpuExecutor owner, int index)
        {
            this.owner = owner; this.index = index;
            Thread = new Thread(Loop) { IsBackground = true, Name = $"SimdYolo CPU {index + 1}" };
        }

        /// <summary>仅在工作线程已声明休眠时设置唤醒事件。</summary>
        internal void WakeIfParked() { if (Volatile.Read(ref parked) != 0) Wake.Set(); }

        /// <summary>读取已发布作业，仅在代际改变且该线程属于参与集合时执行一次。</summary>
        /// <param name="lastEpoch">该线程最后检查过的作业代际，读取新作业时更新。</param>
        /// <returns>当前线程实际参与了一个新作业时为 true；否则为 false。</returns>
        private bool TryJob(ref long lastEpoch)
        {
            var job = Volatile.Read(ref owner.published);
            if (job is null || job.Epoch == lastEpoch) return false;
            lastEpoch = job.Epoch;
            if (index >= job.Participants - 1) return false;
            var previous = Participant;
            Participant = owner;
            // 张量池与设备作用域仅属于推理调用线程，不能继承到工作线程。
            CpuTensorPool.Current = null;
            DeviceScope.Current = null;
            try { job.Participate(); }
            finally { Participant = previous; CpuTensorPool.Current = null; DeviceScope.Current = null; }
            return true;
        }

        /// <summary>短暂自旋等待相邻内核，然后以发布/休眠双向完整屏障握手进入事件等待。</summary>
        private void Loop()
        {
            long lastEpoch = 0;
            while (Volatile.Read(ref owner.stopping) == 0)
            {
                if (TryJob(ref lastEpoch)) continue;
                // 相邻内核之间只进行短暂的计算型自旋。
                // 空闲检测器的工作线程很快进入休眠，停止消耗 CPU。
                bool found = false;
                for (int attempt = 0; attempt < owner.spinAttempts; attempt++)
                {
                    if (Volatile.Read(ref owner.stopping) != 0) return;
                    Thread.SpinWait(32);
                    if (TryJob(ref lastEpoch)) { found = true; break; }
                }
                if (found) continue;
                Interlocked.Exchange(ref parked, 1);
                // 作业发布可能与工作线程休眠并发发生。
                // 声明休眠后再次检查作业，完整屏障与发布方的 Exchange 配对。
                // 线程要么观察到新作业，要么消费发布方留下的唤醒信号。
                if (Volatile.Read(ref owner.stopping) != 0) { Volatile.Write(ref parked, 0); return; }
                if (!TryJob(ref lastEpoch)) Wake.WaitOne();
                Volatile.Write(ref parked, 0);
            }
        }
    }

    /// <summary>表示一次分块作业，协调任务领取、错误汇总和最后参与者的完成发布。</summary>
    /// <param name="from">循环起点，包含此索引。</param>
    /// <param name="to">循环终点，不包含此索引。</param>
    /// <param name="participants">包含调用线程在内的参与线程数。</param>
    /// <param name="grain">每次原子领取的索引数量。</param>
    /// <param name="cancellation">当前作业的取消令牌。</param>
    private abstract class Job(int from, int to, int participants, int grain, CancellationToken cancellation) : IDisposable
    {
        /// <summary>当前作业的发布代际，用于防止同一工作线程重复参与。</summary>
        internal long Epoch;
        /// <summary>本次作业的参与线程数量。</summary>
        /// <value>本次作业的参与线程数量。</value>
        internal int Participants { get; } = participants;
        /// <summary>下一个可领取任务块的起点，由原子加法推进。</summary>
        private long next = from;
        /// <summary>尚未结束的参与线程数量。</summary>
        private int remaining = participants;
        /// <summary>最后参与者完成事件 Set 后发布的标记，保证事件释放时序。</summary>
        private int completionPublished;
        /// <summary>已有线程失败的标记；阻止继续领取任务。</summary>
        private int failed;
        /// <summary>所有参与线程结束时设置的完成事件。</summary>
        private readonly ManualResetEventSlim completed = new(false, 0);
        /// <summary>参与线程执行异常的并发汇总队列。</summary>
        private readonly ConcurrentQueue<Exception> errors = new();

        /// <summary>以原子加法领取下一个不重叠任务块；失败、取消或区间耗尽时停止领取。</summary>
        /// <param name="start">返回本线程领取区间的起点，包含此索引。</param>
        /// <param name="end">返回本线程领取区间的终点，不包含此索引。</param>
        /// <returns>领取到尚未处理的任务块时为 true；失败或任务耗尽时为 false。</returns>
        /// <exception cref="OperationCanceledException">当前并行选项或作业的取消令牌已请求取消。</exception>
        protected bool Take(out int start, out int end)
        {
            start = end = 0;
            if (Volatile.Read(ref failed) != 0) return false;
            cancellation.ThrowIfCancellationRequested();
            long offset = Interlocked.Add(ref next, grain) - grain;
            if (offset >= to) return false;
            start = (int)offset; end = (int)Math.Min(to, offset + grain);
            return true;
        }
        /// <summary>由具体作业实现领取并处理其拥有的索引块。</summary>
        protected abstract void Execute();
        /// <summary>保存线程执行异常并发布失败标记，使其他线程停止领取新任务。</summary>
        /// <param name="error">需要传播给调用方的执行错误。</param>
        protected void Report(Exception error) { errors.Enqueue(error); Volatile.Write(ref failed, 1); }
        /// <summary>执行一个参与线程的工作，捕获错误，并由最后参与者发布完成事件与完成标记。</summary>
        internal void Participate()
        {
            try { Execute(); }
            catch (Exception error) { Report(error); }
            finally
            {
                if (Interlocked.Decrement(ref remaining) == 0)
                {
                    completed.Set();
                    // 最后工作线程的 Set 可能仍在执行。
                    // 调用方必须等完成标记发布后才能释放完成事件。
                    Volatile.Write(ref completionPublished, 1);
                }
            }
        }
        /// <summary>等待所有参与线程完成且最后一次事件 Set 已返回，防止事件被过早释放。</summary>
        internal void Wait()
        {
            for (int attempt = 0; attempt < 128; attempt++)
            {
                if (Volatile.Read(ref completionPublished) != 0) return;
                Thread.SpinWait(32);
            }
            completed.Wait();
            while (Volatile.Read(ref completionPublished) == 0) Thread.SpinWait(1);
        }
        /// <summary>汇总参与线程的异常；全部为同一令牌取消时传播取消，否则抛出聚合异常。</summary>
        /// <exception cref="AggregateException">参与线程、循环回调或局部状态清理出现一个或多个错误。</exception>
        /// <exception cref="OperationCanceledException">当前并行选项或作业的取消令牌已请求取消。</exception>
        internal void ThrowErrors()
        {
            if (errors.IsEmpty) return;
            var failures = errors.ToArray();
            if (cancellation.IsCancellationRequested && failures.All(error => error is OperationCanceledException canceled && canceled.CancellationToken == cancellation))
                throw new OperationCanceledException(cancellation);
            throw new AggregateException(failures);
        }
        /// <summary>在调用方确认全部参与线程完成后，释放该作业的完成事件。</summary>
        public void Dispose() => completed.Dispose();
    }

    /// <summary>表示逐索引回调的分块作业。</summary>
    /// <param name="from">循环起点，包含此索引。</param>
    /// <param name="to">循环终点，不包含此索引。</param>
    /// <param name="participants">包含调用线程在内的参与线程数。</param>
    /// <param name="grain">每次原子领取的索引数量。</param>
    /// <param name="token">当前作业的取消令牌。</param>
    /// <param name="body">对当前索引执行的回调，泛型版本返回更新后的线程局部状态。</param>
    private sealed class ActionJob(int from, int to, int participants, int grain, CancellationToken token, Action<int> body)
        : Job(from, to, participants, grain, token)
    {
        /// <summary>由具体作业实现领取并处理其拥有的索引块。</summary>
        protected override void Execute()
        {
            while (Take(out int start, out int end)) for (int index = start; index < end; index++) body(index);
        }
    }

    /// <summary>表示每个参与线程拥有初始化状态和最终清理回调的分块作业。</summary>
    /// <typeparam name="TLocal">每个参与线程独享的局部状态类型。</typeparam>
    /// <param name="from">循环起点，包含此索引。</param>
    /// <param name="to">循环终点，不包含此索引。</param>
    /// <param name="participants">包含调用线程在内的参与线程数。</param>
    /// <param name="grain">每次原子领取的索引数量。</param>
    /// <param name="token">当前作业的取消令牌。</param>
    /// <param name="init">为参与线程创建局部状态的回调。</param>
    /// <param name="body">对当前索引执行的回调，泛型版本返回更新后的线程局部状态。</param>
    /// <param name="finallyAction">线程结束时释放其局部状态的回调。</param>
    private sealed class LocalJob<TLocal>(int from, int to, int participants, int grain, CancellationToken token,
                                        Func<TLocal> init, Func<int, ParallelLoopState?, TLocal, TLocal> body, Action<TLocal> finallyAction)
        : Job(from, to, participants, grain, token)
    {
        /// <summary>由具体作业实现领取并处理其拥有的索引块。</summary>
        protected override void Execute()
        {
            if (!Take(out int start, out int end)) return;
            TLocal local = init();
            try
            {
                do { for (int index = start; index < end; index++) local = body(index, null, local); }
                while (Take(out start, out end));
            }
            catch (Exception error) { Report(error); }
            finally
            {
                try { finallyAction(local); }
                catch (Exception error) { Report(error); }
            }
        }
    }
}
