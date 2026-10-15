using SimdYolo;

namespace SimdYolo.Training;

/// <summary>配置 YOLOv8/v9/v10/YOLO11/YOLO26 检测头微调；骨干和特征金字塔保持冻结。</summary>
public sealed record TrainingOptions
{
    /// <summary>输入图像的正方形边长，必须为 32 的倍数。</summary><value>默认 128。</value>
    public int ImageSize { get; init; } = 128;
    /// <summary>本次需要完成的总轮数；恢复时不能小于已有轮数。</summary><value>默认 10。</value>
    public int Epochs { get; init; } = 10;
    /// <summary>梯度累加的图像数量；每图 BatchNorm 独立统计，不冒充联合批次 BatchNorm。</summary><value>默认 1。</value>
    public int BatchSize { get; init; } = 1;
    /// <summary>SGD 学习率。</summary><value>默认 0.01。</value>
    public float LearningRate { get; init; } = .01f;
    /// <summary>SGD 动量系数，范围为 [0,1)。</summary><value>默认 0.9。</value>
    public float Momentum { get; init; } = .9f;
    /// <summary>仅卷积权重使用的 L2 衰减。</summary><value>默认 0.0005。</value>
    public float WeightDecay { get; init; } = .0005f;
    /// <summary>初始化和按轮洗牌的确定性随机种子。</summary><value>默认 26。</value>
    public int Seed { get; init; } = 26;
    /// <summary>选择 CPU 或 Vulkan；Vulkan 仅加速前向和反向 GEMM，其余训练运算在 CPU。</summary><value>默认 CPU。</value>
    public ComputeDevice Device { get; init; } = ComputeDevice.Cpu;
    /// <summary>CPU SIMD 后端的线程上限。</summary><value>默认 0，由后端决定。</value>
    public int CpuThreads { get; init; }
    /// <summary>保存每张图像的冻结特征，以加速小型微调数据集。</summary><value>默认 true；数据较大时关闭。</value>
    public bool CacheFeatures { get; init; } = true;

    /// <summary>拒绝无效训练配置和当前尚未实现的训练设备。</summary>
    /// <exception cref="ArgumentException">尺寸、轮数、梯度累加、学习率或线程配置不合法。</exception>
    /// <exception cref="NotSupportedException">请求 Metal 或 Auto 训练。</exception>
    internal void Validate()
    {
        if (ImageSize is < 64 or > 1280 || ImageSize % 32 != 0 || Epochs is < 1 or > 10000 || BatchSize is < 1 or > 128 || CpuThreads < 0 ||
            !float.IsFinite(LearningRate) || LearningRate <= 0 || !float.IsFinite(Momentum) || Momentum is < 0 or >= 1 || !float.IsFinite(WeightDecay) || WeightDecay < 0)
            throw new ArgumentException("训练尺寸、轮数、梯度累加、SGD 参数或线程数不合法。");
        if (Device is not (ComputeDevice.Cpu or ComputeDevice.Vulkan)) throw new NotSupportedException("首期训练仅支持 CPU 或 Vulkan 矩阵运算。");
    }
}

/// <summary>描述已映射到训练图像像素坐标的目标框。</summary>
/// <param name="ClassId">从零开始的类别编号。</param><param name="X1">左边界。</param><param name="Y1">上边界。</param><param name="X2">右边界。</param><param name="Y2">下边界。</param>
public readonly record struct TrainingBox(int ClassId, float X1, float Y1, float X2, float Y2);

/// <summary>保存一张 RGB24 图像及其归一化 YOLO 标注。</summary>
/// <param name="Name">样本名称。</param><param name="Pixels">RGB24 字节，不包含行填充。</param><param name="Width">图像宽度。</param><param name="Height">图像高度。</param><param name="Labels">每行 class cx cy w h 的归一化标注。</param>
public sealed record TrainingImage(string Name, byte[] Pixels, int Width, int Height, string Labels);

/// <summary>记录一次真实优化步骤的损失与累计耗时。</summary>
/// <param name="Step">已完成的优化步数。</param><param name="Epoch">当前轮次，从一开始。</param><param name="TotalLoss">三个加权损失之和。</param><param name="BoxLoss">加权 CIoU 损失。</param><param name="ClassLoss">加权 BCE 损失。</param><param name="DflLoss">加权 DFL；YOLO26 为归一化距离 L1。</param><param name="ElapsedMs">本次运行累计毫秒数。</param>
public sealed record TrainingProgress(int Step, int Epoch, float TotalLoss, float BoxLoss, float ClassLoss, float DflLoss, double ElapsedMs);
