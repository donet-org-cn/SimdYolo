namespace SimdYolo;

internal sealed partial class PtModel
{
    /// <summary>标记本次执行只计算检测头之前的骨干和特征金字塔。</summary>
    private bool extractingFeatures;
    /// <summary>返回原始未融合检测头，供独立训练程序集复制可训练参数。</summary>
    /// <value>保留原始 BN 模块和参数的符号对象。</value>
    internal PyObject DetectionHead => layers[^1];
    /// <summary>运行冻结骨干并返回三个尺度的检测特征，不执行检测头。</summary>
    /// <param name="input">RGB NCHW 输入。</param>
    /// <returns>仍遵守当前 CPU 池或设备帧所有权的特征，调用方须在帧结束前物化。</returns>
    /// <exception cref="NotSupportedException">模型不是三尺度标准 Detect 头。</exception>
    internal Tensor[] ExtractDetectionFeatures(Tensor input)
    {
        TrainingHeadSchema.Read(DetectionHead);
        extractingFeatures = true;
        try { return Run(input); }
        finally { extractingFeatures = false; }
    }
}
