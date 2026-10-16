namespace SimdYolo.Training;

/// <summary>完整轮次的验证指标与早停状态，指标均在实际保存模型重新加载后计算。</summary>
/// <param name="Epoch">完整轮次；零表示训练前基线。</param><param name="Precision">宏平均精确率。</param><param name="Recall">宏平均召回率。</param>
/// <param name="Map50">IoU0.50处的宏平均精度。</param><param name="Map50To95">IoU0.50到0.95的平均精度。</param><param name="Improved">是否更新最佳模型。</param>
/// <param name="BestEpoch">当前最佳模型轮次。</param><param name="StaleEpochs">连续未改善轮数。</param><param name="StoppedEarly">是否正常早停。</param>
public sealed record EpochValidation(int Epoch, double Precision, double Recall, double Map50, double Map50To95, bool Improved, int BestEpoch, int StaleEpochs, bool StoppedEarly);
