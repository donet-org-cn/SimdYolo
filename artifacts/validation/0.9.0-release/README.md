# 0.9.0 发布、文档和性能更新

本目录记录0.9.0首次正式Release的补齐工作。推理/训练程序集保持此前验证的0.9.0冻结二进制，文档、图表和发布附件单独更新；未将文档修改解释为算法优化。

## 已发现并处理的问题

- 主分支原README已有0.9.0介绍，但首页主要速度表仍是0.7.0，两篇文章也链接到0.7.1下载。当前README、两篇图文、模型说明与下载入口已统一到0.9.0。
- GitHub只有v0.7.1 Release，正文中文字已被写成字面问号。原始响应备份在本地缓存；修复说明由UTF-8文件通过结构化JSON提交并回读对比，原有附件不变。
- 首次运行性能脚本时采用了默认随机输入，而0.8.1正式矩阵使用固定实拍图。随机输入中ORT CUDA的YOLO26分割TopK分数4项超出既定容差，完整失败记录保留；正式0.9.0矩阵改为与0.8.1相同的图片张量，单独完整重跑，未混合输入或放宽容差。

## 输入与失败诊断

| 场景 | 输入SHA256 | 记录 |
| --- | --- | --- |
| 首次随机输入 | `dd80e4472fba8829d1550659a8fc6eea6c817b5a85c80eac4b334fcf83166e27` | [诊断](synthetic-input-diagnostic.json)、[失败报告](synthetic-input-failed.json) |
| 固定实拍图 | `4312bccf98b9fcd92bbb73c7530f757d62bebb46da5a585bf1da1d642d8040af` | [两轮预检](image-preflight.json)、[正式矩阵](../../runtime-comparison-0.9.0.json) |

随机输入失败发生在ORT CUDA对照，已通过的62个窗口没有混入新的图片矩阵。TopK分数容差保持 `atol=2e-7, rtol=3e-4`，完整候选唯一配对仍核对所有框、类别和附加字段。失败报告中的638像素量级最大差来自尚未配对的行，不能解释成已配对坐标误差。

正式速度结论仅适用于本次固定输入、模型与机器。额外随机输入的超差说明不能从这组性能图推断所有输入下的等价性。文档ZIP另附精确的 `benchmark/input-640.f32`，便于按原SHA复现。

## 发布附件与图表

- `package-assets.json` 核验33个原模型和冻结程序集，保存模型ZIP、网页版ZIP、文档ZIP、两个NuGet包和校验和的摘要。
- 两个NuGet包保留先前训练、推理及独立消费测试时的字节；最新README、文章和性能图在当前源码与文档ZIP中。
- 当前图表由 `tools/render-release-performance.py` 从完整原始JSON生成，保存PNG、SVG、版本和源报告摘要；旧版图文ZIP与性能记录不覆写。
- `tools/publish-release.py` 使用本地UTF-8文件创建草稿、上传附件，最后发布；中文内容与GitHub回读逐字比较，同名不同内容附件拒绝覆盖。

[当前完整性能报告](../../runtime-comparison-0.9.0.md) · [训练与网页验证](../0.9.0/README.md) · [当前Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.9.0)
