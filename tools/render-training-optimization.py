"""使用完整测量、绑定相同DLL的原始结果生成对照，保留未通过的比较项。"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'artifacts/validation/0.10.1'
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
summary=read(DIRECTORY/'validation-summary.json')
performance=read(DIRECTORY/'performance/training-runtime-comparison.json')
old=read(ROOT/'artifacts/validation/0.10.0/performance/training-runtime-comparison.json')
comparison=read(DIRECTORY/'outcomes/comparison.json')
assert summary['status']==performance['status']=='passed' and comparison['status']=='completed'
assert len(performance['runs'])==40 and len(comparison['results'])==10
models=performance['models']
outcomes={(r['model'],r['device']):read(DIRECTORY/'outcomes'/r['report']) for r in comparison['results']}
for result in outcomes.values():
    assert result['bothSavedModelsReloaded'] and result['status']=='completed'
    for name in ['coreAssemblySha256','trainingAssemblySha256']: assert result[name]==summary[name]
def median(model,engine,raw=performance): return statistics.median(r['medianMs'] for r in raw['runs'] if Path(r['model']).stem==model and r['engineKey']==engine)

sys.path.insert(0,str(ROOT/'.cache/reference'))
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/article-mpl-config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'svg.fonttype':'path'})
def save(fig,name):
    fig.savefig(ROOT/f'artifacts/{name}.png',dpi=155)
    path=ROOT/f'artifacts/{name}.svg';fig.savefig(path)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
for group,original,ours in [('cpu','torch-cpu','net-cpu'),('gpu','torch-cuda','net-vulkan')]:
    fig,ax=plt.subplots(figsize=(12,6.6));y=np.arange(5);height=.24
    for offset,values,label,color in [(-1,[median(m,ours,old) for m in models],'我们 0.10.0（历史）','#b9c2cf'),(0,[median(m,original) for m in models],'原版 PyTorch（本轮）','#e59945'),(1,[median(m,ours) for m in models],'我们 0.10.1（本轮）','#3478bd')]:
        data=old if offset==-1 else performance;engine=original if offset==0 else ours
        lower=[v-min(r['medianMs'] for r in data['runs'] if Path(r['model']).stem==m and r['engineKey']==engine) for m,v in zip(models,values)]
        upper=[max(r['medianMs'] for r in data['runs'] if Path(r['model']).stem==m and r['engineKey']==engine)-v for m,v in zip(models,values)]
        bars=ax.barh(y+offset*height,values,height,label=label,color=color,xerr=[lower,upper],capsize=3,error_kw={'elinewidth':1})
        for bar,value,error in zip(bars,values,upper): ax.text(value+error+max(values)*.015,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontsize=10)
    maximum=max(r['medianMs'] for data in [performance,old] for r in data['runs'] if r['engineKey'] in [original,ours])
    ax.set_yticks(y,models);ax.invert_yaxis();ax.set_xlim(0,maximum*1.20)
    ax.set_xlabel('每次检测头训练更新耗时（ms，越低越快）');ax.set_title(f'0.10.1 训练优化：{group.upper()} 同机对照',loc='left',fontsize=18,pad=18)
    ax.legend(frameon=False,loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.text(.12,.04,'128×128 / 2 类 / FP32 / 4线程 / 相同冻结特征 / 每配置两窗、每窗512次更新\n计入头前向、损失、反向及SGD；不含骨干、预处理和保存。细线为两窗口范围，非置信区间。',fontsize=10,color='#596574')
    fig.subplots_adjust(left=.12,right=.96,top=.87,bottom=.20);save(fig,f'training-performance-0.10.1-{group}')

fig,axes=plt.subplots(1,2,figsize=(14,6.5));x=np.arange(5)
for ax,metric,title in zip(axes,['map50','map50_95'],['mAP50','mAP50–95']):
    for offset,device,engine,label,color in [(-1.5,'cpu','saved_torch_after','PyTorch CPU','#e59a45'),(-.5,'cpu','saved_net_after','我们 CPU','#3478bd'),(.5,'vulkan','saved_torch_after','PyTorch CUDA','#ddbc8c'),(1.5,'vulkan','saved_net_after','我们 Vulkan','#83b4de')]:
        values=[outcomes[m,device]['metrics'][engine][metric]*100 for m in models]
        ax.bar(x+offset*.18,values,.18,label=label,color=color)
    ax.set_xticks(x,models,rotation=15);ax.set_ylabel(title+'（%，越高越好）');ax.set_ylim(0,70);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.set_title(title)
axes[0].legend(frameon=False,ncol=2,loc='upper right',fontsize=9)
fig.suptitle('两边实际保存的模型重新加载：同一32张微调留出图片',x=.07,ha='left',fontsize=18)
fig.text(.07,.03,'COCO128明确拆成96张微调/32张验证，80类、256×256、2轮、每模型192次更新。\n预训练权重见过COCO train2017；这是实现一致性诊断，不能作为未见数据泛化结论。',fontsize=10,color='#596574')
fig.tight_layout(rect=[0,.12,1,.92]);save(fig,'training-results-0.10.1')

fig,axes=plt.subplots(2,3,figsize=(15,8.5));axes=axes.ravel()
for ax,m in zip(axes,models):
    for device,engine,label,color,style in [('cpu','torch','PyTorch CPU','#e59945','-'),('cpu','net','我们 CPU','#3478bd','--'),('vulkan','torch','PyTorch CUDA','#b57b99','-'),('vulkan','net','我们 Vulkan','#388f82',':')]:
        history=outcomes[m,device]['history'][engine]
        ax.plot([p['step'] for p in history],[p['totalLoss'] for p in history],style,color=color,label=label,linewidth=1.1,alpha=.9)
    ax.set_title(m);ax.set_xlabel('实际更新次数');ax.set_ylabel('训练目标');ax.grid(alpha=.15)
axes[-1].axis('off');axes[-1].legend(*axes[0].get_legend_handles_labels(),loc='upper left',frameon=False)
axes[-1].text(.03,.48,'同权重、同顺序、同冻结特征\n192次连续SGD更新\n\n离散TAL匹配与FP32舍入可使\n多步轨迹分叉；逐步误差保留\n于原始JSON，不声称逐位一致。',va='top',fontsize=11,linespacing=1.6)
fig.suptitle('真实图片微调：训练曲线对照',x=.06,ha='left',fontsize=18);fig.tight_layout(rect=[0,0,1,.94]);save(fig,'training-loss-parity-0.10.1')

lines=['# 0.10.1 训练优化与训练后模型对比','',
       '本轮继续保持纯 C#/.NET、FP32 和原有单步误差容差。优化 SIMD 激活/BN/SGD 检查、1×1 零复制、连续行 im2col/col2im、直接转置点积 dW、有界临时矩阵复用，以及 Vulkan 小矩阵的 CPU SIMD 分派。骨干/FPN 冻结，仅微调五代检测头。','',
       '## 速度和排名','',
       'i7-8700K / GTX 1050 Ti；128×128、2类、FP32、4线程、相同初始状态与冻结特征。每配置两个独立串行窗口，每窗预热10步后512次更新。原版为官方 Ultralytics 8.4.0 检测头/损失与 PyTorch SGD，并非带 DataLoader/增强的全参数 train 管线。','',
       '每步包含头部前向、损失、反向、SGD与清零；Vulkan包含上传/回读，CUDA每步同步。排除冻结骨干、预处理、模型加载和保存。0.10.0列来自历史同机数据；PyTorch和0.10.1在本轮重测。①最快，按未舍入值比较；两窗口不是统计显著性检验。','',
       '| 模型 | 原版 CPU ms | 我们 0.10.0 CPU（历史） | 我们 0.10.1 CPU ms | 较原版倍速 | CPU 第一 |','| --- | ---: | ---: | ---: | ---: | --- |']
for m in models:
    ref,net=median(m,'torch-cpu'),median(m,'net-cpu')
    lines.append(f"| {m} | {ref:.2f} {'①' if ref<net else '②'} | {median(m,'net-cpu',old):.2f} | {net:.2f} {'①' if net<ref else '②'} | {ref/net:.2f}× | {'我们' if net<ref else 'PyTorch'} |")
lines+=['','| 模型 | 原版 CUDA ms | 我们 0.10.0 Vulkan（历史） | 我们 0.10.1 Vulkan ms | 较原版倍速 | GPU 第一 |','| --- | ---: | ---: | ---: | ---: | --- |']
for m in models:
    ref,net=median(m,'torch-cuda'),median(m,'net-vulkan')
    lines.append(f"| {m} | {ref:.2f} {'①' if ref<net else '②'} | {median(m,'net-vulkan',old):.2f} | {net:.2f} {'①' if net<ref else '②'} | {ref/net:.2f}× | {'我们' if net<ref else 'PyTorch'} |")
lines+=['','![CPU训练速度](training-performance-0.10.1-cpu.png)','','![GPU训练速度](training-performance-0.10.1-gpu.png)','',
        'Vulkan 为混合训练：矩阵乘加项数不超过4,000,000的 GEMM在CPU SIMD计算，大矩阵在GPU；深度/分组卷积、BN、激活、损失和SGD仍在CPU。该阈值基于本机提交与回读开销，不代表其他GPU的最优值。没有把ORT推理耗时当作训练基线。','',
        '## 两边训练后，保存的模型效果是否接近','',
        '使用官方[COCO128](https://docs.ultralytics.com/datasets/detect/coco128/)的128张真实图片，固定种子2611明确拆成**96张微调、32张微调留出验证**，没有沿用原数据默认train=val配置。80类、256×256、2轮、每模型192次更新，SGD lr=0.0001、momentum=0.9、weight_decay=0.0005；不使用增强、EMA或MuSGD。两边共用逐字节相同的冻结特征、类别、初始权重、每轮顺序及YOLO26调度。','',
        'CPU对比PyTorch CPU训练；Vulkan对比PyTorch CUDA训练。**实际重新打开PyTorch `.pt`与C# `.sytrain`，运行完整冻结骨干和更新后的检测头**，使用同一C# letterbox输入、官方NMS/匹配/AP统计；阈值0.001、NMS IoU0.7、max_det300，IoU0.50:0.95。端到端头按官方TopK不执行普通NMS。','',
        '| 模型 | 训练设备 | 原版保存模型 mAP50 % | 我们保存模型 mAP50 % | 原版 mAP50–95 % | 我们 mAP50–95 % | mAP50–95 差（百分点） |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for m in models:
    for device in ['cpu','vulkan']:
        result=outcomes[m,device];a=result['metrics']['saved_torch_after'];b=result['metrics']['saved_net_after']
        lines.append(f"| {m} | {'CPU' if device=='cpu' else 'CUDA / Vulkan'} | {a['map50']*100:.4f} | {b['map50']*100:.4f} | {a['map50_95']*100:.4f} | {b['map50_95']*100:.4f} | {abs(a['map50_95']-b['map50_95'])*100:.6f} |")
lines+=['','![保存模型真实图片验证](training-results-0.10.1.png)','',
        '| 模型 | 原始权重 mAP50–95 %（共同冻结特征） | PyTorch CPU 微调后 % | 我们 CPU 微调后 % |','| --- | ---: | ---: | ---: |']
for m in models:
    r=outcomes[m,'cpu']['metrics'];lines.append(f"| {m} | {r['torch_before']['map50_95']*100:.4f} | {r['saved_torch_after']['map50_95']*100:.4f} | {r['saved_net_after']['map50_95']*100:.4f} |")
lines+=['','本次短程微调的验证精度有下降，两边趋势相同。该实验检查优化后训练行为与原版是否接近，**不证明训练配方有效或微调提高精度**。COCO128来自COCO train2017，预训练权重已见过该数据；32张只对本次微调留出，不能称为独立于预训练的泛化测试。尚未进行完整COCO val2017或用户业务集收敛评测。','',
        '## 多步参数与输出对照','',
        '| 模型 | 设备 | 192步后参数/BN最大绝对差 | 192步后动量最大绝对差 | 全程训练目标最大绝对差 | 直接参数门槛 |','| --- | --- | ---: | ---: | ---: | --- |']
for m in models:
    for device in ['cpu','vulkan']:
        r=outcomes[m,device];p=r['snapshots'][-1]['parameters']
        lines.append(f"| {m} | {device} | {max(c['maxAbs'] for c in p):.6g} | {max(c['momentum']['maxAbs'] for c in p if 'momentum' in c):.6g} | {r['lossError']['maxAbs']:.6g} | {'通过' if r['directParameterComparisonPassed'] else '**未通过**'} |")
lines+=['',f"**直接参数比较 {summary['directParameterComparisonsPassed']}/10 通过，保存模型指标 {summary['metricComparisonsPassed']}/10 达到门槛，固定PyTorch CPU参考的参数控制 {summary['cpuParameterControlsPassed']}/10 通过。未通过项保留false，没有扩大容差。**",'',
        '| 模型 | 原版自身CPU/CUDA参数/BN最大差 | 我们Vulkan/原版CPU最大差 | 我们Vulkan/原版CUDA最大差 |','| --- | ---: | ---: | ---: |']
for m in models:
    r=outcomes[m,'vulkan'];baseline=r['pytorchCpuVsCuda'][-1]['parameters'];control=r['cpuParameterControl'][-1]['parameters'];direct=r['snapshots'][-1]['parameters']
    lines.append(f"| {m} | {max(c['maxAbs'] for c in baseline):.6g} | {max(c['maxAbs'] for c in control):.6g} | {max(c['maxAbs'] for c in direct):.6g} |")
lines+=['','![逐步训练曲线](training-loss-parity-0.10.1.png)','',
        '在第1、16、96、192步比较全部可训练参数和BN状态，并记录动量、每图原始输出误差与192步损失轨迹。多步参数门槛预先固定为0.002+0.01×|参考值|，保存模型的P/R/mAP绝对差门槛为0.02；原有单步严格容差保持不变。','',
        'v8 Vulkan对CUDA的多步参数门槛未通过。额外对照发现，PyTorch自身CPU/CUDA也产生同量级漂移；Vulkan对固定PyTorch CPU参考仍通过原门槛。首步差很小，后期TAL离散匹配将FP32舍入放大。这是基于控制实验的解释，不将未通过项改称通过，也不把mAP接近解释成参数逐位一致。完整JSON同时保留直接比较、固定CPU控制与PyTorch自身CPU/CUDA对照。','',
        '## 资源与分配','',
        '| 模型 | 引擎 | CPU %（全机12逻辑核心） | RSS峰值 MiB | GPU %（整卡） | 显存 MiB（整卡） | C# 每步分配 MiB |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for m in models:
    for e in ['torch-cpu','net-cpu','torch-cuda','net-vulkan']:
        rows=[r for r in performance['runs'] if Path(r['model']).stem==m and r['engineKey']==e]
        cpu=statistics.median(r['cpuPercentOfMachine'] for r in rows)
        rss=statistics.median(max([s.get('rssMiB',0) for s in r['samples']]+[r.get('workingSetMiB',0)]) for r in rows)
        samples=[s for r in rows for s in r['samples'] if 'gpuPercent' in s]
        gpu=f"{statistics.mean(s['gpuPercent'] for s in samples):.1f}" if samples else '—'
        vram=f"{statistics.mean(s['deviceVramMiB'] for s in samples):.1f}" if samples else '—'
        alloc=f"{statistics.median(r['allocatedMiB']/r['steps'] for r in rows):.2f}" if e.startswith('net-') else '—'
        lines.append(f'| {m} | {e} | {cpu:.1f} | {rss:.1f} | {gpu} | {vram} | {alloc} |')
lines+=['','RSS包含宿主、模型和缓存；GPU/显存是整卡采样，不是进程独占。资源只在训练计时窗口采样，C#分配为累计托管分配/更新次数，不能与PyTorch RSS相减。','',
        '## 验证与复现','',
        f"- 训练回归{summary['trainingTests']}项、推理回归{summary['inferenceTests']}项，单步数值{summary['numericChecks']:,}项；{summary['performanceWindows']}独立窗口、{summary['timedUpdates']:,}次计时更新。",
        '- 单图高学习率第100步附近的收敛测试曾波动未达到10%下降；流程测试延长至200步，仍使用原10%下降要求。真实图片留出验证单独进行，不替代原有梯度有限差分和单步官方数值对照。',
        '- [原始速度与资源](validation/0.10.1/performance/training-runtime-comparison.json)、[十组训练结果对比](validation/0.10.1/outcomes/comparison.json)、[单步数值矩阵](validation/0.10.1/oracle-matrix.json)、[总验证与摘要](validation/0.10.1/validation-summary.json)。',
        f"- 主库 SHA256：`{summary['coreAssemblySha256']}`。",f"- 训练库 SHA256：`{summary['trainingAssemblySha256']}`。",'',
        '```powershell','python tools/validate-training-optimization.py','python tools/render-training-optimization.py','```','',
        '运行库、训练库与网页仍无Python/PyTorch/CUDA/ORT/BLAS依赖；Python、固定官方源码和数据集只用于开发对照。当前仍为检测头微调，不含分割/姿态、全参数或Metal训练。历史0.9.0推理和0.10.0训练报告保留原数据，不覆盖旧排名。','']
(ROOT/'artifacts/training-optimization-0.10.1.md').write_text('\n'.join(lines),encoding='utf-8')
print('Rendered 0.10.1 performance, saved-model accuracy, loss and resource comparisons.')
