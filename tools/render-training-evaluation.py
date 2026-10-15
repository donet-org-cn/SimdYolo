"""仅从完整且通过检查的训练结果生成中文评测、排名和 PNG/SVG，保留原始数据。"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--validation',type=Path,default=Path('artifacts/validation/0.10.0'))
args=parser.parse_args()
directory=args.validation
raw_path=directory/'performance/training-runtime-comparison.json'
raw=json.loads(raw_path.read_text(encoding='utf-8'))
assert raw['status']=='passed' and len(raw['runs'])==40
assert raw['sizes']==128 and raw['threads']==4 and raw['repeats']==2
oracle=json.loads((directory/'oracle-matrix.json').read_text(encoding='utf-8'))
assert oracle['status']=='passed' and oracle['totalGroups']==34
integrations={d:json.loads((directory/f'integration-{d}.json').read_text(encoding='utf-8')) for d in ['cpu','vulkan']}
assert all(d['status']=='passed' and len(d['results'])==5 and all(r['passed'] for r in d['results']) for d in integrations.values())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
core_sha=sha(ROOT/'src/SimdYolo/bin/Release/net10.0/SimdYolo.dll')
train_sha=sha(ROOT/'src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll')
for d in integrations.values():
    assert d['coreAssemblySha256']==core_sha and d['trainingAssemblySha256']==train_sha
for r in oracle['results']:
    value=json.loads((directory/r['report']).read_text(encoding='utf-8'))
    assert value['status']=='passed' and value['coreAssemblySha256']==core_sha and value['trainingAssemblySha256']==train_sha
runs=defaultdict(list)
for r in raw['runs']:
    assert r['passed'] and r['steps']==raw['steps']
    if r['engineKey'].startswith('net-'):
        assert r['coreAssemblySha256']==core_sha and r['trainingAssemblySha256']==train_sha
    runs[r['model'].removesuffix('.pt'),r['engineKey']].append(r)
models=raw['models'];engines=['torch-cpu','net-cpu','torch-cuda','net-vulkan']
assert all(len(runs[m,e])==2 for m in models for e in engines)
for model in models:
    check=json.loads((directory/f'performance/fixture-check-{model}.json').read_text(encoding='utf-8'))
    assert check['status']=='passed' and check['coreAssemblySha256']==core_sha and check['trainingAssemblySha256']==train_sha
    assert len({r['fixtureSha256'] for e in engines for r in runs[model,e]})==1
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib-release'))
for path in ['.cache/article-graphics-deps','.cache/reference']:
    sys.path.insert(0,str(ROOT/path))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
font=Path('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'svg.fonttype':'path','axes.unicode_minus':False,'savefig.facecolor':'white'})
labels={'torch-cpu':'原版 PyTorch CPU','net-cpu':'纯 C# CPU SIMD','torch-cuda':'原版 PyTorch CUDA','net-vulkan':'纯 C# Vulkan 混合训练'}
colors={'torch-cpu':'#9eabc0','net-cpu':'#367ec4','torch-cuda':'#d69b44','net-vulkan':'#319b8d'}
def median(model,engine,key='medianMs'):return statistics.median(r[key] for r in runs[model,engine])
def save(fig,name):
    fig.savefig(ROOT/f'artifacts/{name}.png',dpi=155)
    svg=ROOT/f'artifacts/{name}.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
for group,selected in [('cpu',engines[:2]),('gpu',engines[2:])]:
    fig,ax=plt.subplots(figsize=(12,6));positions=np.arange(len(models));height=.30
    for i,e in enumerate(selected):
        means=[median(m,e) for m in models]
        lower=[v-min(r['medianMs'] for r in runs[m,e]) for m,v in zip(models,means)]
        upper=[max(r['medianMs'] for r in runs[m,e])-v for m,v in zip(models,means)]
        bars=ax.barh(positions+(i-.5)*height,means,height,color=colors[e],label=labels[e],xerr=[lower,upper],capsize=3,error_kw={'elinewidth':1})
        for j,(bar,value) in enumerate(zip(bars,means)):
            ax.text(value+upper[j]+max(means)*.014,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontsize=10)
    ax.set_yticks(positions,models);ax.invert_yaxis();ax.set_xlim(left=0,right=max(max(r['medianMs'] for m in models for r in runs[m,e]) for e in selected)*1.20)
    ax.set_xlabel('每次训练更新耗时（ms，越低越快）');ax.set_title(f'0.10.0 检测头训练：{group.upper()} 实测',loc='left',fontsize=18,pad=18)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right',frameon=False)
    fig.text(.12,.05,f"128×128 / FP32 / 2 类 / 4 线程 / 固定缓存特征 / 每窗 {raw['steps']} 次更新 / 2 独立窗口\n含前向、损失、反向及 SGD；不含骨干、预处理和保存。细线为两窗口范围，不是置信区间。",fontsize=10,color='#5a6575')
    fig.subplots_adjust(left=.12,right=.96,top=.86,bottom=.22);save(fig,f'training-performance-0.10.0-{group}')
fig,axes=plt.subplots(2,3,figsize=(15,8.5));axes=axes.ravel()
for ax,model in zip(axes,models):
    for d,color in [('cpu',colors['net-cpu']),('vulkan',colors['net-vulkan'])]:
        record=next(r for r in integrations[d]['results'] if r['model']==model)
        history=record['history'];ax.plot([p['step'] for p in history],[p['totalLoss'] for p in history],color=color,label=d.upper(),linewidth=1.3)
    ax.set_title(model);ax.set_xlabel('真实 SGD 更新次数');ax.set_ylabel('实际加权训练损失');ax.set_xlim(1,100);ax.grid(alpha=.15)
axes[-1].axis('off');axes[0].legend(frameon=False)
axes[-1].text(.05,.85,'冻结骨干/FPN，仅更新检测头\n单张合成方块图，64×64，100轮\nYOLO26 的训练目标权重随轮次改变\n\n曲线不能当作验证集 mAP\n前后固定口径评估另见表格',va='top',linespacing=1.7,fontsize=12)
fig.suptitle('五代检测模型：CPU / Vulkan 实际训练曲线',x=.06,ha='left',fontsize=19)
fig.tight_layout(rect=[0,0,1,.94]);save(fig,'training-loss-0.10.0')
lines=['# 0.10.0 五代检测头训练测试与评测','',
       '本版把 v8、Ultralytics v9、v10、YOLO11、YOLO26 接入同一个纯 C# 检测头微调入口和网页。骨干/FPN 冻结；v10/26 两个分支均训练。分割/姿态、全参数与 Metal 训练尚未实现。','',
       '## 正确性与实际训练','',
       f"固定官方 Ultralytics 8.4.0 / PyTorch 2.11.0 CPU 参考：**34/34 组、{oracle['totalChecks']:,}/{oracle['totalChecks']:,} 项检查通过**。五模型 × CPU/Vulkan × 普通/小目标/空标签，另含 YOLO26 两设备的中间轮和末轮。检查完整分支前向、官方单/双头组合目标、标签分配、logits 梯度、全部参数梯度、梯度连接状态、BN、SGD 后权重及动量；保留固定容差。性能所用128尺寸另有五组夹具检查。",'',
       '| 实际权重 | CPU 评估损失（前→后） | Vulkan 评估损失（前→后） | 实际更新 | 保存/恢复/推理 |','| --- | ---: | ---: | ---: | --- |']
for model in models:
    a=next(r for r in integrations['cpu']['results'] if r['model']==model);b=next(r for r in integrations['vulkan']['results'] if r['model']==model)
    assert a['after']<a['before']*.9 and b['after']<b['before']*.9
    lines.append(f"| {model}.pt | {a['before']:.4f} → {a['after']:.4f} | {b['before']:.4f} → {b['after']:.4f} | 各100步 | 通过 |")
lines+=['','![实际训练曲线](training-loss-0.10.0.png)','',
        '上表为固定单图合成方块、64×64、SGD学习率0.005、100轮的 BN 推理模式损失；YOLO26 评估固定初始0.8/0.2双头权重。曲线是每步实际优化目标，YOLO26会随轮次调权；两者不能混为一个指标。单图损失下降只证明真实更新和可拟合性，**没有在 COCO 或业务留出集测量 mAP**。','',
        '恢复测试比较同一调度下连续三轮和取消后恢复：参数/BN绝对容差2e-5，训练后的动量使用与梯度相同的5e-4+2e-3×|参考值|；加载时动量字节必须完全一致。重新提取的冻结特征存在 FP32 舍入，未要求 CPU 跨会话训练过程逐位相同。原生检查点元数据、类别、输出形状和 CPU/Vulkan 推理另检。','',
        '修正了两种 SGD 行为：全背景框分支未连接时跳过衰减/动量；有正样本时所有回归尺度获得显式零梯度，未命中尺度仍执行 SGD。两种情况均由官方梯度连接及更新结果检查。','',
        '## 训练速度排名','',
        f"本机 i7-8700K / GTX 1050 Ti。128×128、2类、FP32、单图BN、4线程、固定初始权重和**完全相同的缓存特征**。各引擎独立进程串行运行，预热10步，每窗固定{raw['steps']}次 SGD 更新；两轮反转后端顺序。40/40 窗口、{len(raw['runs'])*raw['steps']:,}次计时更新完成。",'',
        '计时含头部前向、标签分配/损失、反向、SGD及梯度清零；Vulkan包含头部 GEMM 上传/回读，PyTorch CUDA每步等待GPU完成。排除冻结骨干、图像预处理、模型加载和每轮检查点压缩写盘。PyTorch保持训练BN，未做推理融合；优化器均为SGD。YOLO26在此固定调度首轮权重，未运行官方MuSGD全配方。','',
        'PyTorch参考直接使用官方检测头模块、loss及torch.optim.SGD，与C#一样仅替换原预训练头的分类末层为2类、保留隐藏通道。此处的“原版”表示官方算子与损失参考，不是包含DataLoader、数据增强、全参数反向与保存的Ultralytics train全流程耗时。','',
        '每格为两窗口中位数的中位数，单位ms，①较快；只在同一 CPU/GPU 分组内排名，小差异不能证明稳定领先。','',
        '| 模型 | 原版 PyTorch CPU | 我们 CPU SIMD | CPU 最快 | 原版 PyTorch CUDA | 我们 Vulkan 混合 | GPU 最快 |',
        '| --- | ---: | ---: | --- | ---: | ---: | --- |']
for model in models:
    values={e:median(model,e) for e in engines}
    cells=[]
    for selected in [engines[:2],engines[2:]]:
        best=min(selected,key=values.get)
        cells += [(f"**{values[e]:.2f} ①**" if e==best else f"{values[e]:.2f} ②") for e in selected]
        cells.append('PyTorch' if best.startswith('torch') else '我们的实现')
    lines.append('| '+model+' | '+' | '.join(cells)+' |')
lines+=['','![CPU训练速度](training-performance-0.10.0-cpu.png)','','![GPU训练速度](training-performance-0.10.0-gpu.png)','',
        'Vulkan训练是混合路径，头部深度卷积、BN、激活、损失、SGD及数据整理仍在CPU。当前推理专用优化不能直接用于持续变化的训练权重；速度结果以本表为准，不能宣称全面超越PyTorch。普通ONNX Runtime未安装Training/ORTModule，本轮不提供其训练耗时，亦不以ORT推理速度代替训练。','',
        '## CPU、内存与GPU资源','',
        'CPU%按全机逻辑核心归一化，RSS是训练窗口采样峰值的两轮中位数，包含宿主、冻结模型和缓存；GPU%/显存为两窗口整卡样本均值，包含桌面及其他进程，不能解释成训练进程独占。采样数保留在JSON，短窗口的整卡采样只作粗略参考。','',
        '| 模型 | 引擎 | CPU %（全机） | RSS采样峰值 MiB | GPU %（整卡） | 显存 MiB（整卡） |','| --- | --- | ---: | ---: | ---: | ---: |']
for model in models:
    for e in engines:
        records=runs[model,e];cpu=median(model,e,'cpuPercentOfMachine')
        rss=statistics.median(max(s['rssMiB'] for s in r['samples'] if 'rssMiB' in s) for r in records)
        gpu=[s['gpuPercent'] for r in records for s in r['samples'] if 'gpuPercent' in s];vram=[s['deviceVramMiB'] for r in records for s in r['samples'] if 'deviceVramMiB' in s]
        lines.append(f"| {model} | {labels[e]} | {cpu:.1f} | {rss:.1f} | {statistics.mean(gpu):.1f}"+f" | {statistics.mean(vram):.1f} |" if gpu else f"| {model} | {labels[e]} | {cpu:.1f} | {rss:.1f} | — | — |")
lines+=['','## 证据与复现','',
        f'- 主库 SHA256：`{core_sha}`。',f'- 训练库 SHA256：`{train_sha}`。',
        f'- [训练速度原始JSON](validation/0.10.0/performance/training-runtime-comparison.json)，SHA256：`{sha(raw_path)}`。',
        '- [数值对照矩阵](validation/0.10.0/oracle-matrix.json)、[CPU微调](validation/0.10.0/integration-cpu.json)、[Vulkan微调](validation/0.10.0/integration-vulkan.json)。',
        '- [完整复现命令与网页验证](validation/0.10.0/README.md)、[训练API及支持范围](../docs/training.md)。','',
        '历史0.9.0推理图表继续绑定旧DLL；本报告只表示0.10.0检测头训练，不改写历史排名。Python/PyTorch和绘图库仅用于开发评测，运行库继续保持纯C#。','']
(ROOT/'artifacts/training-evaluation-0.10.0.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
print(json.dumps(dict(status='passed',groups=34,checks=oracle['totalChecks'],windows=40,timedSteps=len(raw['runs'])*raw['steps'])))
