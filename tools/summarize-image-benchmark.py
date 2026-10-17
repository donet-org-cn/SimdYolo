"""Generate the Chinese report and a shareable figure from unedited timing evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / 'artifacts/validation/0.11.1'
data = json.loads((EVIDENCE/'image-performance.json').read_text(encoding='utf-8'))
web = json.loads((EVIDENCE/'web-performance.json').read_text(encoding='utf-8'))
if data['status'] != 'passed' or web['status'] != 'passed':
    raise RuntimeError('Incomplete timing evidence')

rows = data['aggregate']
description = lambda row: row['model'] + ' / ' + row['device'] + (' / 2048×1536' if row['image'] == 'large' else ' / 810×1080')
lines = ['# 0.11.1：图片推理、目标分离与背景去除优化', '',
         '实测日期：' + data['dateUtc'][:10] + '。Windows 11 / i7-8700K / GTX 1050 Ti，FP32、batch=1、模型输入640×640、CPU四线程。对照冻结0.11.0与0.11.1，使用相同真实公交照片RGB与权重。核心保持纯C#/.NET，浏览器负责图片解码与RGBA/PNG合成。', '',
         '本轮重点降低原图尺度的掩码处理、上传和浏览器重复处理成本。网络前向内核未修改；没有重新测量PyTorch/ORT，也不改写历史0.9.0排名。Metal没有本机实测。', '',
         '## 完整图片推理', '',
         '单位ms，越低越快。包含预处理、前向、NMS/端到端筛选和原图尺度概率掩码，排除HTTP、JSON与网页合成。主要分割配置按旧→新→新→旧顺序，各两个独立进程窗口；检测控制与大图各一对。每个分段预热12帧，至少测2秒和40帧，多窗口取窗口中位数的中位数。小幅差异不代表稳定领先。', '',
         '| 模型 / 后端 / 原图 | 优化前 ms | 优化后 ms | 耗时下降 |',
         '| --- | ---: | ---: | ---: |']
for row in rows:
    before, after = (row[key]['predict']['medianMs'] for key in ['baseline', 'optimized'])
    lines.append(f'| {description(row)} | {before:.2f} | {after:.2f} | {(1-after/before)*100:.1f}% |')
lines += ['', '![实际图片推理优化前后](image-pipeline-0.11.1.png)', '',
          '## 掩码与资源分段', '',
          '8位转换为网页序列化准备byte数组，单独计时，不包含在上表Predict中。托管分配是每帧累计分配量，不是RSS或峰值内存。CPU核心百分比中100%代表使用一个逻辑核心，不能直接当作全机CPU使用率。原始JSON保留每段CPU核心百分比、P95、分配与全部采样；本轮没有采集显存/GPU利用率，不能据速度推断其下降。', '',
          '| 配置 | 掩码后处理 前→后 ms | 8位转换 前→后 ms | 完整Predict托管分配 前→后 MiB/帧 |',
          '| --- | ---: | ---: | ---: |']
for row in rows:
    if 'maskBytes' not in row['baseline']:
        continue
    a, b = row['baseline'], row['optimized']
    lines.append(f"| {description(row)} | {a['decode']['medianMs']:.3f} → {b['decode']['medianMs']:.3f} | {a['maskBytes']['medianMs']:.3f} → {b['maskBytes']['medianMs']:.3f} | {a['predict']['allocatedBytesPerFrame']/1048576:.2f} → {b['predict']['allocatedBytesPerFrame']/1048576:.2f} |")
lines += ['', '## 网页实际整轮', '',
          '实际点击开始推理，计入RGB准备、HTTP、JSON解析、叠加及目标/背景合成。独立于上面的.NET分段测试；Edge无头画布使用CPU，.NET Vulkan仍使用真实GPU。每配置预热3次、测5次；两版本均包含响应克隆与2ms完成轮询开销，按旧版全部→新版全部执行，属于开发诊断，不是多窗口统计显著性结论。', '',
          '| 图片 / 后端 | 优化前 ms | 优化后 ms | 耗时下降 | RGB请求体 前→后 MiB | 切换类别重复解码 前→后次数 |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
for after in web['records']:
    if after['variant'] != 'after':
        continue
    before = next(row for row in web['records'] if row['variant'] == 'before' and (row['name'],row['backend']) == (after['name'],after['backend']))
    a,b=before['medianClientMs'],after['medianClientMs']
    lines.append(f"| {after['width']}×{after['height']} / {after['backend']} | {a:.1f} | {b:.1f} | {(1-b/a)*100:.1f}% | {before['samples'][0]['requestBytes']/1048576:.2f} → {after['samples'][0]['requestBytes']/1048576:.2f} | {before['switchTarget']['maskDecodes']} → {after['switchTarget']['maskDecodes']} |")
lines += ['', '重复解码次数统计连续九次切换人物/车辆/全部。新结果仍需解码一次；相同结果的类别切换复用缓存。原始RGB数据体减少25%，JSON元数据移至有界请求头。画布缓存最多32 MiB，RGB缓存最多12 MiB，均随图片更换释放；超预算掩码作为临时结果，不常驻缓存。输出仍为有界预览尺寸的两张互补透明PNG，不填补被遮挡背景。', '',
          '## 改动与正确性', '',
          '- 纯C#共享原图到原型坐标表，SIMD累加原型、AVX2 Gather插值及Sigmoid，复用两行插值结果；ArrayPool复用临时logits，返回概率图保持独立所有权。',
          '- 大区域多实例按CpuThreads并行，小区域/单实例串行；结果保留原来的排列顺序。',
          '- MaskConverter.ToBytes使用托管SIMD窄化与中点取偶舍入，System.Text.Json直接序列化byte数组，保留原Base64线格式。',
          '- 网页缓存二值掩码和彩色叠加层，LRU按像素字节上限淘汰，内容或尺寸变化使缓存失效；调整置信度使用同一加载会话的逐次阈值。', '',
          f"实际模型最终帧共{len(data['correctness'])}项对照，核对{sum(row['maskPixels'] for row in data['correctness']):,}个掩码概率。最大框误差{max(row['maxBoxError'] for row in data['correctness']):.6g}像素、最大概率误差{max(row['maxMaskProbabilityError'] for row in data['correctness']):.6g}，0.5阈值翻转{sum(row['thresholdFlips'] for row in data['correctness'])}个。逐项固定门槛为框0.01像素、置信度2e-5、完整模型掩码概率2e-5；独立标量掩码公式另使用3e-6门槛。跨进程网络舍入包含在完整模型误差中。", '',
          'CPU/Vulkan回归116项、训练集成19项；禁用硬件intrinsics的掩码2项；完整浏览器41条检查及15项新增传输/缓存检查。新旧合成在重叠、非整数位置、孔洞/无目标等小型用例的24,576个RGBA字节完全一致；真实照片导出的目标与背景透明度互补。实际PNG下载、大图、标注、训练验证/恢复与低精度入口均已回归。', '',
          '加载与首帧数值保留在原始JSON，只有各窗口样本，未清空OS/驱动缓存，本轮不据此宣称冷启动改善。稳态分段中位数不能相加冒充完整耗时。', '',
          '[原始性能JSON](validation/0.11.1/image-performance.json) · [网页原始采样](validation/0.11.1/web-performance.json) · [验证、摘要、截图与复现](validation/0.11.1/README.md)', '']
(ROOT/'artifacts/image-pipeline-optimization-0.11.1.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')

# 导出静态图片供README/博客引用；仅为开发绘图依赖。
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig, ax = plt.subplots(figsize=(10, 6.6))
y = list(range(len(rows)))
old=[row['baseline']['predict']['medianMs'] for row in rows]
new=[row['optimized']['predict']['medianMs'] for row in rows]
ax.barh([v-.18 for v in y], old, height=.34, color='#9ca3af', label='0.11.0')
ax.barh([v+.18 for v in y], new, height=.34, color='#6558ce', label='0.11.1')
ax.set_yticks(y, [description(row) for row in rows], fontproperties=font, fontsize=9)
ax.invert_yaxis(); ax.set_xlabel('完整图片推理 / ms（越低越快）', fontproperties=font)
ax.set_title('真实图片推理：预处理 + 网络 + 检测筛选 + 原图掩码', fontproperties=font)
for index,(a,b) in enumerate(zip(old,new)):
    ax.text(a+.3,index-.18,f'{a:.2f}',va='center',fontsize=8)
    ax.text(b+.3,index+.18,f'{b:.2f}',va='center',fontsize=8)
ax.set_xlim(0,max(old+new)*1.15); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True); ax.legend()
fig.text(.5,.012,'i7-8700K / GTX 1050 Ti · FP32 · 模型输入640×640 · CPU四线程；不含HTTP/网页',ha='center',fontproperties=font,fontsize=8)
fig.tight_layout(rect=(0,.035,1,1)); fig.savefig(ROOT/'artifacts/image-pipeline-0.11.1.png',dpi=170); plt.close(fig)
print('Generated report and performance PNG')
