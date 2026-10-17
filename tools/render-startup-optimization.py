"""从完成的生命周期记录生成中文表格和独立SVG，避免手抄或混合不同发布方式的数据。"""
import argparse
import json
from pathlib import Path
import statistics
from xml.sax.saxutils import escape


def load(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if value.get('Status') != 'complete':
        raise ValueError(f'Incomplete timing report: {path}')
    if len(value['Records']) != 96 or len(value['Comparisons']) != 12:
        raise ValueError('Expected 3 models x 2 formats x 2 backends x 2 ABBA repeats')
    return value


def svg(path, report, title):
    rows = report['Comparisons']
    width, height = 1500, 120 + len(rows) * 69
    maximum = max(row['Metrics']['ReadyToFirstOutputMs'][key] for row in rows for key in ['Baseline', 'Candidate'])
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
                '<rect width="100%" height="100%" fill="#f8fafc"/>', '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#0f172a">',
                f'<text x="30" y="42" font-size="26" font-weight="700">{escape(title)}</text>',
                '<text x="30" y="72" font-size="16">首次原始输出（ms，越低越好）；每组每版本4个新进程窗口；灰色=基线，蓝色=对照</text>']
    for i, row in enumerate(rows):
        y = 112 + i * 69
        label = f'{row["Model"]} / {row["Backend"]}'
        elements.append(f'<text x="30" y="{y+21}" font-size="17">{escape(label)}</text>')
        for j, (key, color) in enumerate([('Baseline', '#94a3b8'), ('Candidate', '#2563eb')]):
            value = row['Metrics']['ReadyToFirstOutputMs'][key]
            length = value / maximum * 880
            top = y + j * 23
            elements.append(f'<rect x="390" y="{top}" width="{length:.2f}" height="18" rx="3" fill="{color}"/>')
            elements.append(f'<text x="{400+length:.2f}" y="{top+15}" font-size="15">{value:.1f}</text>')
    elements.extend(['</g>', '</svg>'])
    path.write_text('\n'.join(elements) + '\n', encoding='utf-8', newline='\n')


def table(report):
    lines = ['| 模型 / 格式 | 设备 | 加载 ms：基线 → 对照 | 首次输出 ms：基线 → 对照 | 稳态 ms：基线 → 对照 | P95 ms：基线 → 对照 | 每帧分配变化 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['Comparisons']:
        metrics = row['Metrics']
        cells = [f'{metrics[key]["Baseline"]:.2f} → {metrics[key]["Candidate"]:.2f}' for key in ['ModelLoadMs', 'ReadyToFirstOutputMs', 'MedianMs', 'P95Ms']]
        lines.append(f'| {row["Model"]} | {row["Backend"]} | {" | ".join(cells)} | {metrics["AllocatedBytesPerFrame"]["CandidateChangePercent"]:+.2f}% |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--publish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, publish = load(args.source), load(args.publish)
    if source['Variants']['candidate']['SnapshotDllSha256'] != publish['Variants']['baseline']['SnapshotDllSha256']:
        raise ValueError('Publish comparison does not use exactly the optimized IL library')
    if source['InputSha256'] != publish['InputSha256']:
        raise ValueError('Input tensors differ')
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    validation = output.parent / 'validation' / '0.11.3'
    web = json.loads((validation / 'web-startup.json').read_text(encoding='utf-8'))
    if web['Status'] != 'complete':
        raise ValueError('Web startup comparison incomplete')
    reloads = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(validation.glob('reload-*.json'))]
    if len(reloads) != 4 or any(row['Status'] != 'passed' for row in reloads):
        raise ValueError('Expected four successful reload ABBA workers')
    def reload_median(sha):
        values = [cycle['ModelLoadMs'] for row in reloads if row['DllSha256'] == sha for cycle in row['Cycles'][1:]]
        if len(values) != 6:
            raise ValueError('Reload library or cycle count mismatch')
        return statistics.median(values)
    reload_old = reload_median(source['Variants']['baseline']['SnapshotDllSha256'])
    reload_new = reload_median(source['Variants']['candidate']['SnapshotDllSha256'])
    source_svg = output.with_name('startup-source-0.11.3.svg')
    publish_svg = output.with_name('startup-publish-0.11.3.svg')
    svg(source_svg, source, '源码优化：0.11.2 IL 与 0.11.3 IL')
    svg(publish_svg, publish, '同源码发布：0.11.3 IL 与 0.11.3 ReadyToRun')
    frames = sum(row['Frames'] for report in [source, publish] for row in report['Records'])
    hashes = [(label, report['Variants'][variant]['SnapshotDllSha256']) for label, report, variant in [
        ('0.11.2 IL', source, 'baseline'), ('0.11.3 IL', source, 'candidate'), ('0.11.3 ReadyToRun', publish, 'candidate')]]
    text = f'''# 0.11.3 加载、首次推理与稳定性优化

实测：Windows / i7-8700K / GTX 1050 Ti，FP32、batch1、640×640、12线程，固定真实图片输入。检测、分割、姿态各一个模型，PT/ONNX、CPU/Vulkan，共两组 **192 个独立进程窗口、{frames:,} 次稳态推理**；每窗口另预热60帧，稳态至少3秒/40帧。每组每配置每版本4个窗口，先计算窗口内稳态中位数，再取窗口间中位数；加载和首次输出也取窗口间中位数。

当前是0.11.3源码更新，正式下载仍为 [v0.11.2](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.11.2)。本轮没有重跑 Python/ORT，[0.11.2原版排名](runtime-comparison-0.11.2.md)保留原始口径，不据此宣称后续源码在所有配置领先。Metal未新增实机验证。

## 普通 IL：源码优化前后

基线为0.11.2正式版DLL，对照为0.11.3普通托管IL。减少PT解析临时数组、合并BN复制/缩放；ONNX采用整数路由、复用参数和存活工作区，相同形状只分析一次常量依赖；标签使用独立只读字典减少冷启动特化构造。

{table(source)}

![源码优化首次输出对比]({source_svg.with_suffix('.png').name})

[SVG矢量图]({source_svg.name})

## 相同源码：可选快速启动发布

基线为上表**相同SHA256的0.11.3 IL**，对照为0.11.3 ReadyToRun主库。两边使用相同计时程序，只替换主库；表中加载、首次输出、稳态均是这一组独立ABBA测量，不能把两组窗口拼接成同一次调用。实际网页和CLI的FastStartup配置已发布到隔离目录验证。

{table(publish)}

![快速启动发布首次输出对比]({publish_svg.with_suffix('.png').name})

[SVG矢量图]({publish_svg.name})

ReadyToRun是.NET SDK的预编译发布方式，源代码仍为C#，不会引入Python或其他推理运行库。**应用发布程序集包含IL与平台相关预编译代码，NuGet主库及普通构建保持托管IL。** 发布体积和持续运行结果一起评估，使用方式与平台约束见[启动文档](../docs/startup.md)。

## 网页启动与同进程再次加载

| 独立场景 | 基线 | 对照 | 变化 |
| --- | ---: | ---: | ---: |
| 进程启动至33模型API列表可用：0.11.2普通发布 → 0.11.3 FastStartup | {web['Summary']['baseline']['MedianMs']:.2f} ms | {web['Summary']['candidate']['MedianMs']:.2f} ms | {web['CandidateChangePercent']:+.2f}% |
| 同进程再次构造v8n PT/Vulkan会话：0.11.2 IL → 0.11.3 IL | {reload_old:.2f} ms | {reload_new:.2f} ms | {(reload_new/reload_old-1)*100:+.2f}% |

网页数据来自8个ABBA进程，包含dotnet宿主和ASP.NET启动，不包含网页渲染或图片推理；模型列表内容摘要完全一致。[原始JSON](validation/0.11.3/web-startup.json)。再次加载使用4个ABBA进程，每进程4次完整新建/释放，后3次作为再加载样本，每版本2个进程/6个相关样本，不将同进程样本视为6个独立实验。Vulkan仅共享进程实例，各会话设备和显存仍独立，详见[生命周期与稳定性证据](validation/0.11.3/README.md)。

## 口径与原始证据

- 加载包含后端创建、模型解析和权重准备。首次输出从托管Main首条语句开始，包含输入读取、加载和首次PredictRaw；不包含dotnet宿主启动和入口JIT，不等于图片端到端响应时间。
- GPU包含输入上传和所有输出读回；图片缩放、NMS、最终掩码、浏览器绘制不在此计时范围。默认继承应用Vulkan缓存，保留操作系统与驱动缓存，不称为硬件冷启动。
- 稳态列为每帧中位数，分配列为精确托管分配变化，负数表示减少。工作集、CPU占用、P95、首帧单独耗时及完整样本保存在JSON中；本轮没有重新采样GPU利用率。
- [源码ABBA记录](lifecycle-source-0.11.3.json)、[发布方式ABBA记录](lifecycle-publish-0.11.3.json)、[功能/稳定性与独立官方输出校验](validation/0.11.3/README.md)。原始f32位于记录中的本机.cache路径，没有随源码提交，迁移及独立验证方法见[工具说明](../tools/LifecycleBenchmark/README.md)。
- 官方参考采用与实测一致的12线程，首帧/末帧共520/520项通过。另保留4线程参考旧版两个窗口的严格匹配失败，新版窗口通过；参考线程和所有容差在独立校验JSON中明确记录。

| 实测程序集 | SHA256 |
| --- | --- |
'''
    text += '\n'.join(f'| {name} | `{sha}` |' for name, sha in hashes)
    text += f'\n\n输入SHA256：`{source["InputSha256"]}`。复现脚本：[run-startup-comparison.ps1](../tools/run-startup-comparison.ps1)。\n'
    output.write_text(text, encoding='utf-8', newline='\n')
    print(output)


if __name__ == '__main__':
    main()
