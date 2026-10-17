"""用本轮完整实测更新发布说明与README；旧测量及NuGet保持原字节。"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--version',required=True)
args=parser.parse_args(); version=args.version
report=json.loads((ROOT/f'artifacts/runtime-comparison-{version}.json').read_text(encoding='utf-8'))
assert report['Status']=='complete'
assert len(report['Records'])==len(report['Models'])*len(report['Engines'])*report['Repeats']
assert all(c['Passed'] for row in report['Records'] for c in row['Correctness'])
spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/compare-runtimes.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rankings='\n'.join(module.render_speed_rankings(report))
windows=len(report['Records']); frames=sum(row['Iterations'] for row in report['Records'])
checks=sum(len(row['Correctness']) for row in report['Records'])
url='https://github.com/hnlyf168/SimdYolo'
release=url+'/releases/tag/v'+version

def save(name,text):
    (ROOT/name).parent.mkdir(parents=True,exist_ok=True)
    (ROOT/name).write_text(text,encoding='utf-8',newline='\n')

def read(name):
    """保留首次读到的文档，以便重跑生成器而不叠加表格或改写历史章节。"""
    original=ROOT/'.cache'/f'release-{version}'/'docs-baseline'/name
    if not original.exists():
        original.parent.mkdir(parents=True,exist_ok=True)
        original.write_bytes((ROOT/name).read_bytes())
    return original.read_text(encoding='utf-8-sig')

def downloads(prefix=''):
    names=[(f'SimdYolo.{version}.nupkg','推理主库、中文XML及GPU着色器'),
           (f'SimdYolo.Training.{version}.nupkg','五代检测头微调、验证/最佳模型/早停，依赖同版主库'),
           (f'SimdYolo-models-v{version}.zip','11个PT、22个ONNX、来源、SHA256和原许可证'),
           (f'SimdYolo-Web-v{version}.zip','ASP.NET Core网页、启动脚本；模型和.NET运行时另备'),
           (f'SimdYolo-docs-v{version}.zip','当前README、API/训练说明、图文、完整评测与复现输入'),
           (f'SHA256SUMS-v{version}.txt','全部附件的SHA256')]
    return '\n'.join(['| 附件 | 内容 |','| --- | --- |']+[f'| [{name}]({url}/releases/download/v{version}/{name}) | {description} |' for name,description in names])

scope=(f"实测日期：{report['Date']}；{report['Cpu']}，{report['Gpu']['Name']}。FP32、batch1、{report['Size']}×{report['Size']}、CPU{report['Threads']}线程。"
       f"11模型×8配置×两轮，共**{windows}个独立窗口、{frames:,}次推理、{checks}/{checks}项输出校验通过**。预热{report['Warmup']}次，每窗口至少{report['DurationSeconds']:g}秒/{report['MinimumIterations']}帧。")
boundary=('本表计入网络前向与输出头解码；GPU包含输入上传和全部输出回读，排除图片预处理、NMS、最终实例掩码、模型加载和首帧。'
          '原版PyTorch使用官方Ultralytics fuse()，ORT启用全部图优化，CUDA关闭TF32。每格为两个独立窗口中位数的中位数，①最快，按未舍入值排名；接近结果不能解释为稳定胜出。')

page=read('README.md')
page=page.replace('**当前源码与仓库内NuGet包：0.11.1。GitHub最新正式Release：v0.10.1。** 最新功能使用main源码或下方0.11.1包；[历史Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1)中的网页和文档ZIP仍属于0.10.1。',
                  f'**当前源码、NuGet及正式Release：{version}。** [下载本版]({release}) · [中文发布说明](docs/releases/v{version}.md)。网页、33模型和图文包均随本版提供；历史附件保持原内容。')
page=page.replace('当前源码为 **0.11.1**，继续优化实际图片推理、目标分离与背景去除：',f'当前源码为 **{version}**，包含图片推理、目标分离与背景去除优化，并完成与原版PyTorch/ORT的新一轮对照：')
page=page.replace('| 当前0.11.1源码 |',f'| 当前{version}源码 |')
page=page.replace('[0.11.1性能与图](artifacts/image-pipeline-optimization-0.11.1.md)',f'[{version}原版/ORT排名](artifacts/runtime-comparison-{version}.md)')
page=page.replace('[验证与截图](artifacts/validation/0.11.1/README.md)',f'[本版验证与截图](artifacts/validation/{version}/README.md)')
start=page.index('## 下载与启动'); end=page.index('从源码启动需要',start)
page=page[:start]+'## 下载与启动\n\n'+downloads()+'\n\n'+page[end:]
page=page.replace('0.11.1实际推理截图](artifacts/validation/0.11.1/web/web-inference.png)',f'{version}实际推理截图](artifacts/validation/{version}/web/web-inference.png)')
page=page.replace('[0.11.1实际截图与下载验证](artifacts/validation/0.11.1/README.md)',f'[{version}实际截图与下载验证](artifacts/validation/{version}/README.md)')
page=page.replace('![0.11.1车辆分离与背景去除](artifacts/validation/0.11.1/web/cutout-vehicles.png)',f'![{version}车辆分离与背景去除](artifacts/validation/{version}/web/cutout-vehicles.png)')
page=page.replace('这些网页新增功能尚未加入历史0.10.1下载包。',f'本版网页运行包已包含这些功能。')
a=page.index('## 0.11.1 实际图片推理优化'); b=page.index('## 推理性能：0.9.0 冻结版本实测',a)
previous=page[a:b].replace('## 0.11.1 实际图片推理优化','## 0.11.1掩码与网页优化（历史测量）')
previous=previous.replace('本轮没有重测PyTorch/ORT','该次0.11.1没有重测PyTorch/ORT')
page=page[:a]+page[b:]
a=page.index('## 推理性能：0.9.0 冻结版本实测'); b=page.index('## C# 使用',a)
performance=(f'## {version} 与原版PyTorch、ORT重新实测\n\n'+scope+'\n\n'+boundary+'\n\n'+rankings+'\n\n'
             f'![{version} CPU速度](artifacts/performance-{version}-cpu.png)\n\n![{version} GPU速度](artifacts/performance-{version}-gpu.png)\n\n'
             f'![{version} 加载、首次与稳态](artifacts/performance-{version}-startup.png)\n\n'
             f'[CPU计算成本](artifacts/performance-{version}-cpu-time.png) · [CPU内存](docs/articles/assets/{version}/09-cpu-memory.png) · [GPU整卡使用率](artifacts/performance-{version}-gpu-usage.png) · [整卡显存](artifacts/performance-{version}-vram.png) · [完整结果](artifacts/runtime-comparison-{version}.md) · [原始JSON](artifacts/runtime-comparison-{version}.json)\n\n'
             'CPU%按全机逻辑核心归一化；RSS包含宿主、模型和缓存，不能由速度排名推断内存优势。GPU/显存是整卡采样，包含桌面及其他进程。加载、首帧单列，未清空OS/驱动缓存。本轮未比较torch.compile、TensorRT、FP16/INT8，也没有Metal实机结果。\n\n'
             '[0.9.0历史排名](artifacts/runtime-comparison-0.9.0.md)保留原始结果；下方0.11.1图片管线测量包含预处理与原图掩码，口径不同。\n\n')
page=page[:a]+performance+previous+page[b:]
page=page.replace('本轮精度对照使用COCO128','0.10.1精度对照使用COCO128')
save('README.md',page)

notes=(f'# SimdYolo {version}：原版对照重测、纯.NET模型工作流与目标分离\n\n'
       '主库和训练核心为纯C#/.NET10；CPU使用SIMD/intrinsics，GPU使用系统Vulkan/Metal。应用不依赖Python、PyTorch、TorchSharp、ORT或BLAS；Vulkan运行不需要CUDA Toolkit/cuDNN/cuBLAS或Vulkan SDK，需要兼容驱动和loader。Python/ORT仅供独立开发验证。\n\n'
       '## 本版功能\n\n'
       '- 直接加载受支持完整PT、ONNX、SYTRAIN及类别标签，检测/实例分割/姿态统一API，随包提供33份原模型。\n'
       '- 纯C#掩码SIMD、坐标与插值行缓存、多实例有界并行、临时缓冲池；公开输出保持独立所有权。\n'
       '- 网页人物/车辆分离和移除目标背景，两张互补透明PNG；4K/8K图有界缩放，原图坐标映射。\n'
       '- RGB24二进制上传、有界掩码LRU、RGB复用；调整置信度复用模型会话。\n'
       '- 五代检测头微调、独立验证集、P/R/mAP、最佳模型、早停、取消恢复；网页标注与人工审核预标注。\n'
       '- 纯.NET导出标准ONNX；CPU FP16混合/INT8及校准为实验功能，当前没有速度优势，默认FP32，GPU仍为FP32。\n\n'
       '## 原版 / 本库 / ORT速度排名\n\n'+scope+'\n\n'+boundary+'\n\n'+rankings+'\n\n'
       f'![CPU速度](https://raw.githubusercontent.com/hnlyf168/SimdYolo/v{version}/artifacts/performance-{version}-cpu.png)\n\n'
       f'![GPU速度](https://raw.githubusercontent.com/hnlyf168/SimdYolo/v{version}/artifacts/performance-{version}-gpu.png)\n\n'
       f'[完整速度、CPU、RSS、GPU及显存]({url}/blob/v{version}/artifacts/runtime-comparison-{version}.md) · [原始JSON]({url}/blob/v{version}/artifacts/runtime-comparison-{version}.json)。整卡GPU数据包含桌面/其他进程；不是单进程使用率或功耗。\n\n'
       '## 推理、训练与网页验证\n\n'
       f'推理CPU/Vulkan回归116项、训练集成19项及完整网页流程见[本版验证]({url}/tree/v{version}/artifacts/validation/{version})。发布使用与正式计时相同的主库DLL，包、模型、网页、文档及中文UTF-8说明均核验摘要。训练速度及保存模型对照仍绑定0.10.1，没有冒充本版重测。\n\n'
       '训练限于冻结骨干/FPN的检测头微调；没有全参数、分割/姿态或Metal训练。普通ONNX不能初始化训练；SYTRAIN是本项目归档，不能冒充PyTorch PT。其他尺度/设备需独立验证，Metal本机未测试。\n\n'
       '## 下载与启动\n\n'+downloads()+'\n\n'
       f'安装.NET10/ASP.NET Core运行时，将网页ZIP与模型ZIP解压至同一目录，运行start.cmd或start.ps1，打开http://127.0.0.1:5188。[详细步骤]({url}/blob/v{version}/docs/releases/web-runtime.md)。源码构建需要.NET10 SDK及Git LFS。\n\n'
       '原模型、SHA256、AGPL-3.0及来源保持不变，项目源码暂未新增许可证。\n')
save(f'docs/releases/v{version}.md',notes)

# 各使用入口标明同一当前版本，保留带版本号的历史测量和归档链接。
nav=read('docs/README.md')
nav=nav.replace('源码0.11.1','源码'+version).replace('当前源码和仓库内NuGet包为 **0.11.1**；GitHub最新正式Release仍为 **v0.10.1**。0.11.1的优化、标注、验证和导出功能请使用main源码或仓库内0.11.1包，历史Release附件保持原内容。',f'当前源码、NuGet及正式Release为 **{version}**。[下载本版]({release})；新运行包和文档ZIP已包含模型工作流及目标分离。历史Release保持原内容。')
nav=nav.replace('## 0.11.1 下载与启动','## '+version+' 下载与启动')
nav=nav.replace('| 0.11.1优化前后速度、表格和图 |',f'| {version}原版PyTorch/本库/ORT的CPU、GPU排名与资源 | [本版重新实测](../artifacts/runtime-comparison-{version}.md) |\n| 0.11.1优化前后速度、表格和图 |')
nav=nav.replace('主库0.11.1','主库'+version).replace('训练库0.11.1','训练库'+version)
nav=nav.replace('SimdYolo.0.11.1.nupkg',f'SimdYolo.{version}.nupkg').replace('SimdYolo.Training.0.11.1.nupkg',f'SimdYolo.Training.{version}.nupkg')
nav=nav.replace('| 历史正式发布 |', '| 当前正式发布 |').replace('[v0.10.1 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1)，运行包及文档ZIP属于0.10.1',f'[v{version} Release]({release})，含本版网页、模型与文档ZIP')
nav=nav.replace('| 0.11.1 | 原图掩码SIMD',f'| {version} | 与PyTorch/ORT重新实测11模型、8配置，发布模型/网页/文档附件 |\n| 0.11.1 | 原图掩码SIMD')
nav=nav.replace('五代检测头微调速度及保存模型结果对照；最新正式Release','五代检测头微调速度及保存模型结果历史对照')
nav=nav.replace('本轮没有重测PyTorch/ORT，纯检测控制组基本持平。',f'0.11.1的纯检测控制组基本持平；{version}跨引擎重测见[本版报告](../artifacts/runtime-comparison-{version}.md)。')
save('docs/README.md',nav)

for name in ['docs/inference.md','docs/training.md','docs/model-workflows.md','examples/SimdYolo.Web/README.md']:
    text=read(name).replace('当前源码0.11.1','当前源码'+version).replace('当前0.11.1继续提供','当前'+version+'继续提供')
    text=text.replace('已发布下载仍为[0.10.1](releases/v0.10.1.md)',f'已发布下载为[{version}](releases/v{version}.md)')
    text=text.replace('推理性能表继续标明0.9.0冻结历史实测',f'推理排名已按{version}重新实测，0.9.0历史结果单独保留')
    text=text.replace('[性能表格与图](../artifacts/image-pipeline-optimization-0.11.1.md)',f'[本版原版/ORT排名](../artifacts/runtime-comparison-{version}.md) · [0.11.1掩码管线优化](../artifacts/image-pipeline-optimization-0.11.1.md)')
    text=text.replace('[性能表格与图](../../artifacts/image-pipeline-optimization-0.11.1.md)',f'[本版原版/ORT排名](../../artifacts/runtime-comparison-{version}.md) · [0.11.1掩码管线优化](../../artifacts/image-pipeline-optimization-0.11.1.md)')
    text=text.replace('当前发布版0.10.1优化CPU SIMD与Vulkan混合训练，五代检测模型入口保持一致。',f'当前发布版{version}包含训练验证、导出、标注与目标分离；五代检测模型入口保持一致。')
    text=text.replace('[0.10.1运行包](https://github.com/hnlyf168/SimdYolo/releases/download/v0.10.1/SimdYolo-Web-v0.10.1.zip)',f'[{version}运行包]({url}/releases/download/v{version}/SimdYolo-Web-v{version}.zip)')
    text=text.replace('当前0.11.1源码','当前'+version+'源码')
    text=text.replace('0.11.1包的DLL一致性','本版包的DLL一致性')
    for filename in ['package-audit.json','package-consumer-cpu.json','package-consumer-vulkan.json']:
        text=text.replace(f'validation/0.11.1/{filename}',f'validation/{version}/{filename}')
    text=text.replace('验证](../artifacts/validation/0.11.1/README.md)',f'验证](../artifacts/validation/{version}/README.md)')
    text=text.replace('--version 0.11.1','--version '+version)
    text=text.replace('0.11.1没有重新测量训练速度',version+'没有重新测量训练速度')
    text=text.replace('0.11.1推理116项',version+'推理116项')
    text=text.replace('[0.10.1 训练库与网页运行包](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1)',f'[{version}训练库与网页运行包]({release})')
    text=text.replace('/releases/download/v0.10.1/SimdYolo-models-v0.10.1.zip',f'/releases/download/v{version}/SimdYolo-models-v{version}.zip')
    text=text.replace('此修复尚未加入历史0.10.1下载包，使用当前源码启动即可体验。',f'此功能已包含在{version}源码和网页运行包中。')
    text=text.replace('此功能尚未加入历史0.10.1下载包。',f'此功能已包含在{version}网页运行包中。')
    for filename in ['web-annotation.png','web-evaluation.png','web-inference.png','web-training-validation.png','cutout-vehicles.png','cutout-foreground.png','cutout-background.png']:
        text=text.replace(f'0.11.1/web/{filename}',f'{version}/web/{filename}')
    text=text.replace('![0.11.1','!['+version)
    save(name,text)

validation=read('VALIDATION.md')
old='当前源码与仓库内NuGet包为 **0.11.1**；正式Release仍为 **v0.10.1**。'
validation=validation.replace(old,f'当前源码、NuGet与正式Release为 **{version}**。')
section=(f'## {version} 原版/ORT重新实测与正式发布\n\n'+scope+'\n\n'
         f'[本版排名和完整资源](artifacts/runtime-comparison-{version}.md) · [中文发布说明](docs/releases/v{version}.md) · [功能验证](artifacts/validation/{version}/README.md) · [附件审查](artifacts/validation/{version}-release/README.md)。\n\n'
         '本轮性能使用最终冻结主库，包含CPU/PT、CPU/ONNX、Vulkan/PT、Vulkan/ONNX、官方PyTorch CPU/CUDA与.NET ORT CPU/CUDA；固定既有输出容差，不删端到端候选或附加字段。\n\n')
validation=validation.replace('## 0.11.1 图片推理与目标分离优化',section+'## 0.11.1 图片推理与目标分离优化',1)
save('VALIDATION.md',validation)

models=read('models/README.md')
models=models.replace('[0.10.1 Release](https://github.com/hnlyf168/SimdYolo/releases/tag/v0.10.1)',f'[{version} Release]({release})')
save('models/README.md',models)

for name,title in [('docs/articles/wechat.md',f'纯C# YOLO {version}：和PyTorch、ORT重新跑一轮，人物车辆也能分离'),('docs/articles/cnblogs.md',f'SimdYolo {version}：纯.NET与PyTorch/ORT的CPU、GPU性能及目标分离')]:
    original=read(name).split('\n',1)[1].lstrip()
    body=(f'# {title}\n\n**仓库：** [hnlyf168/SimdYolo]({url}) · **本版下载：** [{version} Release]({release})\n\n'
          +scope+'\n\n'+boundary+'\n\n'+rankings+'\n\n'
          f'![本轮CPU](../../artifacts/performance-{version}-cpu.png)\n\n![本轮GPU](../../artifacts/performance-{version}-gpu.png)\n\n'
          f'[CPU、内存、GPU、显存、加载与首帧完整结果](../../artifacts/runtime-comparison-{version}.md)。本版仍保持核心纯C#/.NET，浏览器处理图片和交互；应用不依赖PyTorch/ORT/CUDA工具链，开发对照依赖不进入发布包。\n\n'
          f'![本版车辆分离](../../artifacts/validation/{version}/web/cutout-vehicles.png)\n\n'
          '## 既有掩码优化与训练评测（历史版本保留）\n\n'+original)
    save(name,body)
article_index=read('docs/articles/README.md')
article_index=article_index.replace('# SimdYolo 0.11.1 图文文档','# SimdYolo '+version+' 图文文档')
article_index=article_index.replace('公众号和博客园已更新0.11.1标题、完整图片推理表格、优化图、人物/车辆分离截图及当前源码入口。',f'公众号和博客园已更新{version}跨引擎重测标题、排名及图表，并保留0.11.1的图片管线优化与人物/车辆分离。')
article_index=article_index.replace('[文档与0.11.1包](../README.md) · [0.11.1性能报告](../../artifacts/image-pipeline-optimization-0.11.1.md)',f'[文档与{version}包](../README.md) · [{version}性能报告](../../artifacts/runtime-comparison-{version}.md)')
article_index=article_index.replace('公众号：0.11.1速度',f'公众号：{version}跨引擎速度')
article_index=article_index.replace('博客园：0.11.1 SIMD/行缓存/网页管线',f'博客园：{version}原版/ORT对照、SIMD与网页管线')
article_index=article_index.replace('当前图文使用仓库文件；GitHub正式Release及其中的文档ZIP仍为0.10.1。',f'当前文章与配图随{version}正式Release文档ZIP提供。')
article_index=article_index.replace('[manifest-0.11.1.json](manifest-0.11.1.json) | 当前文章、配图与取数源摘要',f'[manifest-{version}.json](manifest-{version}.json) | 当前文章、配图与取数源摘要')
article_index=article_index.replace('## 当前0.11.1配图',f'## 当前{version}跨引擎配图\n\n[CPU](../../artifacts/performance-{version}-cpu.png) · [GPU](../../artifacts/performance-{version}-gpu.png) · [加载/首帧](../../artifacts/performance-{version}-startup.png) · [整卡GPU](../../artifacts/performance-{version}-gpu-usage.png) · [完整图集](assets/{version}/)\n\n## 0.11.1掩码与分离配图（历史）')
save('docs/articles/README.md',article_index)

runtime=read('docs/releases/web-runtime.md').replace('0.10.1',version)
runtime=runtime.replace('## 示例流程',
    '## 本版新增入口\n\n'
    '分割推理可点击“人物 / 车辆示例”，选择人物、车辆或某一类别，分别下载只保留目标及移除目标的透明PNG。上传大图先在浏览器有界缩放，后端仍使用纯.NET推理；RGB24二进制、掩码缓存和置信度会话复用已包含。\n\n'
    '网页还有数据标注、预标注审核、独立验证集、最佳模型、早停、模型比较、标准ONNX导出和实验CPU低精度。低精度没有速度优势，默认FP32；GPU为FP32。\n\n## 示例流程')
save('docs/releases/web-runtime.md',runtime)

validation_page=(f'# {version} 功能与程序集验证\n\n'
    f'本版包含0.11.0模型工作流及0.11.1掩码/网页优化，并以最终冻结DLL重新比较原版PyTorch和ORT。[{windows}个性能窗口、{frames:,}次推理、{checks}项输出对照](../../runtime-comparison-{version}.json)完成；输出容差沿用既有规则。\n\n'
    '| 检查 | 原始证据 |\n| --- | --- |\n'
    '| CPU/Vulkan推理116项 | [日志](inference-tests.log) |\n'
    '| 训练集成19项 | [日志](training-tests.log) |\n'
    '| 关闭硬件intrinsics的掩码专项2项 | [日志](masks-software.log) |\n'
    '| 完整浏览器41条流程，含大图/PNG/标注/验证/导出/恢复 | [JSON](web/web-browser.json) |\n'
    '| 纯.NET包、中文XML、33模型许可证 | [包审查](package-audit.json)、[训练元数据](training-metadata.json) |\n'
    '| 离线NuGet CPU/Vulkan消费 | [CPU](package-consumer-cpu.json)、[Vulkan](package-consumer-vulkan.json) |\n\n'
    '![本版目标分离](web/cutout-vehicles.png)\n\n'
    f'本轮前向排名排除图片预处理与最终掩码；[0.11.1原图管线对照](../0.11.1/README.md)保留原来的程序集与结果。训练性能与保存模型的官方对照仍为0.10.1历史结果，本版不声称重新执行该训练矩阵。\n\n'
    '主库、训练库与正式计时的DLL摘要在[final-audit.json](final-audit.json)，发布附件及解压运行验证见[发布验证](../'+version+'-release/README.md)。\n')
save(f'artifacts/validation/{version}/README.md',validation_page)
release_page=(f'# {version} 发布附件验证\n\n'
    f'本版对应[中文说明](../../../docs/releases/v{version}.md)。主库、训练库、33模型、网页运行包、图文包及SHA256SUMS全部随Release提供。\n\n'
    '- 发布核心DLL与本轮176窗口性能使用的冻结CLI目录DLL逐字节相同。\n'
    '- 原模型/许可证与manifest全部匹配；网页静态文件与当前源码逐字节一致。\n'
    '- 文档ZIP保持相对图片目录，未包含的历史资料改为版本化GitHub链接。\n'
    '- 发布说明以UTF-8文件发送并回读逐字符核验，附件上传核对GitHub SHA256及长度。\n\n'
    '[文档与排名审查](../../../docs/articles/manifest-'+version+'.json) · [本版功能验证](../'+version+'/README.md) · [附件清单与摘要](package-assets.json) · [解压网页验证](web-package-smoke.json)。\n')
save(f'artifacts/validation/{version}-release/README.md',release_page)
print(json.dumps({'status':'prepared','version':version,'windows':windows,'frames':frames,'checks':checks}))
