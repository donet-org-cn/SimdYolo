"""核对最终训练优化交付的DLL、模型、数值/性能/模型结果、中文文档和图链。"""
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'artifacts/validation/0.10.1'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
summary=read(DIRECTORY/'validation-summary.json')
assert summary['status']=='passed'
for key,path in [('coreAssemblySha256','src/SimdYolo/bin/Release/net10.0/SimdYolo.dll'),('trainingAssemblySha256','src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll')]:
    assert summary[key]==sha(ROOT/path)
for path,digest in summary['evidence'].items(): assert sha(ROOT/path)==digest,path
perf=read(DIRECTORY/'performance/training-runtime-comparison.json')
assert perf['status']=='passed' and len(perf['runs'])==40 and sum(r['steps'] for r in perf['runs'])==20480
assert {r['engineKey'] for r in perf['runs']}=={'net-cpu','torch-cpu','net-vulkan','torch-cuda'}
readme=(ROOT/'README.md').read_text(encoding='utf-8-sig')
for model in perf['models']:
    values=[statistics.median(r['medianMs'] for r in perf['runs'] if Path(r['model']).stem==model and r['engineKey']==engine) for engine in ['torch-cpu','net-cpu','torch-cuda','net-vulkan']]
    expected=f'| {model} | {values[0]:.2f} | **{values[1]:.2f}** | {values[2]:.2f} | **{values[3]:.2f}** |'
    assert expected in readme,(model,'README training table differs')
for r in perf['runs']:
    assert r['passed'] and r['steps']==512 and r['size']==128 and r['threads']==4
    if r['engineKey'].startswith('net-'):
        assert all(r[k]==summary[k] for k in ['coreAssemblySha256','trainingAssemblySha256'])
outcomes=read(DIRECTORY/'outcomes/comparison.json')
assert outcomes['status']=='completed' and len(outcomes['results'])==10
for r in outcomes['results']:
    value=read(DIRECTORY/'outcomes'/r['report'])
    assert value['status']=='completed' and value['bothSavedModelsReloaded'] and value['steps']==192
    assert [s['step'] for s in value['snapshots']]==[1,16,96,192]
    assert all(s['passed'] for s in value['cpuParameterControl']) and value['metricComparisonPassed']
    assert value['directParameterComparisonPassed']==all(s['passed'] for s in value['snapshots'])
    assert value['torchTrainingDevice']==('cpu' if r['device']=='cpu' else 'cuda')
    assert value['trainImages']==96 and value['validationImages']==32
    assert all(value[k]==summary[k] for k in ['coreAssemblySha256','trainingAssemblySha256'])
models=read(ROOT/'models/manifest.json')['Models']
for model in models:
    path=ROOT/model['File']
    assert path.stat().st_size==model['SizeBytes'] and sha(path)==model['Sha256'].lower()
documents=['README.md','VALIDATION.md','docs/training.md','examples/SimdYolo.Web/README.md','docs/articles/README.md','docs/articles/wechat.md','docs/articles/cnblogs.md',
           'artifacts/training-optimization-0.10.1.md','artifacts/validation/0.10.1/README.md']
for name in documents:
    path=ROOT/name;text=path.read_text(encoding='utf-8-sig')
    assert '\ufffd' not in text and '????' not in text,name
    assert text.count('```')%2==0,name
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' in target or target.startswith('#'): continue
        assert (path.parent/target.split('#')[0]).exists(),(name,target)
for name in ['training-performance-0.10.1-cpu','training-performance-0.10.1-gpu','training-results-0.10.1','training-loss-parity-0.10.1']:
    assert (ROOT/f'artifacts/{name}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    ET.parse(ROOT/f'artifacts/{name}.svg')
# 删除本工具创建的计时哨兵文件；先验证最终绝对路径，绝不遍历删除缓存或仓库外目录。
for marker in (DIRECTORY/'performance').rglob('*'):
    if marker.name in {'measuring','measured'} and marker.is_file():
        assert marker.resolve().is_relative_to(DIRECTORY.resolve())
        marker.unlink()
check=subprocess.run(['git','-c','safe.directory=F:/simdyolo','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check'],capture_output=True,text=True)
assert check.returncode==0,check.stdout[:2000]
paths=set(documents)
paths.update(['docs/articles/manifest.json','src/SimdYolo/SimdYolo.csproj','src/SimdYolo.Training/SimdYolo.Training.csproj','examples/SimdYolo.Web/Program.cs','examples/SimdYolo.Web/SimdYolo.Web.csproj'])
paths.update(p.relative_to(ROOT).as_posix() for p in DIRECTORY.rglob('*') if p.is_file() and p.name!='final-audit.json')
# 发布附件由独立打包审计覆盖，避免文档ZIP包含本审计后产生摘要循环。
paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'artifacts').glob('*0.10.1*') if p.is_file() and p.suffix in {'.md','.json','.png','.svg'})
paths.update(p.relative_to(ROOT).as_posix() for folder in ['src/SimdYolo.Training','tests/SimdYolo.Training.Tests'] for p in (ROOT/folder).glob('*.cs'))
paths.update('tools/'+p for p in ['benchmark-training.py','compare-training-outcomes.py','validate-training-optimization.py','render-training-optimization.py','validate-training-optimization-delivery.py'])
value=dict(status='passed',version='0.10.1',**{k:v for k,v in summary.items() if k not in ['status','version','evidence']},modelsVerified=len(models),documentsVerified=len(documents),utf8AndRelativeLinks=True,
           baselineCommit=subprocess.check_output(['git','-c','safe.directory=F:/simdyolo','rev-parse','HEAD'],text=True).strip(),files={p:sha(ROOT/p) for p in sorted(paths)})
(DIRECTORY/'final-audit.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in value.items() if k!='files'}))
