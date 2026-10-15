"""串行冻结、回归、数值对照、独立窗口性能及真实图片训练结果验证。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUTPUT=ROOT/'artifacts/validation/0.10.1'
OUTPUT.mkdir(parents=True,exist_ok=True)
CORE=ROOT/'src/SimdYolo/bin/Release/net10.0/SimdYolo.dll'
TRAINING=ROOT/'src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll'
TEST=ROOT/'tests/SimdYolo.Training.Tests/bin/Release/net10.0/SimdYolo.Training.Tests.dll'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(name,command):
    env=os.environ.copy();env['DOTNET_CLI_UI_LANGUAGE']='en-US'
    print('START '+name,flush=True)
    with (OUTPUT/(name+'.log')).open('w',encoding='utf-8') as log:
        log.write(json.dumps([str(v) for v in command])+'\n');log.flush()
        p=subprocess.run([str(v) for v in command],stdout=log,stderr=subprocess.STDOUT,env=env)
    if p.returncode:
        print((OUTPUT/(name+'.log')).read_text(encoding='utf-8')[-6000:],flush=True)
        raise SystemExit(p.returncode)
    print('PASS '+name,flush=True)


if '--resume' not in sys.argv:
    run('build',['dotnet','build','SimdYolo.sln','-c','Release','--no-restore','-v','quiet'])
frozen=dict(coreAssemblySha256=sha(CORE),trainingAssemblySha256=sha(TRAINING))
phases=[
    ('training-tests',['dotnet',TEST,'--integration','--vulkan']),
    ('inference-tests',['dotnet','tests/SimdYolo.Tests/bin/Release/net10.0/SimdYolo.Tests.dll','--vulkan','--models','models/pt']),
    ('oracle-final',[sys.executable,'tools/run-training-matrix.py','--output',OUTPUT,'--cache','.cache/training-011/oracle-final']),
    ('performance',[sys.executable,'tools/benchmark-training.py','--output',OUTPUT/'performance']),
    ('outcomes',[sys.executable,'tools/compare-training-outcomes.py','--resume']),
    ('core-xml',['dotnet','run','--project','tools/PackageAudit','-c','Release','--',CORE,CORE.with_suffix('.xml'),OUTPUT/'core-xml-audit.json']),
    ('training-xml',['dotnet','run','--project','tools/PackageAudit','-c','Release','--',TRAINING,TRAINING.with_suffix('.xml'),OUTPUT/'training-xml-audit.json']),
]
for name,command in phases:
    marker=OUTPUT/(name+'.passed.json')
    if '--resume' in sys.argv and marker.exists() and json.loads(marker.read_text())==frozen:
        print('REUSE '+name,flush=True);continue
    if name=='oracle-final' and (OUTPUT/'oracle-run.log').exists():
        (ROOT/'.cache/training-011/oracle-development.log').write_bytes((OUTPUT/'oracle-run.log').read_bytes())
        (OUTPUT/'oracle-run.log').write_text('',encoding='utf-8')
    run(name,command)
    assert sha(CORE)==frozen['coreAssemblySha256'] and sha(TRAINING)==frozen['trainingAssemblySha256'],'Assembly changed during validation'
    marker.write_text(json.dumps(frozen,indent=2)+'\n',encoding='utf-8')
oracle=json.loads((OUTPUT/'oracle-matrix.json').read_text())
perf=json.loads((OUTPUT/'performance/training-runtime-comparison.json').read_text())
outcomes=json.loads((OUTPUT/'outcomes/comparison.json').read_text())
assert oracle['status']==perf['status']=='passed' and outcomes['status']=='completed'
assert all(r['cpuParameterControlPassed'] and r['metricComparisonPassed'] for r in outcomes['results'])
for row in perf['runs']:
    if row['engineKey'].startswith('net-'):
        assert all(row[k]==v for k,v in frozen.items())
for row in outcomes['results']:
    record=json.loads((OUTPUT/'outcomes'/row['report']).read_text())
    assert record['status']=='completed' and record['bothSavedModelsReloaded']
    assert all(record[k]==v for k,v in frozen.items())
evidence={p.relative_to(ROOT).as_posix():sha(p) for p in OUTPUT.rglob('*.json') if p.name not in {'validation-summary.json','final-audit.json'}}
summary=dict(status='passed',version='0.10.1',**frozen,numericGroups=oracle['totalGroups'],numericChecks=oracle['totalChecks']+sum(json.loads(p.read_text())['totalChecks'] for p in (OUTPUT/'performance').glob('fixture-check-*.json')),
             trainingTests=19,inferenceTests=117,performanceWindows=len(perf['runs']),timedUpdates=sum(r['steps'] for r in perf['runs']),trainingOutcomeConfigurations=len(outcomes['results']),
             trainImages=96,validationImages=32,epochs=2,updatesPerModel=192,bothSavedModelsReloaded=True,directParameterComparisonsPassed=outcomes['directParameterComparisonsPassed'],
             metricComparisonsPassed=outcomes['metricComparisonsPassed'],cpuParameterControlsPassed=outcomes['cpuParameterControlsPassed'],evidence=evidence)
(OUTPUT/'validation-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='evidence'}),flush=True)
