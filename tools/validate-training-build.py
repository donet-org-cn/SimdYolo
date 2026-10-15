"""构建、训练生命周期/推理回归、最终数值矩阵、中文XML审查和纯NuGet消费验证。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent.parent
os.chdir(ROOT)
directory=ROOT/'artifacts/validation/0.10.0'
directory.mkdir(parents=True,exist_ok=True)
environment=os.environ.copy()
environment['DOTNET_CLI_UI_LANGUAGE']='en-US'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(name,command):
    result=subprocess.run(command,text=True,encoding='utf-8',errors='replace',capture_output=True,env=environment)
    (directory/(name+'.log')).write_text(json.dumps(command)+'\n'+result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        print(result.stdout[-6000:]+result.stderr[-6000:],flush=True)
        raise SystemExit(result.returncode)
    print('PASS '+name+': '+result.stdout.strip().splitlines()[-1],flush=True)
legacy=ROOT/'.cache/training-validation/integration/last.sytrain'
legacy_copy=ROOT/'.cache/training-010/legacy-0.9.0.sytrain'
if legacy.exists() and not legacy_copy.exists():
    with zipfile.ZipFile(legacy) as archive:
        if json.loads(archive.read('manifest.json'))['Format']=='SimdYolo.Training/1':
            shutil.copyfile(legacy,legacy_copy)
resume_audits='--resume-audits' in sys.argv
if not resume_audits:
    run('build',['dotnet','build','SimdYolo.sln','-c','Release','--no-restore','-v','minimal'])
core=ROOT/'src/SimdYolo/bin/Release/net10.0/SimdYolo.dll'
training=ROOT/'src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll'
performance=json.loads((directory/'performance/training-runtime-comparison.json').read_text(encoding='utf-8'))
assert performance['status']=='passed'
assert all(r['coreAssemblySha256']==sha(core) and r['trainingAssemblySha256']==sha(training) for r in performance['runs'] if r['engineKey'].startswith('net-'))
test='tests/SimdYolo.Training.Tests/bin/Release/net10.0/SimdYolo.Training.Tests.dll'
if not resume_audits:
    run('training-tests',['dotnet',test,'--integration','--vulkan'])
    run('inference-tests',['dotnet','tests/SimdYolo.Tests/bin/Release/net10.0/SimdYolo.Tests.dll','--vulkan','--models','models/pt'])
    run('integration-cpu',['dotnet',test,'test-matrix','.cache/training-010/cpu','--epochs','100'])
for device in ['cpu','vulkan']:
    source=ROOT/f'.cache/training-010/{device}/integration.json'
    value=json.loads(source.read_text(encoding='utf-8'))
    assert value['status']=='passed' and value['coreAssemblySha256']==sha(core) and value['trainingAssemblySha256']==sha(training)
    shutil.copyfile(source,directory/f'integration-{device}.json')
# 把旧开发日志留在缓存中；本目录的日志只保留最终程序集的一次完整矩阵。
if not resume_audits and (directory/'oracle-run.log').exists():
    shutil.copyfile(directory/'oracle-run.log',ROOT/'.cache/training-010/oracle-development.log')
    (directory/'oracle-run.log').unlink()
if not resume_audits:
    run('oracle-final',[sys.executable,'tools/run-training-matrix.py'])
else:
    matrix=json.loads((directory/'oracle-matrix.json').read_text(encoding='utf-8'))
    assert matrix['status']=='passed' and matrix['totalGroups']==34
for name,dll in [('core',core),('training',training)]:
    run(name+'-xml',['dotnet','run','--project','tools/PackageAudit','-c','Release','--',str(dll),str(dll.with_suffix('.xml')),str(directory/(name+'-xml-audit.json'))])
    value=json.loads((directory/(name+'-xml-audit.json')).read_text(encoding='utf-8-sig'))
    assert not value['DocumentationProblems'] and not value['UndocumentedPublicSymbols'] and value['IlOnly']
for project in ['SimdYolo','SimdYolo.Training']:
    run('pack-'+project,['dotnet','pack',f'src/{project}/{project}.csproj','-c','Release','--no-build','--no-restore','-o','artifacts'])
run('consumer-restore',['dotnet','restore','tools/TrainingPackageConsumer','--source','artifacts','--packages','.cache/training-package-feed-010'])
run('consumer-build',['dotnet','build','tools/TrainingPackageConsumer','-c','Release','--no-restore','-v','minimal'])
consumers=[]
for model in ['yolov8n','yolov9t','yolov10n','yolo11n','yolo26n']:
    for device in ['Cpu','Vulkan']:
        output=ROOT/f'.cache/training-010/consumer-{model}-{device}'
        run('consumer-'+model+'-'+device,['dotnet','tools/TrainingPackageConsumer/bin/Release/net10.0/TrainingPackageConsumer.dll',f'models/pt/{model}.pt',str(output),device])
        value=json.loads((output/'consumer.json').read_text(encoding='utf-8'))
        assert value['CoreAssemblySha256']==sha(core) and value['TrainingAssemblySha256']==sha(training)
        consumers.append(dict(model=model,**value))
if legacy_copy.exists():
    output=ROOT/'.cache/training-010/consumer-legacy'
    run('consumer-legacy',['dotnet','tools/TrainingPackageConsumer/bin/Release/net10.0/TrainingPackageConsumer.dll',str(legacy_copy),str(output),'Cpu'])
    consumers.append(dict(model='legacy /1 archive',sourceSha256=sha(legacy_copy),**json.loads((output/'consumer.json').read_text(encoding='utf-8'))))
(directory/'package-consumers.json').write_text(json.dumps(dict(status='passed',results=consumers),indent=2)+'\n',encoding='utf-8')
packages={p.name:sha(p) for p in [ROOT/'artifacts/SimdYolo.0.10.0.nupkg',ROOT/'artifacts/SimdYolo.Training.0.10.0.nupkg']}
(directory/'packages.json').write_text(json.dumps(dict(status='passed',coreAssemblySha256=sha(core),trainingAssemblySha256=sha(training),packages=packages),indent=2)+'\n',encoding='utf-8')
print('Final build, numerical matrix, integration and package validation passed.',flush=True)
