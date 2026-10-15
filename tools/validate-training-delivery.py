"""核对训练交付的最终程序集、包、数值/网页证据、模型来源与UTF8文档链接。"""
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent.parent
directory=ROOT/'artifacts/validation/0.10.0'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
core=ROOT/'src/SimdYolo/bin/Release/net10.0/SimdYolo.dll'
training=ROOT/'src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll'
core_sha,training_sha=sha(core),sha(training)
for name in ['core','training']:
    audit=read(directory/f'{name}-xml-audit.json')
    assert audit['Version']=='0.10.0' and audit['IlOnly']
    assert not audit['DocumentationProblems'] and not audit['UndocumentedPublicSymbols']
    allowed={'SimdYolo'} if name=='training' else set()
    assert all(ref.startswith('System') or ref in allowed for ref in audit['AssemblyReferences'])
    with zipfile.ZipFile(ROOT/f'artifacts/{"SimdYolo" if name=="core" else "SimdYolo.Training"}.0.10.0.nupkg') as package:
        module='SimdYolo' if name=='core' else 'SimdYolo.Training'
        assert hashlib.sha256(package.read(f'lib/net10.0/{module}.dll')).hexdigest()==(core_sha if name=='core' else training_sha)
        assert package.read(f'lib/net10.0/{module}.xml')==(core if name=='core' else training).with_suffix('.xml').read_bytes()
        assert package.read('README.md')==(ROOT/('README.md' if name=='core' else 'docs/training.md')).read_bytes()
package_record=read(directory/'packages.json')
assert all(sha(ROOT/'artifacts'/name)==expected for name,expected in package_record['packages'].items())
models=read(ROOT/'models/manifest.json')['Models']
assert len(models)==33
for model in models:
    path=ROOT/model['File']
    assert sha(path)==model['Sha256'] and path.stat().st_size==model['SizeBytes']
    assert model['License'].startswith('AGPL-3.0')
matrix=read(directory/'oracle-matrix.json')
assert matrix['status']=='passed' and matrix['totalGroups']==34 and matrix['totalChecks']==19620
for r in matrix['results']:
    value=read(directory/r['report'])
    assert value['status']=='passed' and value['failedChecks']==0
    assert value['coreAssemblySha256']==core_sha and value['trainingAssemblySha256']==training_sha
    assert value['sourceModelSha256']==sha(value['sourceModel'])
extra=0
for path in (directory/'performance').glob('fixture-check-*.json'):
    value=read(path)
    assert value['status']=='passed' and value['failedChecks']==0
    assert value['coreAssemblySha256']==core_sha and value['trainingAssemblySha256']==training_sha
    extra+=value['totalChecks']
assert extra==2712
performance=read(directory/'performance/training-runtime-comparison.json')
assert performance['status']=='passed' and len(performance['runs'])==40 and sum(r['steps'] for r in performance['runs'])==20480
for device in ['cpu','vulkan']:
    integration=read(directory/f'integration-{device}.json')
    assert integration['status']=='passed' and len(integration['results'])==5
    assert integration['coreAssemblySha256']==core_sha and integration['trainingAssemblySha256']==training_sha
    assert all(r['passed'] and r['steps']==100 and r['allBranchesUpdated'] and r['loadedMomentumBitwiseEqual'] and r['after']<r['before']*.9 for r in integration['results'])
assert '16 training tests passed.' in (directory/'training-tests.log').read_text(encoding='utf-8')
assert '117 total tests passed.' in (directory/'inference-tests.log').read_text(encoding='utf-8')
consumers=read(directory/'package-consumers.json')
assert consumers['status']=='passed' and len(consumers['results'])>=10
assert all(r['Status']=='passed' and r['CoreAssemblySha256']==core_sha and r['TrainingAssemblySha256']==training_sha for r in consumers['results'])
for name in ['web-matrix.json','web-api.json','web-browser/web-browser.json']:
    assert read(directory/name)['status']=='passed'
assert len(read(directory/'web-matrix.json')['results'])==10
assert not read(directory/'web-browser/web-browser.json')['errors']
documents=['README.md','VALIDATION.md','docs/training.md','docs/articles/README.md','docs/articles/wechat.md','docs/articles/cnblogs.md','examples/SimdYolo.Web/README.md','artifacts/training-evaluation-0.10.0.md','artifacts/validation/0.10.0/README.md']
for name in documents:
    path=ROOT/name;text=path.read_text(encoding='utf-8-sig')
    assert '\ufffd' not in text and '????' not in text and text.count('```')%2==0,name
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' in target or target.startswith('#'):continue
        assert (path.parent/target.split('#')[0]).exists(),(name,target)
assets=[]
for stem in ['training-performance-0.10.0-cpu','training-performance-0.10.0-gpu','training-loss-0.10.0']:
    for suffix in ['png','svg']:
        path=ROOT/f'artifacts/{stem}.{suffix}'
        if suffix=='svg':ET.parse(path)
        else:
            data=path.read_bytes();assert data.startswith(b'\x89PNG\r\n\x1a\n') and struct.unpack('>II',data[16:24])[0]>=1000
        assets.append(path)
for path in directory.glob('performance/*/measur*'):
    # 仅清除已完成本轮的短期采样标记，解析绝对路径限制在本目录内。
    assert path.resolve().is_relative_to(directory.resolve()) and path.name in ['measuring','measured']
    path.unlink()
frozen={'SimdYolo.0.9.0.nupkg':'5a2b28313c962d63b789083b590517b969fba3a0a6ce285a5719fa07b73bff3c','SimdYolo.Training.0.9.0.nupkg':'41b76c9a56da1f62e98cfa9493fc690f0c7b9ed33ae373fc6652244d9cb1db42'}
assert all(sha(ROOT/'artifacts'/name)==expected for name,expected in frozen.items())
diff=subprocess.run(['git','-c','safe.directory=F:/simdyolo','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check'],cwd=ROOT,capture_output=True,text=True)
assert diff.returncode==0,diff.stdout[:2000]+diff.stderr[:2000]
files=[ROOT/name for name in documents]+assets
files+=list(directory.rglob('*.json'))+list(ROOT.glob('src/SimdYolo.Training/*.cs'))+[ROOT/'src/SimdYolo/TrainingHeadSchema.cs',ROOT/'src/SimdYolo/TrainingModelState.cs',ROOT/'src/SimdYolo/PtModel.Training.cs',ROOT/'src/SimdYolo/ModelInfo.cs',ROOT/'src/SimdYolo/PickleReader.cs']
files+=[p for p in (ROOT/'tools').glob('*training*.py')]+list((ROOT/'tests/SimdYolo.Training.Tests').glob('*.cs'))
files+=[ROOT/'examples/SimdYolo.Web'/name for name in ['Workspace.cs','TrainingService.cs','Program.cs','wwwroot/index.html','wwwroot/app.js']]
manifest_path=directory/'final-audit.json'
manifest=dict(status='passed',sourceVersion='0.10.0',baselineCommit=subprocess.check_output(['git','-c','safe.directory=F:/simdyolo','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),workingTreeContainsImplementation=True,
              coreAssemblySha256=core_sha,trainingAssemblySha256=training_sha,numericGroups=39,numericChecks=22332,trainingRegressionTests=16,inferenceRegressionTests=117,
              integrationConfigurations=10,epochsPerConfiguration=100,performanceWindows=40,timedUpdates=20480,webTrainingConfigurations=10,packageConsumerConfigurations=len(consumers['results']),modelsVerified=33,
              chineseXmlProblems=0,utf8AndRelativeLinks=True,previousPackagesUnchanged=True,
              files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files)) if p!=manifest_path})
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:manifest[k] for k in ['status','numericChecks','trainingRegressionTests','inferenceRegressionTests','performanceWindows','timedUpdates','modelsVerified']}))
