"""核验本版实测、测试、离线包消费与最终程序集，生成发布冻结清单。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import zipfile

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',required=True)
version=parser.parse_args().version; output=ROOT/f'artifacts/validation/{version}'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name): return json.loads((output/name).read_text(encoding='utf-8-sig'))

# 只规范本轮日志，旧包、旧性能记录及本轮性能JSON均保持原字节。
texts=[file for file in output.rglob('*') if file.suffix in ['.log','.json','.md']]
texts.append(ROOT/f'artifacts/runtime-comparison-{version}.md')
for file in texts:
    raw=file.read_bytes(); text=raw.decode('utf-16') if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else raw.decode('utf-8-sig')
    file.write_text(text.replace('\r',''),encoding='utf-8',newline='\n')
report_path=ROOT/f'artifacts/runtime-comparison-{version}.json'
report=json.loads(report_path.read_text(encoding='utf-8'))
assert report['Status']=='complete'
assert len(report['Records'])==len(report['Models'])*len(report['Engines'])*report['Repeats']==176
assert all(check['Passed'] for record in report['Records'] for check in record['Correctness'])
checks=sum(len(record['Correctness']) for record in report['Records'])
core=sha(ROOT/'src/SimdYolo/bin/Release/net10.0/SimdYolo.dll')
training=sha(ROOT/'src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll')
assert report['SimdYoloDllSha256']==core
assert sha(Path(report['NetCliPath']).with_name('SimdYolo.dll'))==core
assert all(record['Version'].removeprefix('SimdYolo ').split('+')[0]==version for record in report['Records'] if record['Engine'].startswith('net-'))
for filename,count,phrase in [('inference-tests.log',116,'tests passed.'),('training-tests.log',19,'training tests passed.'),('masks-software.log',2,'tests passed.')]:
    assert str(count)+' '+phrase in (output/filename).read_text(encoding='utf-8')
package_audit=load('package-audit.json'); assert package_audit['Status']=='passed' and not package_audit['Issues']
metadata=load('training-metadata.json')
assert metadata['IlOnly'] and metadata['ExternalAssemblyReferences']==['SimdYolo']
assert not metadata['DocumentationProblems'] and not metadata['UndocumentedPublicSymbols'] and not metadata['UnmappedXmlMemberIds']
for backend in ['cpu','vulkan']:
    value=load(f'package-consumer-{backend}.json')
    assert value['status']=='passed' and value['version']==version
    assert value['coreSha256']==core and value['trainingSha256']==training
browser=load('web/web-browser.json'); assert browser['status']=='passed' and not browser['errors'] and len(browser['checks'])==41
assert load('image-web.json')['status']=='passed'
packages=[]
for name,expected in [('SimdYolo',core),('SimdYolo.Training',training)]:
    file=ROOT/f'artifacts/{name}.{version}.nupkg'
    with zipfile.ZipFile(file) as archive: assert hashlib.sha256(archive.read('lib/net10.0/'+name+'.dll')).hexdigest()==expected
    packages.append({'file':file.relative_to(ROOT).as_posix(),'sha256':sha(file)})
old=json.loads((ROOT/'artifacts/validation/0.11.1/manifest.json').read_text(encoding='utf-8'))
assert all(sha(ROOT/item['file'])==item['sha256'] for item in old['packages'])
files=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'ls-files','--cached','--others','--exclude-standard','--','src','tests','tools','examples','docs','README.md','VALIDATION.md'],cwd=ROOT,text=True).splitlines()
sources={name:sha(ROOT/name) for name in sorted(set(files)) if Path(name).suffix in ['.cs','.csproj','.py','.mjs','.md','.html','.css']}
result={'status':'passed','version':version,'recordedUtc':datetime.now(timezone.utc).isoformat(),
        'coreAssemblySha256':core,'trainingAssemblySha256':training,
        'assemblies':{'SimdYolo.dll':core,'SimdYolo.Training.dll':training},
        'scope':'Frozen runtime DLLs, full inference matrix, original-output correctness, runtime regressions, managed/XML audit and actual offline NuGet consumption. Release documentation/assets/publication audited separately.',
        'inferenceTests':116,'trainingTests':19,'softwareMaskTests':2,'browserChecks':41,'imageWebChecks':15,
        'performanceWindows':176,'timedInferences':sum(row['Iterations'] for row in report['Records']),
        'correctnessOutputs':checks,'benchmarkSha256':sha(report_path),'packages':packages,
        'old0111PackagesPreserved':True,'sourceFiles':sources}
(output/'final-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
print(json.dumps({'status':'passed','version':version,'core':core,'training':training,'windows':176,'checks':checks}))
