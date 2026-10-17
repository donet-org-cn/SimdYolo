"""Bind final source, packages, tested DLLs, UI and unedited measurement evidence."""
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'artifacts/validation/0.11.1'


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def load(file):
    return json.loads((OUT/file).read_text(encoding='utf-8-sig'))


# PowerShell 5 creates UTF16 logs; Python defaults to CRLF. Keep new evidence UTF8/LF.
texts = [file for file in OUT.rglob('*') if file.suffix in ['.log','.json','.md']]
texts.append(ROOT/'artifacts/image-pipeline-optimization-0.11.1.md')
for log in texts:
    raw = log.read_bytes()
    value = raw.decode('utf-16') if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else raw.decode('utf-8-sig')
    log.write_text(value.replace('\r',''), encoding='utf-8', newline='\n')

core = digest(ROOT/'src/SimdYolo/bin/Release/net10.0/SimdYolo.dll')
training = digest(ROOT/'src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll')
performance = load('image-performance.json')
assert performance['status'] == 'passed' and len(performance['records']) == 24
assert len(performance['correctness']) == 12
assert all(row['coreAssemblySha256'] == core for row in performance['records'] if row['variant'] == 'optimized')
assert all(row['coreAssemblySha256'] == performance['baselineSha256'] for row in performance['records'] if row['variant'] == 'baseline')
assert load('package-audit.json')['Status'] == 'passed'
metadata = load('training-metadata.json')
assert metadata['IlOnly'] and metadata['ExternalAssemblyReferences'] == ['SimdYolo']
assert not metadata['DocumentationProblems'] and not metadata['UndocumentedPublicSymbols'] and not metadata['UnmappedXmlMemberIds']
for backend in ['cpu','vulkan']:
    consumer = load('package-consumer-'+backend+'.json')
    assert consumer['status'] == 'passed' and consumer['coreSha256'] == core and consumer['trainingSha256'] == training
assert load('image-web.json')['status'] == 'passed'
assert load('web/web-browser.json')['status'] == 'passed'
assert len(load('web/web-browser.json')['checks']) == 41
for filename, count in [('inference-tests.log',116),('training-tests.log',19),('masks-software.log',2)]:
    assert str(count)+' '+('training tests passed.' if filename=='training-tests.log' else 'tests passed.') in (OUT/filename).read_text(encoding='utf-8')
packages = []
for name, expected in [('SimdYolo',core),('SimdYolo.Training',training)]:
    file = ROOT/f'artifacts/{name}.0.11.1.nupkg'
    with zipfile.ZipFile(file) as archive:
        assert hashlib.sha256(archive.read('lib/net10.0/'+name+'.dll')).hexdigest() == expected
        spec = ET.fromstring(archive.read(name+'.nuspec'))
        dependencies = [{**node.attrib} for node in spec.iter() if node.tag.endswith('dependency')]
        assert not dependencies if name == 'SimdYolo' else len(dependencies)==1 and dependencies[0]['id']=='SimdYolo' and dependencies[0]['version']=='0.11.1'
    packages.append(dict(file=str(file.relative_to(ROOT)).replace('\\','/'),sha256=digest(file),dependencies=dependencies))
files = subprocess.check_output(['git','-c','safe.directory='+str(ROOT).replace('\\','/'),'ls-files','--cached','--others','--exclude-standard','--','src','tests','examples','tools'],cwd=ROOT,text=True).splitlines()
sources = {file:digest(ROOT/file) for file in sorted(set(files)) if Path(file).suffix in ['.cs','.csproj','.mjs','.py','.html','.css','.md']}
evidence = {str(file.relative_to(ROOT)).replace('\\','/'):digest(file) for file in sorted(OUT.rglob('*')) if file.is_file() and file.name!='manifest.json'}
local = json.load(urlopen('http://127.0.0.1:5188/api/info'))
assert local['version']=='0.11.1' and local['binaryRgb']
report = dict(status='passed',recordedUtc=datetime.now(timezone.utc).isoformat(),
              gitBaseCommit=subprocess.check_output(['git','-c','safe.directory='+str(ROOT).replace('\\','/'),'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              coreAssemblySha256=core,trainingAssemblySha256=training,baselineAssemblySha256=performance['baselineSha256'],packages=packages,
              sourceFiles=sources,evidenceFiles=evidence,localWeb=dict(url='http://127.0.0.1:5188',version=local['version'],binaryRgb=local['binaryRgb']),
              afterTiming='Final web-only changes add documentation and retain acceptance of valid unpadded Base64. Server responses remain padded. The final 15 transport/cache checks were rerun against port5188, including unpadded masks. All measured managed core DLL bytes match the packaged core. Core/training NuGet README updates do not change these consumed DLL bytes.',
              drawingDependency='Matplotlib 3.11.2 in an isolated workspace-only development target; not included in application packages.')
(OUT/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
print(json.dumps(dict(status='passed',core=core,training=training,sourceFiles=len(sources),evidenceFiles=len(evidence))))
