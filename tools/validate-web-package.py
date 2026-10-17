"""在独立目录解压实际网页/模型ZIP，启动.NET服务并验证真实分割推理。"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',required=True);parser.add_argument('--port',type=int,default=5203)
args=parser.parse_args(); version=args.version
directory=(ROOT/f'.cache/release-{version}/unpacked').resolve(); assert directory.is_relative_to(ROOT)
directory.mkdir(parents=True,exist_ok=True)
for name in [f'SimdYolo-Web-v{version}.zip',f'SimdYolo-models-v{version}.zip']:
    with zipfile.ZipFile(ROOT/'artifacts'/name) as archive:
        for entry in archive.infolist():
            assert (directory/entry.filename).resolve().is_relative_to(directory)
        archive.extractall(directory)
base='http://127.0.0.1:'+str(args.port)
def request(path,body=None,headers=None):
    with urlopen(Request(base+path,data=body,headers=headers or {}),timeout=120) as response:
        return response.read()
log=ROOT/f'.cache/release-{version}/unpacked-web.log'
command=['dotnet',str(directory/'SimdYolo.Web.dll'),'--urls',base,'--contentRoot',str(directory),'--SimdYolo:RepositoryRoot',str(directory),'--SimdYolo:DataRoot',str(directory/'data')]
with log.open('w',encoding='utf-8') as stdout:
    child=subprocess.Popen(command,cwd=directory,stdout=stdout,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        start=time.monotonic(); info=None
        while time.monotonic()-start<60:
            if child.poll() is not None: raise RuntimeError('Unpacked web exited; see '+str(log))
            try: info=json.loads(request('/api/info'));break
            except URLError: time.sleep(.2)
        assert info and info['version']==version and info['binaryRgb'] and len(info['models'])==33
        assert request('/api/sample').startswith(b'\x89PNG\r\n\x1a\n')
        assert b'SimdYolo' in request('/')
        static={}
        for name in ['app.js','cutout.js','annotation.js','workflows.js','index.html']:
            actual=request('/'+name)
            assert actual==(directory/'wwwroot'/name).read_bytes()
            static[name]=hashlib.sha256(actual).hexdigest()
        pixels=(ROOT/'.cache/image-optimization/fixtures/bus.rgb').read_bytes(); cases=[]
        for model_name,backend in [('yolov8n-seg.pt','Cpu'),('yolo26n-seg.onnx','Vulkan')]:
            model=next(item for item in info['models'] if item['name']==model_name)
            metadata={'modelId':model['id'],'backend':backend,'width':810,'height':1080,'confidence':.25}
            result=json.loads(request('/api/infer-rgb',pixels,{'Content-Type':'application/octet-stream','X-SimdYolo':json.dumps(metadata)}))
            assert result['instances'] and result['backend'].startswith('CPU' if backend=='Cpu' else 'Vulkan')
            for item in result['instances']:
                mask=item['mask']; assert mask and len(base64.b64decode(mask['probabilities']))==mask['width']*mask['height']
            cases.append({'model':model_name,'backend':result['backend'],'instances':len(result['instances']),'elapsedMs':result['elapsedMs']})
        report={'status':'passed','version':version,'repositoryRoot':str(directory),'sourceRepositoryFallback':False,'models':33,'staticSha256':static,'inference':cases,'runtime':'dotnet published DLL, actual extracted archives'}
        output=ROOT/f'artifacts/validation/{version}-release/web-package-smoke.json'
        output.write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
        print(json.dumps(report))
    finally:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill();child.wait()
