"""验证网页五代训练模型列表、CPU/Vulkan 训练、恢复、下载上传和原生推理。"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import zipfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:5188')
parser.add_argument('--output',type=Path,default=Path('artifacts/validation/0.10.0/web-matrix.json'))
args=parser.parse_args()
base=args.url.rstrip('/')
def api(path,value=None):
    request=Request(base+path,data=None if value is None else json.dumps(value).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=120) as response:
        return json.load(response)
def wait(job):
    start=time.monotonic()
    while time.monotonic()-start<180:
        value=api('/api/training/'+job['id'])
        if value['status'] in ['completed','cancelled','failed']:
            assert value['status']=='completed',value
            return value
        time.sleep(.2)
    raise AssertionError('Training timeout')
info=api('/api/info')
assert info['version']=='0.10.0'
names=['yolov8n.pt','yolov9t.pt','yolov10n.pt','yolo11n.pt','yolo26n.pt']
original=[m for m in info['models'] if m['name'] in names]
assert len(original)==5 and all(m['trainable'] for m in original)
assert not any(m['trainable'] for m in info['models'] if m['format']=='ONNX' or m['task']!='Detection')
pixels=bytes([25,40,55])*64*64
dataset=api('/api/datasets',dict(name='Five-family API validation',classes=['object-a','object-b'],images=[dict(name='fixture',width=64,height=64,pixels=base64.b64encode(pixels).decode(),labels='0 .4 .4 .5 .5\n1 .6 .6 .4 .4')]))
results=[]
for model in sorted(original,key=lambda m:names.index(m['name'])):
    for backend in ['Cpu','Vulkan']:
        run=wait(api('/api/training',dict(modelId=model['id'],datasetId=dataset['id'],backend=backend,imageSize=64,epochs=2,batchSize=1,learningRate=.003,seed=26)))
        assert run['step']==2 and len(run['history'])==2 and run['checkpointUrl']
        with urlopen(base+run['checkpointUrl']) as response:
            checkpoint=response.read()
        with zipfile.ZipFile(io.BytesIO(checkpoint)) as archive:
            manifest=json.loads(archive.read('manifest.json'))
            assert manifest['Format']=='SimdYolo.Training/2' and manifest['EndToEnd']==(model['name'] in ['yolov10n.pt','yolo26n.pt'])
            assert archive.getinfo('LICENSE-AGPL-3.0.txt').file_size>100
        # 以不含 YOLO 字样的文件名重新上传，验证能力识别不依赖文件名。
        with urlopen(Request(base+'/api/models',data=checkpoint,headers={'X-Filename':'renamed.sytrain','Content-Type':'application/octet-stream'}),timeout=120) as response:
            uploaded=json.load(response)
        assert uploaded['trainable']
        prediction=api('/api/infer',dict(modelId=uploaded['id'],backend=backend,width=64,height=64,pixels=base64.b64encode(pixels).decode(),confidence=.001))
        assert prediction['task']=='Detection' and prediction['elapsedMs']>0
        resumed=wait(api('/api/training/'+run['id']+'/resume',dict(epochs=3)))
        assert resumed['step']==3 and resumed['epoch']==3
        results.append(dict(model=model['name'],backend=backend,steps=run['step'],resumedSteps=resumed['step'],downloadUpload=True,nativeInference=True,checkpointSha256=hashlib.sha256(checkpoint).hexdigest(),passed=True))
        print('PASS',model['name'],backend,flush=True)
report=dict(status='passed',version=info['version'],trainableOriginalModels=[m['name'] for m in original],results=results)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
