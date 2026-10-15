"""网页版真实 API 联调；仅开发验证依赖 Pillow，应用运行不依赖 Python。"""
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
import base64, hashlib, io, json, time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:5188')
parser.add_argument('--image', type=Path, required=True, help='用于检测、分割、姿态验证的真实照片，例如官方 bus.jpg')
parser.add_argument('--output', type=Path, default=Path('artifacts/validation/0.9.0/web-api.json'))
args=parser.parse_args()
BASE=args.url.rstrip('/')
def api(path, value=None):
    data=None if value is None else json.dumps(value).encode()
    request=Request(BASE+path,data=data,headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=120) as response:
        return json.load(response)
def wait(job):
    start=time.monotonic()
    while time.monotonic()-start<120:
        value=api('/api/training/'+job['id'])
        if value['status'] in ('completed','cancelled','failed'):
            assert value['status']!='failed',value
            return value
        time.sleep(.25)
    raise AssertionError('Training did not finish')

info=api('/api/info'); assert info['version']=='0.9.0' and len(info['models'])>=33
model=next(m for m in info['models'] if m['name']=='yolov8n.pt')
with urlopen(BASE+'/api/sample') as response:
    picture=Image.open(io.BytesIO(response.read())).convert('RGB')
pixels=base64.b64encode(picture.tobytes()).decode()
dataset=api('/api/datasets',{'name':'API integration square','classes':['square'],'images':[{'name':'square.png','width':picture.width,'height':picture.height,'pixels':pixels,'labels':'0 .5 .5 .5 .5'}]})
run=api('/api/training',{'modelId':model['id'],'datasetId':dataset['id'],'backend':'Cpu','imageSize':64,'epochs':2,'batchSize':1,'learningRate':.003,'seed':26})
completed=wait(run); assert completed['status']=='completed' and len(completed['history'])==2 and completed['step']==2
assert completed['checkpointUrl'] and completed['modelId']!=model['id']
with urlopen(BASE+completed['checkpointUrl']) as response:
    checkpoint=response.read()
assert checkpoint[:2]==b'PK'
inference=api('/api/infer',{'modelId':completed['modelId'],'backend':'Cpu','width':picture.width,'height':picture.height,'pixels':pixels,'confidence':.001})
assert inference['task']=='Detection' and inference['elapsedMs']>0
repeated=api('/api/infer',{'modelId':completed['modelId'],'backend':'Cpu','width':picture.width,'height':picture.height,'pixels':pixels,'confidence':.001})
assert repeated['reusedSession'] and repeated['modelLoadMs']==0
with urlopen(Request(BASE+'/api/models',data=checkpoint,headers={'X-Filename':'web-api-checkpoint.sytrain','Content-Type':'application/octet-stream'}),timeout=120) as response:
    uploaded=json.load(response)
assert uploaded['format']=='SYTRAIN' and uploaded['trainable']
resumed=wait(api('/api/training/'+run['id']+'/resume',{'epochs':3}))
assert resumed['status']=='completed' and resumed['step']==3 and resumed['epoch']==3

# A real photograph exercises detection, segmentation and pose output mapping.
photograph=Image.open(args.image).convert('RGB')
raw=base64.b64encode(photograph.tobytes()).decode()
predictions=[]
for name, backend in [('yolov8n.pt','Cpu'),('yolov8n-seg.pt','Cpu'),('yolov8n-pose.pt','Cpu'),('yolo26n.onnx','Vulkan')]:
    selected=next(m for m in info['models'] if m['name']==name)
    result=api('/api/infer',{'modelId':selected['id'],'backend':backend,'width':photograph.width,'height':photograph.height,'pixels':raw,'confidence':.25})
    assert result['instances'],(name,result)
    if 'seg' in name: assert any(i['mask'] for i in result['instances'])
    if 'pose' in name: assert all(len(i['keypoints'])==17 for i in result['instances'])
    predictions.append({'model':name,'backend':result['backend'],'instances':len(result['instances']),'task':result['task'],'passed':True})

gpu_trained=False
if any(b['id']=='Vulkan' and b['training'] for b in info['backends']):
    gpu=wait(api('/api/training',{'modelId':model['id'],'datasetId':dataset['id'],'backend':'Vulkan','imageSize':64,'epochs':1,'batchSize':1,'learningRate':.003,'seed':26}))
    assert gpu['status']=='completed' and 'Vulkan' in gpu['backend']
    gpu_trained=True

running=api('/api/training',{'modelId':model['id'],'datasetId':'demo-squares','backend':'Cpu','imageSize':128,'epochs':100,'batchSize':1,'learningRate':.003,'seed':26})
time.sleep(.3); api('/api/training/'+running['id']+'/cancel',{})
cancelled=wait(running); assert cancelled['status']=='cancelled'
try:
    api('/api/infer',{'modelId':model['id'],'backend':'Cpu','width':2,'height':2,'pixels':'AAAA','confidence':.25})
    raise AssertionError('Malformed RGB was accepted')
except HTTPError as error:
    assert error.code==400
with urlopen(BASE+'/api/sample') as response:
    assert response.headers['Content-Type'].startswith('image/png')
report={'status':'passed','version':info['version'],'models':len(info['models']),'trainingSteps':completed['step'],'resumedSteps':resumed['step'],'checkpointSha256':hashlib.sha256(checkpoint).hexdigest(),
        'checkpointInference':True,'cancelledStatus':cancelled['status'],'predictions':predictions,'invalidRgbRejected':True,'reusedInferenceSession':repeated['reusedSession'],'uploadedCheckpoint':True,'vulkanTraining':gpu_trained}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
