"""同初始状态、真实图片、固定顺序和 SGD 训练后，比较 PyTorch 与保存的 C# 模型。

COCO128 只用于实现诊断，显式分成 96 张微调图和 32 张验证图；不作为未见数据泛化结论。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT/'tests/SimdYolo.Training.Tests/bin/Release/net10.0/SimdYolo.Training.Tests.dll'
MODELS = ['yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo26n']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    sys.path.insert(0, str(ROOT/'.cache/reference'))
    from PIL import Image
    images = sorted((args.dataset/'images/train2017').glob('*.jpg'))
    assert len(images) == 128, 'Expected all 128 original images'
    random.Random(2611).shuffle(images)
    result = []
    rgb_dir = args.cache/'rgb'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images):
        image = Image.open(path).convert('RGB')
        rgb = rgb_dir/(path.stem+'.rgb')
        rgb.write_bytes(image.tobytes())
        labels_path = args.dataset/'labels/train2017'/(path.stem+'.txt')
        labels = labels_path.read_text() if labels_path.exists() else ''
        # 原数据小数舍入可能使边界略超图像；两边共用裁剪后的同一份标签，并记录原文件摘要。
        clean = []
        for line in labels.splitlines():
            cls, cx, cy, w, h = map(float, line.split())
            x1, y1, x2, y2 = max(0, cx-w/2), max(0, cy-h/2), min(1, cx+w/2), min(1, cy+h/2)
            if x2 > x1 and y2 > y1:
                clean.append(f'{int(cls)} {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}')
        result.append(dict(id=path.stem, split='train' if index < 96 else 'val', rgbPath=str(rgb.resolve()), width=image.width, height=image.height,
                           labels='\n'.join(clean), imageSha256=sha(path), labelSha256=sha(labels_path) if labels_path.exists() else None))
    value = dict(source='https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip', splitSeed=2611, trainImages=96, validationImages=32,
                 scope='fine-tuning holdout only; COCO train2017 is present in pretrained weights training data', images=result)
    manifest = args.cache/'dataset.json'
    manifest.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    return manifest


def worker(args):
    if args.device == 'cuda':
        sys.path.insert(0, str(ROOT/'.cache/torch-cu126'))
    sys.path.insert(0, str(ROOT/'.cache/yolo26-release-reference'))
    sys.path.append(str(ROOT/'.cache/reference'))
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.cache/yolo26-config')
    import numpy as np
    import torch
    import torch.nn as nn
    import ultralytics
    import ultralytics.nn.tasks as tasks
    import ultralytics.utils.loss as losses
    from types import SimpleNamespace
    assert ultralytics.__version__ == '8.4.0'
    tasks.YOLOv10DetectionModel = tasks.DetectionModel
    losses.v10DetectLoss = losses.E2EDetectLoss
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(26)
    fixture = json.loads((args.fixture/'fixture.json').read_text())
    assert fixture['sourceSha256'] == sha(fixture['sourceModel'])
    checkpoint = torch.load(fixture['sourceModel'], map_location='cpu', weights_only=False)
    model = (checkpoint.get('ema') or checkpoint['model']).float()
    head = model.model[-1]
    prefixes = ['cv2', 'cv3'] + (['one2one_cv2', 'one2one_cv3'] if fixture['headKind'] != 'Dfl' else [])
    state = head.state_dict()
    def read(d):
        return torch.from_numpy(np.fromfile(args.fixture/d['path'], dtype='<f4').reshape(d['shape']).copy())
    for d in fixture['parameters']:
        state[d['name']] = read(d).to(state[d['name']].dtype)
    head.load_state_dict(state, strict=True)
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    for name, p in head.named_parameters():
        p.requires_grad_(any(name.startswith(k+'.') for k in prefixes))
    model.to(args.device).eval()
    head.train()
    nc, bins = len(fixture['classNames']), fixture['regMax']
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([head])
            self.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5, epochs=fixture['epochs'])
    fake = FakeModel()
    criterion = losses.E2ELoss(fake) if fixture['headKind'] == 'ProgressiveEndToEnd' else losses.E2EDetectLoss(fake) if len(prefixes) == 4 else losses.v8DetectionLoss(fake)
    decay = [p for p in head.parameters() if p.requires_grad and p.ndim == 4]
    other = [p for p in head.parameters() if p.requires_grad and p.ndim != 4]
    optimizer = torch.optim.SGD([dict(params=decay, weight_decay=fixture['weightDecay']), dict(params=other, weight_decay=0)], lr=fixture['learningRate'], momentum=fixture['momentum'], foreach=False)
    images = fixture['images']
    features = [[read(d).to(args.device) for d in i['features']] for i in images]
    def predict(prefix, ff):
        return dict(boxes=torch.cat([b(x).reshape(1, 4*bins, -1) for b,x in zip(getattr(head,prefix+'cv2'),ff)],2),
                    scores=torch.cat([b(x).reshape(1,nc,-1) for b,x in zip(getattr(head,prefix+'cv3'),ff)],2), feats=ff)
    def write(name, value):
        value = value.detach().cpu().float().contiguous().numpy()
        value.astype('<f4').tofile(args.output/name)
        return dict(path=name, shape=list(value.shape))
    def snapshot(step):
        state = head.state_dict()
        parameters = dict(head.named_parameters())
        result = []
        for j,d in enumerate(fixture['parameters']):
            row = dict(name=d['name'], **write(f'step-{step}-parameter-{j}.f32', state[d['name']]))
            p = parameters.get(d['name'])
            velocity = optimizer.state.get(p, {}).get('momentum_buffer')
            row['velocityPath'] = write(f'step-{step}-velocity-{j}.f32', velocity if velocity is not None else torch.zeros_like(p))['path'] if p is not None else None
            result.append(row)
        return dict(step=step, parameters=result)
    def evaluate(stage):
        head.eval()
        result=[]
        with torch.no_grad():
            for index,i in enumerate(images):
                if i['split'] != 'val': continue
                prefix = 'one2one_' if len(prefixes) == 4 else ''
                boxes = [b(x) for b,x in zip(getattr(head,prefix+'cv2'),features[index])]
                classes = [b(x) for b,x in zip(getattr(head,prefix+'cv3'),features[index])]
                result.append(dict(id=i['id'], boxes=[write(f'{stage}-{i["id"]}-boxes-{j}.f32',v) for j,v in enumerate(boxes)],
                                   classes=[write(f'{stage}-{i["id"]}-classes-{j}.f32',v) for j,v in enumerate(classes)]))
        head.train()
        return result
    before = evaluate('before')
    train = [i for i in images if i['split'] == 'train']
    history, snapshots, step = [], [], 0
    for epoch,order in enumerate(fixture['trainOrder']):
        for index in order:
            image, ff = train[index], features[index]
            truth = image['truth']; size = fixture['size']
            batch=dict(batch_idx=torch.zeros(len(truth),device=args.device), cls=torch.tensor([t['classId'] for t in truth],device=args.device).float().reshape(-1,1),
                       bboxes=torch.tensor([[(t['x1']+t['x2'])/(2*size),(t['y1']+t['y2'])/(2*size),(t['x2']-t['x1'])/size,(t['y2']-t['y1'])/size] for t in truth],device=args.device).float().reshape(-1,4))
            pred = predict('', ff)
            if len(prefixes) == 4: pred = dict(one2many=pred,one2one=predict('one2one_',ff))
            loss, items = criterion(pred,batch)
            loss=loss.sum()
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step+=1
            history.append(dict(step=step,epoch=epoch+1,totalLoss=float(loss.detach().cpu())))
            if step in [1,16] or step % len(train) == 0: snapshots.append(snapshot(step))
        if fixture['headKind'] == 'ProgressiveEndToEnd': criterion.update()
    after = evaluate('after')
    torch.save(dict(model=model.cpu(),source_sha256=fixture['sourceSha256'],fine_tuning_only=True),args.output/'trained.pt')
    value=dict(status='completed',engine='PyTorch',device=args.device,steps=step,fixtureSha256=sha(args.fixture/'fixture.json'),history=history,snapshots=snapshots,before=before,after=after,
               checkpointSha256=sha(args.output/'trained.pt'),torchVersion=torch.__version__,ultralyticsVersion=ultralytics.__version__)
    (args.output/'outcome.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(f'Completed PyTorch {args.device}: {step} updates',flush=True)


def compare(fixture_dir, torch_dir, net_dir, output):
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.cache/yolo26-config')
    sys.path.insert(0,str(ROOT/'.cache/yolo26-release-reference'))
    sys.path.append(str(ROOT/'.cache/reference'))
    import numpy as np
    import torch
    from types import SimpleNamespace
    from ultralytics.utils.metrics import ap_per_class, box_iou
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.engine.validator import BaseValidator
    import ultralytics.nn.tasks as tasks
    import ultralytics.utils.loss as losses
    tasks.YOLOv10DetectionModel = tasks.DetectionModel
    losses.v10DetectLoss = losses.E2EDetectLoss
    torch.set_num_threads(4)
    fixture=json.loads((fixture_dir/'fixture.json').read_text())
    reference=json.loads((torch_dir/'outcome.json').read_text())
    actual=json.loads((net_dir/'outcome.json').read_text())
    assert reference['fixtureSha256']==actual['fixtureSha256']==sha(fixture_dir/'fixture.json')
    assert reference['steps']==actual['steps']==len(fixture['trainOrder'])*96
    def read(directory,d,path='path'):
        return np.fromfile(directory/d[path],dtype='<f4').reshape(d['shape'])
    def error(a,b):
        a,b=a.astype(np.float64).ravel(),b.astype(np.float64).ravel()
        assert np.isfinite(a).all() and np.isfinite(b).all()
        delta=np.abs(a-b)
        return dict(maxAbs=float(delta.max(initial=0)),rms=float(np.sqrt(np.mean(delta**2))),relativeL2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)))
    def state_checks(ref,ref_dir,value,value_dir):
        snapshots=[]
        for rr,aa in zip(ref['snapshots'],value['snapshots'],strict=True):
            assert rr['step']==aa['step']
            checks=[]
            for r,a in zip(rr['parameters'],aa['parameters'],strict=True):
                assert r['name']==a['name'] and r['shape']==a['shape']
                expected,observed=read(ref_dir,r),read(value_dir,a)
                e=error(expected,observed)
                # 固定门槛：CUDA多步对照未通过时仍记录false，不改阈值、不删除差异。
                e.update(name=r['name'],passed=bool(np.all(np.abs(expected-observed)<=.002+.01*np.abs(expected))))
                if r['velocityPath'] is not None:
                    e['momentum']=error(read(ref_dir,r,'velocityPath'),read(value_dir,a,'velocityPath'))
                checks.append(e)
            snapshots.append(dict(step=rr['step'],passed=all(c['passed'] for c in checks),parameters=checks))
        return snapshots
    snapshots=state_checks(reference,torch_dir,actual,net_dir)
    cpu_dir=fixture_dir/'torch'; cpu_reference=json.loads((cpu_dir/'outcome.json').read_text())
    cpu_control=state_checks(cpu_reference,cpu_dir,actual,net_dir)
    baseline_drift=state_checks(cpu_reference,cpu_dir,reference,torch_dir) if reference['device']=='cuda' else []
    def dense(directory,row):
        boxes=torch.cat([torch.from_numpy(read(directory,d).reshape(1,4,fixture['regMax'],-1)) for d in row['boxes']],3)
        classes=torch.cat([torch.from_numpy(read(directory,d).reshape(1,len(fixture['classNames']),-1)) for d in row['classes']],2).sigmoid()
        distances=boxes[:,:,0,:] if fixture['regMax']==1 else (boxes.softmax(2)*torch.arange(fixture['regMax']).reshape(1,1,-1,1)).sum(2)
        anchors,strides=[],[]
        for d,stride in zip(row['boxes'],fixture['strides']):
            h,w=d['shape'][2:]
            yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
            anchors.append(torch.stack((xx.reshape(-1)+.5,yy.reshape(-1)+.5),0)); strides.extend([stride]*(h*w))
        centers=torch.cat(anchors,1); scale=torch.tensor(strides).reshape(1,1,-1)
        xy1=(centers-distances[:,:2])*scale; xy2=(centers+distances[:,2:])*scale
        xywh=torch.cat(((xy1+xy2)/2,xy2-xy1),1)
        return torch.cat((xywh,classes),1)
    def detections(pred,end2end):
        if end2end and pred.shape[-1]!=6:
            # 与官方端到端头相同：先选 anchor top300，再选 class/anchor top300。
            boxes=pred[:,:4].transpose(1,2); scores=pred[:,4:].transpose(1,2)
            nc=scores.shape[-1]; k=min(300,scores.shape[1]); idx=scores.amax(2).topk(k)[1]
            bb=boxes.gather(1,idx[:,:,None].expand(-1,-1,4)); ss=scores.gather(1,idx[:,:,None].expand(-1,-1,nc))
            conf,ii=ss.flatten(1).topk(k); bb=bb.gather(1,(ii//nc)[:,:,None].expand(-1,-1,4))
            xy1=bb[:,:,:2]-bb[:,:,2:]/2; xy2=bb[:,:,:2]+bb[:,:,2:]/2
            pred=torch.cat((xy1,xy2,conf[:,:,None],(ii%nc).float()[:,:,None]),2)
        return non_max_suppression(pred,conf_thres=.001,iou_thres=.7,multi_label=True,max_det=300,nc=len(fixture['classNames']),max_time_img=10,end2end=end2end)[0]
    end2end=fixture['headKind']!='Dfl'
    val={i['id']:i for i in fixture['images'] if i['split']=='val'}
    # 重新打开实际 PyTorch 权重，执行完整冻结骨干和更新后的头，避免只比较内存中的 head。
    saved_path=torch_dir/'saved-predictions.json'
    saved=json.loads(saved_path.read_text()) if saved_path.exists() else None
    if saved is None or saved['checkpointSha256']!=reference['checkpointSha256']:
        model=torch.load(torch_dir/'trained.pt',map_location='cpu',weights_only=False)['model'].float().eval()
        saved_rows=[]
        with torch.no_grad():
            for i in val.values():
                pred=model(torch.from_numpy(read(fixture_dir,i['input'])))
                if isinstance(pred,(tuple,list)): pred=pred[0]
                array=pred.detach().cpu().float().numpy(); name=f'saved-{i["id"]}-native.f32'
                array.astype('<f4').tofile(torch_dir/name)
                saved_rows.append(dict(id=i['id'],native=dict(path=name,shape=list(array.shape))))
        saved=dict(checkpointSha256=reference['checkpointSha256'],rows=saved_rows)
        saved_path.write_text(json.dumps(saved,indent=2)+'\n',encoding='utf-8')
    def metrics(directory,rows,native=False):
        correct,conf,cls,truths=[],[],[],[]
        raw=[]
        for row in rows:
            pred=torch.from_numpy(read(directory,row['native'])) if native else dense(directory,row)
            det=detections(pred,end2end)
            tt=val[row['id']]['truth']; gt=torch.tensor([[t['x1'],t['y1'],t['x2'],t['y2']] for t in tt]).float().reshape(-1,4); tc=torch.tensor([t['classId'] for t in tt])
            match=BaseValidator.match_predictions(SimpleNamespace(iouv=torch.linspace(.5,.95,10)),det[:,5],tc,box_iou(gt,det[:,:4]))
            correct.append(match.numpy()); conf.append(det[:,4].numpy()); cls.append(det[:,5].numpy()); truths.append(tc.numpy())
            raw.append(dict(id=row['id'],detections=det.numpy().tolist()))
        result=ap_per_class(np.concatenate(correct),np.concatenate(conf),np.concatenate(cls),np.concatenate(truths),names=dict(enumerate(fixture['classNames'])))
        p,r,ap=result[2],result[3],result[5]
        return dict(precision=float(p.mean()),recall=float(r.mean()),map50=float(ap[:,0].mean()),map50_95=float(ap.mean()),classesWithGroundTruth=len(ap),images=len(rows),detections=sum(len(i['detections']) for i in raw)),raw
    measures={}; detections_all={}; output_errors={}
    for stage in ['before','after']:
        measures['torch_'+stage],detections_all['torch_'+stage]=metrics(torch_dir,reference[stage])
        measures['net_'+stage],detections_all['net_'+stage]=metrics(net_dir,actual[stage])
        errors=[]
        for rr,aa in zip(reference[stage],actual[stage],strict=True):
            assert rr['id']==aa['id']
            errors.append(dict(id=rr['id'],**error(dense(torch_dir,rr).numpy(),dense(net_dir,aa).numpy())))
        output_errors[stage]=errors
    measures['saved_net_after'],detections_all['saved_net_after']=metrics(net_dir,actual['after'],True)
    measures['saved_torch_after'],detections_all['saved_torch_after']=metrics(torch_dir,saved['rows'],True)
    metric_deltas={k:abs(measures['saved_torch_after'][k]-measures['saved_net_after'][k]) for k in ['precision','recall','map50','map50_95']}
    loss_ref=np.array([i['totalLoss'] for i in reference['history']]); loss_net=np.array([i['totalLoss'] for i in actual['history']])
    parameter_passed=all(s['passed'] for s in snapshots); metric_passed=all(v<=.02 for v in metric_deltas.values())
    control_passed=all(s['passed'] for s in cpu_control)
    result=dict(status='completed',comparisonPassed=parameter_passed and metric_passed,directParameterComparisonPassed=parameter_passed,metricComparisonPassed=metric_passed,cpuParameterControlPassed=control_passed,
                model=Path(fixture['sourceModel']).name,device=actual['device'],steps=actual['steps'],trainImages=96,validationImages=32,size=fixture['size'],
                fixtureSha256=sha(fixture_dir/'fixture.json'),sourceSha256=fixture['sourceSha256'],datasetSha256=fixture['datasetSha256'],coreAssemblySha256=actual['coreAssemblySha256'],trainingAssemblySha256=actual['trainingAssemblySha256'],torchTrainingDevice=reference['device'],bothSavedModelsReloaded=True,
                metrics=measures,metricAbsoluteDeltas=metric_deltas,metricTolerance=.02,multiStepParameterTolerance=dict(absolute=.002,relative=.01),lossError=error(loss_ref,loss_net),
                snapshots=snapshots,cpuParameterControl=cpu_control,pytorchCpuVsCuda=baseline_drift,outputErrors=output_errors,history=dict(torch=reference['history'],net=actual['history']),torchCheckpointSha256=reference['checkpointSha256'],nativeCheckpointSha256=actual['nativeCheckpointSha256'],
                scope='96/32 disjoint fine-tuning split of COCO128, 80 classes, 256px, 2 epochs, frozen shared features, SGD; no augmentations. Pretrained models have seen COCO train2017.',validatorSha256=sha(__file__))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (net_dir/'detections.json').write_text(json.dumps(detections_all),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','model','device','directParameterComparisonPassed','metricComparisonPassed','cpuParameterControlPassed','metricAbsoluteDeltas']}),flush=True)
    return result


def orchestrate(args):
    args.cache.mkdir(parents=True,exist_ok=True); args.output.mkdir(parents=True,exist_ok=True)
    manifest=prepare(args)
    rows=[]
    for model in args.model or MODELS:
        fixture=args.cache/model; torch_out=fixture/'torch'; torch_out.mkdir(parents=True,exist_ok=True)
        def run(command):
            subprocess.run(command,check=True)
        if not args.resume or not (fixture/'fixture.json').exists(): run(['dotnet',str(TEST),'outcome-emit',str(fixture),'--dataset',str(manifest),'--model',f'models/pt/{model}.pt'])
        if not args.resume or not (torch_out/'outcome.json').exists(): run([sys.executable,__file__,'--worker','--fixture',str(fixture),'--output',str(torch_out)])
        for device in ['cpu','vulkan']:
            net_out=fixture/device
            reference_dir=torch_out if device=='cpu' else fixture/'torch-cuda'
            if device=='vulkan':
                reference_dir.mkdir(exist_ok=True)
                if not args.resume or not (reference_dir/'outcome.json').exists(): run([sys.executable,__file__,'--worker','--device','cuda','--fixture',str(fixture),'--output',str(reference_dir)])
            cached=json.loads((net_out/'outcome.json').read_text()) if (net_out/'outcome.json').exists() else None
            current=sha(ROOT/'src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll')
            if not args.resume or cached is None or cached['trainingAssemblySha256']!=current: run(['dotnet',str(TEST),'outcome-run',str(fixture),str(net_out)]+(['--vulkan'] if device=='vulkan' else []))
            report=args.output/f'{model}-{device}.json'
            result=compare(fixture,reference_dir,net_out,report)
            rows.append(dict(model=model,device=device,report=report.name,completed=True,directParameterComparisonPassed=result['directParameterComparisonPassed'],metricComparisonPassed=result['metricComparisonPassed'],cpuParameterControlPassed=result['cpuParameterControlPassed']))
            (args.output/'comparison.json').write_text(json.dumps(dict(status='running',results=rows),indent=2)+'\n',encoding='utf-8')
    (args.output/'comparison.json').write_text(json.dumps(dict(status='completed',results=rows,datasetSha256=sha(manifest),directParameterComparisonsPassed=sum(r['directParameterComparisonPassed'] for r in rows),metricComparisonsPassed=sum(r['metricComparisonPassed'] for r in rows),cpuParameterControlsPassed=sum(r['cpuParameterControlPassed'] for r in rows)),indent=2)+'\n',encoding='utf-8')
    assert all(r['cpuParameterControlPassed'] and r['metricComparisonPassed'] for r in rows),'Implementation control or saved-model metrics differ; inspect raw comparison'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--fixture',type=Path); parser.add_argument('--cache',type=Path,default=Path('.cache/training-011/outcomes'))
    parser.add_argument('--dataset',type=Path,default=Path('.cache/training-011/coco128'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/validation/0.10.1/outcomes'))
    parser.add_argument('--model',action='append'); parser.add_argument('--resume',action='store_true')
    arguments=parser.parse_args()
    (worker if arguments.worker else orchestrate)(arguments)
