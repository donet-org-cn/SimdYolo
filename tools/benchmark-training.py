"""串行比较纯 C# 与原版 PyTorch 的检测头训练时间和进程资源。

缓存相同的冻结特征，默认固定512次更新与初始权重，计时包括前向、损失、反向和 SGD。
不把 ONNX Runtime 推理当训练基线；没有安装其 Training/ORTModule。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
MODELS = ['yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo26n']
TEST = ROOT / 'tests/SimdYolo.Training.Tests/bin/Release/net10.0/SimdYolo.Training.Tests.dll'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    if args.device == 'cuda':
        sys.path.insert(0, str(ROOT/'.cache/torch-cu126'))
    sys.path.insert(0, str(ROOT/'.cache/yolo26-release-reference'))
    sys.path.append(str(ROOT/'.cache/reference'))
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.cache/yolo26-config')
    import numpy as np
    import psutil
    import torch
    import torch.nn as nn
    import ultralytics
    import ultralytics.nn.tasks as tasks
    import ultralytics.utils.loss as losses
    from types import SimpleNamespace
    assert ultralytics.__version__ == '8.4.0'
    assert (torch.version.cuda is not None) == (args.device == 'cuda'), 'CPU and CUDA must use isolated torch builds'
    tasks.YOLOv10DetectionModel = tasks.DetectionModel
    losses.v10DetectLoss = losses.E2EDetectLoss
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    fixture = json.loads((args.fixture/'fixture.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(fixture['sourceModel'], map_location='cpu', weights_only=False)
    model = (checkpoint.get('ema') or checkpoint['model']).float()
    head = model.model[-1]
    prefixes = ['cv2', 'cv3'] + (['one2one_cv2', 'one2one_cv3'] if len(fixture['branches']) == 2 else [])
    for prefix in prefixes:
        if prefix.endswith('cv3'):
            for branch in getattr(head, prefix):
                c = branch[-1]
                branch[-1] = nn.Conv2d(c.in_channels, len(fixture['classNames']), c.kernel_size, c.stride, c.padding, c.dilation, c.groups, c.bias is not None)
    head.nc = len(fixture['classNames'])
    head.no = head.nc + head.reg_max*4
    head.stride = torch.tensor(fixture['strides'])
    def read(d):
        return torch.from_numpy(np.frombuffer((args.fixture/d['path']).read_bytes(), dtype='<f4').reshape(d['shape']).copy())
    state = head.state_dict()
    for d in fixture['parameters']:
        state[d['name']] = read(d).to(state[d['name']].dtype)
    head.load_state_dict(state, strict=True)
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    for name, p in head.named_parameters():
        p.requires_grad_(any(name.startswith(k+'.') for k in prefixes))
    model.to(args.device).eval()
    head.train()
    features = [read(d).to(args.device) for d in fixture['features']]
    truth = fixture['truth']
    height, width = args.size, args.size
    batch = dict(batch_idx=torch.zeros(len(truth), device=args.device), cls=torch.tensor([t['classId'] for t in truth], device=args.device).float().reshape(-1,1),
                 bboxes=torch.tensor([[(t['x1']+t['x2'])/(2*width), (t['y1']+t['y2'])/(2*height), (t['x2']-t['x1'])/width, (t['y2']-t['y1'])/height] for t in truth], device=args.device).float().reshape(-1,4))
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([head])
            self.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5, epochs=5)
    fake = FakeModel()
    criterion = losses.E2ELoss(fake) if fixture['headKind'] == 'ProgressiveEndToEnd' else losses.E2EDetectLoss(fake) if len(prefixes) == 4 else losses.v8DetectionLoss(fake)
    decay, other = [], []
    for p in head.parameters():
        if p.requires_grad:
            (decay if p.ndim == 4 else other).append(p)
    optimizer = torch.optim.SGD([dict(params=decay, weight_decay=.0005), dict(params=other, weight_decay=0)], lr=.0001, momentum=.9, foreach=False)
    def predict(prefix):
        return dict(boxes=torch.cat([b(x).reshape(1, 4*head.reg_max, -1) for b,x in zip(getattr(head, prefix+'cv2'), features)],2),
                    scores=torch.cat([b(x).reshape(1, head.nc, -1) for b,x in zip(getattr(head, prefix+'cv3'), features)],2), feats=features)
    def step():
        pred = predict('')
        if len(prefixes) == 4:
            pred = dict(one2many=pred, one2one=predict('one2one_'))
        loss = criterion(pred,batch)[0].sum()
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if args.device == 'cuda':
            torch.cuda.synchronize()
    for _ in range(10):
        step()
    durations = []
    process = psutil.Process()
    (args.output/'measuring').write_text('ready')
    cpu = sum(process.cpu_times()[:2])
    begin = time.perf_counter()
    for _ in range(args.steps):
        start = time.perf_counter()
        step()
        durations.append((time.perf_counter()-start)*1000)
    elapsed = time.perf_counter()-begin
    (args.output/'measured').write_text('done')
    result = dict(engine='PyTorch', model=Path(fixture['sourceModel']).name, sourceSha256=sha(fixture['sourceModel']), device=args.device, size=args.size, threads=args.threads, warmup=10, steps=len(durations),
                  medianMs=statistics.median(durations), meanMs=statistics.mean(durations), p95Ms=sorted(durations)[int(len(durations)*.95)],
                  cpuPercentOfMachine=(sum(process.cpu_times()[:2])-cpu)/elapsed/(os.cpu_count() or 1)*100, workingSetMiB=process.memory_info().rss/1048576,
                  fixtureSha256=sha(args.fixture/'fixture.json'), torchVersion=torch.__version__, ultralyticsVersion=ultralytics.__version__, cuda=torch.version.cuda,
                  scope='cached frozen features; head forward+loss+backward+SGD; excludes backbone, preprocessing, checkpoint IO', passed=True)
    (args.output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


def orchestrate(args):
    sys.path.append(str(ROOT/'.cache/reference'))
    import psutil
    args.output.mkdir(parents=True,exist_ok=True)
    runs = []
    for model in MODELS:
        fixture = ROOT/'.cache/training-010'/f'benchmark-fixture-{model}-{args.size}'
        subprocess.run(['dotnet',str(TEST),'emit-matrix',str(fixture),'--model',f'models/pt/{model}.pt','--size',str(args.size)],check=True)
        # 计时前对实际128尺寸夹具再次数值核验，性能比较只接受已通过的初始数据。
        subprocess.run([sys.executable,'tools/validate-training-matrix.py','--fixture',str(fixture),'--output',str(args.output/f'fixture-check-{model}.json')],check=True)
        for repeat in range(1, args.repeats+1):
            engines = ['net-cpu','torch-cpu','net-vulkan','torch-cuda']
            if repeat % 2 == 0:
                engines.reverse()
            for engine in engines:
                directory = args.output/f'{model}-{engine}-{repeat}'
                directory.mkdir(parents=True,exist_ok=True)
                for name in ['measuring','measured']:
                    (directory/name).unlink(missing_ok=True)
                command = (['dotnet',str(TEST),'bench-head',str(directory),'--model',f'models/pt/{model}.pt','--fixture',str(fixture),*(['--vulkan'] if engine=='net-vulkan' else [])] if engine.startswith('net-') else
                           [sys.executable,__file__,'--worker','--device','cuda' if engine=='torch-cuda' else 'cpu','--fixture',str(fixture),'--output',str(directory)])
                command += ['--size',str(args.size),'--threads',str(args.threads),'--steps',str(args.steps)]
                samples = []
                env = os.environ.copy()
                env.pop('PYTHONPATH',None)
                with (directory/'worker.log').open('w',encoding='utf-8') as log:
                    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    while process.poll() is None:
                        if (directory/'measuring').exists() and not (directory/'measured').exists():
                            sample = dict(time=time.time())
                            try:
                                sample['rssMiB'] = psutil.Process(process.pid).memory_info().rss/1048576
                            except psutil.Error:
                                pass
                            if engine in ['net-vulkan','torch-cuda']:
                                query = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits','-i','0'],text=True,capture_output=True,timeout=5)
                                if query.returncode == 0:
                                    fields=query.stdout.strip().split(',')
                                    try: sample.update(gpuPercent=float(fields[0]),deviceVramMiB=float(fields[1]))
                                    except ValueError: pass
                            samples.append(sample)
                        time.sleep(.2)
                    if process.returncode:
                        raise RuntimeError(f'{engine} failed: '+(directory/'worker.log').read_text(encoding='utf-8')[-5000:])
                result = json.loads((directory/'benchmark.json').read_text(encoding='utf-8'))
                result.update(engineKey=engine,repeat=repeat,samples=samples)
                assert result['fixtureSha256'] == sha(fixture/'fixture.json') and result['steps']==args.steps
                runs.append(result)
                (args.output/'training-runtime-comparison.json').write_text(json.dumps(dict(status='running',runs=runs),indent=2)+'\n',encoding='utf-8')
                print(f"{model} {engine} repeat {repeat}: {result['medianMs']:.3f} ms",flush=True)
    report = dict(status='passed',scope='Detection-head fine-tuning only; fixed cached features, two classes, FP32, gradient accumulation=1; no checkpoints in timed region',
                  sizes=args.size,threads=args.threads,repeats=args.repeats,steps=args.steps,models=MODELS,cores=os.cpu_count(),gpuResourceScope='whole device, not process attributed; sampled only in timed window',runs=runs,toolSha256=sha(__file__))
    (args.output/'training-runtime-comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--output',type=Path,default=Path('artifacts/validation/0.10.0/performance'))
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    (worker if args.worker else orchestrate)(args)
