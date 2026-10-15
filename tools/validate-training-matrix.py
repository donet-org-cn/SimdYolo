"""用固定的官方 Ultralytics 8.4.0 独立验证五代检测头、损失、全部参数梯度和 BN。

仅用于开发对照；Python/PyTorch 不进入 C# 运行时。每项固定容差，失败保留明细并退出非零。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(args):
    root = Path(__file__).resolve().parent.parent
    reference = args.ultralytics_dir.resolve()
    sys.path.insert(0, str(reference))
    sys.path.append(str(root / '.cache/reference'))
    os.environ.setdefault('YOLO_CONFIG_DIR', str(root / '.cache/yolo26-config'))
    import numpy as np
    import torch
    import torch.nn as nn
    import ultralytics
    import ultralytics.nn.tasks as tasks
    import ultralytics.utils.loss as loss_module
    from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss, E2ELoss
    assert ultralytics.__version__ == '8.4.0'
    assert Path(ultralytics.__file__).resolve().parent.parent == reference
    # 官方兼容层相同的旧类名映射；直接读取可信本仓库检查点，避免导入无关任务依赖。
    tasks.YOLOv10DetectionModel = tasks.DetectionModel
    loss_module.v10DetectLoss = E2EDetectLoss
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    directory = args.fixture.resolve()
    fixture_path = directory / 'fixture.json'
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    checks, files = [], {}

    def read(d, field='path'):
        path = (directory / d[field]).resolve()
        assert path.is_relative_to(directory)
        raw = path.read_bytes()
        a = np.frombuffer(raw, dtype='<f4').reshape(d['shape']).copy()
        assert np.isfinite(a).all()
        files[str(path.relative_to(directory))] = sha(path)
        return a

    def compare(name, actual, expected, category, tolerance=(5e-4, 2e-3)):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        difference = np.abs(a - b)
        allowed = tolerance[0] + tolerance[1] * np.abs(b)
        failed = difference > allowed
        record = dict(name=name, category=category, shape=list(a.shape), elements=a.size,
                      atol=tolerance[0], rtol=tolerance[1], maxAbsError=float(difference.max(initial=0)),
                      maxToleranceRatio=float((difference / np.maximum(allowed, 1e-300)).max(initial=0)),
                      failedElements=int(failed.sum()), passed=bool(np.isfinite(a).all() and np.isfinite(b).all() and not failed.any()))
        checks.append(record)
        if not record['passed']:
            print(json.dumps(record), flush=True)

    source = Path(fixture['sourceModel'])
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    model = (checkpoint.get('ema') or checkpoint['model']).float()
    head = model.model[-1]
    branches = fixture['branches']
    prefixes = ['cv2', 'cv3'] + (['one2one_cv2', 'one2one_cv3'] if len(branches) == 2 else [])
    assert head.reg_max == fixture['regMax']
    assert type(head).__name__ in ('Detect', 'v10Detect')
    for prefix in prefixes:
        assert len(getattr(head, prefix)) == 3
        if prefix.endswith('cv3'):
            for branch in getattr(head, prefix):
                conv = branch[-1]
                branch[-1] = nn.Conv2d(conv.in_channels, len(fixture['classNames']), conv.kernel_size,
                                       conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None)
    head.nc = len(fixture['classNames'])
    head.no = head.nc + 4 * head.reg_max
    head.stride = torch.tensor(fixture['strides'])
    for name, p in head.named_parameters():
        p.requires_grad_(any(name.startswith(prefix + '.') for prefix in prefixes))
    state = head.state_dict()
    expected_names = {n for n in state if any(n.startswith(p + '.') for p in prefixes)}
    assert expected_names == {d['name'] for d in fixture['parameters']}
    assert {n for n, p in head.named_parameters() if p.requires_grad} == {d['name'] for d in fixture['parameters'] if d['gradientPath']}
    for descriptor in fixture['parameters']:
        values = read(descriptor)
        assert list(state[descriptor['name']].shape) == list(values.shape)
        state[descriptor['name']] = torch.from_numpy(values).to(state[descriptor['name']].dtype)
    head.load_state_dict(state, strict=True)
    features = [torch.from_numpy(read(d)) for d in fixture['features']]
    height, width = int(features[0].shape[2] * head.stride[0]), int(features[0].shape[3] * head.stride[0])
    truth = fixture['truth']
    batch = dict(batch_idx=torch.zeros(len(truth)), cls=torch.tensor([t['classId'] for t in truth]).float().reshape(-1, 1),
                 bboxes=torch.tensor([[(t['x1']+t['x2'])/(2*width), (t['y1']+t['y2'])/(2*height),
                                       (t['x2']-t['x1'])/width, (t['y2']-t['y1'])/height] for t in truth]).float().reshape(-1, 4))

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([head])
            self.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5, epochs=fixture['scheduleEpochs'])

    fake = FakeModel()
    wrapper = E2ELoss(fake) if fixture['headKind'] == 'ProgressiveEndToEnd' else E2EDetectLoss(fake) if len(branches) == 2 else None
    if fixture['headKind'] == 'ProgressiveEndToEnd':
        for _ in range(fixture['epoch']):
            wrapper.update()

    def predictions(boxes, classes):
        return dict(boxes=torch.cat([x.reshape(1, 4*head.reg_max, -1) for x in boxes], 2),
                    scores=torch.cat([x.reshape(1, head.nc, -1) for x in classes], 2), feats=features)

    def criterion_for(d):
        if wrapper is not None:
            return getattr(wrapper, d['name'])
        return v8DetectionLoss(fake, tal_topk=10)

    def official_loss(d, boxes, classes, label):
        criterion = criterion_for(d)
        expected_weight = (wrapper.o2m if d['name'] == 'one2many' else wrapper.o2o) if fixture['headKind'] == 'ProgressiveEndToEnd' else 1
        compare(label+'.weight', d['weight'], expected_weight, 'schedule', (1e-7, 1e-7))
        compare(label+'.topk', d['topk'], criterion.assigner.topk, 'assignmentConfig', (0, 0))
        compare(label+'.topk2', d['topk2'], criterion.assigner.topk2, 'assignmentConfig', (0, 0))
        assigned, losses, _ = criterion.get_assigned_targets_and_loss(predictions(boxes, classes), batch)
        for k, value in zip(('box', 'classification', 'dfl'), losses.detach().numpy()):
            compare(label+'.'+k, d['loss'][k], value, 'loss', (1e-4, 3e-4))
        compare(label+'.total', d['loss']['total'], losses.sum().item(), 'loss', (1e-4, 3e-4))
        compare(label+'.positives', d['loss']['positives'], int(assigned[0].sum()), 'assignment', (0, 0))
        return losses, expected_weight

    def gradient(x):
        return x.grad.detach().numpy() if x.grad is not None else np.zeros(tuple(x.shape), dtype=np.float32)

    head.train()
    groups = []
    total = 0
    combined_predictions = {}
    for d in branches:
        boxes = [branch(x) for branch, x in zip(getattr(head, d['prefix']+'cv2'), features)]
        classes = [branch(x) for branch, x in zip(getattr(head, d['prefix']+'cv3'), features)]
        for x in boxes + classes:
            x.retain_grad()
        losses, weight = official_loss(d, boxes, classes, d['name']+'.officialHead')
        total = total + losses.sum() * weight
        combined_predictions[d['name']] = predictions(boxes, classes)
        groups.append((d, boxes, classes))
    if wrapper is not None:
        # 独立验证组合目标本身，避免只分别验证分支却漏掉版本特定的组合规则。
        combined = wrapper(combined_predictions, batch)[0].sum()
        compare('combinedOfficialWrapper', total.detach().item(), combined.detach().item(), 'combinedObjective', (1e-6, 1e-6))
    total.backward()
    for d, boxes, classes in groups:
        for key, tensors in [('boxes', boxes), ('classes', classes)]:
            for i, (desc, x) in enumerate(zip(d[key], tensors)):
                compare(f"{d['name']}.{key}.{i}", read(desc), x.detach().numpy(), 'headForward')
                compare(f"{d['name']}.{key}.{i}.gradient", read(desc, 'gradientPath'), gradient(x), 'headOutputGradient')
    parameters, after = dict(head.named_parameters()), head.state_dict()
    for d in fixture['parameters']:
        name = d['name']
        if d['gradientPath']:
            compare(name+'.gradient', read(d, 'gradientPath'), gradient(parameters[name]), 'parameterGradient')
            compare(name+'.connected', d['connected'], parameters[name].grad is not None, 'gradientConnectivity', (0,0))
        buffer = name not in parameters
        compare(name+'.after', read(d, 'afterPath'), after[name].detach().numpy(), 'batchNormRunningStatistics' if buffer else 'parameterUnchangedBeforeSgd',
                (5e-5, 3e-4) if buffer and not name.endswith('num_batches_tracked') else (0, 0))
    optimizer = torch.optim.SGD([dict(params=[parameters[d['name']]], weight_decay=fixture['optimizer']['weightDecay'] if d['decay'] else 0) for d in fixture['sgdParameters']], lr=fixture['optimizer']['learningRate'], momentum=fixture['optimizer']['momentum'], foreach=False)
    optimizer.step()
    for d in fixture['sgdParameters']:
        p = parameters[d['name']]
        compare(d['name']+'.sgd', read(d), p.detach().numpy(), 'sgdParameter', (5e-5, 3e-4))
        momentum = optimizer.state[p].get('momentum_buffer', torch.zeros_like(p))
        compare(d['name']+'.momentum', read(d, 'velocityPath'), momentum.numpy(), 'sgdMomentum')
    for d in branches:
        boxes = [torch.from_numpy(read(x)).requires_grad_() for x in d['boxes']]
        classes = [torch.from_numpy(read(x)).requires_grad_() for x in d['classes']]
        losses, weight = official_loss(d, boxes, classes, d['name']+'.fixtureLogits')
        (losses.sum()*weight).backward()
        for key, tensors in [('boxes', boxes), ('classes', classes)]:
            for i, (desc, x) in enumerate(zip(d[key], tensors)):
                compare(f"{d['name']}.isolated.{key}.{i}.gradient", read(desc, 'gradientPath'), gradient(x), 'isolatedLossGradient')
    return dict(status='passed' if all(c['passed'] for c in checks) else 'failed', sourceModel=str(source), sourceModelSha256=sha(source),
                fixtureSha256=sha(fixture_path), coreAssemblySha256=fixture['coreAssemblySha256'], trainingAssemblySha256=fixture['trainingAssemblySha256'],
                headKind=fixture['headKind'], regMax=head.reg_max, csharpDevice=fixture['device'], epoch=fixture['epoch'], scheduleEpochs=fixture['scheduleEpochs'],
                reference=dict(ultralyticsVersion=ultralytics.__version__, torchVersion=torch.__version__, device='cpu', threads=args.threads,
                               sources={p:sha(reference/'ultralytics'/p) for p in ['nn/modules/head.py','utils/loss.py','utils/tal.py','utils/metrics.py']}),
                validatorSha256=sha(__file__), totalChecks=len(checks), passedChecks=sum(c['passed'] for c in checks), failedChecks=sum(not c['passed'] for c in checks), checks=checks, files=files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--ultralytics-dir', type=Path, default=Path('.cache/yolo26-release-reference'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    try:
        report = validate(args)
    except Exception as error:
        report = dict(status='error', error=str(error), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:report.get(k) for k in ['status','sourceModel','totalChecks','failedChecks','error']}), flush=True)
    if report['status'] == 'error':
        print(report['traceback'], file=sys.stderr)
    sys.exit(0 if report['status'] == 'passed' else 1)
