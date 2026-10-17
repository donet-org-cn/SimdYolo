"""Run frozen/current managed image pipelines serially; never import inference runtimes."""
import array
import hashlib
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / '.cache/image-optimization'
OUT = ROOT / 'artifacts/validation/0.11.1'
OUT.mkdir(parents=True, exist_ok=True)
records = []
checks = []


def compare(before, after):
    """Check actual final-frame geometry and all float mask pixels against fixed gates."""
    a = json.loads(Path(str(before) + '.instances.json').read_text(encoding='utf-8'))
    b = json.loads(Path(str(after) + '.instances.json').read_text(encoding='utf-8'))
    if len(a) != len(b):
        raise ValueError('Instance count changed')
    box_error = confidence_error = probability_error = threshold_flips = pixels = 0
    for i, (left, right) in enumerate(zip(a, b)):
        if left['Box']['ClassId'] != right['Box']['ClassId']:
            raise ValueError('Class changed')
        box_error = max(box_error, *(abs(left['Box'][key] - right['Box'][key]) for key in ['X', 'Y', 'Width', 'Height']))
        confidence_error = max(confidence_error, abs(left['Box']['Confidence'] - right['Box']['Confidence']))
        if left['mask'] != right['mask'] or left['Keypoints'] != right['Keypoints']:
            raise ValueError('Mask geometry or keypoints changed')
        if left['mask']:
            x, y = array.array('f'), array.array('f')
            x.frombytes(Path(str(before) + f'.mask-{i}.f32').read_bytes())
            y.frombytes(Path(str(after) + f'.mask-{i}.f32').read_bytes())
            if len(x) != len(y):
                raise ValueError('Mask array size changed')
            for p, q in zip(x, y):
                error = abs(p - q)
                if not error <= 2e-5:
                    raise ValueError('Whole-model probability tolerance exceeded: ' + str(error))
                probability_error = max(probability_error, error)
                threshold_flips += (p >= .5) != (q >= .5)
            pixels += len(x)
    if box_error > .01 or confidence_error > 2e-5:
        raise ValueError('Box/confidence tolerance exceeded')
    return dict(before=str(before.relative_to(ROOT)), after=str(after.relative_to(ROOT)), maxBoxError=box_error,
                maxConfidenceError=confidence_error, maxMaskProbabilityError=probability_error,
                maskPixels=pixels, thresholdFlips=threshold_flips)


def run(model, image, device, order):
    """Use fresh child processes and retain every timing window, including control regressions."""
    width, height = (810, 1080) if image == 'bus' else (2048, 1536)
    baselines = []
    for iteration, variant in enumerate(order):
        prefix = WORK / 'measurements' / f'{model}-{image}-{device}-{iteration}-{variant}'
        command = ['dotnet', str(WORK / f'{variant}-worker/ImagePipelineBenchmark.dll'),
                   str(ROOT / f'models/pt/{model}.pt'), str(WORK / f'fixtures/{image}.rgb'),
                   str(prefix), str(width), str(height), device, '4']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError(result.stdout[-1000:] + result.stderr[-2000:])
        value = json.loads(Path(str(prefix) + '.json').read_text(encoding='utf-8'))
        value.update(variant=variant, modelName=model, image=image, device=device, iteration=iteration,
                     command=command)
        records.append(value)
        if variant == 'baseline':
            baselines.append(prefix)
        else:
            checks.append(compare(baselines[-1], prefix))
        save()
        print(json.dumps(dict(model=model, image=image, device=device, iteration=iteration, variant=variant,
                              predictMs=value['stages']['predict']['medianMs'], decodeMs=value['stages']['decode']['medianMs'])), flush=True)


def save():
    report = dict(status='running', dateUtc=datetime.now(timezone.utc).isoformat(), host=platform.platform(),
                  scope='FP32 batch1 input640x640 CPU4 threads. Actual RGB photograph; Predict includes resize, forward, NMS and original-resolution instance masks. Sequential fresh processes; 12 warmups/stage; >=2s and >=40 frames/stage. Mask-byte conversion measured separately, not included in Predict. No Python inference, ORT or PyTorch measured this round. CPU% is process core-percent (100=one logical core); allocations are managed allocation volume, not retained RSS. Startup does not clear OS/driver caches.',
                  baselineCommit='039ee0f0c178da68b68d6ae69e46d6e719726b36',
                  baselineSha256=hashlib.sha256((WORK/'baseline/SimdYolo.dll').read_bytes()).hexdigest(),
                  records=records, correctness=checks)
    (OUT/'image-performance.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8', newline='\n')


for family in ['yolov8n-seg', 'yolo26n-seg']:
    for backend in ['Cpu', 'Vulkan']:
        run(family, 'bus', backend, ['baseline', 'optimized', 'optimized', 'baseline'])
for backend in ['Cpu', 'Vulkan']:
    run('yolov8n', 'bus', backend, ['baseline', 'optimized'])
    run('yolov8n-seg', 'large', backend, ['baseline', 'optimized'])

report = json.loads((OUT/'image-performance.json').read_text(encoding='utf-8'))
report['status'] = 'passed'
report['aggregate'] = []
for key in sorted({(r['modelName'], r['image'], r['device']) for r in records}):
    model, image, device = key
    row = dict(model=model, image=image, device=device)
    for variant in ['baseline', 'optimized']:
        windows = [r for r in records if (r['modelName'], r['image'], r['device'], r['variant']) == (*key, variant)]
        row[variant] = {stage: {field: statistics.median(r['stages'][stage][field] for r in windows)
                               for field in ['medianMs', 'p95Ms', 'allocatedBytesPerFrame', 'cpuCorePercent']}
                        for stage in windows[0]['stages']}
        row[variant]['windows'] = len(windows)
    report['aggregate'].append(row)
(OUT/'image-performance.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8', newline='\n')
print('PASS all windows and fixed-tolerance correctness comparisons', flush=True)
