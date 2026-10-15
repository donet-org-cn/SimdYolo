"""串行重建五代检测训练夹具并运行独立官方数值对照，避免同时争用 CPU/GPU。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('artifacts/validation/0.10.0'))
parser.add_argument('--cache', type=Path, default=Path('.cache/training-010'))
parser.add_argument('--model', action='append')
parser.add_argument('--cpu-only', action='store_true')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
test = 'tests/SimdYolo.Training.Tests/bin/Release/net10.0/SimdYolo.Training.Tests.dll'
results = []
for model in args.model or ['yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo26n']:
    for device in ['cpu'] if args.cpu_only else ['cpu', 'vulkan']:
        cases = [('normal', []), ('small', ['--small-boxes']), ('empty', ['--empty'])]
        if model == 'yolo26n':
            cases += [('middle', ['--epoch', '2']), ('final', ['--epoch', '4'])]
        for case, options in cases:
            name = f'{model}-{device}-{case}'
            fixture = args.cache / name
            report = args.output / f'oracle-{name}.json'
            for command in [
                ['dotnet', test, 'emit-matrix', str(fixture), '--model', f'models/pt/{model}.pt', *(['--vulkan'] if device == 'vulkan' else []), *options],
                [sys.executable, 'tools/validate-training-matrix.py', '--fixture', str(fixture), '--output', str(report)],
            ]:
                result = subprocess.run(command, text=True, encoding='utf-8', errors='replace', capture_output=True)
                with (args.output / 'oracle-run.log').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(command) + '\n' + result.stdout + result.stderr)
                if result.returncode:
                    print(result.stdout[-5000:] + result.stderr[-5000:], flush=True)
                    raise SystemExit(result.returncode)
            value = json.loads(report.read_text(encoding='utf-8'))
            results.append(dict(model=model, device=device, case=case, report=report.name, checks=value['totalChecks'], passed=value['status']=='passed'))
            (args.output / f'fixture-{name}.json').write_bytes((fixture/'fixture.json').read_bytes())
            (args.output / 'oracle-matrix.json').write_text(json.dumps(dict(status='running', results=results), indent=2)+'\n', encoding='utf-8')
            print(f"PASS {name}: {value['totalChecks']} checks", flush=True)
(args.output / 'oracle-matrix.json').write_text(json.dumps(dict(status='passed', totalGroups=len(results), totalChecks=sum(r['checks'] for r in results), results=results), indent=2)+'\n', encoding='utf-8')
