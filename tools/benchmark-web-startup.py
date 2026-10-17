"""串行启动真实 ASP.NET 应用，计时进程启动至33模型列表可用；仅使用 Python 标准库。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5205)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or not 1024 <= args.port <= 65535:
        parser.error('Invalid repeats or port')
    if args.output.exists():
        parser.error('Output exists; use a new evidence path')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    variants = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    for directory in variants.values():
        if not (directory / 'SimdYolo.Web.dll').is_file():
            parser.error(f'Published web application missing: {directory}')
    run = ROOT / '.cache' / 'web-startup' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    report = {'Status': 'running', 'Definition': 'Controller process launch to completed HTTP /api/info with 33 registered models; includes dotnet host and ASP.NET startup. No inference or browser loading.',
              'Order': 'ABBA; independent process and data directory per window; OS file cache preserved', 'Records': [],
              'Libraries': {name: {'Directory': str(directory), 'Sha256': hashlib.sha256((directory / 'SimdYolo.dll').read_bytes()).hexdigest()} for name, directory in variants.items()}}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    expected = None
    for repeat in range(args.repeats):
        for variant in ['baseline', 'candidate', 'candidate', 'baseline']:
            index = len(report['Records']) + 1
            directory = variants[variant]
            data = run / f'data-{index}'
            command = ['dotnet', str(directory / 'SimdYolo.Web.dll'), '--urls', f'http://127.0.0.1:{args.port}', '--contentRoot', str(directory),
                       '--SimdYolo:RepositoryRoot', str(ROOT), '--SimdYolo:DataRoot', str(data)]
            with (run / f'{index}-{variant}.log').open('wb') as log:
                started = time.perf_counter()
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f'Web worker exited; see {log.name}')
                        try:
                            with opener.open(f'http://127.0.0.1:{args.port}/api/info', timeout=.5) as response:
                                info = json.load(response)
                            ready_ms = (time.perf_counter() - started) * 1000
                            break
                        except (urllib.error.URLError, TimeoutError):
                            if time.perf_counter() - started > 60:
                                raise TimeoutError(f'Web startup timeout; see {log.name}')
                            time.sleep(.01)
                    if len(info['models']) != 33:
                        raise ValueError(f'Expected 33 models, got {len(info["models"])}')
                    fingerprint = hashlib.sha256(json.dumps(info['models'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    if expected is not None and fingerprint != expected:
                        raise ValueError('Model labels/tasks/capability listing changed')
                    expected = fingerprint
                    report['Records'].append({'Variant': variant, 'Repeat': repeat + 1, 'ProcessToApiReadyMs': ready_ms, 'Version': info['version'], 'Models': 33, 'ModelListSha256': fingerprint})
                    print(f'{index}: {variant} {ready_ms:.2f} ms', flush=True)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
    report['Summary'] = {name: {'Windows': args.repeats * 2, 'MedianMs': statistics.median(row['ProcessToApiReadyMs'] for row in report['Records'] if row['Variant'] == name)} for name in variants}
    report['CandidateChangePercent'] = (report['Summary']['candidate']['MedianMs'] / report['Summary']['baseline']['MedianMs'] - 1) * 100
    report['Status'] = 'complete'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
