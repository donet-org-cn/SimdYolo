"""Compare .NET and stock Ultralytics/PyTorch inference, serially on the same machine.

Development tool only; it is not a dependency of the .NET inference library.
Requires torch/ultralytics/numpy. Reuses .cache/reference when available.
Only load trusted model checkpoints. No packages are installed by this script.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache"))
if importlib.util.find_spec("ultralytics") is None and (ROOT / ".cache/reference/ultralytics").is_dir():
    sys.path.insert(0, str(ROOT / ".cache/reference"))
import numpy as np
import torch
import ultralytics

def summarize(samples):
    median = statistics.median(samples)
    return {"MedianMs": median, "P95Ms": float(np.percentile(samples, 95)),
            "MinMs": min(samples), "MeanMs": statistics.mean(samples),
            "FramesPerSecond": 1000 / median, "SamplesMs": samples}

def tensors(output):
    if isinstance(output, torch.Tensor):
        return [output]
    return [t for item in output for t in tensors(item)]

def cpu_name():
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or "unknown"

def dotnet_measure(path, backend, args, input_path):
    cli = ROOT / "src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll"
    run = subprocess.run(["dotnet", str(cli), "bench", str(path), backend, str(args.size),
                          str(args.iterations), str(args.threads), str(args.warmup), str(input_path)],
                         cwd=ROOT, text=True, capture_output=True, check=True)
    header, body = run.stdout.strip().split("\n", 1)
    record = json.loads(body)
    record.update({"Engine": f"net-{backend}", "Backend": header.strip(), "Fusion": "managed load-time optimizations"})
    return record

def python_measure(path, array, fusion, device, timing, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
    if fusion:
        model.fuse(verbose=False)
    head = model.model[-1]
    # Match PredictRaw: decoded head / prototypes / keypoints, without auxiliary training outputs.
    head.export, head.format, head.dynamic = True, "onnx", False
    model = model.to(device)
    host_input = torch.from_numpy(array)
    prepared = host_input.to(device)
    def infer():
        if timing == "host-to-host":
            return [t.cpu() for t in tensors(model(host_input.to(device)))]
        return tensors(model(prepared))
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(args.warmup):
            infer()
        sync()
        samples = []
        for _ in range(args.iterations):
            sync()
            started = time.perf_counter_ns()
            result = infer()
            sync()
            samples.append((time.perf_counter_ns() - started) / 1e6)
    output_shapes = [list(t.shape) for t in result]
    if any(not bool(torch.isfinite(t).all()) for t in result):
        raise RuntimeError(f"Non-finite Python output: {path}")
    record = {"Model": path.name, "Engine": f"python-{device}-{'fused' if fusion else 'unfused'}" + (f"-{timing}" if device == "cuda" else ""),
              "Backend": f"PyTorch {device.upper()}", "Shape": list(array.shape), "Outputs": output_shapes,
              "Fusion": fusion, "Timing": timing, "Iterations": args.iterations,
              "Warmup": args.warmup, "CpuThreads": args.threads, **summarize(samples)}
    del result, prepared, model, checkpoint
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return record

def write_report(report, destination):
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    records = report["Records"]
    groups = {}
    for r in records:
        groups.setdefault(r["Model"], {})[r["Engine"]] = r
    lines = ["# .NET 与 Python + PyTorch 速度对比", "", f"输入：{report['Size']}×{report['Size']}，batch=1，FP32；CPU 线程上限 {report['Threads']}；预热 {report['Warmup']} 次，测量 {report['Iterations']} 次。", "",
             "计时包含模型前向与输出头解码，不包含加载、图片预处理、NMS、分割 mask 后处理或姿态可视化。所有模型各配置串行执行，输入文件和权重相同。", "",
             f"CPU：{report['Cpu']}；PyTorch：{report['Torch']}；Ultralytics：{report['Ultralytics']}；.NET SDK：{report['DotnetSdk']}。", "",
             "| 模型 | .NET CPU ms | Python CPU 原始 ms | Python CPU 融合 ms | .NET Vulkan ms | .NET CPU / Python 融合 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, g in groups.items():
        def ms(key):
            return f"{g[key]['MedianMs']:.2f}" if key in g else "—"
        ratio = g["net-cpu"]["MedianMs"] / g["python-cpu-fused"]["MedianMs"] if "net-cpu" in g and "python-cpu-fused" in g else None
        lines.append(f"| {name} | {ms('net-cpu')} | {ms('python-cpu-unfused')} | {ms('python-cpu-fused')} | {ms('net-vulkan')} | {ratio:.2f}× |" if ratio is not None else f"| {name} | {ms('net-cpu')} | {ms('python-cpu-unfused')} | {ms('python-cpu-fused')} | {ms('net-vulkan')} | — |")
    lines += ["", "所有耗时均为中位数，数值越小越快。比值大于 1 表示当前 .NET CPU 比 Python/PyTorch CPU 慢。Vulkan 是 GPU 路径，不能据此把 CPU/GPU 的差异归因于语言。", "",
              "Python 融合版调用原版 Ultralytics 的 fuse()，与 .NET 加载时的 Conv/BN 融合保持接近的优化配置；原版还会重参数化部分 RepConv。未使用 torch.compile、ONNX Runtime 或 TensorRT。", ""]
    if not report["CudaAvailable"]:
        lines += [f"本机 Python CUDA 未测试：{report['CudaSkipReason']}。因此不能得出 .NET Vulkan 与 PyTorch GPU 谁更快的结论。", ""]
    else:
        lines += ["CUDA host-to-host 包含输入上传与结果回读；device-only 仅测已驻留设备的输入输出，二者在 JSON 中分别记录。", ""]
        lines += ["| 模型 | .NET Vulkan ms | PyTorch CUDA 含上传/回读 ms | PyTorch CUDA 设备内 ms | Vulkan / CUDA 含传输 |", "| --- | ---: | ---: | ---: | ---: |"]
        for name, g in groups.items():
            host = g.get("python-cuda-fused-host-to-host")
            resident = g.get("python-cuda-fused-device-only")
            gpu = g.get("net-vulkan")
            if host is None:
                continue
            ratio = f"{gpu['MedianMs'] / host['MedianMs']:.2f}×" if gpu else "—"
            lines.append(f"| {name} | {gpu['MedianMs']:.2f} | {host['MedianMs']:.2f} | {resident['MedianMs']:.2f} | {ratio} |" if gpu and resident else f"| {name} | — | {host['MedianMs']:.2f} | — | {ratio} |")
        lines += [""]
    lines += ["完整采样、P95、FPS、模型/input SHA256 和配置见同名 JSON；复跑命令见 README。", "",
              "融合行为参考：[Ultralytics fuse](https://docs.ultralytics.com/reference/nn/tasks/#ultralytics.nn.tasks.BaseModel.fuse)。CUDA 计时使用 [torch.cuda.synchronize](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)。", ""]
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

def plot(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(r["Model"] for r in report["Records"]))
    engines = [("net-cpu", ".NET CPU SIMD", "#4774da"), ("python-cpu-unfused", "Python/PyTorch CPU", "#94a3b8"),
               ("python-cpu-fused", "Python/PyTorch CPU fused", "#059669"), ("net-vulkan", ".NET Vulkan GPU", "#a855f7")]
    if report["CudaAvailable"]:
        engines.append(("python-cuda-fused-host-to-host", "Python/PyTorch CUDA + transfers", "#d97706"))
    bar_height = .75 / len(engines)
    fig, ax = plt.subplots(figsize=(12, max(5, len(models) * .95)))
    for j, (engine, label, color) in enumerate(engines):
        values = [next((r["MedianMs"] for r in report["Records"] if r["Model"] == model and r["Engine"] == engine), float("nan")) for model in models]
        bars = ax.barh(np.arange(len(models)) + (j - (len(engines) - 1) / 2) * bar_height, values, height=bar_height * .9, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_yticks(np.arange(len(models)), [m.removesuffix(".pt") for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Median inference latency (ms), lower is better")
    ax.set_title(f"Same-machine YOLO comparison | FP32, batch 1, {report['Size']}x{report['Size']}\n{report['Threads']} CPU threads | {report['Warmup']} warmups + {report['Iterations']} measured runs")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolov9t", "yolov10n", "yolo11n"])
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--backends", nargs="+", choices=["cpu", "vulkan", "metal"], default=["cpu", "vulkan"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/python-comparison.json")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--only-cuda", action="store_true", help="Append CUDA measurements to an existing matching CPU report")
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32 or args.iterations < 1 or args.warmup < 1 or args.threads < 1:
        parser.error("size must be a positive multiple of 32; iterations, warmup and threads must be positive")
    if args.vision:
        args.models += ["yolov8n-seg", "yolov8n-pose", "yolo11n-seg", "yolo11n-pose"]
    paths = [args.model_dir / (m if m.endswith(".pt") else m + ".pt") for m in dict.fromkeys(args.models)]
    for path in paths:
        if not path.is_file():
            parser.error(f"Model missing: {path}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    array = np.full((1, 3, args.size, args.size), .5, dtype=np.float32)
    work = ROOT / ".cache/speed-comparison"
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / "input.f32"
    array.tofile(input_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"Cpu": cpu_name(), "Platform": platform.platform(), "Torch": torch.__version__,
              "Ultralytics": ultralytics.__version__, "Python": platform.python_version(),
              "DotnetSdk": subprocess.check_output(["dotnet", "--version"], cwd=ROOT, text=True).strip(),
              "CudaAvailable": torch.cuda.is_available(), "CudaBuild": torch.version.cuda,
              "CudaSkipReason": "Installed torch build has no CUDA support" if torch.version.cuda is None else "CUDA device unavailable",
              "Size": args.size, "Batch": 1, "Precision": "FP32", "Threads": args.threads,
              "Warmup": args.warmup, "Iterations": args.iterations,
              "InputSha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "ModelSha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, "Records": []}
    report["TorchBuild"] = torch.__config__.show()
    if args.only_cuda:
        if not report["CudaAvailable"]:
            parser.error("--only-cuda requires an available CUDA GPU")
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        for field in ["Size", "Threads", "Warmup", "Iterations", "InputSha256", "ModelSha256"]:
            if existing[field] != report[field]:
                parser.error(f"Cannot append: existing {field} does not match")
        existing["CudaReference"] = {"Torch": report["Torch"], "Build": report["CudaBuild"], "TorchBuild": report["TorchBuild"], "Gpu": torch.cuda.get_device_name(0), "Capability": list(torch.cuda.get_device_capability(0))}
        existing["CudaAvailable"], existing["CudaBuild"], existing["CudaSkipReason"] = True, report["CudaBuild"], None
        existing["Records"] = [r for r in existing["Records"] if not r["Engine"].startswith("python-cuda-")]
        report = existing
    for path in paths:
        print(f"Measuring {path.name} ...", flush=True)
        records = []
        for backend in ([] if args.only_cuda else args.backends):
            r = dotnet_measure(path, backend, args, input_path)
            records.append(r)
            print(f"  {r['Engine']}: {r['MedianMs']:.2f} ms ({r['FramesPerSecond']:.2f} FPS)", flush=True)
        for fused in ([] if args.only_cuda else [False, True]):
            r = python_measure(path, array, fused, "cpu", "host-to-host", args)
            records.append(r)
            print(f"  {r['Engine']}: {r['MedianMs']:.2f} ms ({r['FramesPerSecond']:.2f} FPS)", flush=True)
        if report["CudaAvailable"]:
            for timing in ["host-to-host", "device-only"]:
                r = python_measure(path, array, True, "cuda", timing, args)
                records.append(r)
                print(f"  {r['Engine']}: {r['MedianMs']:.2f} ms", flush=True)
        if len({json.dumps(r["Outputs"]) for r in records}) != 1:
            raise RuntimeError(f"Output shapes differ between engines: {path.name}")
        report["Records"] += records
        write_report(report, args.output)
    if not args.no_plot:
        plot(report, args.output)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)

if __name__ == "__main__":
    main()
