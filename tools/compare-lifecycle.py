"""Compare frozen/candidate SimdYolo lifecycle windows in serial fresh processes.

The worker is pure .NET. This development controller uses only Python's standard
library; it is never a dependency of the SimdYolo package. Each case/repeat runs
baseline, candidate, candidate, baseline (ABBA). Output files remain available
for a separate numerical check; this tool never declares accuracy or victory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ["yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo26n",
                  "yolov8n-seg", "yolo11n-seg", "yolo26n-seg",
                  "yolov8n-pose", "yolo11n-pose", "yolo26n-pose"]
ENVIRONMENT_FLAGS = ["SIMDYOLO_VULKAN_CACHE", "SIMDYOLO_VULKAN_CACHE_DIR",
                     "SIMDYOLO_VULKAN_REPLAY", "SIMDYOLO_CPU_EXECUTOR",
                     "DOTNET_ReadyToRun", "DOTNET_TieredPGO",
                     "DOTNET_TieredCompilation", "DOTNET_PROCESSOR_COUNT"]
METRICS = ["ModelLoadMs", "FirstInferenceMs", "ReadyToFirstOutputMs", "MedianMs",
           "MeanMs", "P95Ms", "AllocatedBytesPerFrame", "WorkingSetMiB", "CpuPercent"]


def sha256(path):
    """Hash a file without loading the entire file into another byte array."""
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
        return digest.hexdigest()


def write_json(path, value):
    """Atomically save progress; a failed worker leaves completed windows intact."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def hardware():
    """Collect descriptive hardware once, outside every timed worker."""
    cpu = platform.processor()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    gpu = None
    command = shutil.which("nvidia-smi")
    if command:
        try:
            query = subprocess.run([command, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True)
            gpu = query.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            pass
    return {"Cpu": cpu, "LogicalProcessors": os.cpu_count(), "Os": platform.platform(),
            "Architecture": platform.machine(), "NvidiaGpuQuery": gpu,
            "GpuQueryScope": "Optional driver utility; actual worker backend/device is recorded per window."}


def cache_snapshot(environment):
    """Describe the requested application cache; driver/OS caches are not flushed."""
    configured = environment.get("SIMDYOLO_VULKAN_CACHE_DIR")
    path = Path(configured) if configured else None
    files = []
    if path and path.is_dir():
        files = [{"Name": item.name, "Bytes": item.stat().st_size, "Sha256": sha256(item)}
                 for item in sorted(path.iterdir()) if item.is_file()]
    return {"Environment": {key: environment.get(key) for key in ENVIRONMENT_FLAGS},
            "ExplicitCacheDirectory": str(path) if path else None, "Files": files,
            "DriverCacheState": "Uncontrolled; a fresh process does not imply a cold driver or OS filesystem cache."}


def summarize(records):
    """Aggregate window medians, retaining all individual windows as evidence."""
    groups = {}
    for record in records:
        key = (record["Model"], record["Format"], record["RequestedBackend"], record["Variant"])
        groups.setdefault(key, []).append(record)
    output = []
    for key, windows in groups.items():
        values = {metric: statistics.median(item[metric] for item in windows) for metric in METRICS}
        output.append({"Model": key[0], "Format": key[1], "Backend": key[2], "Variant": key[3],
                       "Windows": len(windows), **values})
    return output


def compare_summaries(summary):
    """Expose numerical before/after ratios without inferring significance."""
    groups = {}
    for row in summary:
        key = (row["Model"], row["Format"], row["Backend"])
        groups.setdefault(key, {})[row["Variant"]] = row
    comparisons = []
    for key, rows in groups.items():
        if "baseline" not in rows or "candidate" not in rows:
            continue
        before, after = rows["baseline"], rows["candidate"]
        metrics = {}
        for metric in METRICS:
            value_before, value_after = before[metric], after[metric]
            metrics[metric] = {"Baseline": value_before, "Candidate": value_after,
                               "BaselineOverCandidate": value_before / value_after if value_after else None,
                               "CandidateChangePercent": (value_after / value_before - 1) * 100 if value_before else None}
        comparisons.append({"Model": key[0], "Format": key[1], "Backend": key[2],
                            "BaselineWindows": before["Windows"], "CandidateWindows": after["Windows"],
                            "Metrics": metrics})
    return comparisons


def main():
    """Prepare identical hosts and execute independent ABBA lifecycle windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Frozen harness/CLI/library directory containing SimdYolo.dll")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Candidate harness/CLI/library directory containing SimdYolo.dll")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--pt-dir", type=Path, default=ROOT / "models/pt")
    parser.add_argument("--onnx-dir", type=Path, default=ROOT / "models/onnx")
    parser.add_argument("--formats", nargs="+", choices=["pt", "onnx"], default=["pt", "onnx"])
    parser.add_argument("--backends", nargs="+", choices=["cpu", "vulkan", "metal"], default=["cpu", "metal" if sys.platform == "darwin" else "vulkan"])
    parser.add_argument("--repeats", type=int, default=2, help="ABBA blocks per model/format/backend; each block has two windows per variant")
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--warmup", type=int, default=30, help="Total premeasurement frames, including the measured first frame")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--input", type=Path, required=True, help="Preprocessed little-endian FP32 NCHW RGB input shared by every worker")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/lifecycle-comparison.json")
    parser.add_argument("--cache-mode", choices=["inherited", "disabled", "isolated"], default="inherited", help="inherited preserves environment/defaults; disabled turns application caches off; isolated uses a fresh per-window cache directory")
    parser.add_argument("--timeout", type=float, default=600, help="Timeout for each isolated worker, in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Replace the report; earlier raw-output directories remain intact")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the schedule without building or measuring")
    args = parser.parse_args()
    if (args.repeats < 1 or args.warmup < 1 or args.duration < 0 or not args.duration < float("inf")
            or args.size < 32 or args.size % 32 or args.threads < 1 or args.min_frames < 1
            or not 0 < args.timeout < float("inf")):
        parser.error("Invalid benchmark settings")
    variants = {"baseline": args.baseline_dir.resolve() / "SimdYolo.dll",
                "candidate": args.candidate_dir.resolve() / "SimdYolo.dll"}
    for path in [*variants.values(), args.input.resolve()]:
        if not path.is_file():
            parser.error(f"Required file does not exist: {path}")
    if args.input.stat().st_size != 3 * args.size * args.size * 4:
        parser.error("Input FP32 byte count does not match --size")
    cases = []
    for model in args.models:
        if Path(model).name != model or model in (".", ".."):
            parser.error("--models must contain model stems, not paths")
        for fmt in args.formats:
            path = (args.pt_dir if fmt == "pt" else args.onnx_dir).resolve() / f"{model}.{fmt}"
            if not path.is_file():
                parser.error(f"Model does not exist: {path}")
            for backend in args.backends:
                cases.append((model, fmt, backend, path))
    if len(set(args.models)) != len(args.models) or len(set(args.formats)) != len(args.formats) or len(set(args.backends)) != len(args.backends):
        parser.error("Repeated models, formats or backends would duplicate windows")
    if args.dry_run:
        print(json.dumps({"Cases": len(cases), "Windows": len(cases) * args.repeats * 4,
                          "Order": ["baseline", "candidate", "candidate", "baseline"],
                          "WarmupFrames": args.warmup, "Size": args.size, "Threads": args.threads,
                          "CacheMode": args.cache_mode}, indent=2))
        return
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"Report exists; use a new --output or --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output.parent / (output.stem + "-windows") / batch
    directory.mkdir(parents=True)
    template = directory / "host-common"
    build = ["dotnet", "build", str(ROOT / "tools/LifecycleBenchmark/LifecycleBenchmark.csproj"),
             "-c", "Release", "-o", str(template), f"-p:SimdYoloLibraryPath={variants['baseline']}"]
    subprocess.run(build, cwd=ROOT, check=True)
    hosts = {}
    variant_info = {}
    for variant, library in variants.items():
        host = directory / ("host-" + variant)
        shutil.copytree(template, host)
        shutil.copyfile(library, host / "SimdYolo.dll")
        hosts[variant] = host
        variant_info[variant] = {"SourceDirectory": str(library.parent), "SourceDllSha256": sha256(library),
                                 "SnapshotDll": str(host / "SimdYolo.dll"), "SnapshotDllSha256": sha256(host / "SimdYolo.dll")}
        if variant_info[variant]["SourceDllSha256"] != variant_info[variant]["SnapshotDllSha256"]:
            raise RuntimeError("Library changed while preparing isolated host")
    sources = [ROOT / "tools/LifecycleBenchmark/Program.cs", ROOT / "tools/LifecycleBenchmark/LifecycleBenchmark.csproj", Path(__file__).resolve()]
    report = {"Status": "running", "StartedUtc": batch, "Order": "ABBA for each case/repeat; case order reverses on alternating repeats",
              "Variants": variant_info, "Hardware": hardware(), "Input": str(args.input.resolve()), "InputSha256": sha256(args.input.resolve()),
              "Models": [{"Path": str(path), "Sha256": sha256(path)} for path in dict.fromkeys(case[3] for case in cases)],
              "ToolSources": [{"Path": str(path.relative_to(ROOT)), "Sha256": sha256(path)} for path in sources],
              "WorkerSha256": sha256(template / "SimdYolo.LifecycleBenchmark.dll"), "CacheMode": args.cache_mode,
              "Settings": {"Size": args.size, "Threads": args.threads, "WarmupFrames": args.warmup, "DurationSeconds": args.duration, "MinimumFrames": args.min_frames, "Repeats": args.repeats},
              "Definitions": {
                  "ModelLoadMs": "YoloDetector construction, including backend creation, model parsing and weight preparation.",
                  "FirstInferenceMs": "First PredictRaw, including GPU upload/compute/output materialization when applicable.",
                  "ReadyToFirstOutputMs": "From the first managed Main statement through input reading, construction and first PredictRaw; excludes dotnet host startup and entry-method JIT before Main executes.",
                  "ProcessTotalMs": "Controller wall time for the complete worker, including dotnet startup, warmup, steady window, output/hashes/JSON and disposal; this is not process-to-first-output latency.",
                  "AllocatedBytesPerFrame": "Precise total managed allocation over the steady loop minus measured telemetry-list growth, divided by frames; input/model preparation and output writing are outside this interval.",
                  "CpuPercent": "Process CPU seconds / steady wall seconds / worker Environment.ProcessorCount * 100; includes managed controller-loop work in the worker.",
                  "PipelineCacheLoaded": "Actual application Vulkan pipeline cache restoration, not a claim about driver or filesystem coldness.",
                  "Summary": "Median of individual window metrics; all raw samples and independent outputs are preserved; no statistical significance or accuracy claim is inferred."},
              "Records": [], "Summary": [], "Comparisons": []}
    write_json(output, report)
    try:
        sequence = 0
        for repeat in range(args.repeats):
            ordered = cases if repeat % 2 == 0 else list(reversed(cases))
            for model, fmt, backend, path in ordered:
                for position, variant in enumerate(["baseline", "candidate", "candidate", "baseline"]):
                    sequence += 1
                    prefix = directory / f"{sequence:04}-{model}-{fmt}-{backend}-{variant}"
                    environment = os.environ.copy()
                    if args.cache_mode == "disabled":
                        environment["SIMDYOLO_VULKAN_CACHE"] = "0"
                    elif args.cache_mode == "isolated":
                        environment["SIMDYOLO_VULKAN_CACHE"] = "1"
                        environment["SIMDYOLO_VULKAN_CACHE_DIR"] = str(directory / f"cache-{sequence:04}")
                    before = cache_snapshot(environment)
                    command = ["dotnet", str(hosts[variant] / "SimdYolo.LifecycleBenchmark.dll"),
                               str(path), str(args.input.resolve()), backend, str(prefix), str(args.warmup),
                               str(args.duration), str(args.size), str(args.threads), str(args.min_frames)]
                    print(f"[{sequence}/{len(cases) * args.repeats * 4}] {model} {fmt} {backend} {variant}", flush=True)
                    started = time.perf_counter()
                    worker = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True,
                                            encoding="utf-8", errors="replace", timeout=args.timeout)
                    process_ms = (time.perf_counter() - started) * 1000
                    Path(str(prefix) + ".stdout.txt").write_text(worker.stdout, encoding="utf-8")
                    Path(str(prefix) + ".stderr.txt").write_text(worker.stderr, encoding="utf-8")
                    if worker.returncode:
                        raise RuntimeError(f"Worker failed ({worker.returncode}); see {prefix}.stderr.txt")
                    record = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8-sig"))
                    if record["FallbackReason"] is not None:
                        raise RuntimeError("Worker fell back to another backend")
                    record.update({"Variant": variant, "Repeat": repeat + 1, "AbbaPosition": position + 1,
                                   "Sequence": sequence, "Format": fmt, "ProcessTotalMs": process_ms,
                                   "DllSha256": variant_info[variant]["SnapshotDllSha256"], "CacheBefore": before,
                                   "CacheAfter": cache_snapshot(environment), "Command": command})
                    report["Records"].append(record)
                    report["Summary"] = summarize(report["Records"])
                    report["Comparisons"] = compare_summaries(report["Summary"])
                    write_json(output, report)
        report["Status"] = "complete"
        report["FinishedUtc"] = datetime.now(timezone.utc).isoformat()
        write_json(output, report)
    except Exception as error:
        report["Status"] = "failed"
        report["Error"] = str(error)
        write_json(output, report)
        raise
    print(f"Completed {len(report['Records'])} independent lifecycle windows: {output}", flush=True)


if __name__ == "__main__":
    main()
