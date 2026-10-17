"""Serial, isolated YOLO latency / CPU / RSS / GPU benchmark.

Development dependencies: numpy, psutil, stock PyTorch/Ultralytics, ONNX Runtime.
Uses existing isolated .cache/reference, .cache/torch-cu126 and .cache/ort-gpu.
Never installs packages and never becomes a dependency of SimdYolo.
"""
import argparse
import ast
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache"))
import numpy as np
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_validation import compare

ENGINES = {
    "net-cpu": ".NET SIMD CPU (.pt)",
    "torch-cpu": "Python/PyTorch CPU fused",
    "ort-cpu": "ONNX Runtime CPU (Python host)",
    "net-vulkan": ".NET Vulkan GPU (.pt)",
    "torch-cuda": "Python/PyTorch CUDA fused",
    "ort-cuda": "ONNX Runtime CUDA (Python host)",
    "net-onnx-cpu": ".NET SIMD CPU (.onnx)",
    "net-onnx-vulkan": ".NET Vulkan GPU (.onnx)",
    "ort-net-cpu": "ONNX Runtime CPU (.NET host)",
    "ort-net-cuda": "ONNX Runtime CUDA (.NET host)",
}
DEFAULT_ENGINES = ["net-cpu", "torch-cpu", "ort-net-cpu", "net-vulkan", "torch-cuda", "ort-net-cuda"]
ENGINE_COLORS = {
    "torch-cpu": "#b45309",
    "net-cpu": "#2563eb",
    "net-onnx-cpu": "#06b6d4",
    "ort-net-cpu": "#4b5563",
    "ort-cpu": "#64748b",
    "torch-cuda": "#f97316",
    "net-vulkan": "#16a34a",
    "net-onnx-vulkan": "#84cc16",
    "ort-net-cuda": "#9333ea",
    "ort-cuda": "#c026d3",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class GpuSampler:
    """NVML is sampled in the controller, never in the timed worker.

    Utilization and VRAM are device-wide, not per-process under WDDM.
    Unavailable metrics are null, never manufactured zeros.
    """
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.error = None
        self.lib = None
        self.info = {}
        try:
            self.lib = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() != 0:
                raise RuntimeError("NVML initialization failed")
            self.handle = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
                raise RuntimeError("NVML device 0 unavailable")
            name, driver = ctypes.create_string_buffer(128), ctypes.create_string_buffer(128)
            self.lib.nvmlDeviceGetName(self.handle, name, len(name))
            self.lib.nvmlSystemGetDriverVersion(driver, len(driver))
            self.info = {"Name": name.value.decode(), "Driver": driver.value.decode(), "Index": 0, "Scope": "whole-device"}
        except (OSError, RuntimeError) as error:
            self.error = str(error)
            self.lib = None

    def sample(self):
        if self.lib is None:
            return {"GpuPercent": None, "VramMiB": None, "VramPercent": None}
        util, memory = self.Util(), self.Memory()
        have_util = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)) == 0
        have_memory = self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) == 0
        return {"GpuPercent": util.gpu if have_util else None,
                "VramMiB": memory.used / 1048576 if have_memory else None,
                "VramPercent": memory.used / memory.total * 100 if have_memory and memory.total else None}

    def close(self):
        if self.lib is not None:
            self.lib.nvmlShutdown()


def activate_packages(engine, ultralytics_dir=None):
    # Select one build in a fresh process. ORT workers do not import torch, so
    # the Python torch module cannot inflate the ORT RSS comparison.
    if engine.startswith("torch"):
        selected = Path(ultralytics_dir or ROOT / ".cache/reference").resolve()
        if not (selected / "ultralytics").is_dir():
            raise RuntimeError(f"Ultralytics package directory is missing: {selected}")
        cuda_packages = (ROOT / ".cache/torch-cu126").resolve()
        if engine == "torch-cpu":
            # 在导入 torch 前移除已知 CUDA wheel 路径，再静态检查最终选中的 wheel。
            sys.path[:] = [entry for entry in sys.path if not Path(entry or os.curdir).resolve().is_relative_to(cuda_packages)]
        directories = [selected, cuda_packages if engine == "torch-cuda" else None, ROOT / ".cache/reference"]
        for directory in reversed(list(dict.fromkeys(directory for directory in directories if directory is not None and directory.is_dir()))):
            sys.path.insert(0, str(directory))
    elif engine == "ort-cuda" and (ROOT / ".cache/ort-gpu").is_dir():
        sys.path.insert(0, str(ROOT / ".cache/ort-gpu"))


def ensure_cpu_torch_package():
    """在导入前读取官方 wheel 的版本声明，禁止 CPU 基准意外加载 CUDA torch。"""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("A CPU-only PyTorch package is required for torch-cpu")
    version = Path(next(iter(spec.submodule_search_locations))) / "version.py"
    if not version.is_file():
        raise RuntimeError(f"Cannot verify that selected PyTorch wheel is CPU-only: {version}")
    for statement in ast.parse(version.read_text(encoding="utf-8")).body:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "cuda" for target in targets):
            if not isinstance(statement.value, ast.Constant) or statement.value.value is not None:
                raise RuntimeError(f"torch-cpu selected a CUDA/unknown PyTorch wheel; CUDA torch was not imported: {version}")
            return
    raise RuntimeError(f"PyTorch cuda build declaration is missing: {version}")


def flatten_tensors(value, torch):
    if isinstance(value, torch.Tensor):
        return [value]
    return [tensor for item in value for tensor in flatten_tensors(item, torch)]


def worker(args):
    started = time.perf_counter()
    baseline_rss = psutil.Process().memory_info().rss / 1048576
    array = np.fromfile(args.input, dtype=np.float32).reshape(1, 3, args.size, args.size)
    metadata = {"Engine": args.engine, "BaselineWorkingSetMiB": baseline_rss, "CpuThreads": args.threads}
    import_started = time.perf_counter()
    activate_packages(args.engine, args.ultralytics_dir)
    if args.engine.startswith("torch"):
        if args.engine == "torch-cpu":
            ensure_cpu_torch_package()
        import torch
        import ultralytics
        import_ms = (time.perf_counter() - import_started) * 1000
        if args.engine == "torch-cpu" and torch.version.cuda is not None:
            raise RuntimeError("CPU benchmark must use a CPU-only PyTorch build")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        model_started = time.perf_counter()
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
        model.fuse(verbose=False)
        head = model.model[-1]
        head.export, head.format, head.dynamic = True, "onnx", False
        cuda = args.engine == "torch-cuda"
        if cuda and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; CPU fallback is forbidden")
        model = model.to("cuda" if cuda else "cpu")
        model_load_ms = (time.perf_counter() - model_started) * 1000
        end_to_end = bool(getattr(head, "end2end", False))
        del checkpoint
        gc.collect()
        host_input = torch.from_numpy(array)

        def infer():
            tensors = flatten_tensors(model(host_input.to("cuda") if cuda else host_input), torch)
            # All public outputs must be host-visible, just like PredictRaw.
            if cuda:
                result = [t.cpu() for t in tensors]
                torch.cuda.synchronize()
                return result
            return tensors

        def as_arrays(result):
            return [t.numpy() for t in result]

        metadata.update({"Version": torch.__version__, "Ultralytics": ultralytics.__version__,
                         "UltralyticsDirectory": str(args.ultralytics_dir.resolve()), "EndToEnd": end_to_end, "ClassCount": int(head.nc),
                         "Cuda": torch.version.cuda, "Cudnn": torch.backends.cudnn.version() if cuda else None,
                         "Backend": "PyTorch CUDA" if cuda else "PyTorch CPU", "Fusion": "stock model.fuse()"})
        context = torch.inference_mode()
    else:
        import onnxruntime as ort
        import_ms = (time.perf_counter() - import_started) * 1000
        model_started = time.perf_counter()
        cuda = args.engine == "ort-cuda"
        if cuda:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime CUDA provider is missing")
            dll_path = ROOT / ".cache/torch-cu126/torch/lib"
            ort.preload_dlls(directory=str(dll_path) if dll_path.is_dir() else None)
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Keep ORT default spinning; disabling it would change CPU and speed.
        providers = [("CUDAExecutionProvider", {"use_tf32": "0", "cudnn_conv_algo_search": "EXHAUSTIVE", "cudnn_conv_use_max_workspace": "1"}), "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(args.model), sess_options=options, providers=providers)
        session.disable_fallback()
        if cuda and session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime unexpectedly fell back to CPU")
        model_load_ms = (time.perf_counter() - model_started) * 1000
        input_name = session.get_inputs()[0].name

        def infer():
            return session.run(None, {input_name: array})

        def as_arrays(result):
            return result

        metadata.update({"Version": ort.__version__, "Backend": session.get_providers()[0],
                         "Providers": session.get_providers(), "ProviderOptions": session.get_provider_options(),
                         "Fusion": "ORT_ENABLE_ALL", "TorchImported": "torch" in sys.modules})
        import contextlib
        context = contextlib.nullcontext()
    with context:
        first_ms = None
        ready_to_first_output_ms = None
        warmup_started = time.perf_counter()
        for _ in range(args.warmup):
            tick = time.perf_counter_ns()
            infer()
            if first_ms is None:
                first_ms = (time.perf_counter_ns() - tick) / 1e6
                ready_to_first_output_ms = (time.perf_counter() - started) * 1000
        warmup_ms = (time.perf_counter() - warmup_started) * 1000
        metadata.update({"ImportMs": import_ms, "ModelLoadMs": model_load_ms,
                         "StartupMs": (time.perf_counter() - started) * 1000, "FirstInferenceMs": first_ms,
                         "ReadyToFirstOutputMs": ready_to_first_output_ms, "WarmupMs": warmup_ms})
        write_json(args.signal / "ready.json", metadata)
        deadline = time.perf_counter() + 120
        while not (args.signal / "go").exists():
            if time.perf_counter() > deadline:
                raise TimeoutError("Benchmark start signal timed out")
            time.sleep(.005)
        process = psutil.Process()
        before_cpu = sum(process.cpu_times()[:2])
        samples = []
        tick = time.perf_counter()
        while len(samples) < args.iterations or time.perf_counter() - tick < args.duration:
            begin = time.perf_counter_ns()
            result = infer()
            samples.append((time.perf_counter_ns() - begin) / 1e6)
        seconds = time.perf_counter() - tick
        cpu_seconds = sum(process.cpu_times()[:2]) - before_cpu
        (args.signal / "done").write_text("done")
        outputs = as_arrays(result)
        if not all(np.isfinite(o).all() for o in outputs):
            raise RuntimeError("Non-finite inference output")
        np.savez(args.signal / "outputs.npz", *outputs)
        if args.engine == "torch-cpu" and end_to_end:
            # 完整候选只在计时及资源窗口结束后生成；既不改变稳态延迟，也不污染采样 RSS。
            postprocess = head.postprocess
            try:
                head.postprocess = lambda predictions, *unused: predictions
                dense = flatten_tensors(model(host_input), torch)[0].detach().numpy()
            finally:
                head.postprocess = postprocess
            if dense.ndim != 3 or not np.isfinite(dense).all():
                raise RuntimeError("Invalid complete decoded candidate oracle")
            dense_path = args.signal / "dense-reference.npz"
            np.savez(dense_path, dense_reference=dense, nc=np.array(int(head.nc), dtype=np.int32))
            metadata["Oracle"] = {"EndToEnd": True, "ClassCount": int(head.nc), "DenseReferenceFile": str(dense_path)}
        median = statistics.median(samples)
        write_json(args.signal / "result.json", {**metadata, "Model": args.model.name, "Shape": list(array.shape),
                   "Outputs": [list(o.shape) for o in outputs], "Iterations": len(samples), "Warmup": args.warmup,
                   "WindowSeconds": seconds, "CpuSeconds": cpu_seconds,
                   "CpuPercent": cpu_seconds / seconds / psutil.cpu_count() * 100,
                   "CpuMsPerFrame": cpu_seconds * 1000 / len(samples), "ThroughputFps": len(samples) / seconds,
                   "MedianMs": median, "MeanMs": statistics.mean(samples), "P95Ms": float(np.percentile(samples, 95)),
                   "MinMs": min(samples), "FramesPerSecond": 1000 / median, "SamplesMs": samples})


def process_sample(process, previous_cpu, previous_time, start, gpu):
    current = time.perf_counter()
    cpu = sum(process.cpu_times()[:2])
    memory = process.memory_info()
    sample = {"Seconds": current - start, "CpuPercent": (cpu - previous_cpu) / (current - previous_time) / psutil.cpu_count() * 100,
              "WorkingSetMiB": memory.rss / 1048576, "WorkingSetPercent": memory.rss / psutil.virtual_memory().total * 100,
              "PrivateMiB": getattr(memory, "private", memory.vms) / 1048576,
              "SystemCpuPercent": psutil.cpu_percent(), **gpu.sample()}
    return sample, cpu, current


def summarize_resources(samples, idle, ram):
    def values(key, skip=0):
        return [s[key] for s in samples if s["Seconds"] >= skip and s[key] is not None]

    def avg(key, skip=0):
        v = values(key, skip)
        return statistics.mean(v) if v else None

    def peak(key):
        v = values(key)
        return max(v) if v else None

    # NVML GPU utilization has a rolling history, so discard the first second.
    gpu_mean = avg("GpuPercent", 1)
    intervals = [b["Seconds"]-a["Seconds"] for a, b in zip(samples, samples[1:])]
    return {"SampleCount": len(samples), "SampledCpuPercent": avg("CpuPercent"),
            "MeanSampleIntervalMs": statistics.mean(intervals)*1000 if intervals else None,
            "MaxSampleIntervalMs": max(intervals)*1000 if intervals else None,
            "MeanWorkingSetMiB": avg("WorkingSetMiB"), "PeakWorkingSetMiB": peak("WorkingSetMiB"),
            "MeanWorkingSetPercent": avg("WorkingSetPercent"), "MeanPrivateMiB": avg("PrivateMiB"),
            "MeanSystemCpuPercent": avg("SystemCpuPercent"), "MeanGpuPercent": gpu_mean,
            "PeakGpuPercent": max(values("GpuPercent", 1), default=None),
            "MeanDeviceVramMiB": avg("VramMiB", 1), "PeakDeviceVramMiB": peak("VramMiB"),
            "MeanDeviceVramPercent": avg("VramPercent", 1), "IdleGpu": idle,
            "ApproximateVramDeltaMiB": avg("VramMiB", 1) - idle["VramMiB"] if avg("VramMiB", 1) is not None and idle["VramMiB"] is not None else None,
            "GpuMetricScope": "whole-device; includes desktop/other processes", "TotalRamMiB": ram / 1048576,
            "Samples": samples}


def run_case(model, engine, repeat, args, input_path, gpu):
    signal = ROOT / ".cache/runtime-comparison" / f"{model}-{engine}-{repeat}-{time.time_ns()}"
    signal.mkdir(parents=True)
    # Let the previous process's GPU context and NVML rolling sample settle.
    time.sleep(1.1)
    idle = gpu.sample()
    pt = args.model_dir / f"{model}.pt"
    onnx = args.onnx_dir / f"{model}.onnx"
    if engine.startswith("ort-net"):
        path = onnx
        cli = ROOT / "tools/OrtBenchmark/bin/Release/net10.0/OrtBenchmark.dll"
        command = ["dotnet", str(cli), str(path), "cuda" if engine.endswith("cuda") else "cpu", str(args.size), str(args.iterations), str(args.threads), str(args.warmup), str(input_path), str(args.duration), str(signal), str(signal / "output"), str(ROOT / ".cache/torch-cu126/torch/lib"), str(ROOT / ".cache/ort-gpu/onnxruntime/capi")]
    elif engine.startswith("net"):
        path = onnx if "onnx" in engine else pt
        backend = "vulkan" if "vulkan" in engine else "cpu"
        cli = getattr(args, "net_cli", ROOT / "src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll")
        command = ["dotnet", str(cli), "bench", str(path), backend, str(args.size), str(args.iterations), str(args.threads), str(args.warmup),
                   str(input_path), str(args.duration), str(signal), str(signal / "output")]
    else:
        path = onnx if engine.startswith("ort") else pt
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--engine", engine, "--model", str(path),
                   "--size", str(args.size), "--threads", str(args.threads), "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                   "--duration", str(args.duration), "--input", str(input_path), "--signal", str(signal),
                   "--ultralytics-dir", str(getattr(args, "ultralytics_dir", ROOT / ".cache/reference"))]
    environment = os.environ.copy()
    # Ambient PYTHONPATH must not accidentally select CUDA torch for CPU workers.
    environment.pop("PYTHONPATH", None)
    with (signal / "stdout.log").open("w", encoding="utf-8") as stdout, (signal / "stderr.log").open("w", encoding="utf-8") as stderr:
        process_started = time.perf_counter()
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=environment)
        try:
            deadline = time.perf_counter() + 240
            while not (signal / "ready.json").exists():
                if child.poll() is not None or time.perf_counter() > deadline:
                    raise RuntimeError(f"Worker failed before ready: {engine} {model}; logs: {signal}")
                time.sleep(.05)
            # 外部进程启动到全部预热完成的 ready 观测；这个数不是首个输出时间。
            process_startup_ms = (time.perf_counter() - process_started) * 1000
            process = psutil.Process(child.pid)
            previous_cpu = sum(process.cpu_times()[:2])
            start = previous_time = time.perf_counter()
            psutil.cpu_percent()
            (signal / "go").write_text("go")
            samples = []
            while not (signal / "done").exists():
                time.sleep(args.sample_interval)
                if (signal / "done").exists():
                    break
                if child.poll() is not None:
                    raise RuntimeError(f"Worker failed during timing: {engine} {model}; logs: {signal}")
                if time.perf_counter() - start > args.duration + 120:
                    raise TimeoutError(f"Worker timing exceeded deadline: {signal}")
                sample, previous_cpu, previous_time = process_sample(process, previous_cpu, previous_time, start, gpu)
                samples.append(sample)
            child.wait(timeout=120)
            if child.returncode:
                raise RuntimeError(f"Worker failed after timing: {signal}")
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait()
            raise
    if engine.startswith("net") or engine.startswith("ort-net"):
        stdout_text = (signal / "stdout.log").read_text(encoding="utf-8-sig")
        record = json.loads(stdout_text[stdout_text.index("{"):])
        arrays = [np.fromfile(signal / f"output.{i}.f32", dtype=np.float32).reshape(shape) for i, shape in enumerate(record["Outputs"])]
        metadata = json.loads((signal / "ready.json").read_text(encoding="utf-8-sig"))
        record.update({"Version": record.get("Version", "SimdYolo（版本未记录）"), "Engine": engine, "BaselineWorkingSetMiB": metadata["BaselineWorkingSetMiB"], "Backend": metadata["Backend"]})
        record.update({key: metadata[key] for key in ("ImportMs", "ModelLoadMs", "FirstInferenceMs", "ReadyToFirstOutputMs", "WarmupMs", "StartupMs") if key in metadata})
    else:
        record = json.loads((signal / "result.json").read_text(encoding="utf-8"))
        with np.load(signal / "outputs.npz") as archive:
            arrays = [archive[key].copy() for key in archive.files]
    record.update({"Model": model, "Repeat": repeat, "ProcessId": child.pid, "ProcessStartupMs": process_startup_ms, "ModelFormat": path.suffix,
                   "Timing": "host-to-host FP32 forward and decoded head; no image processing or NMS",
                   "Resources": summarize_resources(samples, idle, psutil.virtual_memory().total)})
    return record, arrays


def check_outputs(model, actual, reference, end_to_end=None, dense=None, nc=0):
    if len(actual) != len(reference):
        raise AssertionError(f"Output count mismatch: {model}")
    checks = []
    if end_to_end is None:
        end_to_end = model.startswith(("yolov10", "yolo26"))
    for index, (a, b) in enumerate(zip(actual, reference)):
        if a.shape != b.shape or not np.isfinite(a).all():
            raise AssertionError(f"Invalid output shape/values: {model}")
        selected = end_to_end and index == 0
        a, b, passed = compare(a, b, selected, dense if selected else None, nc)
        checks.append({"Shape": list(a.shape), "MaxAbsError": float(np.max(np.abs(a-b), initial=0)),
                       "Validation": "TopK scores and unique complete anchor/class/payload pairing" if selected and dense is not None
                           else "Unique complete row pairing" if selected else "Elementwise FP32",
                       "Passed": passed})
        if not passed:
            raise AssertionError(f"Output mismatch {model}: {checks[-1]}")
    return checks


def cpu_name():
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def cpu_only_report(report):
    engines = set(report["Engines"]) | {record["Engine"] for record in report["Records"]}
    return not any("cuda" in engine or "vulkan" in engine for engine in engines)


def render_speed_rankings(report):
    """Render four-engine CPU/GPU rankings without changing measured records."""
    groups = {}
    for record in report["Records"]:
        groups.setdefault(record["Model"], {}).setdefault(record["Engine"], []).append(record)
    repeats = report.get("Repeats", 1)
    symbols = {1: "①", 2: "②", 3: "③", 4: "④"}
    lines = ["## CPU 速度排名" if cpu_only_report(report) else "## CPU / GPU 速度排名", "",
             "每格为中位耗时 ms + 名次，①最快、④最慢，第一名加粗；重复窗口取各窗口中位数的中位数，按未舍入值排序。小幅差距不代表稳定胜出或统计显著性。", ""]
    categories = [
        ("CPU", ["torch-cpu", "net-cpu", "net-onnx-cpu", "ort-net-cpu"],
         ["原版 PyTorch CPU", "我们的 `.pt` CPU", "我们的 ONNX CPU", "ORT CPU（.NET）"]),
        ("GPU", ["torch-cuda", "net-vulkan", "net-onnx-vulkan", "ort-net-cuda"],
         ["原版 PyTorch CUDA", "我们的 `.pt` Vulkan", "我们的 ONNX Vulkan", "ORT CUDA（.NET）"]),
    ]
    if cpu_only_report(report):
        categories = categories[:1]
    for name, engines, labels in categories:
        if not cpu_only_report(report):
            lines += [f"### {name} 速度排名", ""]
        lines += ["| 模型 | " + " | ".join(labels) + " |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        winners = {engine: 0 for engine in engines}
        eligible_count = 0
        for model in report["Models"]:
            runs = groups.get(model, {})
            values = {engine: statistics.median(record["MedianMs"] for record in runs[engine])
                      for engine in engines if runs.get(engine)}
            complete = all(len(runs.get(engine, [])) >= repeats for engine in engines)
            if complete:
                eligible_count += 1
            cells = []
            for engine in engines:
                if engine not in values:
                    cells.append("N/A")
                    continue
                cell = f"{values[engine]:.2f}"
                if complete:
                    rank = 1 + sum(value < values[engine] for value in values.values())
                    cell += " " + symbols[rank]
                    if rank == 1:
                        winners[engine] += 1
                        cell = "**" + cell + "**"
                cells.append(cell)
            lines.append("| " + model + " | " + " | ".join(cells) + " |")
        lines += [f"| 第一名次数（完整四配置 {eligible_count}/{len(report['Models'])} 模型） | " +
                  " | ".join(str(winners[engine]) for engine in engines) + " |", ""]
    return lines


def render_repeat_limits(report):
    """Describe observed repeat variation without assigning its cause."""
    groups = {}
    for record in report["Records"]:
        groups.setdefault(record["Model"], {}).setdefault(record["Engine"], []).append(record)
    lines = ["## 跨轮差异与限制", ""]
    varying = []
    conservative = {"net-cpu": [], "net-onnx-cpu": []}
    near_ties = []
    near_gpu_baselines = []
    near_ort_baselines = []
    for model in report["Models"]:
        runs = groups.get(model, {})
        reference = [r["MedianMs"] for r in runs.get("torch-cpu", [])]
        if len(reference) >= 2 and max(reference) / min(reference) >= 1.2:
            varying.append(f"`{model}` {min(reference):.5f}–{max(reference):.5f} ms")
        if len(reference) >= max(2, report.get("Repeats", 1)):
            for engine in conservative:
                ours = [r["MedianMs"] for r in runs.get(engine, [])]
                if len(ours) >= report.get("Repeats", 1):
                    conservative[engine].append(1 - max(ours) / min(reference))
        for device, first, second in [("CPU", "net-cpu", "net-onnx-cpu"),
                                      ("GPU", "net-vulkan", "net-onnx-vulkan")]:
            a = {r.get("Repeat", 1): r["MedianMs"] for r in runs.get(first, [])}
            b = {r.get("Repeat", 1): r["MedianMs"] for r in runs.get(second, [])}
            shared = a.keys() & b.keys()
            if len(shared) < 2:
                continue
            differences = [a[i] - b[i] for i in shared]
            mean_a, mean_b = statistics.median(a.values()), statistics.median(b.values())
            difference = abs(mean_a - mean_b)
            fraction = difference / min(mean_a, mean_b)
            if min(differences) < 0 < max(differences) and fraction < .01:
                near_ties.append(f"`{model}` {device} 的 `.pt` / ONNX 综合仅差 {difference:.6f} ms（约 {fraction*100:.3f}%）")
        ort_cpu = {r.get("Repeat", 1): r["MedianMs"] for r in runs.get("ort-net-cpu", [])}
        for engine, label in [("net-cpu", ".pt"), ("net-onnx-cpu", "ONNX")]:
            ours = {r.get("Repeat", 1): r["MedianMs"] for r in runs.get(engine, [])}
            shared = ort_cpu.keys() & ours.keys()
            if len(shared) < max(2, report.get("Repeats", 1)):
                continue
            a, b = statistics.median(ours.values()), statistics.median(ort_cpu.values())
            fraction = abs(a - b) / min(a, b)
            if fraction < .01:
                differences = [ours[i] - ort_cpu[i] for i in shared]
                swapped = "，两轮顺序交换" if min(differences) < 0 < max(differences) else ""
                near_ort_baselines.append(f"`{model}` {label} / ORT CPU 差 {abs(a-b):.6f} ms（约 {fraction*100:.3f}%）{swapped}")
        cuda = {r.get("Repeat", 1): r["MedianMs"] for r in runs.get("torch-cuda", [])}
        for engine, label in [("net-vulkan", ".pt"), ("net-onnx-vulkan", "ONNX")]:
            ours = {r.get("Repeat", 1): r["MedianMs"] for r in runs.get(engine, [])}
            shared = cuda.keys() & ours.keys()
            if len(shared) < max(2, report.get("Repeats", 1)):
                continue
            a, b = statistics.median(ours.values()), statistics.median(cuda.values())
            fraction = abs(a - b) / min(a, b)
            if fraction < .01:
                differences = [ours[i] - cuda[i] for i in shared]
                swapped = "，两轮顺序交换" if min(differences) < 0 < max(differences) else ""
                near_gpu_baselines.append(f"`{model}` {label} / PyTorch CUDA 差约 {fraction*100:.3f}%{swapped}")
    if varying:
        lines += ["PyTorch CPU 有明显跨轮波动：" + "；".join(varying) +
                  "。原因尚未定位；两个原始窗口都保留，两轮汇总为两个窗口中位数的平均值。", ""]
    if all(len(v) == len(report["Models"]) and min(v) > 0 for v in conservative.values()):
        pt, onnx = conservative["net-cpu"], conservative["net-onnx-cpu"]
        lines += [f"逐模型取本库较慢窗口与 PyTorch CPU 较快窗口比较，两种格式的全部 {len(report['Models'])} 个模型仍更快：`.pt` 耗时降低 {min(pt)*100:.2f}–{max(pt)*100:.2f}%，ONNX 降低 {min(onnx)*100:.2f}–{max(onnx)*100:.2f}%。这支持本次观测的 CPU 结论，不构成统计显著性检验。", ""]
    if near_ties:
        lines += ["近似并列且两轮换位：" + "；".join(near_ties) + "。表格保留数字顺序，不能据此声称某条路径稳定更快。", ""]
    if near_gpu_baselines:
        lines += ["与原版 GPU 接近（本次汇总耗时差不足 1%）：" + "；".join(near_gpu_baselines) + "。数值排名第一不代表稳定领先；两次重复不足以判断统计显著性。", ""]
    if near_ort_baselines:
        lines += ["与 ORT CPU 接近（本次汇总耗时差不足 1%）：" + "；".join(near_ort_baselines) + "。表格保留未舍入数字的排名，不能据微小差距声称稳定超越。", ""]
    lines += ["每个模型固定先运行 PyTorch CPU 以生成输出对照，再随机排列其余后端；模型顺序固定。仍可能存在顺序与热状态偏差；两次重复不能提供跨环境的置信区间。", ""]
    return lines


def render_startup_timings(report):
    """启动拆分独立展示，外部等待全部预热就绪的时间不冒充首次输出时间。"""
    if not any("ProcessStartupMs" in record for record in report["Records"]):
        return []
    lines = ["## 首次推理与启动拆分", "",
             "下表是各独立进程重复值的中位数，单位 ms。模型加载只计构造/权重读取、融合和设备或 ORT 会话创建，不含读输入文件及推理。首次推理只计第一次前向及全部输出回读；程序内至首输出从基准设置入口计起，不含 Python/.NET 宿主启动。", "",
             "全部预热包含首次推理。外部启动至 ready 从控制器调用 Popen 到观测预热完成文件，包含宿主、导入、设置、首次推理及剩余预热，还包含约 50 ms 轮询粒度的观测误差；它不等于首次输出时间。原 StartupMs 仍按程序内入口至全部预热完成定义写入 JSON。Python 导入列只计 torch/Ultralytics 或 ORT 导入及选包检查；numpy/psutil 在 worker 入口前导入，其成本由外部启动指标覆盖。N/A 表示此后端未提供该字段，不估算或补零。", "",
             "每轮使用新进程，但系统文件缓存及已存在的 GPU 管线缓存可能已热；不能将这些数据解释为清空所有系统/磁盘缓存后的首次安装启动。", "",
             "| 模型 | 后端 | 模型加载 | 首次推理 | 程序内至首输出 | 全部预热 | 外部启动至 ready | Python 导入 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    groups = {}
    for record in report["Records"]:
        groups.setdefault((record["Model"], record["Engine"]), []).append(record)
    for (model, engine), records in groups.items():
        cells = [model, ENGINES[engine]]
        for field in ("ModelLoadMs", "FirstInferenceMs", "ReadyToFirstOutputMs", "WarmupMs", "ProcessStartupMs", "ImportMs"):
            values = [record.get(field) for record in records]
            cells.append(f"{statistics.median(values):.2f}" if all(value is not None for value in values) else "N/A")
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def write_cpu_report(report, destination, no_plot):
    """Render this CPU run without borrowing GPU results from another release."""
    versions = sorted({str(record["Version"]) for record in report["Records"]
                       if record["Engine"].startswith("net-") and record.get("Version")})
    runtime_title = " / ".join(versions) if versions else "SimdYolo"
    expected = len(report["Models"]) * len(report["Engines"]) * report.get("Repeats", 1)
    comparisons = [comparison for record in report["Records"] for comparison in record.get("Correctness", [])]
    ort_intervals = [record["Resources"]["MeanSampleIntervalMs"] for record in report["Records"]
                     if record["Engine"] == "ort-net-cpu" and record["Resources"].get("MeanSampleIntervalMs") is not None]
    interval_note = (f"本次 ORT CPU 各窗口的平均采样间隔为 {min(ort_intervals):.0f}–{max(ort_intervals):.0f} ms。"
                     if ort_intervals else "")
    lines = [f"# {runtime_title}：CPU 推理速度与资源对比", "",
             f"日期：{report['Date']}；CPU：{report['Cpu']}（{report['LogicalProcessors']} 逻辑核心）；内存：{report['TotalRamGiB']:.2f} GiB。", "",
             ("本报告提取同一次完整比较的 64 个 CPU 窗口，未重复计时；本轮 GPU 的 64 个新窗口见 [CPU/GPU 完整报告](optimized-runtime-comparison.md)。"
              if report.get("ParentReport") else
              "本报告只统计本次 CPU 实测窗口，不包含旧版 GPU 结果。GPU 性能未在本轮重新计时；正确性验证见 VALIDATION.md。"), "",
             f"状态：{report.get('Status', 'complete' if len(report['Records']) == expected else 'partial')}；完成 {len(report['Records'])}/{expected} 个独立窗口，共 {sum(record['Iterations'] for record in report['Records']):,} 次推理，输出校验 {sum(comparison['Passed'] for comparison in comparisons)}/{len(comparisons)} 通过。", "",
             f"统一 FP32 / batch=1 / {report['Size']}×{report['Size']}、同一输入张量，CPU 线程上限 {report['Threads']}；预热 {report['Warmup']} 次，每窗口至少 {report['DurationSeconds']:g} 秒、{report['MinimumIterations']} 次，每配置 {report.get('Repeats', 1)} 个独立窗口。排除图片预处理、NMS、mask 后处理及冷首帧。", ""]
    lines += render_speed_rankings(report)
    lines += render_repeat_limits(report)
    lines += render_startup_timings(report)
    groups = {}
    for record in report["Records"]:
        groups.setdefault(record["Model"], {}).setdefault(record["Engine"], []).append(record)
    lines += ["## CPU 与基线的差距", "",
              "耗时倍数 = 本库耗时 / 基线耗时，大于 1 表示本库更慢；RSS 倍数大于 1 表示本库进程内存更高。各项取独立窗口值的中位数；缺少完整重复窗口时显示 N/A。", "",
              "| 模型 | `.pt` / PyTorch 耗时 | `.pt` / ORT 耗时 | ONNX / PyTorch 耗时 | ONNX / ORT 耗时 | `.pt` / ORT RSS | ONNX / ORT RSS |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    metrics = [("net-cpu", "torch-cpu", lambda record: record["MedianMs"]),
               ("net-cpu", "ort-net-cpu", lambda record: record["MedianMs"]),
               ("net-onnx-cpu", "torch-cpu", lambda record: record["MedianMs"]),
               ("net-onnx-cpu", "ort-net-cpu", lambda record: record["MedianMs"]),
               ("net-cpu", "ort-net-cpu", lambda record: record["Resources"]["MeanWorkingSetMiB"]),
               ("net-onnx-cpu", "ort-net-cpu", lambda record: record["Resources"]["MeanWorkingSetMiB"])]
    for model in report["Models"]:
        runs = groups.get(model, {})
        cells = []
        for engine, baseline, metric in metrics:
            if min(len(runs.get(engine, [])), len(runs.get(baseline, []))) < report.get("Repeats", 1):
                cells.append("N/A")
                continue
            ours = statistics.median(metric(record) for record in runs[engine])
            reference = statistics.median(metric(record) for record in runs[baseline])
            cells.append(f"{ours / reference:.2f}×" if reference > 0 else "N/A")
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    lines += ["", "CPU 每帧耗时、累计 CPU 时间和进程内存成本应分别评价。较高 CPU 占用不代表更高效率；较低 CPU 占用也不能单独证明效率优势。更快的延迟可能伴随更多 CPU 时间，需要同时查看 CPU ms/帧和进程 RSS，不能据速度排名宣称资源成本全面领先。进程 RSS 包含运行库、模型权重和缓存，不等于算子工作区大小。", "",
              "## 计时与资源口径", "",
              "各配置独立进程、串行运行。显示值相同仍可能有名次差异；原始中位数完全相同则并列，第一名次数分别计入，因此合计可能超过模型数。四配置及全部重复窗口齐全后才排名和统计第一。", "",
              "计时包含网络前向与输出头解码，排除加载、首次编译、图片预处理、NMS 和 mask 后处理。PyTorch 使用原版 Ultralytics fuse()，没有 torch.compile；ORT 开启所有图优化，顺序执行、默认线程自旋。", "",
              "CPU% = 进程 CPU 秒 / 计时窗口秒 / 逻辑核心数 × 100；它不是单核心口径。CPU ms/帧由窗口累计进程 CPU 时间计算。内存是进程工作集 RSS；内存%以物理 RAM 总量为分母。Private MiB 是 Windows 进程私有提交量。", "",
              f"外部采样间隔目标为 {report.get('SampleIntervalSeconds', .25)*1000:g} ms；CPU 满载时调度可能延迟。{interval_note}整机 CPU、RSS 为按采样点计算的采样平均或采样峰值，不按实际采样间隔加权。真实时间戳与采样次数保留在 JSON；进程 CPU%和每帧延迟不依赖该采样平均方式。", "",
              "主要 ORT 对照使用官方 .NET 宿主，与本库共享相同 .NET 运行时，并把全部输出复制为托管 float[]。主库没有新增 ORT 或 Python 依赖；它们只用于对照测试。内存对比是各宿主进程的总成本。", "",
              f"输入：`{report.get('InputDescription', '未记录')}`；SHA256：`{report.get('InputSha256', '未记录')}`。主库 SHA256：`{report.get('SimdYoloDllSha256', '未记录')}`。", "",
              "## 资源与延迟明细", "",
              "外部资源列为采样平均；进程 CPU% / CPU ms/帧由窗口累计时间计算。每个 Repeat 保留独立行。", "",
              "| 模型 | 后端 | Repeat | 中位 ms | P95 ms | 吞吐 FPS | 进程 CPU% | 整机 CPU%（采样） | CPU ms/帧 | RSS MiB（采样） | 内存%（采样） |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for record in report["Records"]:
        resources = record["Resources"]
        lines.append(f"| {record['Model']} | {ENGINES[record['Engine']]} | {record.get('Repeat', 1)} | {record['MedianMs']:.2f} | {record['P95Ms']:.2f} | {record['ThroughputFps']:.1f} | {record['CpuPercent']:.2f} | {resources['MeanSystemCpuPercent']:.2f} | {record['CpuMsPerFrame']:.2f} | {resources['MeanWorkingSetMiB']:.1f} | {resources['MeanWorkingSetPercent']:.2f} |")
    lines += ["", "单进程采样、全部延迟、CPU 时间、RSS/私有提交量峰值、输出正确性、版本、输入与模型 SHA256 均在同名 JSON。启动/首帧时间单独保留，不计入稳态速度。输出字段按 `atol=0.003, rtol=0.0003` 对照本次原版 PyTorch CPU；端到端 TopK 分数使用 `atol=0.0000002, rtol=0.0003`，并对完整解码候选作唯一 anchor/class 配对，连同 mask 系数及关键点整行校验，允许合法同分排序差异。完整候选在计时后生成，未计入延迟或资源窗口。", "",
              "本次结果取决于 CPU、宿主、线程策略、模型和输入，不能外推到其他机器、模型规模、FP16/INT8 或 torch.compile。", "",
              "参考：[ORT 线程与自旋](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)。", ""]
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    if not no_plot and report["Records"]:
        write_charts(report, destination, runtime_title)


def write_report(report, destination, no_plot):
    if cpu_only_report(report):
        write_cpu_report(report, destination, no_plot)
        return
    versions = sorted({str(r["Version"]) for r in report["Records"]
                       if r["Engine"].startswith("net-") and r.get("Version")})
    runtime_title = " / ".join(versions) if versions else "SimdYolo"
    ort_intervals = [r["Resources"]["MeanSampleIntervalMs"] for r in report["Records"]
                     if r["Engine"] == "ort-net-cpu" and r["Resources"].get("MeanSampleIntervalMs") is not None]
    interval_note = (f"本次 ORT CPU 各窗口的平均采样间隔为 {min(ort_intervals):.0f}–{max(ort_intervals):.0f} ms。"
                     if ort_intervals else "")
    expected = len(report["Models"])*len(report["Engines"])*report.get("Repeats", 1)
    comparisons = [c for r in report["Records"] for c in r.get("Correctness", [])]
    lines = [f"# {runtime_title}：推理速度与资源对比", "", f"日期：{report['Date']}；CPU：{report['Cpu']}（{report['LogicalProcessors']} 逻辑核心）；内存：{report['TotalRamGiB']:.2f} GiB；GPU：{report['Gpu'].get('Name', '不可用')}，驱动 {report['Gpu'].get('Driver', '未记录')}。", "",
             f"状态：{report.get('Status', 'complete' if len(report['Records']) == expected else 'partial')}；完成 {len(report['Records'])}/{expected} 个独立窗口，共 {sum(r['Iterations'] for r in report['Records']):,} 次推理，输出校验 {sum(c['Passed'] for c in comparisons)}/{len(comparisons)} 通过。", "",
             f"统一 FP32 / batch=1 / {report['Size']}×{report['Size']}、同一输入张量，CPU 线程上限 {report['Threads']}；预热 {report['Warmup']} 次，每窗口至少 {report['DurationSeconds']:g} 秒、{report['MinimumIterations']} 次，每配置 {report.get('Repeats', 1)} 个独立窗口。GPU 包含输入上传和全部输出回读；排除图片预处理、NMS、mask 后处理及冷首帧。", ""]
    lines += render_speed_rankings(report)
    lines += render_repeat_limits(report)
    lines += render_startup_timings(report)
    methods = ["## 计时与资源口径", "",
             "各配置独立进程、串行运行。显示值相同仍可能有名次差异；原始中位数完全相同则并列，第一名次数分别计入，因此合计可能超过模型数。缺项显示 N/A，四配置及全部重复窗口齐全后才排名和统计第一。", "",
             "GPU 耗时全部包含输入上传、全部输出回读。计时包含网络前向与输出头解码，排除加载、首次编译、图片预处理、NMS、mask 后处理。PyTorch 使用原版 Ultralytics fuse()，没有 torch.compile；ORT 开启所有图优化，顺序执行、默认线程自旋，CUDA 卷积自动选择算法，关闭 TF32。", "",
             "CPU% = 进程 CPU 秒 / 计时窗口秒 / 逻辑核心数 × 100；它不是单核心口径。CPU ms/帧用于判断同样工作量消耗的 CPU 时间，避免把低占用直接解释成效率高。内存是进程工作集 RSS，包括运行库、模型和缓存；内存%以物理 RAM 总量为分母。Private MiB 是 Windows 进程私有提交量。", "",
             f"外部采样间隔目标为 {report.get('SampleIntervalSeconds', .25)*1000:g} ms；CPU 满载时调度可能延迟。{interval_note}整机 CPU、RSS、GPU/显存均为按采样点计算的采样平均或采样峰值，不按实际采样间隔加权。真实时间戳与采样次数保留在 JSON；进程 CPU%来自窗口累计 CPU 时间，每帧延迟由后端自计时，两者不依赖该采样平均方式。", "",
             "GPU% 和显存由独立控制进程通过 NVML 采样，为整张显卡，包含桌面及其他进程；不伪装成单个推理进程的使用率。舍弃 GPU 窗口第一秒，避免滚动采样混入启动阶段；记录每次启动前的 GPU/显存快照及显存增量。启动前快照不保证完全空闲，可能含前一个进程的滚动历史；显存增量也不是准确的进程显存。CPU 推理期间 GPU 数字是背景活动，不能归因于 CPU 后端。", "",
             "主要 ORT 对照使用 .NET 宿主，与我们库共享相同 .NET 运行时，并把全部输出复制为托管 float[]。可选 Python ORT 行使用 Python 绑定，不导入 PyTorch。CUDA ORT 直接预加载现有 cuDNN/CUDA DLL，主库没有新增 ORT 依赖。内存对比是宿主进程总成本，包括不同运行库和缓存。", "",
             "此机器的 NuGet ORT 1.26 GPU 原生构建在 QuickGelu 报 `cudaErrorNoKernelImageForDevice`；正式 .NET ORT 改用官方 Python GPU wheel 中的同版本原生 DLL，通过官方 .NET Managed 绑定调用。该构建已在 1050 Ti 上执行和校验，DLL 哈希写入 JSON；运行时没有调用 Python。", "",
             f"输入：`{report.get('InputDescription', '未记录')}`；SHA256：`{report.get('InputSha256', '未记录')}`。主库 SHA256：`{report.get('SimdYoloDllSha256', '未记录')}`。", ""]
    lines += methods
    resources = ["## 资源与延迟明细", "",
                 "外部资源列为采样平均；进程 CPU% / CPU ms/帧由窗口累计时间计算。每个 Repeat 保留独立行。", "",
                 "| 模型 | 后端 | Repeat | 中位 ms | P95 ms | 吞吐 FPS | 进程 CPU% | 整机 CPU%（采样） | CPU ms/帧 | RSS MiB（采样） | 内存%（采样） | GPU%（整卡采样） | 显存 MiB（整卡采样） |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda value: "N/A" if value is None else f"{value:.2f}"
    for r in report["Records"]:
        s = r["Resources"]
        resources.append(f"| {r['Model']} | {ENGINES[r['Engine']]} | {r.get('Repeat', 1)} | {r['MedianMs']:.2f} | {r['P95Ms']:.2f} | {r['ThroughputFps']:.1f} | {r['CpuPercent']:.2f} | {s['MeanSystemCpuPercent']:.2f} | {r['CpuMsPerFrame']:.2f} | {s['MeanWorkingSetMiB']:.1f} | {s['MeanWorkingSetPercent']:.2f} | {fmt(s['MeanGpuPercent'])} | {fmt(s['MeanDeviceVramMiB'])} |")
    resources += ["", "单进程采样、全部延迟、CPU 时间、RSS/私有提交量峰值、GPU/显存原始时间序列、输出正确性、版本、输入与模型 SHA256 均在同名 JSON。启动/首帧时间单独保留，不计入稳态速度。输出字段按 `atol=0.003, rtol=0.0003` 对照本次原版 PyTorch CPU；端到端 TopK 分数使用 `atol=0.0000002, rtol=0.0003`，并对完整解码候选作唯一 anchor/class 配对，连同 mask 系数及关键点整行校验，允许合法同分排序差异。完整候选在计时后生成，未计入延迟或资源窗口。", "",
              "参考：[ORT CUDA 配置](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、[ORT 线程与自旋](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)、[NVML GPU 忙碌率定义](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlUtilization__t.html)。", ""]
    groups = {}
    for r in report["Records"]:
        groups.setdefault(r["Model"], {}).setdefault(r["Engine"], []).append(r)
    summary = []
    cpu_key = "net-cpu" if "net-cpu" in report["Engines"] else "net-onnx-cpu"
    gpu_key = "net-vulkan" if "net-vulkan" in report["Engines"] else "net-onnx-vulkan"
    keys = [cpu_key, "torch-cpu", "ort-net-cpu", gpu_key, "torch-cuda", "ort-net-cuda"]
    summary += ["## 本库 .pt 与基线的差距", "", "耗时倍数 = 本库 .pt 耗时 / 基线耗时，大于 1 表示本库更慢；各项先取独立窗口值的中位数。内存节省按相同模型的进程 RSS 采样平均计算；CPU 节省按每帧进程 CPU 时间计算。", "",
                "| 模型 | CPU / PyTorch | CPU / ORT | Vulkan / PyTorch CUDA | Vulkan / ORT CUDA | Vulkan RSS 节省 vs PyTorch | Vulkan RSS 节省 vs ORT | Vulkan CPU时间/帧 节省 vs PyTorch | Vulkan CPU时间/帧 节省 vs ORT |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, engines in groups.items():
        if not all(e in engines for e in keys):
            continue
        def val(engine, metric):
            return statistics.median(metric(r) for r in engines[engine])
        latency = lambda r: r["MedianMs"]
        rss = lambda r: r["Resources"]["MeanWorkingSetMiB"]
        cpu = lambda r: r["CpuMsPerFrame"]
        ratios = [val(cpu_key, latency)/val("torch-cpu", latency), val(cpu_key, latency)/val("ort-net-cpu", latency),
                  val(gpu_key, latency)/val("torch-cuda", latency), val(gpu_key, latency)/val("ort-net-cuda", latency)]
        savings = [(1-val(gpu_key, metric)/val(baseline, metric))*100 for metric in [rss, cpu] for baseline in ["torch-cuda", "ort-net-cuda"]]
        summary.append("| " + model + " | " + " | ".join([*(f"{v:.2f}×" for v in ratios), *(f"{v:.1f}%" for v in savings)]) + " |")
    cpu_memory = []
    for engine, label in [("net-cpu", ".pt CPU"), ("net-onnx-cpu", "ONNX CPU")]:
        for baseline, reference in [("torch-cpu", "PyTorch CPU"), ("ort-net-cpu", "ORT CPU")]:
            excess = []
            for model in report["Models"]:
                runs = groups.get(model, {})
                if min(len(runs.get(engine, [])), len(runs.get(baseline, []))) < report.get("Repeats", 1):
                    continue
                ours = statistics.median(r["Resources"]["MeanWorkingSetMiB"] for r in runs[engine])
                other = statistics.median(r["Resources"]["MeanWorkingSetMiB"] for r in runs[baseline])
                excess.append((ours / other - 1) * 100)
            if len(excess) == len(report["Models"]) and min(excess) > 0:
                cpu_memory.append(f"{label} 进程 RSS 比 {reference} 高 {min(excess):.1f}–{max(excess):.1f}%")
    if cpu_memory:
        summary += ["", "CPU 内存存在劣势：" + "；".join(cpu_memory) + "。CPU 每帧耗时和内存应分别评估。"]
    summary += ["", "GPU 的低进程 CPU 时间与低进程 RSS 是本次观察到的资源成本优势，不能解读为显存或功耗优势。高 GPU 忙碌率只说明有内核在执行，不能证明算子效率、能效或计算单元占用率。显卡、驱动、宿主、线程策略和本次输入都会影响结果，不能外推到其他机器、模型规模、FP16/INT8、TensorRT、torch.compile 或 Metal。", ""]
    lines += summary + resources
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    if no_plot or not report["Records"]:
        return
    write_charts(report, destination, runtime_title)


def write_charts(report, destination, runtime_title):
    if (ROOT / ".cache/reference").is_dir():
        sys.path.insert(0, str(ROOT / ".cache/reference"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = report["Models"]
    engines = report["Engines"]
    fig, axes = plt.subplots(2, 2, figsize=(17, max(10, len(models)*1.45)))
    metrics = [("Median latency (ms), lower is better", lambda r: r["MedianMs"]),
               ("Process CPU (% of all logical CPUs)", lambda r: r["CpuPercent"]),
               ("Process working set (MiB, includes host runtime)", lambda r: r["Resources"]["MeanWorkingSetMiB"]),
               ("GPU busy (%, whole device; CPU rows omitted)", lambda r: r["Resources"]["MeanGpuPercent"] if "cuda" in r["Engine"] or "vulkan" in r["Engine"] else None)]
    if cpu_only_report(report):
        metrics[-1] = ("Process CPU time (ms/frame), lower is better", lambda record: record["CpuMsPerFrame"])
    height = .8 / len(engines)
    for ax, (title, metric) in zip(axes.flat, metrics):
        for j, engine in enumerate(engines):
            values = []
            for model in models:
                v = [metric(r) for r in report["Records"] if r["Model"] == model and r["Engine"] == engine]
                v = [x for x in v if x is not None]
                values.append(statistics.median(v) if v else float("nan"))
            ax.barh(np.arange(len(models)) + (j-(len(engines)-1)/2)*height, values, height=height*.92, color=ENGINE_COLORS.get(engine, "#111827"), label=ENGINES[engine])
        ax.set_yticks(np.arange(len(models)), models)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    hardware = report["Cpu"] if cpu_only_report(report) else f"{report['Cpu']} / {report['Gpu'].get('Name', 'GPU unavailable')}"
    fig.suptitle(f"{runtime_title} | FP32, batch 1, {report['Size']}x{report['Size']} | {report['Threads']} threads | {report.get('Repeats', 1)} windows/config\n{hardware}", fontsize=13)
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(destination.with_suffix(".png"), dpi=170)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--onnx-dir", type=Path, default=ROOT / ".cache/validation")
    parser.add_argument("--net-cli", type=Path, default=ROOT / "src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll",
                        help="冻结基线或候选 SimdYolo CLI DLL；主库哈希取同目录 SimdYolo.dll")
    parser.add_argument("--ultralytics-dir", type=Path, default=ROOT / ".cache/reference",
                        help="原版 Ultralytics 包目录；YOLO26 可用 .cache/yolo26-release-reference")
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolov9t", "yolov10n", "yolo11n"])
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--engines", nargs="+", choices=list(ENGINES), default=DEFAULT_ENGINES)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--threads", type=int, default=psutil.cpu_count())
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--duration", type=float, default=8)
    parser.add_argument("--sample-interval", type=float, default=.25)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runtime-comparison.json")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Rebuild Markdown/plots from an existing output JSON without measuring")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", choices=list(ENGINES), help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--signal", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.net_cli = args.net_cli.resolve()
    args.ultralytics_dir = args.ultralytics_dir.resolve()
    if min(args.size, args.threads, args.warmup, args.iterations, args.repeats) < 1 or args.size % 32 or args.duration < 2 or args.sample_interval <= 0:
        parser.error("Use positive sizes/threads/counts, size divisible by 32, duration >=2 seconds")
    if args.worker:
        worker(args)
        return
    if args.render_only:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        expected = len(report["Models"])*len(report["Engines"])*report["Repeats"]
        report.setdefault("Status", "complete" if len(report["Records"]) == expected else "partial")
        write_report(report, args.output, args.no_plot)
        return
    if "torch-cpu" not in args.engines:
        parser.error("torch-cpu must be included as the output oracle")
    if not (args.ultralytics_dir / "ultralytics").is_dir():
        parser.error(f"Missing Ultralytics package directory: {args.ultralytics_dir}")
    if any(engine.startswith("net-") for engine in args.engines):
        for path in (args.net_cli, args.net_cli.with_name("SimdYolo.dll")):
            if not path.is_file():
                parser.error(f"Missing selected .NET CLI/library: {path}")
    if args.vision:
        args.models += ["yolov8n-seg", "yolov8n-pose", "yolo11n-seg", "yolo11n-pose"]
    args.models = list(dict.fromkeys(args.models))
    args.engines = list(dict.fromkeys(args.engines))
    for model in args.models:
        for path in [args.model_dir / f"{model}.pt", args.onnx_dir / f"{model}.onnx"]:
            if not path.is_file():
                parser.error(f"Missing fixture: {path}")
    work = ROOT / ".cache/runtime-comparison"
    work.mkdir(parents=True, exist_ok=True)
    input_path = args.input.resolve() if args.input else work / "input.f32"
    if not args.input:
        np.random.default_rng(20261013).uniform(0, 1, (1, 3, args.size, args.size)).astype(np.float32).tofile(input_path)
    if input_path.stat().st_size != 3 * args.size * args.size * 4:
        parser.error("Input tensor byte count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gpu = GpuSampler()
    report = {"Date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "Platform": platform.platform(), "Cpu": cpu_name(),
              "LogicalProcessors": psutil.cpu_count(), "TotalRamGiB": psutil.virtual_memory().total / 1073741824,
              "Gpu": gpu.info, "GpuSamplingError": gpu.error, "DotnetSdk": subprocess.check_output(["dotnet", "--version"], text=True).strip(),
              "Python": sys.version, "Size": args.size, "Threads": args.threads, "Warmup": args.warmup,
              "DurationSeconds": args.duration, "MinimumIterations": args.iterations, "SampleIntervalSeconds": args.sample_interval,
              "Status": "running", "Repeats": args.repeats, "Models": args.models, "Engines": args.engines, "InputSha256": sha256(input_path),
              "InputDescription": str(input_path),
              "NetCliPath": str(args.net_cli), "UltralyticsDirectory": str(args.ultralytics_dir),
              "SimdYoloDllSha256": sha256(args.net_cli.with_name("SimdYolo.dll")) if args.net_cli.with_name("SimdYolo.dll").is_file() else None,
              "StartupTimingDefinitions": {
                  "ProcessStartupMs": "External Popen invocation to ready-file observation, including runtime startup, setup, imports, first inference and all warmup; target readiness poll interval is 50 ms. Not time to first output.",
                  "ImportMs": "Python framework/package selection and import. numpy/psutil were imported before worker entry; their cost is captured by ProcessStartupMs.",
                  "ModelLoadMs": "Model construction/loading/fusion and device placement; ORT includes provider/DLL preparation and session creation. Excludes input-file read and inference.",
                  "FirstInferenceMs": "First host-visible warmup forward only, including input transfer and all output synchronization when using GPU.",
                  "ReadyToFirstOutputMs": "Worker benchmark-setup entry through the first host-visible output; excludes interpreter/dotnet host startup.",
                  "WarmupMs": "All warmup forwards, including the first one.",
                  "StartupMs": "Original definition retained: worker benchmark-setup entry through completion of all warmup; excludes host startup."},
              "OrtNativeDllSha256": {p.name: sha256(p) for p in (ROOT / ".cache/ort-gpu/onnxruntime/capi").glob("onnxruntime*.dll")},
              "ModelsSha256": {model: {"pt": sha256(args.model_dir / f"{model}.pt"), "onnx": sha256(args.onnx_dir / f"{model}.onnx")} for model in args.models}, "Records": []}
    randomizer = random.Random(20261013)
    try:
        for repeat in range(1, args.repeats + 1):
            for model in args.models:
                # Run the original reference first, then shuffle the remaining
                # engines, to reduce systematic order bias across models.
                rest = [e for e in args.engines if e != "torch-cpu"]
                randomizer.shuffle(rest)
                reference = None
                reference_dense, reference_nc, reference_end_to_end = None, 0, False
                for engine in ["torch-cpu", *rest]:
                    print(f"START repeat={repeat} {model} {engine}", flush=True)
                    record, outputs = run_case(model, engine, repeat, args, input_path, gpu)
                    if reference is None:
                        reference = outputs
                        reference_end_to_end = bool(record.get("EndToEnd", False))
                        reference_nc = int(record.get("ClassCount", 0))
                        if reference_end_to_end:
                            oracle = record.get("Oracle")
                            if not oracle:
                                raise AssertionError(f"Complete end-to-end candidate oracle missing: {model}")
                            dense_path = Path(oracle["DenseReferenceFile"])
                            with np.load(dense_path) as archive:
                                reference_dense = archive["dense_reference"].copy()
                                if int(archive["nc"]) != reference_nc:
                                    raise AssertionError(f"Oracle class count mismatch: {model}")
                            oracle.update({"Sha256": sha256(dense_path), "Shape": list(reference_dense.shape)})
                    record["Correctness"] = check_outputs(model, outputs, reference, reference_end_to_end, reference_dense, reference_nc)
                    report["Records"].append(record)
                    write_json(args.output, report)
                    s = record["Resources"]
                    print(f"DONE {record['MedianMs']:.2f} ms | CPU {record['CpuPercent']:.1f}% | RSS {s['MeanWorkingSetMiB']:.0f} MiB | GPU {s['MeanGpuPercent']}% | n={record['Iterations']}", flush=True)
        report["Status"] = "complete"
    except BaseException as error:
        report["Status"] = "failed"
        report["Error"] = str(error)
        raise
    finally:
        gpu.close()
        write_json(args.output, report)
        write_report(report, args.output, args.no_plot)
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
