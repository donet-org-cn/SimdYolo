"""Render verified 0.8.0 -> 0.8.1 lifecycle and eight-engine runtime evidence.

Standard library only. Reject incomplete, duplicate, mixed-version, mixed-input
or failed-correctness reports. No inference, performance sampling or rewriting
of source reports occurs. Optional lifecycle validation is bound by file SHA256.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path, PureWindowsPath
import re
import statistics

ROOT = Path(__file__).resolve().parent.parent
ENGINES = ["torch-cpu", "net-cpu", "net-onnx-cpu", "ort-net-cpu",
           "torch-cuda", "net-vulkan", "net-onnx-vulkan", "ort-net-cuda"]
LABELS = {"torch-cpu": "原版 PyTorch", "net-cpu": "我们的 .pt SIMD", "net-onnx-cpu": "我们的 ONNX SIMD", "ort-net-cpu": "ORT .NET CPU",
          "torch-cuda": "原版 PyTorch CUDA", "net-vulkan": "我们的 .pt Vulkan", "net-onnx-vulkan": "我们的 ONNX Vulkan", "ort-net-cuda": "ORT .NET CUDA"}
LIFECYCLE_METRICS = [("ModelLoadMs", "加载 ms"), ("FirstInferenceMs", "首帧 ms"),
                     ("ReadyToFirstOutputMs", "准备至首输出 ms"), ("MedianMs", "稳态 ms"),
                     ("AllocatedBytesPerFrame", "托管分配 MiB/帧")]


def require(condition, message):
    """Reject ambiguous or incomplete evidence instead of generating a ranking."""
    if not condition:
        raise ValueError(message)


def read(path):
    """Load immutable input evidence and bind it to the exact file digest."""
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def digest(value, name):
    """Require an actual SHA256 value, never compare missing hashes as equal."""
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None, f"Invalid {name} SHA256")
    return value.lower()


def filename(path):
    """Read recorded Windows file names correctly when rendering on another OS."""
    return PureWindowsPath(path).name if "\\" in path else Path(path).name


def number(value, name, positive=False):
    """Check finite nonnegative measurements; timing windows must be positive."""
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0), f"Invalid {name}: {value}")
    return float(value)


def verify(runtime, lifecycle, validation, lifecycle_sha):
    """Validate full matrices, repeat identities, hashes and output check coverage."""
    require(runtime.get("Status") == lifecycle.get("Status") == "complete", "Both source reports must be complete")
    models = runtime.get("Models", [])
    require(len(models) == 11 and len(set(models)) == 11, "Expected exactly 11 unique models")
    require(set(runtime.get("Engines", [])) == set(ENGINES) and len(runtime["Engines"]) == 8, "Expected the complete eight-engine matrix")
    repeats = runtime.get("Repeats")
    life_repeats = lifecycle.get("Settings", {}).get("Repeats")
    require(type(repeats) is int and repeats > 0 and type(life_repeats) is int and life_repeats > 0, "Invalid repeat counts")
    candidate = digest(lifecycle.get("Variants", {}).get("candidate", {}).get("SnapshotDllSha256"), "candidate library")
    baseline = digest(lifecycle.get("Variants", {}).get("baseline", {}).get("SnapshotDllSha256"), "baseline library")
    require(candidate != baseline, "Baseline and candidate must be distinct snapshots")
    require(candidate == digest(runtime.get("SimdYoloDllSha256"), "runtime library"), "Runtime and lifecycle candidate libraries differ")
    input_sha = digest(runtime.get("InputSha256"), "runtime input")
    require(input_sha == digest(lifecycle.get("InputSha256"), "lifecycle input"), "Source input tensors differ")
    settings = lifecycle["Settings"]
    require(runtime.get("Size") == settings.get("Size") == 640 and runtime.get("Threads") == settings.get("Threads") == 12, "Expected matched 640 RGB input and 12-thread configuration")
    require(runtime.get("Cpu") == lifecycle.get("Hardware", {}).get("Cpu") and runtime.get("LogicalProcessors") == lifecycle.get("Hardware", {}).get("LogicalProcessors"), "Source CPU/hardware descriptions differ")
    require(runtime.get("Platform") == lifecycle.get("Hardware", {}).get("Os"), "Source OS descriptions differ")
    for variant in ("baseline", "candidate"):
        info = lifecycle["Variants"][variant]
        require(digest(info.get("SourceDllSha256"), variant + " source") == digest(info.get("SnapshotDllSha256"), variant + " snapshot"), "Variant changed while preparing its snapshot")
    model_hashes = {}
    for item in lifecycle.get("Models", []):
        name = filename(item["Path"])
        require(name not in model_hashes, "Duplicate lifecycle model file")
        model_hashes[name] = digest(item.get("Sha256"), name)
    require(set(model_hashes) == {f"{model}.{fmt}" for model in models for fmt in ("pt", "onnx")}, "Lifecycle model hash matrix is incomplete")
    for model in models:
        for fmt in ("pt", "onnx"):
            require(model_hashes[f"{model}.{fmt}"] == digest(runtime.get("ModelsSha256", {}).get(model, {}).get(fmt), model + " " + fmt), "Model files differ between source reports")
    runtime_records = runtime.get("Records", [])
    expected = set(itertools.product(models, ENGINES, range(1, repeats + 1)))
    seen = set(); correctness = 0
    for record in runtime_records:
        key = (record.get("Model"), record.get("Engine"), record.get("Repeat"))
        require(key not in seen, "Duplicate runtime window: " + str(key)); seen.add(key)
        require(key in expected, "Unexpected runtime window: " + str(key))
        fmt = ".onnx" if "onnx" in record["Engine"] or record["Engine"].startswith("ort-") else ".pt"
        require(record.get("ModelFormat") == fmt, "Runtime engine/model format mismatch")
        require(record.get("CpuThreads") == 12 and record.get("Shape") == [1, 3, 640, 640] and record.get("Warmup") == runtime.get("Warmup"), "Runtime input/thread/warmup configuration differs")
        if record["Engine"].startswith("net-"):
            require(re.search(r"\b0\.8\.1(?:\.0)?\b", str(record.get("Version"))) is not None, "Runtime .NET window is not 0.8.1")
        checks = record.get("Correctness")
        require(isinstance(checks, list) and checks and all(check.get("Passed") is True for check in checks), "Missing or failed runtime correctness: " + str(key))
        require(len(checks) == len(record.get("Outputs", [])), "Runtime correctness does not cover every output")
        require(all(check.get("Shape") == shape for check, shape in zip(checks, record["Outputs"])), "Runtime checked output shapes differ")
        correctness += len(checks)
        samples = record.get("SamplesMs")
        require(isinstance(samples, list) and len(samples) == record.get("Iterations") and samples, "Runtime sample/frame count mismatch")
        require(len(samples) >= runtime.get("MinimumIterations", 1) and number(record.get("WindowSeconds"), "runtime window", True) + 1e-6 >= runtime.get("DurationSeconds", 0), "Runtime measurement window is too short")
        for value in samples:
            number(value, "runtime sample", positive=True)
        require(math.isclose(statistics.median(samples), number(record.get("MedianMs"), "runtime median", True), rel_tol=1e-9, abs_tol=1e-8), "Runtime median does not match its samples")
    require(seen == expected and len(runtime_records) == len(expected), "Incomplete runtime matrix")
    life_records = lifecycle.get("Records", [])
    expected_life = set(itertools.product(models, ("pt", "onnx"), ("cpu", "vulkan"), range(1, life_repeats + 1), range(1, 5)))
    seen = set(); sequences = set()
    for record in life_records:
        fmt = record.get("Format"); name = record.get("Model", "")
        require(name.endswith("." + str(fmt)), "Lifecycle model filename/format mismatch")
        model = name[:-(len(fmt) + 1)]
        backend = str(record.get("RequestedBackend", "")).lower()
        key = (model, fmt, backend, record.get("Repeat"), record.get("AbbaPosition"))
        require(key in expected_life and key not in seen, "Unexpected or duplicate lifecycle window: " + str(key)); seen.add(key)
        position = record["AbbaPosition"]; variant = "baseline" if position in (1, 4) else "candidate"
        require(record.get("Variant") == variant, "Lifecycle ABBA position/variant mismatch")
        require(record.get("Version") == ("0.8.0" if variant == "baseline" else "0.8.1"), "Mixed library version in lifecycle windows")
        require(digest(record.get("DllSha256"), "window library") == (baseline if variant == "baseline" else candidate), "Mixed DLL in lifecycle windows")
        require(record.get("FallbackReason") is None and record.get("Size") == 640 and record.get("CpuThreads") == 12, "Lifecycle window backend/config mismatch")
        require(record.get("WarmupFrames") == settings.get("WarmupFrames") and record.get("MinimumFrames") == settings.get("MinimumFrames") and record.get("DurationSeconds") == settings.get("DurationSeconds"), "Lifecycle window settings differ")
        require(filename(record.get("ModelPath", "")) == name, "Lifecycle model path/filename mismatch")
        sequence = record.get("Sequence")
        require(type(sequence) is int and sequence > 0 and sequence not in sequences, "Duplicate/invalid lifecycle sequence"); sequences.add(sequence)
        for metric, _ in LIFECYCLE_METRICS:
            number(record.get(metric), "lifecycle " + metric, positive=metric != "AllocatedBytesPerFrame")
        samples = record.get("SamplesMs")
        require(isinstance(samples, list) and len(samples) == record.get("Frames") and samples, "Lifecycle sample/frame count mismatch")
        require(len(samples) >= settings.get("MinimumFrames", 1) and number(record.get("Seconds"), "lifecycle window", True) + 1e-6 >= settings.get("DurationSeconds", 0), "Lifecycle measurement window is too short")
        require(record["ReadyToFirstOutputMs"] + 1e-6 >= record["ModelLoadMs"] + record["FirstInferenceMs"], "Lifecycle ready time is inconsistent with load/first timings")
        for value in samples:
            number(value, "lifecycle sample", positive=True)
        require(math.isclose(statistics.median(samples), record["MedianMs"], rel_tol=1e-9, abs_tol=1e-8), "Lifecycle median does not match samples")
    require(seen == expected_life and len(life_records) == len(expected_life) and sequences == set(range(1, len(life_records) + 1)), "Incomplete lifecycle ABBA matrix")
    if validation is not None:
        require(validation.get("Status") == "passed" and validation.get("Passed") is True and validation.get("ComparisonStatus") == "complete", "Lifecycle validation is not fully passed")
        require(digest(validation.get("ComparisonSha256"), "validation source") == lifecycle_sha, "Lifecycle validation belongs to another source file")
        windows = validation.get("Windows", [])
        require(len(windows) == len(life_records) == validation.get("ExpectedWindows"), "Lifecycle validation window count mismatch")
        checks_total = 0
        for index, (window, record) in enumerate(zip(windows, life_records), 1):
            require(window.get("Window") == index and window.get("Passed") is True, "Missing/failed validation window")
            for field in ("Sequence", "Model", "Format", "RequestedBackend", "Variant", "Repeat", "AbbaPosition", "Version", "DllSha256"):
                require(window.get(field) == record.get(field), "Validation identity mismatch: " + field)
            require(digest(window.get("Input", {}).get("Sha256"), "validated input") == input_sha, "Validation input differs")
            require(digest(window.get("ModelSha256"), "validated model") == model_hashes[record["Model"]], "Validation model differs")
            required_checks = {}
            for phase, descriptor_key in (("first", "FirstOutputs"), ("last", "Outputs")):
                descriptors = record.get(descriptor_key, [])
                require(descriptors and sorted(item.get("Index") for item in descriptors) == list(range(len(descriptors))), "Invalid lifecycle output indices")
                for descriptor in descriptors:
                    required_checks[(phase, descriptor["Index"])] = descriptor
            actual = {}
            for check in window.get("Checks", []):
                key = (check.get("Phase"), check.get("Output"))
                require(key in required_checks and key not in actual and check.get("Passed") is True and check.get("Finite") is True, "Duplicate, missing or failed output check")
                descriptor = required_checks[key]
                require(digest(check.get("Sha256"), "validated output") == digest(descriptor.get("Sha256"), "worker output") and check.get("Shape") == descriptor.get("Shape") and check.get("Bytes") == descriptor.get("Bytes"), "Validated raw output differs from lifecycle evidence")
                actual[key] = check
            require(set(actual) == set(required_checks), "Validation does not cover all first/last outputs")
            checks_total += len(actual)
        summary = validation.get("Summary", {})
        require(summary.get("Windows") == summary.get("PassedWindows") == len(life_records) and summary.get("OutputChecks") == summary.get("PassedOutputChecks") == checks_total, "Validation summary count mismatch")
    return models, correctness


def grouped(records, key):
    """Preserve raw windows while grouping their scalar metrics for rendering."""
    groups = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return groups


def scalar(windows, metric):
    """Return the median of window metrics, not a pooled/frame-weighted median."""
    return statistics.median(record[metric] for record in windows)


def window_range(windows, metric, scale=1):
    """Display the window median and actual min/max of independent windows."""
    values = [record[metric] / scale for record in windows]
    return f"{statistics.median(values):.4f} [{min(values):.4f}, {max(values):.4f}]"


def lower(before, after):
    """Preserve regressions as negative reductions; a zero baseline has no ratio."""
    return f"{(1 - after / before) * 100:+.2f}%" if before else "N/A"


def variability_findings(groups, models):
    """Flag observed spread and nominal ORT wins that cannot establish stability."""
    varying = []
    uncertain = []
    for model, engine in itertools.product(models, ENGINES):
        windows = groups[(model, engine)]
        values = [window["MedianMs"] for window in windows]
        if max(values) / min(values) >= 1.2:
            varying.append((model, engine, min(values), max(values), scalar(windows, "MedianMs"), max(values) / min(values), windows))
    for model in models:
        for ours, reference in (("net-cpu", "ort-net-cpu"), ("net-onnx-cpu", "ort-net-cpu"),
                                ("net-vulkan", "ort-net-cuda"), ("net-onnx-vulkan", "ort-net-cuda")):
            own_windows, ort_windows = groups[(model, ours)], groups[(model, reference)]
            own_median, ort_median = scalar(own_windows, "MedianMs"), scalar(ort_windows, "MedianMs")
            if own_median >= ort_median:
                continue
            own_values = [window["MedianMs"] for window in own_windows]
            ort_values = [window["MedianMs"] for window in ort_windows]
            own_by_repeat = {window["Repeat"]: window["MedianMs"] for window in own_windows}
            ort_by_repeat = {window["Repeat"]: window["MedianMs"] for window in ort_windows}
            differences = [own_by_repeat[repeat] - ort_by_repeat[repeat] for repeat in sorted(own_by_repeat.keys() & ort_by_repeat.keys())]
            reasons = []
            if min(own_values) >= min(ort_values):
                reasons.append("我们的最快窗口仍不快于 ORT 最快窗口")
            if any(value < 0 for value in differences) and any(value > 0 for value in differences):
                reasons.append("重复窗口中的快慢顺序交换")
            if max(own_values) >= min(ort_values):
                reasons.append("两者窗口范围未完全分离")
            if reasons:
                uncertain.append((model, ours, reference, own_median, ort_median,
                                  min(own_values), max(own_values), min(ort_values), max(ort_values), reasons))
    return varying, uncertain


def link(path, output):
    """Link evidence relative to the generated report, with portable slashes."""
    import os
    return Path(os.path.relpath(path, output.parent)).as_posix()


def render(runtime, life, validation, models, correctness, paths, hashes, output):
    """Render rank tables first, followed by honest lifecycle and window evidence."""
    rg = grouped(runtime["Records"], lambda r: (r["Model"], r["Engine"]))
    lg = grouped(life["Records"], lambda r: (r["Model"].rsplit(".", 1)[0], r["Format"], r["RequestedBackend"].lower(), r["Variant"]))
    varying, uncertain = variability_findings(rg, models)
    marked = {(item[0], item[1]) for item in uncertain}
    counts = {}
    for metric in ("ReadyToFirstOutputMs", "MedianMs"):
        count = 0
        for model, fmt, backend in itertools.product(models, ("pt", "onnx"), ("cpu", "vulkan")):
            before = lg[(model, fmt, backend, "baseline")]; after = lg[(model, fmt, backend, "candidate")]
            count += max(r[metric] for r in after) < min(r[metric] for r in before)
        counts[metric] = count
    lines = ["# SimdYolo 0.8.1 实测：原版、我们的实现与 ORT", "",
             f"本轮使用同一台 {runtime['Cpu']}、{runtime['Gpu'].get('Name', 'GPU 名称未记录')}，FP32 / batch=1 / 640×640、12 线程。以下排名只来自本轮 0.8.1 的 {len(runtime['Records'])} 个完整窗口；不拼接历史 0.7.x 测量。稳态输出与原版逐窗口对照 {correctness}/{correctness} 项通过。", "",
             f"0.8.0 → 0.8.1 的独立 ABBA 窗口中，{counts['ReadyToFirstOutputMs']}/44 组配置的新版本准备至首输出时间范围全部低于旧版本范围，{counts['MedianMs']}/44 组配置的稳态中位耗时范围全部低于旧版本范围。这里描述本次窗口的观察结果，不推断统计显著性或跨硬件表现。", "",
             "每格为 ms + 名次；①最快，第一名加粗。各配置取独立窗口中位数的中位数，按未舍入值排名；完全相同的数值共享名次。只有两个窗口时，这个汇总值恰好是两个窗口中位耗时的平均数，一个异常慢窗口就可能改变排名。微小差距及范围重叠不能作为稳定胜出的依据。", ""]
    if varying:
        lines += [f"本轮有 {len(varying)} 个模型/引擎组合的最慢窗口与最快窗口之比达到或超过 1.20；下方逐项列出实际范围，涵盖 ORT、原版与我们的实现，不把波动隐藏在汇总排名中。", ""]
    if uncertain:
        lines += ["表中的 † 表示：汇总数值虽比对应 ORT 小，但最快窗口对照、重复顺序或范围重叠不支持宣称稳定领先。不能借 ORT 的一个异常慢窗口宣称全面超越。", ""]
    for title, engines in (("CPU 速度排名", ENGINES[:4]), ("GPU 速度排名", ENGINES[4:])):
        lines += ["## " + title, "", "| 模型 | " + " | ".join(LABELS[e] for e in engines) + " |", "| --- | " + " | ".join(["---:"] * 4) + " |"]
        for model in models:
            values = [scalar(rg[(model, engine)], "MedianMs") for engine in engines]; ordered = sorted(values)
            cells = []
            for engine, value in zip(engines, values):
                rank = ordered.index(value); cell = f"{value:.4f} {['①', '②', '③', '④'][rank]}"
                cell = "**" + cell + "**" if rank == 0 else cell
                if (model, engine) in marked:
                    cell += " †"
                cells.append(cell)
            lines.append("| " + model + " | " + " | ".join(cells) + " |")
        lines += [""]
    lines += ["## 跨窗口波动与排名限制", "",
              "这里的波动阈值是最慢窗口中位耗时 / 最快窗口中位耗时 ≥ 1.20，用于提示观察到的跨进程变化，不是统计显著性检验。Repeat 相同只表示同一轮次，不表示两个引擎同时测量或处于相同瞬时负载。不能用下午或历史版本的绝对耗时替换本轮较慢的结果。", ""]
    if varying:
        lines += ["| 模型 | 配置 | 最快–最慢窗口 ms | 最慢/最快 | 汇总中位 ms | 各 Repeat 的窗口中位 ms |",
                  "| --- | --- | ---: | ---: | ---: | --- |"]
        for model, engine, minimum, maximum, median, ratio, windows in varying:
            detail = "；".join(f"R{window['Repeat']}={window['MedianMs']:.4f}" for window in sorted(windows, key=lambda r: r["Repeat"]))
            lines.append(f"| {model} | {LABELS[engine]} | {minimum:.4f}–{maximum:.4f} | {ratio:.3f}× | {median:.4f} | {detail} |")
        lines += [""]
    else:
        lines += ["本次完整矩阵没有达到上述阈值的组合；这不等于已经证明跨设备或长期运行稳定性。", ""]
    if uncertain:
        lines += ["### 汇总领先 ORT，但不能宣称稳定领先的组合", "",
                  "| 模型 | 我们的配置 / 对应 ORT | 我们汇总 ms | ORT 汇总 ms | 我们窗口范围 ms | ORT 窗口范围 ms | 原因 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for model, own, reference, own_median, ort_median, own_minimum, own_maximum, ort_minimum, ort_maximum, reasons in uncertain:
            lines.append(f"| {model} | {LABELS[own]} / {LABELS[reference]} | {own_median:.4f} | {ort_median:.4f} | {own_minimum:.4f}–{own_maximum:.4f} | {ort_minimum:.4f}–{ort_maximum:.4f} | {'；'.join(reasons)} |")
        lines += [""]
    else:
        lines += ["没有出现上述汇总领先而窗口证据相矛盾的组合；有限窗口仍不能构成统计显著性或全面超越的证明。", ""]
    lines += ["## 首次使用与稳态：0.8.0 → 0.8.1", "",
              "每格为旧版 → 新版（降低百分比）。正数表示下降，负数表示增加；百分比按未舍入值计算。前四项单位为 ms，分配为 MiB/帧。各指标均取独立进程窗口的中位数。", ""]
    for fmt, backend in itertools.product(("pt", "onnx"), ("cpu", "vulkan")):
        lines += [f"### {'.pt' if fmt == 'pt' else 'ONNX'} · {'CPU SIMD' if backend == 'cpu' else 'Vulkan GPU'}", "",
                  "| 模型 | " + " | ".join(label for _, label in LIFECYCLE_METRICS) + " |", "| --- | " + " | ".join(["---:"] * 5) + " |"]
        for model in models:
            before = lg[(model, fmt, backend, "baseline")]; after = lg[(model, fmt, backend, "candidate")]; cells = []
            for metric, _ in LIFECYCLE_METRICS:
                a, b = scalar(before, metric), scalar(after, metric); scale = 1048576 if metric == "AllocatedBytesPerFrame" else 1
                cells.append(f"{a / scale:.4f} → {b / scale:.4f}（{lower(a, b)}）")
            lines.append("| " + model + " | " + " | ".join(cells) + " |")
        lines += [""]
    lines += ["## 测量边界与正确性", "",
              f"稳态对照每配置 {runtime['Repeats']} 个进程，预热 {runtime['Warmup']} 帧，至少 {runtime['DurationSeconds']:g} 秒和 {runtime['MinimumIterations']} 帧。生命周期每配置 {life['Settings']['Repeats']} 轮 ABBA，每个版本每轮两个全新进程；预热 {life['Settings']['WarmupFrames']} 帧（包括已测首帧），稳态至少 {life['Settings']['DurationSeconds']:g} 秒及 {life['Settings']['MinimumFrames']} 帧。", "",
              "加载包含 YoloDetector 构造、后端初始化、模型解析和权重准备。首帧为首次 PredictRaw；GPU 包括输入上传、计算和全部输出回读。准备至首输出从托管 Main 首条语句开始，包含输入文件读取、会话构造和首帧，排除 dotnet 宿主启动和进入 Main 之前的入口 JIT。ProcessTotalMs 包含进程启动、预热、稳态、写盘和释放，不能用作进程至首输出延迟。", "",
              f"生命周期缓存模式：`{life.get('CacheMode', '未记录')}`。全新进程不会清空操作系统文件缓存或驱动缓存；应用管线缓存是否加载、命令重放次数及缓存文件状态保留在每个原始窗口中。管线缓存加载窗口：{sum(r['PipelineCacheLoaded'] is True for r in life['Records'])}/{len(life['Records'])}；不把这些窗口称为硬件冷启动。", "",
              "托管分配来自稳态 GC 累计分配量，扣除采样列表扩容；它是每帧创建量，既不是 RSS、存活堆大小，也不是峰值内存。公开输出数组计入分配。进程 CPU% 按累计 CPU 时间 / 墙钟 / 逻辑处理器数归一化；CPU ms/帧累加所有线程，因此可能大于墙钟 ms/帧。RSS 是进程驻留内存；GPU% 和显存是整卡采样，包含桌面及其他进程，不能解释为模型独占占用。", "",
              "排名排除图像预处理、NMS 和掩码后处理，比较 FP32 原始模型输出；ORT CUDAExecutionProvider 允许部分节点运行于 CPU，不能描述为整图全部 CUDA。此轮没有 Metal 实机结果。", ""]
    if validation:
        lines += [f"生命周期首帧与末帧已独立对照原版 CPU FP32：{validation['Summary']['PassedWindows']}/{validation['Summary']['Windows']} 个窗口、{validation['Summary']['PassedOutputChecks']}/{validation['Summary']['OutputChecks']} 个输出检查通过；验证文件通过生命周期 JSON 的 SHA256 绑定。端到端头按完整候选和实例附加字段配对，不用逐行排序误判并列 TopK。", ""]
    else:
        lines += ["未提供生命周期独立正确性报告：此处首帧/末帧数据只描述性能，不能据此声称这些输出已与原版独立核验。", ""]
    lines += ["## 稳态资源对照", "", "资源列取独立窗口指标的中位数；RSS、GPU 和显存各窗口先取采样平均。不可用指标为 N/A。", "",
              "| 模型 | 配置 | CPU% | CPU ms/帧 | RSS MiB | GPU%（整卡） | 显存 MiB（整卡） |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model, engine in itertools.product(models, ENGINES):
        windows = rg[(model, engine)]
        def resource(name):
            values = [r.get("Resources", {}).get(name) for r in windows]
            return f"{statistics.median(values):.2f}" if all(type(v) in (int, float) and math.isfinite(v) for v in values) else "N/A"
        lines.append(f"| {model} | {LABELS[engine]} | {scalar(windows, 'CpuPercent'):.2f} | {scalar(windows, 'CpuMsPerFrame'):.4f} | {resource('MeanWorkingSetMiB')} | {resource('MeanGpuPercent')} | {resource('MeanDeviceVramMiB')} |")
    lines += ["", "## 独立窗口附表", "", "中位数 [最小窗口值, 最大窗口值]。这些范围来自独立进程窗口，不是置信区间。", "",
              "### 原版 / 我们 / ORT 的稳态窗口范围", "", "| 模型 | 配置 | 中位 ms [min, max] | 窗口数 |", "| --- | --- | ---: | ---: |"]
    for model, engine in itertools.product(models, ENGINES):
        windows = rg[(model, engine)]; lines.append(f"| {model} | {LABELS[engine]} | {window_range(windows, 'MedianMs')} | {len(windows)} |")
    lines += ["", "### 0.8.0 / 0.8.1 生命周期窗口范围", "", "| 模型 | 格式 / 设备 | 版本 | 窗口数 | " + " | ".join(label for _, label in LIFECYCLE_METRICS) + " |",
              "| --- | --- | --- | ---: | " + " | ".join(["---:"] * 5) + " |"]
    for model, fmt, backend, variant in itertools.product(models, ("pt", "onnx"), ("cpu", "vulkan"), ("baseline", "candidate")):
        windows = lg[(model, fmt, backend, variant)]; cells = [window_range(windows, metric, 1048576 if metric == "AllocatedBytesPerFrame" else 1) for metric, _ in LIFECYCLE_METRICS]
        lines.append(f"| {model} | {fmt} / {backend} | {'0.8.0' if variant == 'baseline' else '0.8.1'} | {len(windows)} | " + " | ".join(cells) + " |")
    lines += ["", "## 原始证据与版本绑定", "",
              f"[本轮八引擎报告]({link(paths[0], output)})：SHA256 `{hashes[0]}`。", "",
              f"[生命周期 ABBA 报告]({link(paths[1], output)})：SHA256 `{hashes[1]}`。", "",
              f"候选 0.8.1 主库 SHA256 `{runtime['SimdYoloDllSha256']}`；冻结 0.8.0 SHA256 `{life['Variants']['baseline']['SnapshotDllSha256']}`；共享输入 SHA256 `{runtime['InputSha256']}`。模型文件 SHA256 已在两个来源逐一核对一致。", ""]
    if validation:
        lines += [f"[生命周期独立输出验证]({link(paths[2], output)})：SHA256 `{hashes[2]}`。", ""]
    return "\n".join(lines)


def main():
    """Read-only render; invalid evidence fails before writing the target report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--lifecycle-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/optimization-0.8.1.md")
    args = parser.parse_args(); paths = [args.runtime_report.resolve(), args.lifecycle_report.resolve()]
    runtime, runtime_sha = read(paths[0]); lifecycle, lifecycle_sha = read(paths[1]); validation = None; hashes = [runtime_sha, lifecycle_sha]
    if args.validation_report:
        paths.append(args.validation_report.resolve()); validation, value = read(paths[2]); hashes.append(value)
    models, correctness = verify(runtime, lifecycle, validation, lifecycle_sha)
    output = args.output.resolve(); require(output not in paths, "Output must not overwrite input evidence")
    text = render(runtime, lifecycle, validation, models, correctness, paths, hashes, output)
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text, encoding="utf-8")
    print(f"Rendered verified optimization evidence: {output}")


if __name__ == "__main__":
    main()
