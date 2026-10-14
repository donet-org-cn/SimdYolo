"""独立校验生命周期窗口的首帧和末帧原始输出，不执行任何性能计时。

这是开发工具；SimdYolo 主库不依赖 Python、NumPy、PyTorch 或 Ultralytics。
需要 NumPy、CPU-only PyTorch 和官方 Ultralytics 8.4.0。仅加载已经授权的
官方或用户自有 PT 检查点；PyTorch 的完整模型反序列化在本开发工具中进行。
端到端输出按完整候选配对，允许合法 TopK 同分次序，仍验证类别、分数及
每个 mask/关键点字段；分割原型独立逐元素比较。文件缺失、摘要不符、
参考包不符或数值失败都返回非零状态，并在 JSON 中保存失败证据。
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    """分块读取文件，返回 SHA256，不额外保留整份模型或输出字节。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value, label):
    """校验记录中的 SHA256 文本格式；缺失或不完整的来源证据直接失败。"""
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"Missing or invalid SHA256 for {label}")
    return value.lower()


def write_json(path, value):
    """原子写入进度及最终结果，拒绝 JSON 中的 NaN 和 Infinity。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def resolve_file(value, report_path, raw_root=None):
    """优先使用仍有效的记录路径；迁移原始输出时在显式 raw_root 中按唯一文件名恢复。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("File path is missing from the lifecycle record")
    native = Path(value).expanduser()
    candidates = [native] if native.is_absolute() else [report_path.parent / native, ROOT / native, native]
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate.resolve()
    if raw_root is not None:
        # 跨平台读取 Windows 报告时也使用正确的最后一个路径分量。
        name = value.replace("\\", "/").rsplit("/", 1)[-1]
        relocated = [path for path in raw_root.rglob(name) if path.is_file()]
        if len(relocated) == 1:
            return relocated[0].resolve()
        if len(relocated) > 1:
            raise ValueError(f"Ambiguous relocated output {name}; preserve a unique raw filename")
    raise FileNotFoundError(f"Recorded file is unavailable: {value}")


def model_filename(value, fmt):
    """把模型文件名或名称转换为固定扩展名的单一路径分量，拒绝目录、驱动器和路径穿越。"""
    if not isinstance(value, str) or not value.strip() or value in (".", "..") or any(character in value for character in ("/", "\\", ":", "\0")):
        raise ValueError("Window Model must be a model filename or stem without directories or drive names")
    if fmt not in ("pt", "onnx"):
        raise ValueError("Window model format must be pt or onnx")
    filename = value if Path(value).suffix else value + "." + fmt
    if Path(filename).suffix.lower() != "." + fmt:
        raise ValueError("Window Model extension does not match Format")
    return filename


def bounded_model_path(directory, filename):
    """解析显式模型目录内的文件，拒绝通过子路径或符号链接越出该目录。"""
    directory = directory.resolve()
    target = (directory / filename).resolve()
    if not target.is_relative_to(directory):
        raise ValueError("Relocated model must remain inside its model directory")
    return target


def locate_model(recorded_path, filename, fmt, report_path, model_root, expected_sha256):
    """优先读取原模型路径；仅在不可读时从 model_root/格式/文件名恢复，并始终核对记录摘要。"""
    filename = model_filename(filename, fmt)
    path = None
    if recorded_path is not None:
        try:
            path = resolve_file(recorded_path, report_path)
            with path.open("rb"):
                pass
        except OSError:
            path = None
    if path is None:
        path = bounded_model_path(model_root, fmt + "/" + filename)
        if not path.is_file():
            raise FileNotFoundError(f"Relocated model is unavailable: {path}")
    if sha256(path) != digest(expected_sha256, "recorded model"):
        raise ValueError(f"Model SHA256 mismatch: {path}")
    return path


def records_from(report):
    """接受正式 Records 或兼容 Windows 列表，保证每个窗口都进入校验。"""
    records = report.get("Records", report.get("Windows"))
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("Lifecycle report must contain a nonempty Records or Windows list")
    if report.get("Records") and report.get("Windows") and report["Records"] != report["Windows"]:
        raise ValueError("Different Records and Windows lists would leave windows unvalidated")
    return records


def record_dll_digest(report, record):
    """取得本窗口冻结的 DLL 摘要，兼容报告级的变体或单版本来源声明。"""
    variant = report.get("Variants", {}).get(record.get("Variant"), {})
    return digest(record.get("DllSha256") or variant.get("SnapshotDllSha256") or report.get("DllSha256"), "window DLL")


def model_digest(report, record, filename):
    """从窗口或模型清单取得被测 PT/ONNX 的预期摘要，拒绝含糊或未声明的模型来源。"""
    if record.get("ModelSha256"):
        return digest(record["ModelSha256"], filename)
    entries = report.get("Models", [])
    if isinstance(entries, list):
        matches = [entry for entry in entries if isinstance(entry, dict) and
                   str(entry.get("Path", "")).replace("\\", "/").rsplit("/", 1)[-1] == filename]
        hashes = {digest(entry.get("Sha256"), filename) for entry in matches}
        if len(hashes) == 1:
            return hashes.pop()
        if len(hashes) > 1:
            raise ValueError(f"Conflicting model SHA256 entries for {filename}")
    known = report.get("ModelsSha256", {}).get(Path(filename).stem, {})
    return digest(known.get(Path(filename).suffix.lstrip(".")), filename)


def select_reference_packages(directory):
    """固定官方参考目录，并在导入前排除本工作区 CUDA wheel、检查 Torch 的 CPU-only 声明。"""
    directory = directory.resolve()
    if not (directory / "ultralytics/__init__.py").is_file():
        raise FileNotFoundError(f"Ultralytics package directory is missing: {directory}")
    cuda_directory = (ROOT / ".cache/torch-cu126").resolve()
    sys.path[:] = [entry for entry in sys.path if not Path(entry or os.curdir).resolve().is_relative_to(cuda_directory)]
    dependencies = ROOT / ".cache/reference"
    if dependencies.is_dir():
        sys.path.insert(0, str(dependencies))
    sys.path.insert(0, str(directory))
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Install a CPU-only PyTorch wheel to run this development validator")
    version = Path(next(iter(spec.submodule_search_locations))) / "version.py"
    if not version.is_file():
        raise RuntimeError(f"Cannot verify the selected PyTorch wheel: {version}")
    for statement in ast.parse(version.read_text(encoding="utf-8")).body:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "cuda" for target in targets):
            if not isinstance(statement.value, ast.Constant) or statement.value.value is not None:
                raise RuntimeError(f"Selected PyTorch is not CPU-only; CUDA wheel was not imported: {version}")
            return directory
    raise RuntimeError(f"PyTorch CPU-only build declaration is missing: {version}")


def flatten_tensors(value, torch):
    """按官方导出顺序展开 Tensor 或嵌套列表/元组，拒绝其他输出类型。"""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in flatten_tensors(item, torch)]
    raise TypeError(f"Unexpected official reference output type: {type(value).__name__}")


def reference_outputs(path, array, torch, np):
    """以官方 FP32 eval/fuse/export CPU 路径计算所有输出及端到端完整候选，不计时。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = (checkpoint.get("ema") or checkpoint.get("model")) if isinstance(checkpoint, dict) else checkpoint
    if model is None or not hasattr(model, "model"):
        raise ValueError(f"Official reference requires a complete YOLO nn.Module checkpoint: {path}")
    model = model.cpu().float().eval()
    model.fuse(verbose=False)
    head = model.model[-1]
    head.export, head.format, head.dynamic = True, "onnx", False
    end_to_end = bool(getattr(head, "end2end", False))
    task = "segment" if type(head).__name__ in ("Segment", "Segment26") else "pose" if type(head).__name__ in ("Pose", "Pose26") else "detect"
    x = torch.from_numpy(array.copy())
    with torch.inference_mode():
        expected = [tensor.detach().cpu().numpy().copy() for tensor in flatten_tensors(model(x), torch)]
        dense = None
        if end_to_end:
            # 只临时关闭筛选，保留官方解码、mask 与关键点；退出后恢复原有属性归属。
            missing = object()
            saved = head.__dict__.get("postprocess", missing)
            head.postprocess = lambda predictions, *unused: predictions
            try:
                dense = flatten_tensors(model(x), torch)[0].detach().cpu().numpy().copy()
            finally:
                if saved is missing:
                    head.__dict__.pop("postprocess", None)
                else:
                    head.postprocess = saved
    if not expected or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in expected):
        raise ValueError(f"Official reference returned invalid FP32 outputs: {path}")
    if dense is not None and (dense.ndim != 3 or not np.isfinite(dense).all()):
        raise ValueError(f"Official reference returned invalid dense candidates: {path}")
    info = {"Path": str(path), "Sha256": sha256(path), "Task": task, "EndToEnd": end_to_end,
            "ClassCount": int(head.nc), "CheckpointVersion": checkpoint.get("version") if isinstance(checkpoint, dict) else None,
            "Shapes": [list(value.shape) for value in expected], "DenseShape": list(dense.shape) if dense is not None else None,
            "OutputSha256": [hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest() for value in expected]}
    return expected, dense, info


def read_output(descriptor, report_path, raw_root, np):
    """验证原始输出的摘要、字节数、元素类型及完整形状后读取 FP32 数据。"""
    path = resolve_file(descriptor.get("Path"), report_path, raw_root)
    declared = digest(descriptor.get("Sha256"), str(path))
    # 摘要与数值共享同一份文件快照，避免两次打开之间文件变化却复用旧摘要。
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != declared:
        raise ValueError(f"Raw output SHA256 mismatch: {path}")
    shape = descriptor.get("Shape")
    if not isinstance(shape, list) or any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise ValueError(f"Invalid raw output shape: {path}")
    required = math.prod(shape) * 4
    if type(descriptor.get("Bytes")) is not int or descriptor["Bytes"] != required or len(raw) != required:
        raise ValueError(f"Raw output byte count does not match shape: {path}")
    output = np.frombuffer(raw, dtype="<f4").reshape(shape)
    return output, {"Path": str(path), "RecordedPath": descriptor["Path"], "Sha256": actual, "Bytes": required, "Shape": shape}


def compare_output(actual, expected, end_to_end, dense, nc, np):
    """调用共享的一对一完整候选配对，统计配对后的误差；原型及普通头逐元素验证。"""
    from output_validation import compare
    left, right, passed = compare(actual, expected, end_to_end, dense if end_to_end else None, nc)
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    return {"ExpectedShape": list(expected.shape), "MaxAbsError": float(delta.max(initial=0)) if finite else None,
            "MeanAbsError": float(delta.mean()) if finite and delta.size else 0.0 if finite else None,
            "Finite": finite, "Validation": "TopK scores and unique complete anchor/class/payload pairing" if end_to_end else "Elementwise FP32",
            "Tolerance": {"Atol": 3e-3, "Rtol": 3e-4, **({"ScoreAtol": 2e-7} if end_to_end else {})}, "Passed": bool(passed)}


class NumericalComparisonCache:
    """仅为同一权重及输入保留数值比较结果；文件路径和帧归属永不进入缓存值。"""

    def __init__(self):
        """固定共享校验代码摘要，建立最多一个参考模型的比较缓存。"""
        self.reference = None
        self.results = {}
        self.helper_sha256 = sha256(Path(__file__).with_name("output_validation.py"))

    def compare(self, actual, expected, dense, info, index, source, np):
        """在文件校验之后按权重、输入、输出索引、完整字节摘要、类型和策略复用结果。"""
        reference = (digest(info.get("Sha256"), "reference model"), digest(info.get("InputSha256"), "reference input"))
        if reference != self.reference:
            self.results.clear()
            self.reference = reference
        selected = info["EndToEnd"] and index == 0
        policy = ("finite-and-unique-full-payload-v1", self.helper_sha256, 3e-3, 3e-4, 2e-7,
                  bool(selected), int(info["ClassCount"]), dense is not None,
                  tuple(dense.shape) if dense is not None else None, dense.dtype.str if dense is not None else None)
        reference_hashes = info.get("OutputSha256", [])
        expected_hash = digest(reference_hashes[index] if index < len(reference_hashes) else None, "reference output")
        key = (reference, index, digest(source["Sha256"], "raw output"), tuple(source["Shape"]),
               actual.dtype.str, tuple(expected.shape), expected.dtype.str, expected_hash, policy)
        reused = key in self.results
        if not reused:
            # 失败结果也按原值缓存；绝不把“已检查”当作“已通过”。
            self.results[key] = compare_output(actual, expected, selected, dense, info["ClassCount"], np)
        return copy.deepcopy(self.results[key]), reused


def validate_group(descriptors, expected, dense, info, phase, report_path, raw_root, np, comparison_cache=None):
    """校验首帧或末帧的全部输出；单个文件失败仍记录并检查其他输出。"""
    if not isinstance(descriptors, list) or len(descriptors) != len(expected):
        raise ValueError(f"{phase} output count does not match official reference ({len(expected)})")
    indices = [descriptor.get("Index") if isinstance(descriptor, dict) else None for descriptor in descriptors]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(expected))):
        raise ValueError(f"{phase} output indices must cover every official output exactly once")
    checks = []
    for descriptor in sorted(descriptors, key=lambda item: item["Index"]):
        index = descriptor["Index"]
        check = {"Phase": phase, "Output": index, "Path": descriptor.get("Path"), "Shape": descriptor.get("Shape"), "ComparisonReused": False}
        try:
            # 每个文件仍独立读取并核对 SHA、字节数和形状，随后才允许数值结果命中缓存。
            actual, source = read_output(descriptor, report_path, raw_root, np)
            check.update(source)
            if comparison_cache is None:
                check.update(compare_output(actual, expected[index], info["EndToEnd"] and index == 0, dense, info["ClassCount"], np))
            else:
                numeric, reused = comparison_cache.compare(actual, expected[index], dense, info, index, source, np)
                check.update(numeric)
                check["ComparisonReused"] = reused
        except Exception as error:
            check.update({"Passed": False, "Error": f"{type(error).__name__}: {error}"})
        checks.append(check)
    return checks


def main():
    """读取生命周期证据，逐窗口检查首帧和末帧，并保存独立参考与失败摘要。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", "--comparison", dest="report", type=Path, required=True, help="compare-lifecycle.py 生成的 JSON；兼容 Records / Windows 列表")
    parser.add_argument("--ultralytics-dir", type=Path, default=ROOT / ".cache/yolo26-release-reference", help="官方 Ultralytics 8.4.0 包目录")
    parser.add_argument("--model-root", type=Path, default=ROOT / "models", help="原模型路径不可读时按 pt/onnx 子目录定位，仍严格核对报告 SHA256")
    parser.add_argument("--pt-dir", type=Path, help="显式指定 ONNX 对应的原始 PT 目录；默认使用 --model-root 下的 pt 子目录")
    parser.add_argument("--input", type=Path, help="输入文件被迁移时的显式替代路径，摘要仍必须匹配报告")
    parser.add_argument("--raw-root", type=Path, help="原始输出迁移后的根目录；记录的绝对路径可读时仍优先使用")
    parser.add_argument("--threads", type=int, default=4, help="官方 CPU 参考线程数；本工具不测量速度")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/validation/0.8.1/lifecycle-outputs.json")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    report_path, output_path = args.report.resolve(), args.output.resolve()
    pt_directory = args.pt_dir.resolve() if args.pt_dir else args.model_root.resolve() / "pt"
    if report_path == output_path:
        parser.error("Validation output must not overwrite the lifecycle comparison")
    result = {"Status": "running", "StartedUtc": datetime.now(timezone.utc).isoformat(),
              "Comparison": str(report_path), "Tool": {"Path": "tools/validate-lifecycle.py", "Sha256": sha256(Path(__file__).resolve())},
              "Scope": "Every first and final file/index is read and independently checked for SHA256, bytes and shape. "
                       "Numerical comparison is reused only for identical complete raw bytes, reference-model/input hashes, "
                       "output index, dtypes, shapes and validation policy; no files or indices are skipped. "
                       "At most one reference model is retained in the numerical comparison cache. No timing or performance ranking.",
              "ModelResolution": {"ModelRoot": str(args.model_root.resolve()), "ReferencePtDirectory": str(pt_directory),
                                  "Policy": "Use readable recorded ModelPath first; otherwise locate Format/Model below ModelRoot and verify recorded SHA256"},
              "References": [], "Windows": []}
    try:
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes.decode("utf-8-sig"))
        records = records_from(report)
        result["ExpectedWindows"] = len(records)
        result["ComparisonSha256"] = hashlib.sha256(report_bytes).hexdigest()
        result["ComparisonStatus"] = report.get("Status")
        selected = select_reference_packages(args.ultralytics_dir)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/yolo26-config"))
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/yolo26-config"))
        import numpy as np
        import torch
        import ultralytics
        if torch.version.cuda is not None:
            raise RuntimeError("Official reference must use a CPU-only PyTorch build")
        if ultralytics.__version__ != "8.4.0":
            raise RuntimeError(f"Official reference must use Ultralytics 8.4.0, got {ultralytics.__version__}")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        result["Oracle"] = {"Torch": torch.__version__, "TorchPath": str(Path(torch.__file__).resolve()), "TorchCudaBuild": torch.version.cuda,
                            "Ultralytics": ultralytics.__version__, "UltralyticsDirectory": str(selected), "CpuThreads": args.threads,
                            "Mode": "CPU FP32 eval().fuse(); export=True, format=onnx, dynamic=False"}
        references, inputs = {}, {}
        comparison_cache = NumericalComparisonCache()
        write_json(output_path, result)
        for number, record in enumerate(records, 1):
            window = {key: record.get(key) for key in ("Sequence", "Model", "ModelPath", "Format", "RequestedBackend", "Backend", "Variant", "Repeat", "AbbaPosition", "Version")}
            window.update({"Window": number, "Checks": [], "Passed": False})
            result["Windows"].append(window)
            try:
                window["DllSha256"] = record_dll_digest(report, record)
                if record.get("FallbackReason") is not None:
                    raise ValueError("Lifecycle window used a fallback backend")
                model_name = str(record.get("Model", ""))
                fmt = str(record.get("Format") or Path(model_name).suffix.lstrip(".")).lower()
                filename = model_filename(model_name, fmt)
                declared_model_hash = model_digest(report, record, filename)
                model_path = locate_model(record.get("ModelPath"), filename, fmt, report_path, args.model_root, declared_model_hash)
                window.update({"RecordedModelPath": record.get("ModelPath"), "ModelPath": str(model_path), "ModelSha256": declared_model_hash})
                input_path = args.input.resolve() if args.input else resolve_file(record.get("InputPath") or report.get("Input"), report_path)
                input_hash = digest(record.get("InputSha256") or report.get("InputSha256"), "input")
                size = record.get("Size", report.get("Settings", {}).get("Size"))
                if type(size) is not int or size < 32 or size % 32:
                    raise ValueError("Window Size must be a positive multiple of 32")
                input_key = (str(input_path), input_hash, size)
                if input_key not in inputs:
                    if sha256(input_path) != input_hash or input_path.stat().st_size != 3 * size * size * 4:
                        raise ValueError(f"Input SHA256/FP32 size mismatch: {input_path}")
                    array = np.fromfile(input_path, dtype="<f4").reshape(1, 3, size, size)
                    if not np.isfinite(array).all():
                        raise ValueError("Lifecycle input contains non-finite values")
                    inputs[input_key] = array
                window["Input"] = {"Path": str(input_path), "Sha256": input_hash, "Shape": [1, 3, size, size]}
                pt_path = model_path if fmt == "pt" else bounded_model_path(pt_directory, Path(filename).stem + ".pt")
                if not pt_path.is_file():
                    raise FileNotFoundError(f"Original PT reference is missing: {pt_path}")
                reference_key = (sha256(pt_path), input_hash, size)
                if reference_key not in references:
                    expected, dense, info = reference_outputs(pt_path, inputs[input_key], torch, np)
                    if info["Sha256"] != reference_key[0]:
                        raise ValueError(f"Original PT reference changed during validation: {pt_path}")
                    info["InputSha256"] = input_hash
                    info["Reference"] = len(result["References"]) + 1
                    references[reference_key] = (expected, dense, info)
                    result["References"].append(info)
                expected, dense, info = references[reference_key]
                window["Reference"] = info["Reference"]
                window["ReferenceModelSha256"] = info["Sha256"]
                for phase, field in (("first", "FirstOutputs"), ("last", "Outputs")):
                    try:
                        window["Checks"].extend(validate_group(record.get(field), expected, dense, info, phase, report_path, args.raw_root, np, comparison_cache))
                    except Exception as error:
                        window["Checks"].append({"Phase": phase, "Passed": False, "Error": f"{type(error).__name__}: {error}"})
                window["Passed"] = bool(window["Checks"]) and all(check["Passed"] for check in window["Checks"])
            except Exception as error:
                window["Error"] = f"{type(error).__name__}: {error}"
            write_json(output_path, result)
            print(json.dumps({"Window": number, "Model": window["Model"], "Variant": window["Variant"], "Backend": window["Backend"],
                              "Checks": len(window["Checks"]), "Passed": window["Passed"], **({"Error": window["Error"]} if "Error" in window else {})}), flush=True)
        result["Passed"] = all(window["Passed"] for window in result["Windows"])
        result["Status"] = "passed" if result["Passed"] else "failed"
    except Exception as error:
        result.update({"Status": "failed", "Passed": False, "Error": f"{type(error).__name__}: {error}"})
        print(result["Error"], file=sys.stderr, flush=True)
    result["FinishedUtc"] = datetime.now(timezone.utc).isoformat()
    result["Summary"] = {"Windows": len(result["Windows"]), "PassedWindows": sum(window["Passed"] for window in result["Windows"]),
                         "OutputChecks": sum(len(window["Checks"]) for window in result["Windows"]),
                         "PassedOutputChecks": sum(check["Passed"] for window in result["Windows"] for check in window["Checks"])}
    write_json(output_path, result)
    print(f"Lifecycle validation {result['Status']}: {output_path}", flush=True)
    return 0 if result["Passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
