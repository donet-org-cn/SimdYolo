"""Development validation only. The C# library never invokes Python or converts PT files.

Run in an environment with torch, ultralytics and (for --onnx) onnx installed.
Only load trusted official/user-approved PT fixtures with this PyTorch reference tool.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache"))
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_validation import compare

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+")
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--backends", nargs="+", default=["cpu", "vulkan"])
parser.add_argument("--onnx", action="store_true")
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--export-size", type=int)
parser.add_argument("--opset", type=int, default=17)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--work-directory", type=Path)
parser.add_argument("--report", type=Path)
parser.add_argument("--ort", action="store_true", help="Also check exported ONNX against ONNX Runtime")
args = parser.parse_args()
torch.set_num_threads(4)
work = args.work_directory or root / ".cache" / "validation"
work.mkdir(parents=True, exist_ok=True)
cli = root / "src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll"
rng = np.random.default_rng(20261013)
array = rng.uniform(0, 1, (args.batch, 3, args.size, args.size)).astype(np.float32)
array.tofile(work / "input.f32")
reports = []

for path in args.models:
    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    model = model.float().eval()
    head = model.model[-1]
    head.export = True
    head.format = "onnx"
    head.dynamic = args.dynamic
    task = "segment" if type(head).__name__ in ("Segment", "Segment26") else "pose" if type(head).__name__ in ("Pose", "Pose26") else "detect"
    end_to_end = bool(getattr(head, "end2end", False))
    x = torch.from_numpy(array)
    with torch.inference_mode():
        expected = model(x)
        dense = None
        if end_to_end:
            postprocess = head.postprocess
            head.postprocess = lambda predictions, *unused: predictions
            decoded = model(x)
            dense = (decoded[0] if isinstance(decoded, (list, tuple)) else decoded).detach().numpy()
            head.postprocess = postprocess
    expected = list(expected) if isinstance(expected, (list, tuple)) else [expected]
    expected = [t.detach().numpy() for t in expected]
    files = [path]
    if args.onnx:
        import onnx
        onnx_path = work / (path.stem + ("-dynamic" if args.dynamic else "") + (f"-opset{args.opset}" if args.opset != 17 else "") + ".onnx")
        export_input = torch.zeros(1, 3, args.export_size, args.export_size) if args.export_size else x
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}, "output0": {0: "batch", 1 if end_to_end else 2: "anchors"}} if args.dynamic else None
        if args.dynamic and task == "segment":
            dynamic_axes["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}
        with torch.inference_mode():
            torch.onnx.export(model, export_input, str(onnx_path), input_names=["images"], output_names=[f"output{i}" for i in range(len(expected))], opset_version=args.opset, dynamo=False, dynamic_axes=dynamic_axes)
        exported = onnx.load(onnx_path)
        onnx.helper.set_model_props(exported, {"task": task, "names": str(model.names), "nc": str(head.nc), "end2end": str(end_to_end), **({"kpt_shape": str(head.kpt_shape)} if task == "pose" else {})})
        onnx.checker.check_model(exported)
        onnx.save(exported, onnx_path)
        files.append(onnx_path)
        if args.ort:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
            for i, output in enumerate(session.run(None, {"images": array})):
                actual, reference, passed = compare(output, expected[i], end_to_end and i == 0, dense if i == 0 else None, head.nc)
                report = {"model": onnx_path.name, "backend": "ort-cpu", "output": i, "shape": list(output.shape), "max_abs_error": float(np.abs(actual-reference).max()), "mean_abs_error": float(np.abs(actual-reference).mean()), "passed": passed}
                reports.append(report)
                if not passed:
                    raise AssertionError(f"ONNX Runtime mismatch: {report}")
    for file in files:
        for backend in args.backends:
            prefix = work / (file.stem + "." + file.suffix[1:] + "." + backend)
            start = time.perf_counter()
            subprocess.run(["dotnet", str(cli), "raw", str(file), str(work / "input.f32"), str(prefix), backend, str(args.size), str(args.batch)], check=True)
            for i, original_reference in enumerate(expected):
                shape = json.loads(Path(str(prefix) + f".{i}.json").read_text())
                actual = np.fromfile(str(prefix) + f".{i}.f32", dtype=np.float32).reshape(shape)
                actual, reference, passed = compare(actual, original_reference, end_to_end and i == 0, dense if i == 0 else None, head.nc)
                diff = np.abs(actual - reference)
                report = {"model": file.name, "backend": backend, "output": i, "shape": shape, "max_abs_error": float(diff.max()), "mean_abs_error": float(diff.mean()), "passed": passed}
                reports.append(report)
                print(json.dumps(report), flush=True)
                if not passed:
                    idx = np.unravel_index(np.argmax(diff), diff.shape)
                    raise AssertionError(f"Numerical mismatch at {idx}: {actual[idx]} != {reference[idx]}")
    del model
report_path = args.report or work / f"results-{args.size}{'-dynamic' if args.dynamic else ''}{f'-opset{args.opset}' if args.opset != 17 else ''}.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(reports, indent=2))
print(f"{len(reports)} output comparisons passed: {report_path}")
