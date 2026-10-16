"""开发验证工具：用官方ONNX checker与独立ORT核验纯.NET导出的真实模型；产品不依赖Python。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default=".cache/features-011")
    parser.add_argument("--output", default="artifacts/validation/0.11.0/onnx-export.json")
    args = parser.parse_args()
    checks = []
    for reference in sorted(Path(args.fixtures).rglob("*.onnx.reference.json")):
        model = Path(str(reference).removesuffix(".reference.json"))
        manifest = json.loads(reference.read_text(encoding="utf-8"))
        onnx.checker.check_model(str(model), full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        tensor = np.fromfile(str(model) + ".input.f32", dtype="<f4").reshape(manifest["inputShape"])
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        if len(outputs) != len(manifest["outputShapes"]):
            raise AssertionError("Output count differs: " + str(model))
        errors = []
        for index, output in enumerate(outputs):
            expected = np.fromfile(str(model) + f".output-{index}.f32", dtype="<f4").reshape(manifest["outputShapes"][index])
            if output.shape != expected.shape or not np.isfinite(output).all():
                raise AssertionError("Invalid shape or nonfinite output: " + str(model))
            reordered = 0
            # TopK边界处近似相等的分数可能因不同Sigmoid实现发生排序交换。
            # 只允许2e-6以内的分数交换，且每行必须一对一匹配相同类别与完整坐标。
            if index == 0 and session.get_modelmeta().custom_metadata_map.get("end2end") == "True" and output.ndim == 3 and output.shape[-1] >= 6:
                available = set(range(output.shape[1]))
                aligned = output.copy()
                for row, target in enumerate(expected[0]):
                    candidates = [j for j in available if output[0, j, 5] == target[5] and abs(float(output[0, j, 4] - target[4])) <= 2e-6 and abs(float(expected[0, row, 4] - expected[0, j, 4])) <= 2e-6]
                    if not candidates:
                        raise AssertionError(f"TopK row cannot be matched one-to-one: {model}, row {row}")
                    chosen = min(candidates, key=lambda j: float(np.max(np.abs(output[0, j, :4] - target[:4]) / (1 + np.abs(target[:4])))))
                    aligned[0, row] = output[0, chosen]
                    available.remove(chosen)
                    reordered += chosen != row
                output = aligned
            difference = np.abs(output - expected)
            relative = difference / (1 + np.abs(expected))
            if relative.max(initial=0) > manifest["tolerance"]:
                raise AssertionError(f"Numerical mismatch {model}: max scaled error {relative.max()}")
            errors.append({"shape": list(output.shape), "maxAbsoluteError": float(difference.max(initial=0)), "maxScaledError": float(relative.max(initial=0)), "nearTieRowsReordered": reordered, "nearTieScoreTolerance": 2e-6})
        checks.append({"model": str(model).replace("\\", "/"), "sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "checker": "passed", "ort": "passed", "tolerance": manifest["tolerance"], "outputs": errors})
        print("PASS", model)
    if len(checks) < 10:
        raise AssertionError("Expected original and trained exports of all five families")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"status": "passed", "onnxVersion": onnx.__version__, "ortVersion": ort.__version__, "checks": checks}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
