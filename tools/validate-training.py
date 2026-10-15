"""使用官方 PyTorch 实现独立验证纯 C# YOLOv8 检测头训练的开发工具。

Python、NumPy、PyTorch 和 Ultralytics 仅用于开发对照，不进入训练库或网页运行时。
该工具会加载夹具指定的可信 PT 检查点；不会自动下载模型或修改任何检查点。

示例：
  python tools/validate-training.py --fixture .cache/training-validation/oracle-cpu \
    --ultralytics-dir .cache/yolo26-release-reference \
    --output artifacts/validation/training-cpu.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


FORWARD_TOLERANCE = (5e-4, 2e-3)
GRADIENT_TOLERANCE = (5e-4, 2e-3)
LOSS_TOLERANCE = (1e-4, 3e-4)
BUFFER_TOLERANCE = (5e-5, 3e-4)


def sha256(path: Path) -> str:
    """返回文件内容的 SHA-256，用于把结果绑定到实际输入与参考实现。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", required=True, type=Path, help="emit-oracle 生成的目录或 fixture.json 文件")
    parser.add_argument("--ultralytics-dir", required=True, type=Path, help="固定版本 Ultralytics 8.4.0 的源码根目录")
    parser.add_argument("--output", required=True, type=Path, help="独立对照结果 JSON 文件")
    parser.add_argument("--threads", type=int, default=2, help="官方 CPU 对照所使用的线程数，默认 2")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须大于 0")
    return args


def validate(args: argparse.Namespace) -> dict:
    """验证训练头前向、损失、原始输出梯度、参数梯度及 BN 运行统计。"""
    fixture_path = args.fixture.resolve()
    if fixture_path.is_dir():
        fixture_path /= "fixture.json"
    directory = fixture_path.parent
    fixture = json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    reference_root = args.ultralytics_dir.resolve()
    if not (reference_root / "ultralytics" / "__init__.py").is_file():
        raise FileNotFoundError(f"Ultralytics 源码目录无效：{reference_root}")
    sys.path.insert(0, str(reference_root))
    project_root = Path(__file__).resolve().parent.parent
    # 仓库先前的参考工具把依赖安装在这里；指定版本的源码始终优先。
    dependency_root = project_root / ".cache" / "reference"
    if dependency_root.is_dir():
        sys.path.append(str(dependency_root))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / ".cache" / "yolo26-config"))
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".cache" / "matplotlib"))
    import numpy as np
    import torch
    import torch.nn as nn
    import ultralytics
    from ultralytics.utils.loss import v8DetectionLoss

    actual_reference_root = Path(ultralytics.__file__).resolve().parent.parent
    if ultralytics.__version__ != "8.4.0" or actual_reference_root != reference_root:
        raise ValueError(f"必须使用指定的官方 8.4.0 源码，实际为 {ultralytics.__version__} @ {actual_reference_root}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(26)
    torch.backends.mkldnn.enabled = True

    files = {}

    def read_array(descriptor: dict, field: str = "path"):
        path = (directory / descriptor[field]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"夹具二进制文件必须位于夹具目录内：{path}")
        shape = descriptor["shape"]
        if any(not isinstance(v, int) or v < 0 for v in shape):
            raise ValueError(f"无效的张量形状：{shape}")
        raw = path.read_bytes()
        expected_bytes = math.prod(shape) * 4
        if len(raw) != expected_bytes:
            raise ValueError(f"张量字节数与形状不匹配：{path}，{len(raw)} != {expected_bytes}")
        array = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        if not np.isfinite(array).all():
            raise ValueError(f"夹具包含非有限值：{path}")
        files[str(path.relative_to(directory))] = {
            "shape": shape,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return array

    checks = []
    disconnected_gradients = []

    def gradient_array(tensor, name: str):
        # 没有正样本时官方 box 分支不连接到 loss；其数学导数为零。
        # 保留未连接记录，区分显式零张量和 PyTorch 延迟分配的 None。
        if tensor.grad is None:
            disconnected_gradients.append(name)
            return np.zeros(tuple(tensor.shape), dtype=np.float32)
        return tensor.grad.detach().numpy()

    def compare(name: str, actual, reference, tolerance: tuple[float, float], category: str):
        actual = np.asarray(actual, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        atol, rtol = tolerance
        shape_match = actual.shape == reference.shape
        finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
        record = {
            "name": name,
            "category": category,
            "shape": list(actual.shape),
            "referenceShape": list(reference.shape),
            "shapeMatch": shape_match,
            "finite": finite,
            "atol": atol,
            "rtol": rtol,
        }
        if shape_match and finite:
            difference = np.abs(actual - reference)
            allowed = atol + rtol * np.abs(reference)
            failures = difference > allowed
            ratio = difference / np.maximum(allowed, 1e-300)
            relative = np.divide(difference, np.maximum(np.abs(reference), 1e-12))
            worst = tuple(int(i) for i in np.unravel_index(int(ratio.argmax()), actual.shape)) if actual.size else ()
            record.update({
                "elements": int(actual.size),
                "maxAbsError": float(difference.max()) if actual.size else 0.0,
                "meanAbsError": float(difference.mean()) if actual.size else 0.0,
                "maxRelativeError": float(relative.max()) if actual.size else 0.0,
                "maxToleranceRatio": float(ratio.max()) if actual.size else 0.0,
                "relativeL2Error": float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(reference.ravel()), 1e-12)),
                "failedElements": int(failures.sum()),
                "worstIndex": list(worst),
                "actualAtWorst": float(actual[worst]) if actual.size else None,
                "referenceAtWorst": float(reference[worst]) if actual.size else None,
                "passed": not bool(failures.any()),
            })
        else:
            record["passed"] = False
        checks.append(record)
        if not record["passed"]:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        return record

    source_model = Path(fixture["sourceModel"]).resolve()
    checkpoint = torch.load(str(source_model), map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    model = model.float()
    head = model.model[-1]
    if type(head).__name__ != "Detect" or getattr(head, "end2end", False) or head.reg_max != 16:
        raise ValueError("此开发对照仅接受 YOLOv8 的三尺度标准 Detect 头。")
    class_names = fixture["classNames"]
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("夹具类别名应为非空且不重复的列表。")
    if len(head.cv2) != 3 or len(head.cv3) != 3:
        raise ValueError("参考检测头不含三个尺度。")
    for branch in head.cv3:
        original = branch[-1]
        if type(original) is not nn.Conv2d:
            raise ValueError("参考分类分支最后一层不是标准 Conv2d。")
        branch[-1] = nn.Conv2d(
            original.in_channels,
            len(class_names),
            original.kernel_size,
            original.stride,
            original.padding,
            original.dilation,
            original.groups,
            original.bias is not None,
            original.padding_mode,
        )
    head.nc = len(class_names)
    head.no = head.nc + head.reg_max * 4
    head.stride = torch.tensor(fixture["strides"], dtype=torch.float32)
    # 发布检查点常把全部 requires_grad 清为 false；此处明确只训练 cv2/cv3。
    for name, parameter in head.named_parameters():
        parameter.requires_grad_(name.startswith(("cv2.", "cv3.")))
    state = head.state_dict()
    names = [descriptor["name"] for descriptor in fixture["parameters"]]
    if len(names) != len(set(names)):
        raise ValueError("夹具重复声明参数。")
    expected_names = set(state) - {"dfl.conv.weight"}
    if set(names) != expected_names:
        raise ValueError(f"训练参数/BN 统计清单不完整：missing={sorted(expected_names-set(names))}, extra={sorted(set(names)-expected_names)}")
    trainable_names = {name for name, parameter in head.named_parameters() if parameter.requires_grad}
    fixture_trainable_names = {d["name"] for d in fixture["parameters"] if d.get("gradientPath")}
    if fixture_trainable_names != trainable_names:
        raise ValueError(f"梯度清单不完整：missing={sorted(trainable_names-fixture_trainable_names)}, extra={sorted(fixture_trainable_names-trainable_names)}")
    for descriptor in fixture["parameters"]:
        name = descriptor["name"]
        values = read_array(descriptor)
        if list(state[name].shape) != list(values.shape):
            raise ValueError(f"{name} 参数形状不匹配：{list(values.shape)} != {list(state[name].shape)}")
        state[name] = torch.from_numpy(values).to(dtype=state[name].dtype)
    head.load_state_dict(state, strict=True)
    features = [torch.from_numpy(read_array(descriptor)) for descriptor in fixture["features"]]
    if len(features) != 3 or any(x.ndim != 4 or x.shape[0] != 1 for x in features):
        raise ValueError("首期训练对照要求 batch=1 的三个 NCHW 特征。")
    height = int(features[0].shape[2] * head.stride[0])
    width = int(features[0].shape[3] * head.stride[0])
    for feature, stride in zip(features, head.stride):
        if int(feature.shape[2] * stride) != height or int(feature.shape[3] * stride) != width:
            raise ValueError("三个尺度的图像尺寸不一致。")

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([head])
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    fake_model = FakeModel()
    criterion = v8DetectionLoss(fake_model, tal_topk=10)
    truth = fixture["truth"]
    batch = {
        "batch_idx": torch.zeros(len(truth), dtype=torch.float32),
        "cls": torch.tensor([item["classId"] for item in truth], dtype=torch.float32).reshape(-1, 1),
        "bboxes": torch.tensor([
            [(b["x1"] + b["x2"]) / (2 * width), (b["y1"] + b["y2"]) / (2 * height),
             (b["x2"] - b["x1"]) / width, (b["y2"] - b["y1"]) / height]
            for b in truth
        ], dtype=torch.float32).reshape(-1, 4),
    }

    def official_loss(boxes, classes):
        predictions = {
            "boxes": torch.cat([value.reshape(1, 64, -1) for value in boxes], dim=2),
            "scores": torch.cat([value.reshape(1, head.nc, -1) for value in classes], dim=2),
            "feats": features,
        }
        assignment, losses, _ = criterion.get_assigned_targets_and_loss(predictions, batch)
        return losses, int(assignment[0].sum().item())

    def compare_loss(losses, positives, prefix):
        expected = fixture["loss"]
        for key, value in zip(("box", "classification", "dfl"), losses.detach().numpy()):
            compare(prefix + "." + key, expected[key], value, LOSS_TOLERANCE, "loss")
        compare(prefix + ".total", expected["total"], losses.sum().item(), LOSS_TOLERANCE, "loss")
        compare(prefix + ".positives", expected["positives"], positives, (0.0, 0.0), "assignment")
        return {
            "box": float(losses[0].detach()),
            "classification": float(losses[1].detach()),
            "dfl": float(losses[2].detach()),
            "total": float(losses.sum().detach()),
            "positives": positives,
        }

    head.train()
    head.zero_grad(set_to_none=True)
    reference_boxes = [branch(feature) for branch, feature in zip(head.cv2, features)]
    reference_classes = [branch(feature) for branch, feature in zip(head.cv3, features)]
    for tensor in reference_boxes + reference_classes:
        tensor.retain_grad()
    losses, positives = official_loss(reference_boxes, reference_classes)
    full_reference_loss = compare_loss(losses, positives, "officialHeadLoss")
    losses.sum().backward()
    for group, tensors in (("boxes", reference_boxes), ("classes", reference_classes)):
        if len(fixture[group]) != 3:
            raise ValueError(f"{group} 夹具必须包含三个尺度。")
        for i, (descriptor, reference) in enumerate(zip(fixture[group], tensors)):
            compare(f"{group}.{i}", read_array(descriptor), reference.detach().numpy(), FORWARD_TOLERANCE, "headForward")
            compare(f"{group}.{i}.gradient", read_array(descriptor, "gradientPath"), gradient_array(reference, f"{group}.{i}"), GRADIENT_TOLERANCE, "headOutputGradient")
    named_parameters = dict(head.named_parameters())
    after_state = head.state_dict()
    for descriptor in fixture["parameters"]:
        name = descriptor["name"]
        if descriptor.get("gradientPath"):
            compare(name + ".gradient", read_array(descriptor, "gradientPath"), gradient_array(named_parameters[name], name), GRADIENT_TOLERANCE, "parameterGradient")
        if descriptor.get("afterPath"):
            is_buffer = name not in named_parameters
            tolerance = (0.0, 0.0) if not is_buffer or name.endswith("num_batches_tracked") else BUFFER_TOLERANCE
            compare(name + ".after", read_array(descriptor, "afterPath"), after_state[name].detach().numpy(), tolerance, "batchNormRunningStatistics" if is_buffer else "parameterUnchangedBeforeSgd")

    # 直接使用 C# 的原始输出再做一次官方损失，区分头前向误差与损失数学误差。
    fixture_boxes = [torch.from_numpy(read_array(d)).requires_grad_() for d in fixture["boxes"]]
    fixture_classes = [torch.from_numpy(read_array(d)).requires_grad_() for d in fixture["classes"]]
    isolated_losses, isolated_positives = official_loss(fixture_boxes, fixture_classes)
    isolated_reference_loss = compare_loss(isolated_losses, isolated_positives, "fixtureLogitsLoss")
    isolated_losses.sum().backward()
    for group, tensors in (("boxes", fixture_boxes), ("classes", fixture_classes)):
        for i, (descriptor, reference) in enumerate(zip(fixture[group], tensors)):
            compare(f"fixtureLogits.{group}.{i}.gradient", read_array(descriptor, "gradientPath"), gradient_array(reference, f"fixtureLogits.{group}.{i}"), GRADIENT_TOLERANCE, "isolatedLossGradient")

    reference_files = [
        reference_root / "ultralytics" / "utils" / "loss.py",
        reference_root / "ultralytics" / "utils" / "tal.py",
        reference_root / "ultralytics" / "utils" / "metrics.py",
        reference_root / "ultralytics" / "nn" / "modules" / "head.py",
    ]
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "scope": "YOLOv8 three-scale detection-head fine-tuning; frozen features; batch=1; FP32; no optimizer step in fixture",
        "reference": {
            "ultralyticsVersion": ultralytics.__version__,
            "ultralyticsDirectory": str(reference_root),
            "torchVersion": torch.__version__,
            "torchCudaVersion": torch.version.cuda,
            "device": "cpu",
            "threads": args.threads,
            "mkldnnEnabled": torch.backends.mkldnn.enabled,
            "sources": {str(path.relative_to(reference_root)): sha256(path) for path in reference_files},
        },
        "fixture": str(fixture_path),
        "fixtureSha256": sha256(fixture_path),
        "sourceModel": str(source_model),
        "sourceModelSha256": sha256(source_model),
        "validatorSha256": sha256(Path(__file__)),
        "classNames": class_names,
        "imageSize": [height, width],
        "strides": fixture["strides"],
        "fixtures": files,
        "csharpLoss": fixture["loss"],
        "officialHeadLoss": full_reference_loss,
        "officialLossFromFixtureLogits": isolated_reference_loss,
        "disconnectedGradientsComparedAsZero": disconnected_gradients,
        "checks": checks,
        "totalChecks": len(checks),
        "passedChecks": sum(check["passed"] for check in checks),
        "failedChecks": sum(not check["passed"] for check in checks),
    }


def main() -> int:
    args = parse_arguments()
    try:
        report = validate(args)
    except Exception as error:
        report = {"status": "error", "error": str(error), "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "totalChecks": report.get("totalChecks"),
                      "failedChecks": report.get("failedChecks"), "output": str(args.output)}, ensure_ascii=False), flush=True)
    if report["status"] == "error":
        print(report["traceback"], file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
