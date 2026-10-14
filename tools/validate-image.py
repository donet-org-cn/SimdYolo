"""Optional development oracle for real image boxes, masks and keypoints.
Uses C# letterboxing as the shared input and PyTorch for the independent forward pass.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache"))
import numpy as np
from PIL import Image
import torch
from ultralytics.utils import ops

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("models", nargs="+")
parser.add_argument("--backend", default="vulkan")
args = parser.parse_args()
torch.set_num_threads(4)
work = root / ".cache/image-validation"
work.mkdir(parents=True, exist_ok=True)
image = Image.open(args.image).convert("RGB")
w, h = image.size
image.save(work / "image.ppm")
cli = root / "src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll"
subprocess.run(["dotnet", str(cli), "preprocess", str(work / "image.ppm"), str(work / "input.f32")], check=True)
tx = json.loads((work / "input.f32.json").read_text())
x = torch.from_numpy(np.fromfile(work / "input.f32", dtype=np.float32).reshape(1, 3, 640, 640))
reports = []
for path in args.models:
    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
    head = model.model[-1]
    head.export, head.format = True, "onnx"
    task = type(head).__name__
    with torch.inference_mode():
        out = model(x)
    prototype = out[1] if isinstance(out, tuple) else None
    out = out[0] if isinstance(out, tuple) else out
    data = out[0].T.numpy()
    nc = head.nc
    scores = data[:, 4:4 + nc].max(axis=1)
    classes = data[:, 4:4 + nc].argmax(axis=1)
    boxes = np.concatenate((data[:, :2] - data[:, 2:4] / 2, data[:, :2] + data[:, 2:4] / 2), axis=1)
    candidates = np.flatnonzero(scores >= .25)
    candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
    kept = []
    for a in candidates:
        if kept:
            other = np.array(kept)
            wh = np.maximum(0, np.minimum(boxes[a, 2:], boxes[other, 2:]) - np.maximum(boxes[a, :2], boxes[other, :2]))
            intersection = wh.prod(axis=1)
            area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
            iou = intersection / np.maximum(1e-12, area[a] + area[other] - intersection)
            if np.any((iou > .45) & (classes[a] == classes[other])):
                continue
        kept.append(int(a))
        if len(kept) == 300:
            break
    original = boxes[kept].copy()
    original[:, [0, 2]] = ((original[:, [0, 2]] - tx["PadX"]) / tx["ScaleX"]).clip(0, w)
    original[:, [1, 3]] = ((original[:, [1, 3]] - tx["PadY"]) / tx["ScaleY"]).clip(0, h)
    destination = work / path.stem
    subprocess.run(["dotnet", str(cli), "predict", str(path), str(work / "image.ppm"), args.backend, str(destination)], check=True)
    actual = json.loads((destination / "predictions.json").read_text())["Instances"]
    if len(actual) != len(kept):
        raise AssertionError(f"Detection count mismatch: {len(actual)} != {len(kept)}")
    mask_iou = []
    max_keypoint_error = 0.
    max_box_error = 0.
    reference_masks = None
    if task == "Segment":
        # Use the actual integer letterbox padding. Older scale_masks implementations
        # truncate recomputed floating padding (19.99999 -> 19), shifting masks a pixel.
        mh, mw = prototype.shape[2:]
        left = round(tx["PadX"] * mw / tx["TargetWidth"])
        top = round(tx["PadY"] * mh / tx["TargetHeight"])
        right = round((tx["PadX"] + w * tx["ScaleX"]) * mw / tx["TargetWidth"])
        bottom = round((tx["PadY"] + h * tx["ScaleY"]) * mh / tx["TargetHeight"])
        logits = torch.from_numpy(data[kept, 4 + nc:]) @ prototype[0].reshape(head.nm, -1)
        cropped = logits.reshape(len(kept), mh, mw)[:, top:bottom, left:right]
        resized = torch.nn.functional.interpolate(cropped[None], (h, w), mode="bilinear", align_corners=False)[0]
        reference_masks = (ops.crop_mask(resized, torch.from_numpy(original)) > 0).numpy()
    for i, (item, anchor) in enumerate(zip(actual, kept)):
        box = item["Box"]
        if box["ClassId"] != int(classes[anchor]):
            raise AssertionError("Class mismatch")
        expected_box = original[i]
        received = np.array([box["X"], box["Y"], box["X"] + box["Width"], box["Y"] + box["Height"]])
        error = float(np.abs(received - expected_box).max())
        max_box_error = max(error, max_box_error)
        np.testing.assert_allclose(received, expected_box, atol=.01, rtol=1e-4)
        if task == "Pose":
            points = data[anchor, 4 + nc:].reshape(head.kpt_shape).copy()
            points[:, 0] = (points[:, 0] - tx["PadX"]) / tx["ScaleX"]
            points[:, 1] = (points[:, 1] - tx["PadY"]) / tx["ScaleY"]
            received = np.array([[p["X"], p["Y"], p["Confidence"]] for p in item["Keypoints"]])
            max_keypoint_error = max(max_keypoint_error, float(np.abs(points - received).max()))
            np.testing.assert_allclose(received, points, atol=.01, rtol=1e-4)
        if task == "Segment":
            mask = item["Mask"]
            crop = np.asarray(Image.open(destination / mask["File"])) > 127
            canvas = np.zeros((h, w), dtype=bool)
            canvas[mask["Y"]:mask["Y"] + mask["Height"], mask["X"]:mask["X"] + mask["Width"]] = crop
            reference = reference_masks[i].astype(bool)
            intersection = np.logical_and(canvas, reference).sum()
            union = np.logical_or(canvas, reference).sum()
            similarity = float(intersection / max(1, union))
            mask_iou.append(similarity)
            if similarity < .99:
                raise AssertionError(f"Mask IoU too low: {similarity}")
    report = {"model": path.name, "backend": args.backend, "instances": len(actual), "max_box_error": max_box_error, "max_keypoint_error": max_keypoint_error, "mask_iou": mask_iou, "passed": True}
    print(json.dumps(report), flush=True)
    reports.append(report)
(work / "results.json").write_text(json.dumps(reports, indent=2))
