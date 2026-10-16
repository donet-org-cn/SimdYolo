"""记录最终源码、程序集和实际验证文件摘要；不制造数值结果，不改写测量时的SHA。"""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from urllib.request import urlopen
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/validation/0.11.0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = {name: json.loads((DEST / name).read_text(encoding="utf-8")) for name in
            ["features.json", "onnx-export.json", "precision.json", "package-audit.json", "training-metadata.json", "package-consumer.json", "web-workflows/web-browser.json"]}
    assert data["features.json"]["status"] == "passed" and len(data["features.json"]["checks"]) == 23
    assert data["onnx-export.json"]["status"] == "passed" and len(data["onnx-export.json"]["checks"]) == 16
    assert data["precision.json"]["status"] == "completed" and len(data["precision.json"]["runs"]) == 20
    assert data["package-audit.json"]["Status"] == "passed" and not data["package-audit.json"]["Issues"]
    metadata = data["training-metadata.json"]
    assert metadata["IlOnly"] and not metadata["DocumentationProblems"] and not metadata["UndocumentedPublicSymbols"] and metadata["ExternalAssemblyReferences"] == ["SimdYolo"]
    assert data["package-consumer.json"]["status"] == "passed"
    assert data["web-workflows/web-browser.json"]["status"] == "passed" and not data["web-workflows/web-browser.json"]["errors"]
    assert "113 tests passed." in (DEST / "inference-tests.log").read_text(encoding="utf-8-sig")
    assert "19 training tests passed." in (DEST / "training-tests.log").read_text(encoding="utf-8-sig")
    core = ROOT / "src/SimdYolo/bin/Release/net10.0/SimdYolo.dll"
    training = ROOT / "src/SimdYolo.Training/bin/Release/net10.0/SimdYolo.Training.dll"
    assert data["package-consumer.json"]["coreAssemblySha256"] == sha(core)
    assert data["package-consumer.json"]["trainingAssemblySha256"] == sha(training)
    assert data["precision.json"]["coreAssemblySha256"] == sha(core)
    assert data["precision.json"]["trainingAssemblySha256"] == sha(training)
    assert data["features.json"]["coreAssemblySha256"] == sha(core) and data["features.json"]["trainingAssemblySha256"] == sha(training)
    packages = []
    for name in ["SimdYolo", "SimdYolo.Training"]:
        package = ROOT / ("artifacts/" + name + ".0.11.0.nupkg")
        with zipfile.ZipFile(package) as archive:
            spec = ET.fromstring(archive.read(name + ".nuspec"))
            dependencies = [node.attrib for node in spec.findall(".//{*}dependency")]
            assert [item["id"] for item in dependencies] == (["SimdYolo"] if name.endswith("Training") else [])
            expected = training if name.endswith("Training") else core
            assert hashlib.sha256(archive.read("lib/net10.0/" + name + ".dll")).hexdigest() == sha(expected)
            packages.append(dict(file=package.relative_to(ROOT).as_posix(), sha256=sha(package), dependencies=dependencies))
    source_paths = set()
    for directory in [ROOT / "src", ROOT / "examples/SimdYolo.Web", ROOT / "tests/SimdYolo.Training.Tests"]:
        source_paths.update(path for path in directory.rglob("*") if path.is_file() and path.suffix in [".cs", ".csproj", ".js", ".css", ".html"] and not {"bin", "obj"}.intersection(path.parts))
    source_paths.update([ROOT / "Directory.Build.props", ROOT / "tools/validate-managed-onnx.py", ROOT / "tools/validate-web-browser.mjs", ROOT / "tools/validate-web-workflows.mjs", ROOT / "tools/render-precision-validation.mjs", Path(__file__)])
    measured_training = data["precision.json"]["trainingAssemblySha256"]
    record = dict(status="passed", recordedUtc=datetime.now(timezone.utc).isoformat(),
                  gitBaseCommit=subprocess.check_output(["git", "-c", "safe.directory=F:/simdyolo", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  coreAssemblySha256=sha(core), trainingAssemblySha256=sha(training),
                  measuredTrainingAssemblySha256=measured_training,
                  afterMeasurement="Feature tests, precision measurements and offline package consumption bind to the final core and training assemblies. Existing FP32/Vulkan regression and browser records are retained; the final cache correction only changes mutable low-precision convolution weights and is covered by an independent mutation test.",
                  packages=packages,
                  sourceFiles={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(source_paths)},
                  evidenceFiles={name: sha(DEST / name) for name in data},
                  webVersion=json.load(urlopen("http://127.0.0.1:5199/api/info")).get("version"))
    (DEST / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="passed", sourceFiles=len(source_paths), packages=len(packages), coreSha256=sha(core), trainingSha256=sha(training))))


if __name__ == "__main__":
    main()
