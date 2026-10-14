"""Repeatable package, managed metadata, Chinese XML and model-manifest audit.

Uses Python's standard library and tools/PackageAudit (.NET standard libraries).
No models are executed and no performance measurements are taken. The default
packs already-built Release outputs; use --skip-pack to audit an existing nupkg.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_MODEL_LICENSE_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"


def sha256(path):
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    """Persist the audit including actual gaps, not a manufactured pass count."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    """Pack existing binaries, inspect their identity/metadata, then verify models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--version", help="Default: Version from src/SimdYolo/SimdYolo.csproj")
    parser.add_argument("--compiled-dir", type=Path, help="Default: src/SimdYolo/bin/Release/net10.0")
    parser.add_argument("--package", type=Path, help="Default: artifacts/SimdYolo.VERSION.nupkg")
    parser.add_argument("--output", type=Path, help="Default: artifacts/validation/VERSION/package-audit.json")
    parser.add_argument("--expected-models", type=int, default=33)
    parser.add_argument("--skip-pack", action="store_true", help="Inspect an existing package instead of running dotnet pack --no-build")
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "src/SimdYolo/SimdYolo.csproj"
    version = args.version or ET.parse(project).getroot().findtext("./PropertyGroup/Version")
    if not version or Path(version).name != version:
        parser.error("Project version is missing or invalid")
    compiled = args.compiled_dir.resolve() if args.compiled_dir else root / "src/SimdYolo/bin/Release/net10.0"
    package = args.package.resolve() if args.package else root / f"artifacts/SimdYolo.{version}.nupkg"
    output = args.output.resolve() if args.output else root / f"artifacts/validation/{version}/package-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    issues = []
    report = {"Version": version, "AuditStartedUtc": datetime.now(timezone.utc).isoformat(),
              "Status": "running", "Issues": issues,
              "Scope": "Package content identity, managed PE/references, compiled Chinese XML structure and local model manifest. No inference/performance or full semantic exception-coverage claim."}
    save_json(output, report)
    try:
        if not args.skip_pack:
            command = ["dotnet", "pack", str(project), "-c", "Release", "--no-build", "-o", str(package.parent)]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
            (output.parent / "package-pack.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError("dotnet pack failed; see package-pack.log")
            report["PackCommand"] = command
        report["PackagePath"] = str(package)
        report["PackageSha256"] = sha256(package)
        report["PackageSizeBytes"] = package.stat().st_size
        extracted = root / f".cache/package-audit/{version}"
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(package) as archive:
            entries = archive.namelist()
            report["Entries"] = entries
            nuspec_name = next(name for name in entries if name.endswith(".nuspec"))
            nuspec = ET.fromstring(archive.read(nuspec_name))
            report["ExternalNugetDependencies"] = [node.attrib for node in nuspec.findall(".//{*}dependency")]
            report["PackageMetadataVersion"] = nuspec.findtext(".//{*}metadata/{*}version")
            if report["PackageMetadataVersion"] != version:
                issues.append("PackageVersionMismatch")
            if report["ExternalNugetDependencies"]:
                issues.append("ExternalNugetDependencies")
            report["PythonAssets"] = [name for name in entries if name.lower().endswith((".py", ".pyc", ".pyd"))]
            report["OtherRuntimeBinaries"] = [name for name in entries if name.lower().endswith((".dll", ".so", ".dylib", ".exe")) and name != "lib/net10.0/SimdYolo.dll"]
            if report["PythonAssets"] or report["OtherRuntimeBinaries"]:
                issues.append("UnexpectedRuntimeAssets")
            for name in ["SimdYolo.dll", "SimdYolo.xml"]:
                data = archive.read("lib/net10.0/" + name)
                packed = extracted / name
                packed.write_bytes(data)
                packed_hash = hashlib.sha256(data).hexdigest()
                compiled_hash = sha256(compiled / name)
                key = "Dll" if name.endswith(".dll") else "Xml"
                report[key + "Sha256"] = packed_hash
                report[key + "MatchesCompiled"] = packed_hash == compiled_hash
                if packed_hash != compiled_hash:
                    issues.append(key + "DoesNotMatchCompiled")
            documentation = {}
            for name in ["README.md", "VALIDATION.md"]:
                matches = archive.read(name) == (root / name).read_bytes()
                documentation[name] = matches
                if not matches:
                    issues.append("PackedDocumentationMismatch:" + name)
            report["PackedDocumentationMatchesWorkspace"] = documentation
        helper = root / "tools/PackageAudit/PackageAudit.csproj"
        build = subprocess.run(["dotnet", "build", str(helper), "-c", "Release"], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (output.parent / "package-helper-build.log").write_text(build.stdout + build.stderr, encoding="utf-8")
        if build.returncode:
            raise RuntimeError("Metadata audit helper build failed")
        metadata_path = extracted / "metadata-audit.json"
        result = subprocess.run(["dotnet", str(helper.parent / "bin/Release/net10.0/PackageAudit.dll"),
                                 str(extracted / "SimdYolo.dll"), str(extracted / "SimdYolo.xml"), str(metadata_path)],
                                cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (output.parent / "package-helper.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Metadata/XML inspection failed")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        report["ManagedMetadataAndXml"] = metadata
        if metadata["Version"] != version:
            issues.append("AssemblyVersionMismatch")
        if not metadata["IlOnly"] or metadata["ExternalAssemblyReferences"]:
            issues.append("ManagedLibraryOrFrameworkReferencesCheckFailed")
        if metadata["DocumentationProblems"] or metadata["UndocumentedPublicSymbols"] or metadata["UnmappedXmlMemberIds"]:
            issues.append("XmlDocumentationRequiresReview")
        manifest_path = root / "models/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        model_rows = []; listed = set(); total_bytes = 0
        for item in manifest["Models"]:
            relative = item["File"]
            path = (root / relative).resolve()
            try:
                path.relative_to(root / "models")
            except ValueError:
                raise RuntimeError("Manifest model path escapes the models directory")
            if relative in listed:
                issues.append("DuplicateManifestFile:" + relative)
            listed.add(relative)
            exists = path.is_file()
            actual_hash = sha256(path) if exists else None
            actual_bytes = path.stat().st_size if exists else None
            size_matches = actual_bytes == item["SizeBytes"]
            hash_matches = actual_hash == item["Sha256"]
            license_present = item.get("License", "").startswith("AGPL-3.0")
            source_url = urlparse(item.get("SourceUrl", ""))
            source_present = source_url.scheme == "https" and bool(source_url.netloc)
            source_weight = item.get("SourceWeight")
            source_exists = bool(source_weight and (root / source_weight).is_file())
            if not (size_matches and hash_matches and license_present and source_present and source_exists):
                issues.append("ModelManifestMismatchOrMissingProvenance:" + relative)
            total_bytes += actual_bytes or 0
            model_rows.append({"File": relative, "SizeBytes": actual_bytes, "Sha256": actual_hash,
                               "SizeMatches": size_matches, "HashMatches": hash_matches,
                               "LicensePresent": license_present, "SourceUrlPresent": source_present,
                               "SourceWeightPresent": source_exists})
        actual_models = {str(path.relative_to(root)).replace("\\", "/") for folder in [root / "models/pt", root / "models/onnx"]
                         for path in folder.iterdir() if path.suffix.lower() in (".pt", ".onnx")}
        if actual_models != listed:
            issues.append("ModelFilesAndManifestDiffer")
        if len(listed) != args.expected_models or manifest.get("Count") != len(listed) or manifest.get("TotalBytes") != total_bytes:
            issues.append("ModelManifestCountOrTotalBytesMismatch")
        license_path = root / "models/LICENSE-AGPL-3.0.txt"
        notices = root / "THIRD_PARTY_NOTICES.md"
        license_hash = sha256(license_path) if license_path.is_file() else None
        notice_text = notices.read_text(encoding="utf-8-sig") if notices.is_file() else ""
        license_preserved = license_hash == ORIGINAL_MODEL_LICENSE_SHA256
        notice_present = "AGPL-3.0" in notice_text and "v8.3.0" in notice_text and "v8.4.0" in notice_text
        if not license_preserved or not notice_present:
            issues.append("OriginalModelLicenseOrSourceNoticesMissing")
        report["ModelManifest"] = {"Path": str(manifest_path), "Sha256": sha256(manifest_path), "Files": len(listed),
                                   "ExpectedFiles": args.expected_models, "TotalBytes": total_bytes, "Models": model_rows,
                                   "UnlistedFiles": sorted(actual_models - listed), "MissingFiles": sorted(listed - actual_models)}
        report["ModelLicensing"] = {"LicensePath": str(license_path), "LicenseSha256": license_hash,
                                    "OriginalLicenseBytesPreserved": license_preserved, "SourceNoticesPresent": notice_present,
                                    "ThirdPartyNoticesSha256": sha256(notices) if notices.is_file() else None,
                                    "Scope": "Presence, recorded provenance and byte preservation; no legal compatibility opinion is inferred."}
        report["Status"] = "passed" if not issues else "requires-review"
        report["AuditFinishedUtc"] = datetime.now(timezone.utc).isoformat()
        save_json(output, report)
        print(json.dumps({"Status": report["Status"], "Package": str(package), "DllMatchesCompiled": report["DllMatchesCompiled"],
                          "XmlMembers": metadata["XmlMembers"], "Models": len(listed), "Issues": issues}, ensure_ascii=False))
        return 0 if not issues else 1
    except Exception as error:
        report["Status"] = "failed"
        report["Error"] = str(error)
        save_json(output, report)
        raise


if __name__ == "__main__":
    sys.exit(main())
