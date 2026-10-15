"""制作模型、网页和文档附件，核验冻结程序集，并明确保留历史推理测量版本。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    """流式计算SHA256，不把模型包整块读入内存。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    """测量完成后运行，压缩期间不并行做性能计时。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.9.0")
    parser.add_argument("--inference-version", help="推理测量版本；与发布版本不同时保留为明确标注的历史结果")
    parser.add_argument("--publish-dir", type=Path, default=ROOT/".cache/training-validation/published-web")
    parser.add_argument("--reuse-model-web", action="store_true", help="保留已验证模型/网页附件的原字节，仅刷新文档和校验和")
    args = parser.parse_args(); version=args.version
    measured=args.inference_version or version
    report=json.loads((ROOT/f"artifacts/runtime-comparison-{measured}.json").read_text(encoding="utf-8"))
    assert report["Status"]=="complete" and all(c["Passed"] for r in report["Records"] for c in r["Correctness"])
    frozen=json.loads((ROOT/f"artifacts/validation/{version}/final-audit.json").read_text(encoding="utf-8"))
    assemblies=frozen.get("assemblies") or {"SimdYolo.dll":frozen["coreAssemblySha256"],"SimdYolo.Training.dll":frozen["trainingAssemblySha256"]}
    assert frozen.get('status')=='passed'
    if measured==version: assert report["SimdYoloDllSha256"]==assemblies["SimdYolo.dll"]
    for name in ["SimdYolo.dll","SimdYolo.Training.dll"]:
        assert sha(args.publish_dir/name)==assemblies[name]
    # 页面必须来自当前源码，防止运行包附上旧的CSS/JS。
    for source in (ROOT/"examples/SimdYolo.Web/wwwroot").rglob("*"):
        if source.is_file(): assert source.read_bytes()==(args.publish_dir/"wwwroot"/source.relative_to(ROOT/"examples/SimdYolo.Web/wwwroot")).read_bytes()
    output=ROOT/"artifacts"; assets=[]
    manifest=json.loads((ROOT/"models/manifest.json").read_text(encoding="utf-8"))
    model_files=[ROOT/m["File"] for m in manifest["Models"]]; assert len(model_files)==33
    for model,path in zip(manifest["Models"],model_files):
        assert sha(path)==model["Sha256"] and path.stat().st_size==model["SizeBytes"]
    model_archive=output/f"SimdYolo-models-v{version}.zip"
    if not args.reuse_model_web:
        with zipfile.ZipFile(model_archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in model_files+[ROOT/"models/README.md",ROOT/"models/manifest.json",ROOT/"models/LICENSE-AGPL-3.0.txt",ROOT/"THIRD_PARTY_NOTICES.md"]:
                archive.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(model_archive) as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if n.endswith((".pt",".onnx"))])==33
        for model in manifest['Models']:
            assert hashlib.sha256(archive.read(model['File'])).hexdigest()==model['Sha256']
    assets.append(model_archive)
    print(json.dumps({"packaged":model_archive.name,"bytes":model_archive.stat().st_size}),flush=True)

    web_archive=output/f"SimdYolo-Web-v{version}.zip"
    if not args.reuse_model_web:
        with zipfile.ZipFile(web_archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in args.publish_dir.rglob("*"):
                if path.is_file() and path.suffix!=".exe":archive.write(path,path.relative_to(args.publish_dir).as_posix())
            archive.write(ROOT/"docs/releases/web-runtime.md","README.md")
            archive.writestr("start.cmd",'@echo off\r\ndotnet "%~dp0SimdYolo.Web.dll" --contentRoot "%~dp0." --SimdYolo:RepositoryRoot "%~dp0." --SimdYolo:DataRoot "%~dp0data" --urls http://127.0.0.1:5188\r\npause\r\n')
            archive.writestr("start.ps1",'$ErrorActionPreference = "Stop"\n& dotnet (Join-Path $PSScriptRoot "SimdYolo.Web.dll") --contentRoot $PSScriptRoot --SimdYolo:RepositoryRoot $PSScriptRoot --SimdYolo:DataRoot (Join-Path $PSScriptRoot "data") --urls http://127.0.0.1:5188\n')
            archive.write(ROOT/"THIRD_PARTY_NOTICES.md","THIRD_PARTY_NOTICES.md")
    with zipfile.ZipFile(web_archive) as archive:
        for name in ["SimdYolo.dll","SimdYolo.Training.dll"]:assert hashlib.sha256(archive.read(name)).hexdigest()==assemblies[name]
        assert archive.read("README.md")== (ROOT/"docs/releases/web-runtime.md").read_bytes()
    assets.append(web_archive)

    # 文档包保持仓库相对布局；未打包的历史资料链接转到版本化GitHub地址。
    files={ROOT/"README.md",ROOT/"VALIDATION.md",ROOT/"THIRD_PARTY_NOTICES.md",ROOT/"models/README.md",ROOT/"models/manifest.json",ROOT/"models/LICENSE-AGPL-3.0.txt",ROOT/"examples/SimdYolo.Web/README.md"}
    files.update((ROOT/"docs").rglob("*.md"))
    files.update(p for p in (ROOT/"docs/articles/assets").rglob("*") if p.is_file())
    files.add(ROOT/"docs/articles/manifest.json")
    files.update(p for p in output.glob(f"*{version}*") if p.is_file() and p.suffix in {".json",".md",".png",".svg"})
    for directory in [output/f"validation/{version}",output/f"validation/{version}-release"]:
        files.update(p for p in directory.rglob("*") if p.is_file() and p.name not in {"package-assets.json","release-publication.json","final-release-audit.json"})
    # 收集正文实际引用的图片、历史报告与原始数据，保持离线文档里的相对图片完整。
    allowed={'.md','.json','.png','.svg','.txt','.log'}
    pending=list(files)
    while pending:
        page=pending.pop()
        if page.suffix!='.md': continue
        for target in re.findall(r'\]\(([^)]+)\)',page.read_text(encoding='utf-8-sig')):
            if '://' in target or target.startswith('#'): continue
            path=(page.parent/target.split('#')[0]).resolve()
            if path.is_relative_to(ROOT) and path.is_file() and path.suffix in allowed and path not in files:
                files.add(path);pending.append(path)
    packaged={p.relative_to(ROOT).as_posix() for p in files}

    def markdown(path):
        """相对图片随包保留；未包含的历史文件改为可访问的GitHub链接。"""
        text=path.read_text(encoding="utf-8-sig")
        def replace(match):
            target=match[1]
            if "://" in target or target.startswith("#"):return match[0]
            file=(path.parent/target.split("#")[0]).resolve()
            if not file.is_relative_to(ROOT):return match[0]
            name=file.relative_to(ROOT).as_posix()
            if name in packaged or file.is_dir() and any(p.startswith(name.rstrip("/")+"/") for p in packaged):return match[0]
            route="tree" if file.is_dir() else "blob"
            return "](https://github.com/hnlyf168/SimdYolo/"+route+"/v"+version+"/"+name+")"
        return re.sub(r"\]\(([^)]+)\)",replace,text)

    docs_archive=output/f"SimdYolo-docs-v{version}.zip"
    with zipfile.ZipFile(docs_archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(files):
            name=path.relative_to(ROOT).as_posix()
            if path.suffix==".md":archive.writestr(name,markdown(path).encode("utf-8"))
            else:archive.write(path,name)
        input_path=Path(report["InputDescription"])
        assert sha(input_path)==report["InputSha256"] and input_path.stat().st_size==3*640*640*4
        archive.write(input_path,"benchmark/input-640.f32")
        archive.writestr("benchmark/README.md",("# 历史推理固定实拍图输入\n\ninput-640.f32 对应"+measured+"推理历史测量，与0.8.1使用相同输入；不是"+version+"训练性能或训练精度的输入。little-endian FP32、NCHW [1,3,640,640]、RGB、0到1，含letterbox。\n\nSHA256：`"+report["InputSha256"]+"`。在compare-runtimes.py的 --input 参数传入此文件即可保持历史输入一致。\n\n原图为Ultralytics官方bus.jpg：https://ultralytics.com/images/bus.jpg 。相关原许可证随文档包models目录保留；仅用于复现实测，不代表独立准确率数据集。\n").encode("utf-8"))
    assets.append(docs_archive)
    assets += [output/f"SimdYolo.{version}.nupkg",output/f"SimdYolo.Training.{version}.nupkg"]
    for path in assets:
        with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
    checksums=output/f"SHA256SUMS-v{version}.txt"
    checksums.write_text("".join(f"{sha(path)}  {path.name}\n" for path in sorted(assets)),encoding="utf-8",newline="\n")
    audit={"Status":"passed","Version":version,"Models":33,"SourceLicenseAdded":False,"OriginalModelsAndLicensePreserved":True,
           "FrozenAssemblies":assemblies,"InferenceMeasuredVersion":measured,"BenchmarkSha256":sha(output/f"runtime-comparison-{measured}.json"),
           "Assets":[{"Name":p.name,"SizeBytes":p.stat().st_size,"Sha256":sha(p)} for p in assets+[checksums]]}
    audit_path=output/f"validation/{version}-release/package-assets.json";audit_path.parent.mkdir(parents=True,exist_ok=True)
    audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8",newline="\n")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    main()
