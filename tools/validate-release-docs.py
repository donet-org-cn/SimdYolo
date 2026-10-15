"""检查当前README、中文文章、排名和图片，再写出可追溯的发布文档清单。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent.parent


def sha(path):
    """计算发布文档或源结果的字节摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """只允许完整且数值检查通过的结果进入当前版本文章。"""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--version",default="0.9.0");parser.add_argument("--source-version");args=parser.parse_args();version=args.version
    report_path=ROOT/f"artifacts/runtime-comparison-{version}.json";report=json.loads(report_path.read_text(encoding="utf-8"))
    assert report["Status"]=="complete"
    expected=len(report["Models"])*len(report["Engines"])*report["Repeats"]
    assert len(report["Records"])==expected and all(c["Passed"] for r in report["Records"] for c in r["Correctness"])
    spec=importlib.util.spec_from_file_location("comparison",ROOT/"tools/compare-runtimes.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rankings="\n".join(module.render_speed_rankings(report));articles=[]
    names=["README.md","VALIDATION.md","models/README.md","docs/inference.md","docs/training.md","docs/articles/README.md","docs/articles/wechat.md","docs/articles/cnblogs.md","docs/releases/v0.7.1.md","docs/releases/v0.9.0.md","docs/releases/web-runtime.md","examples/SimdYolo.Web/README.md","artifacts/validation/0.9.0-release/README.md"]
    for name in names:
        path=ROOT/name;text=path.read_text(encoding="utf-8-sig")
        assert "\ufffd" not in text and "????" not in text,(name,"broken Unicode")
        assert "<!-- CURRENT_RANKINGS -->" not in text and "<!-- TIMED_FRAMES -->" not in text,(name,"unfilled data")
        assert text.count("```")%2==0,(name,"unclosed code fence")
        if name in {"README.md","docs/articles/wechat.md","docs/articles/cnblogs.md","docs/releases/v0.9.0.md"}:assert rankings in text,(name,"rank table differs from raw report")
        images=[]
        for target in re.findall(r"\]\(([^)]+)\)",text):
            if "://" in target or target.startswith("#"):continue
            local=path.parent/target.split("#")[0]
            assert local.exists(),(name,target)
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)",text):
            if "://" not in target:images.append(target)
        articles.append({"File":name,"Sha256":sha(path),"ChineseCharacters":len(re.findall(r"[\u4e00-\u9fff]",text)),"Images":images,"Utf8AndLinksPassed":True})
    asset_dir=ROOT/f"docs/articles/assets/{version}";data=json.loads((asset_dir/"benchmark-data.json").read_text(encoding="utf-8"));assert data["SourceSha256"]==sha(report_path) and data["Version"]==version
    assert len(list(asset_dir.glob("*.png")))==10 and len(list(asset_dir.glob("*.svg")))==10
    files=[]
    for path in sorted(asset_dir.iterdir()):
        if not path.is_file():continue
        item={"File":path.relative_to(ROOT/"docs/articles").as_posix(),"Sha256":sha(path),"Bytes":path.stat().st_size}
        if path.suffix==".png":
            content=path.read_bytes();assert content.startswith(b"\x89PNG\r\n\x1a\n");item["Dimensions"]=list(struct.unpack(">II",content[16:24]));assert item["Dimensions"][0]>=1000
        elif path.suffix==".svg":ET.parse(path)
        files.append(item)
    for name in ["cpu","gpu","cpu-time","gpu-usage","vram","startup"]:
        for suffix in ["png","svg"]:assert (ROOT/f"artifacts/performance-{version}-{name}.{suffix}").is_file()
    manifest={"SourceReportSha256":sha(report_path),"MeasuredVersion":version,"CurrentRelease":version,"AssemblySha256":report["SimdYoloDllSha256"],
              "MeasurementWindows":len(report["Records"]),"TimedFrames":sum(r["Iterations"] for r in report["Records"]),"CorrectnessOutputs":sum(len(r["Correctness"]) for r in report["Records"]),
              "Articles":articles,"Files":files,"Checks":{"AllRankTablesMatchRawReport":True,"Utf8WithoutReplacementCharacters":True,"RelativeLinksExist":True,"PngAndSvgValidated":True}}
    manifest["CurrentSourceVersion"]=args.source_version or version
    (ROOT/"docs/articles/manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8",newline="\n")
    print(json.dumps({"status":"passed","documents":len(articles),"png":10,"svg":10,"version":version}))


if __name__=="__main__":main()
