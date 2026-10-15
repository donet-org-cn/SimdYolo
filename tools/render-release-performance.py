"""从完整实测 JSON 生成当前发布的中文性能图和文章配图，不修改原始测量。"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib-release"))
CPU = ["torch-cpu", "net-cpu", "net-onnx-cpu", "ort-net-cpu"]
GPU = ["torch-cuda", "net-vulkan", "net-onnx-vulkan", "ort-net-cuda"]
LABELS = {"torch-cpu": "原版 PyTorch CPU", "net-cpu": "C# .pt / CPU SIMD", "net-onnx-cpu": "C# ONNX / CPU SIMD", "ort-net-cpu": "ORT CPU / .NET",
          "torch-cuda": "原版 PyTorch CUDA", "net-vulkan": "C# .pt / Vulkan", "net-onnx-vulkan": "C# ONNX / Vulkan", "ort-net-cuda": "ORT CUDA / .NET"}
COLORS = ["#8390a2", "#2876bd", "#1b9b8f", "#c78a35"]


def main():
    """严格检查版本、窗口和输出对照，再绘制带单位、范围及来源的图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    args.report = args.report.resolve()
    raw = args.report.read_bytes(); report = json.loads(raw)
    runs = defaultdict(list)
    for record in report["Records"]:
        runs[record["Model"], record["Engine"]].append(record)
        if record["Engine"].startswith("net-"):
            assert record["Version"].removeprefix("SimdYolo ").split("+", 1)[0] == args.version
        assert all(item["Passed"] for item in record["Correctness"])
    assert report["Status"] == "complete"
    assert len(report["Records"]) == len(report["Models"]) * len(report["Engines"]) * report["Repeats"]
    assert all(len(runs[m, e]) == report["Repeats"] for m in report["Models"] for e in report["Engines"])
    for directory in [".cache/article-graphics-deps", ".cache/reference"]:
        if (ROOT / directory).is_dir(): sys.path.insert(0, str(ROOT / directory))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "sans-serif"]
    plt.rcParams.update({"svg.fonttype": "path", "axes.unicode_minus": False, "savefig.facecolor": "white"})
    assets = ROOT / "docs/articles/assets" / args.version; assets.mkdir(parents=True, exist_ok=True)
    prefix = ROOT / "artifacts" / f"performance-{args.version}"
    models = report["Models"]
    subtitle = f"FP32 · batch=1 · {report['Size']}×{report['Size']} · {report['Threads']}线程 · 每配置{report['Repeats']}个独立窗口"
    source = f"SimdYolo {args.version} · {report['Date']} · i7-8700K / GTX 1050 Ti"

    def values(model, engine, metric):
        """取得独立窗口数值，资源指标支持其嵌套路径。"""
        return [metric(r) for r in runs[model, engine]]

    def median(model, engine, metric=lambda r: r["MedianMs"]):
        """按窗口中位数聚合，不把逐帧样本池化。"""
        return statistics.median(values(model, engine, metric))

    def save(fig, stem):
        """同时保留可上传的 PNG 与文字转轮廓的 SVG。"""
        for extension in ["png", "svg"]: fig.savefig(str(stem) + "." + extension, dpi=140)
        svg=Path(str(stem)+".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
        plt.close(fig)

    def bars(stem, subset, engines, title, metric=lambda r: r["MedianMs"], unit="毫秒 / 帧，越短越快", note=None):
        """绘制零起点四配置分组条形图，细线为各独立窗口的数值范围。"""
        fig, ax = plt.subplots(figsize=(12, max(5.5, len(subset) * .82 + 2.0)))
        y = np.arange(len(subset)); bar = .18; maximum = 0
        for j, engine in enumerate(engines):
            grouped = [values(m, engine, metric) for m in subset]
            aggregate = np.array([statistics.median(v) for v in grouped]); maximum = max(maximum, max(max(v) for v in grouped))
            ax.barh(y + (j-1.5)*bar, aggregate, height=bar*.9, color=COLORS[j], label=LABELS[engine])
            ax.errorbar(aggregate, y+(j-1.5)*bar, xerr=np.array([[a-min(v) for a,v in zip(aggregate,grouped)],[max(v)-a for a,v in zip(aggregate,grouped)]]), fmt="none", color="#334155", linewidth=.7, capsize=2)
            for i, value in enumerate(aggregate): ax.annotate(f"{value:.2f}", (max(grouped[i]), y[i]+(j-1.5)*bar), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set(yticks=y, yticklabels=subset, xlim=(0, maximum*1.18)); ax.invert_yaxis(); ax.set_xlabel(unit)
        ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.suptitle(title, fontsize=19, weight="bold", y=.98); fig.text(.5,.945,subtitle,ha="center",fontsize=10,color="#526079")
        fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(.54,.048), frameon=False, fontsize=10)
        fig.text(.02,.025,note or "细线为重复窗口范围，不是置信区间；数值排名不表示稳定领先。",fontsize=8,color="#526079")
        fig.text(.02,.007,source,fontsize=8,color="#526079"); fig.tight_layout(rect=(0,.115,1,.925)); save(fig, stem)

    bars(str(prefix)+"-cpu",models,CPU,"CPU 推理速度：原版 / C# / ONNX Runtime")
    bars(str(prefix)+"-gpu",models,GPU,"GPU 推理速度：CUDA / Vulkan / ONNX Runtime")
    detection = [m for m in models if not m.endswith(("-seg", "-pose"))]
    vision = [m for m in models if m not in detection]
    for stem, subset, engines, title in [("04-cpu-detection",detection,CPU,"目标检测：CPU 速度对比"),("05-cpu-vision",vision,CPU,"分割与姿态：CPU 速度对比"),("06-gpu-detection",detection,GPU,"目标检测：GPU 速度对比"),("07-gpu-vision",vision,GPU,"分割与姿态：GPU 速度对比")]:
        bars(assets/stem,subset,engines,title)
    bars(assets/"09-cpu-memory",models,CPU,"CPU 进程内存：包含宿主、模型与缓存",lambda r:r["Resources"]["MeanWorkingSetMiB"],"采样工作集 RSS / MiB，越低占用越少")
    bars(str(prefix)+"-cpu-time",models,CPU,"CPU 计算成本：每帧进程 CPU 时间",lambda r:r["CpuMsPerFrame"],"CPU 毫秒 / 帧，越低消耗越少")
    bars(str(prefix)+"-gpu-usage",models,GPU,"GPU 使用率：整张显卡的采样值",lambda r:r["Resources"]["MeanGpuPercent"],"GPU 忙碌时间 / %",note="整卡数值包含桌面及其他进程，不能解释为单进程 GPU 使用率或效率排名。")
    bars(str(prefix)+"-vram",models,GPU,"GPU 显存：整张显卡的采样值",lambda r:r["Resources"]["MeanDeviceVramMiB"],"整卡显存 / MiB",note="包含桌面及其他进程的显存；不是单模型或单进程显存。")

    # 首次前向与加载分别计时；不把全部预热或宿主启动当成首次输出。
    first = models[0]; fig, axes = plt.subplots(1,3,figsize=(18,7))
    engines = CPU + GPU
    for ax, (key,title) in zip(axes,[("ModelLoadMs","模型加载"),("FirstInferenceMs","首次前向与回读"),("MedianMs","预热后的稳态")]):
        samples = [median(first,e,lambda r:r[key]) for e in engines]
        ax.barh(range(len(engines)),samples,color=COLORS*2); ax.set_yticks(range(len(engines)),[LABELS[e] for e in engines],fontsize=9)
        ax.invert_yaxis();ax.set_title(title);ax.set_xlabel("毫秒");ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
        for i,value in enumerate(samples):ax.annotate(f"{value:.2f}",(value,i),xytext=(4,0),textcoords="offset points",va="center",fontsize=8)
        ax.set_xlim(0,max(samples)*1.25)
    fig.suptitle(f"{first}：加载、首次与稳态分开看 · {args.version}",fontsize=18,weight="bold")
    fig.text(.02,.04,"首次前向排除模型加载，GPU 包含输入上传及全部输出回读；新进程不表示清空系统/驱动缓存。",fontsize=10)
    fig.text(.02,.01,source,fontsize=9);fig.tight_layout(rect=(0,.08,1,.92));save(fig,str(prefix)+"-startup")

    def concept(stem,title,items,footer):
        """生成实现边界示意，明确区分结构说明与实测曲线。"""
        rows=(len(items)+1)//2;fig,ax=plt.subplots(figsize=(12,rows*2.5+2));ax.set(xlim=(0,12),ylim=(0,rows*2.5+2));ax.axis("off")
        ax.text(.35,rows*2.5+1.3,title,fontsize=23,weight="bold",color="#18324e")
        for i,(label,body) in enumerate(items):
            x=.35+(i%2)*6;y=rows*2.5-1.9-(i//2)*2.5
            ax.add_patch(FancyBboxPatch((x,y),5.3,2,boxstyle="round,pad=.12",facecolor="#eef4fa",edgecolor="#d9e5ef"))
            ax.text(x+.2,y+1.45,label,fontsize=15,weight="bold",color="#2876bd")
            ax.text(x+.2,y+.68,body,fontsize=12,color="#526079",va="center",linespacing=1.7)
        ax.text(.35,.15,footer,fontsize=9,color="#526079",va="bottom");fig.tight_layout();save(fig,assets/stem)

    for stem,technical in [("01-wechat-cover",False),("02-cnblogs-cover",True)]:
        fig,ax=plt.subplots(figsize=(16,8));bg="#edf4fa" if technical else "#132a43";fg="#132a43" if technical else "white"
        fig.patch.set_facecolor(bg);ax.set_facecolor(bg);ax.axis("off")
        ax.text(.04,.85,f"SimdYolo {args.version}  /  C# · .NET 10",transform=ax.transAxes,color="#39aaa4",fontsize=26)
        ax.text(.04,.58,"纯 C# YOLO：推理与检测头微调",transform=ax.transAxes,color=fg,fontsize=35,weight="bold")
        ax.text(.04,.40,"PT + ONNX  ·  CPU SIMD + Vulkan",transform=ax.transAxes,color=fg,fontsize=25)
        ax.text(.04,.23,"11 模型 CPU/GPU 实测  ·  33 个模型  ·  网页示例",transform=ax.transAxes,color=fg,fontsize=23)
        ax.text(.04,.08,"训练限 YOLOv8 检测头；Metal 未在本机实测",transform=ax.transAxes,color=fg,fontsize=16)
        fig.tight_layout();save(fig,assets/stem)
    concept("03-architecture","推理与训练：共享纯 C# 模型执行能力",[
        ("模型解析","PT / ONNX / SYTRAIN\n元数据、类别与结构校验"),("推理执行","CPU SIMD / Vulkan / Metal\n检测、分割、姿态公开结果"),
        ("冻结特征","YOLOv8 PT 骨干与 FPN 冻结\n特征缓存，不回传骨干梯度"),("检测头微调","Conv + BN + SiLU，真实反向\nTAL / BCE / CIoU / DFL / SGD"),
        ("保存与恢复","SYTRAIN：原 PT、头参数、BN\nSGD 动量、类别、数据摘要"),("网页版示例","浏览器图片和标签导入\n推理、曲线、取消、恢复、下载")],
        f"{args.version} 架构示意；Vulkan 训练为 GPU GEMM + CPU BN/损失/SGD；Metal 训练尚未实现。")
    concept("08-cpu-optimization","CPU SIMD：当前实现的计算与内存路径",[
        ("矩阵乘法","FP32 SIMD / intrinsics\n分块、尾部与可变权重"),("卷积执行","Winograd 与专用点卷积\n按形状选择路径并保留回退"),
        ("并行与所有权","按实际地址划分缓存行\n独立公开输出与临时张量池"),("训练中的差异","参数持续更新，不复用旧权重缓存\n冻结特征可以复用")],
        "这是结构示意，不代表单项优化收益；性能结论只来自同名原始实测 JSON。")
    concept("10-vulkan-deployment","Vulkan：减少 CUDA 计算环境依赖",[
        ("运行环境","64 位 .NET 10\n兼容显卡驱动与 Vulkan loader"),("无需 CUDA 工具链","本库不调用 CUDA / cuDNN / cuBLAS\n不要求安装 Vulkan SDK 来运行"),
        ("着色器随包提供","嵌入预编译 SPIR-V\n修改内核后才需要重新编译"),("跨平台与验证边界","Vulkan 是跨平台图形/计算 API\n本轮仅 Windows / NVIDIA 实测")],
        "跨厂商 API 不代表所有显卡/平台已验证；Metal 有独立实现，本轮无 macOS 实测。")
    data={"Version":args.version,"Source":args.report.relative_to(ROOT).as_posix(),"SourceSha256":hashlib.sha256(raw).hexdigest(),
          "AssemblySha256":report["SimdYoloDllSha256"],"Date":report["Date"],"Models":models,
          "LatencyMs":{m:{e:median(m,e) for e in report["Engines"]} for m in models},
          "CpuRssMiB":{m:{e:median(m,e,lambda r:r["Resources"]["MeanWorkingSetMiB"]) for e in CPU} for m in models},
          "TimedFrames":sum(r["Iterations"] for r in report["Records"]),"Windows":len(report["Records"]),
          "OutputChecks":sum(len(r["Correctness"]) for r in report["Records"]),
          "Aggregation":"Median of independent window medians; ranges are window ranges, not confidence intervals."}
    (assets/"benchmark-data.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8",newline="\n")
    spec=importlib.util.spec_from_file_location("comparison",ROOT/"tools/compare-runtimes.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tables="\n".join(module.render_speed_rankings(report))+"\n"
    (ROOT/".cache/release-0.9.0/rankings.md").write_text(tables,encoding="utf-8",newline="\n")
    print(json.dumps({"status":"passed","version":args.version,"sourceSha256":data["SourceSha256"],"windows":data["Windows"],"frames":data["TimedFrames"],"outputChecks":data["OutputChecks"]}))


if __name__ == "__main__":
    main()
