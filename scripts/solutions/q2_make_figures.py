from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
Q2_DIR = BASE_DIR / "outputs" / "q2"
FIG_DIR = Q2_DIR / "figures"

THRESHOLD_FILE = Q2_DIR / "q2_vehicle_speed_threshold_reference.csv"
EVENT_FILE = Q2_DIR / "q2_high_event_judgement.csv"
L3_FILE = Q2_DIR / "q2_l3_after_20240301_judgement.csv"
SUMMARY_FILE = Q2_DIR / "q2_summary.txt"


def set_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def save_flowchart() -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")

    steps = [
        ("输入数据", "清洗后检测数据\n每3分钟上传"),
        ("门限匹配", "按 车型+车速分箱\n匹配自适应门限"),
        ("等级复核", "计算 corrected_level\n比较 level_is_correct"),
        ("高值归因", "仅 corrected_level>=3\n3天+50米时空复现规则"),
        ("输出结果", "事件类型/置信度\n汇总指标与明细表"),
    ]

    x_positions = np.linspace(0.08, 0.92, len(steps))
    y = 0.52
    w = 0.16
    h = 0.38

    for idx, (title, body) in enumerate(steps):
        x = x_positions[idx] - w / 2
        box = plt.Rectangle((x, y - h / 2), w, h, ec="#243447", fc="#e8f1fa", lw=1.6)
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.08, title, ha="center", va="center", fontsize=12, fontweight="bold", color="#0f2740")
        ax.text(x + w / 2, y - 0.06, body, ha="center", va="center", fontsize=10, color="#153a5b", linespacing=1.4)
        if idx < len(steps) - 1:
            x0 = x + w
            x1 = x_positions[idx + 1] - w / 2
            ax.annotate(
                "",
                xy=(x1 - 0.01, y),
                xytext=(x0 + 0.01, y),
                arrowprops=dict(arrowstyle="->", color="#375a7f", lw=2),
            )

    ax.set_title("Q2 实时判定流程图", fontsize=14, fontweight="bold", color="#0b1f33", pad=10)
    fig.tight_layout()
    out = FIG_DIR / "q2_fig1_realtime_flow.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _heatmap_on_axis(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    arr = data.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=0, fontsize=8)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index.astype(str), fontsize=8)
    ax.set_title(title, fontsize=10, pad=4)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.8, color="#2a1200")
    return im


def save_threshold_heatmap() -> Path:
    th = pd.read_csv(THRESHOLD_FILE)
    speed_order = ["40-60", "60-80", "80-100", "100-120", "120-140", "140-200"]
    th["speed_bin"] = pd.Categorical(th["speed_bin"], categories=speed_order, ordered=True)
    th = th.sort_values(["车型", "speed_bin"]).reset_index(drop=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    titles = [
        ("垂加_t1", "垂向门限 t1"),
        ("垂加_t2", "垂向门限 t2"),
        ("垂加_t3", "垂向门限 t3"),
        ("垂加_t4", "垂向门限 t4"),
        ("水加_t1", "横向门限 t1"),
        ("水加_t2", "横向门限 t2"),
        ("水加_t3", "横向门限 t3"),
        ("水加_t4", "横向门限 t4"),
    ]

    ims = []
    for ax, (col, title) in zip(axes.flatten(), titles):
        pivot = th.pivot(index="车型", columns="speed_bin", values=col)
        im = _heatmap_on_axis(ax, pivot, title)
        ims.append(im)
        ax.set_xlabel("速度分箱", fontsize=8)
        ax.set_ylabel("车型", fontsize=8)

    cbar = fig.colorbar(ims[0], ax=axes, shrink=0.82, location="right")
    cbar.set_label("门限值", fontsize=9)
    fig.suptitle("Q2 车型-车速自适应门限热力图", fontsize=14, fontweight="bold")
    out = FIG_DIR / "q2_fig2_threshold_heatmaps.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _parse_metrics(summary_text: str) -> dict[str, float]:
    patterns = {
        "precision": r"精确率:\s*([0-9.]+)",
        "recall": r"召回率:\s*([0-9.]+)",
        "f1": r"F1:\s*([0-9.]+)",
    }
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, summary_text)
        out[key] = float(m.group(1)) if m else np.nan
    return out


def save_event_distribution() -> Path:
    events = pd.read_csv(EVENT_FILE)
    l3 = pd.read_csv(L3_FILE)
    summary_text = SUMMARY_FILE.read_text(encoding="utf-8")
    metrics = _parse_metrics(summary_text)

    order = ["设备或干扰", "线路问题", "不确定"]
    left = events["event_type"].value_counts().reindex(order).fillna(0).astype(int)
    right = l3["event_type"].value_counts().reindex(order).fillna(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    axes[0].bar(left.index, left.values, color=colors)
    axes[0].set_title("修正等级>=3 事件类型分布", fontsize=11)
    axes[0].set_ylabel("事件数")
    axes[0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(left.values):
        axes[0].text(i, v + max(3, 0.01 * left.max()), str(v), ha="center", fontsize=9)

    axes[1].bar(right.index, right.values, color=colors)
    axes[1].set_title("2024-03-01 后原始三级报警判定", fontsize=11)
    axes[1].grid(axis="y", alpha=0.25)
    for i, v in enumerate(right.values):
        axes[1].text(i, v + max(1, 0.03 * max(right.max(), 1)), str(v), ha="center", fontsize=9)

    metric_text = (
        f"弱监督指标: Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
    )
    fig.suptitle("Q2 高值事件归因统计", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02, metric_text, ha="center", fontsize=10, color="#16324F")

    out = FIG_DIR / "q2_fig3_event_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    out1 = save_flowchart()
    out2 = save_threshold_heatmap()
    out3 = save_event_distribution()

    captions = [
        "图7-1 Q2实时判定流程图：展示数据输入、门限匹配、等级复核与归因输出的在线流程。",
        "图7-2 车型-车速自适应门限热力图：按车型与速度分箱展示垂向/横向 t1~t4 门限。",
        "图7-3 高值事件归因分布图：左为全量高值事件，右为2024-03-01后原始三级报警判定结果，并附弱监督指标。",
    ]
    caption_file = FIG_DIR / "q2_figure_captions.txt"
    caption_file.write_text("\n".join(captions), encoding="utf-8")

    print("已生成图像文件:")
    print(f"- {out1}")
    print(f"- {out2}")
    print(f"- {out3}")
    print(f"- {caption_file}")


if __name__ == "__main__":
    main()
