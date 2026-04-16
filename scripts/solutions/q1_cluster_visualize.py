from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "outputs" / "q1" / "q1_device_scores_cluster.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "q1" / "visualization"


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"结果文件不存在: {path}")
    return pd.read_csv(path)


def build_embedding(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    x = df[feat_cols].to_numpy(dtype=float)
    x = RobustScaler().fit_transform(x)
    x = np.clip(x, -8.0, 8.0)
    emb = PCA(n_components=2, random_state=42).fit_transform(x)
    return emb


def ensure_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "device_status" not in out.columns:
        if "is_low_evidence" not in out.columns:
            out["is_low_evidence"] = 0
        out["device_status"] = np.where(
            out["is_abnormal_device"] == 1,
            "abnormal",
            np.where(out["is_low_evidence"] == 1, "low_evidence", "normal"),
        )
    return out


def plot_pca_by_cluster(df: pd.DataFrame, emb: np.ndarray, out_file: Path) -> None:
    plt.figure(figsize=(10, 7))
    labels = df["cluster_label"].to_numpy()
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len([x for x in unique_labels if x >= 0]), 1))

    color_idx = 0
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            plt.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=45,
                c="black",
                alpha=0.75,
                label="noise(-1)",
                marker="x",
            )
        else:
            plt.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=40,
                c=[cmap(color_idx)],
                alpha=0.75,
                label=f"cluster {lab}",
            )
            color_idx += 1

    plt.title("Q1 Clustering by DBSCAN Label (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_pca_by_status(df: pd.DataFrame, emb: np.ndarray, out_file: Path) -> None:
    plt.figure(figsize=(10, 7))
    status_order = ["normal", "low_evidence", "abnormal"]
    color_map = {"normal": "#4c78a8", "low_evidence": "#f58518", "abnormal": "#e45756"}
    marker_map = {"normal": "o", "low_evidence": "^", "abnormal": "x"}

    for s in status_order:
        mask = df["device_status"] == s
        if mask.sum() == 0:
            continue
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=46 if s != "abnormal" else 58,
            c=color_map[s],
            alpha=0.80,
            label=f"{s} ({int(mask.sum())})",
            marker=marker_map[s],
        )

    plt.title("Q1 Device Status View (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_reliability_hist(df: pd.DataFrame, out_file: Path) -> None:
    scores = df["reliability_score"].to_numpy(dtype=float)
    p10 = float(np.quantile(scores, 0.10))

    plt.figure(figsize=(10, 6))
    status_order = ["normal", "low_evidence", "abnormal"]
    color_map = {"normal": "#4c78a8", "low_evidence": "#f58518", "abnormal": "#e45756"}
    bins = np.linspace(0, 100, 26)
    stacked_vals = [df.loc[df["device_status"] == s, "reliability_score"].to_numpy(dtype=float) for s in status_order]
    plt.hist(
        stacked_vals,
        bins=bins,
        stacked=True,
        color=[color_map[s] for s in status_order],
        label=[f"{s} ({len(v)})" for s, v in zip(status_order, stacked_vals)],
        alpha=0.85,
        edgecolor="white",
    )
    plt.axvline(p10, color="red", linestyle="--", linewidth=1.8, label=f"P10={p10:.2f}")
    plt.title("Reliability Score Distribution")
    plt.xlabel("reliability_score")
    plt.ylabel("device count")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_abnormal_reason_bar(df: pd.DataFrame, out_file: Path) -> None:
    reason_cols = [
        "is_noise",
        "rule_peer_residual_p95",
        "rule_isolated_l3plus_rate",
        "rule_weekly_mean_slope",
        "rule_rolling_max_slope",
        "rule_vh_level_gap_rate",
        "rule_max_acc_std",
    ]
    present = [c for c in reason_cols if c in df.columns]
    if not present:
        return

    abnormal_df = df[df["device_status"] == "abnormal"].copy()
    if len(abnormal_df) == 0:
        return

    counts = abnormal_df[present].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5.5))
    plt.bar(counts.index, counts.values, color="#54a24b", alpha=0.85)
    plt.title("Abnormal Device Trigger Counts")
    plt.ylabel("count")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_feature_heatmap(df: pd.DataFrame, feat_cols: list[str], out_file: Path) -> None:
    # 仅展示风险最高(分数最低)的前30台设备，便于识别模式
    sub = df.sort_values("reliability_score", ascending=True).head(30).copy()
    x = sub[feat_cols].to_numpy(dtype=float)
    x = RobustScaler().fit_transform(x)
    x = np.clip(x, -3, 3)

    plt.figure(figsize=(12, 8))
    im = plt.imshow(x, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    plt.colorbar(im, shrink=0.8, label="robust z")
    plt.yticks(range(len(sub)), sub["设备ID"].tolist(), fontsize=8)
    plt.xticks(range(len(feat_cols)), feat_cols, rotation=35, ha="right", fontsize=9)
    plt.title("Top-30 Risk Devices Feature Heatmap")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = ensure_status(load_scores(INPUT_FILE))

    feat_cols = [
        "peer_signed_residual_mean",
        "l3plus_rate",
        "isolated_l3plus_rate",
        "conflict_rate",
        "peer_residual_mean",
        "peer_residual_p95",
        "peer_level_disagree_rate",
        "weekly_mean_slope",
        "piecewise_mean_gap",
        "rolling_max_slope",
        "vh_ratio_mean",
        "vh_ratio_std",
        "vh_level_gap_rate",
        "max_acc_p95",
        "max_acc_std",
    ]
    emb = build_embedding(df, feat_cols)

    pca_cluster_path = OUTPUT_DIR / "q1_cluster_pca_scatter.png"
    pca_status_path = OUTPUT_DIR / "q1_status_pca_scatter.png"
    hist_path = OUTPUT_DIR / "q1_reliability_hist.png"
    reason_bar_path = OUTPUT_DIR / "q1_abnormal_reason_bar.png"
    heatmap_path = OUTPUT_DIR / "q1_top30_feature_heatmap.png"

    plot_pca_by_cluster(df, emb, pca_cluster_path)
    plot_pca_by_status(df, emb, pca_status_path)
    plot_reliability_hist(df, hist_path)
    plot_abnormal_reason_bar(df, reason_bar_path)
    plot_feature_heatmap(df, feat_cols, heatmap_path)

    print("Q1可视化已生成:")
    print(f"- {pca_cluster_path}")
    print(f"- {pca_status_path}")
    print(f"- {hist_path}")
    print(f"- {reason_bar_path}")
    print(f"- {heatmap_path}")


if __name__ == "__main__":
    main()
