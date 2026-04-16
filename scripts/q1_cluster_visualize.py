from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "outputs" / "q1" / "q1_device_scores_cluster.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "q1"


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


def plot_pca_clusters(df: pd.DataFrame, emb: np.ndarray, out_file: Path) -> None:
    plt.figure(figsize=(10, 7))
    labels = df["cluster_label"].to_numpy()
    abnormal = df["is_abnormal_device"].to_numpy() == 1

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

    # 异常设备描边突出
    plt.scatter(
        emb[abnormal, 0],
        emb[abnormal, 1],
        s=120,
        facecolors="none",
        edgecolors="red",
        linewidths=1.6,
        label="abnormal ring",
    )

    plt.title("Q1 Device Clustering (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_reliability_hist(df: pd.DataFrame, out_file: Path) -> None:
    scores = df["reliability_score"].to_numpy(dtype=float)
    p10 = float(np.quantile(scores, 0.10))

    plt.figure(figsize=(9, 5.5))
    plt.hist(scores, bins=25, alpha=0.8, color="#4c78a8", edgecolor="white")
    plt.axvline(p10, color="red", linestyle="--", linewidth=1.8, label=f"P10={p10:.2f}")
    plt.title("Reliability Score Distribution")
    plt.xlabel("reliability_score")
    plt.ylabel("device count")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
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
    df = load_scores(INPUT_FILE)

    feat_cols = [
        "l3plus_rate",
        "isolated_l3plus_rate",
        "conflict_rate",
        "peer_residual_mean",
        "peer_residual_p95",
        "peer_level_disagree_rate",
        "weekly_mean_slope",
        "max_acc_p95",
        "max_acc_std",
    ]
    emb = build_embedding(df, feat_cols)

    pca_path = OUTPUT_DIR / "q1_cluster_pca_scatter.png"
    hist_path = OUTPUT_DIR / "q1_reliability_hist.png"
    heatmap_path = OUTPUT_DIR / "q1_top30_feature_heatmap.png"

    plot_pca_clusters(df, emb, pca_path)
    plot_reliability_hist(df, hist_path)
    plot_feature_heatmap(df, feat_cols, heatmap_path)

    print("Q1可视化已生成:")
    print(f"- {pca_path}")
    print(f"- {hist_path}")
    print(f"- {heatmap_path}")


if __name__ == "__main__":
    main()
