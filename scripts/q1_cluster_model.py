import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "csv_cleaned"
OUTPUT_DIR = BASE_DIR / "outputs" / "q1"


def load_all_cleaned(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*_清洗后.csv"))
    if not files:
        raise FileNotFoundError(f"未找到清洗后文件: {input_dir}")
    frames = []
    for fp in files:
        df = pd.read_csv(fp, parse_dates=["日期"])
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("日期").reset_index(drop=True)
    return data


def build_device_features(df: pd.DataFrame) -> pd.DataFrame:
    # 先按线路+行别+50米网格+日期 形成同点同刻群体基准
    group_cols = ["线号", "行别", "50米网格里程", "日期"]
    peer = df.groupby(group_cols).agg(
        peer_count=("设备ID", "size"),
        peer_med_max_acc=("max_acc", "median"),
        peer_med_level=("数据等级", "median"),
    ).reset_index()
    df2 = df.merge(peer, on=group_cols, how="left")

    # 剔除没有“其他设备可对照”的样本（peer_count<=1）
    df2["has_peer"] = (df2["peer_count"] > 1).astype(int)
    df2["peer_abs_residual"] = (df2["max_acc"] - df2["peer_med_max_acc"]).abs()
    df2["peer_level_disagree"] = (df2["数据等级"] != df2["peer_med_level"].round()).astype(int)

    # 设备级聚合
    g = df2.groupby("设备ID", as_index=False)
    feat = g.agg(
        n=("设备ID", "size"),
        coverage_days=("日期", lambda s: s.dt.date.nunique()),
        dominant_vehicle_type=("车型", lambda s: int(s.mode().iloc[0])),
        dominant_line=("线号", lambda s: int(s.mode().iloc[0])),
        l3plus_rate=("数据等级", lambda s: (s >= 3).mean()),
        conflict_rate=("等级冲突标记", "mean"),
        max_acc_p95=("max_acc", lambda s: s.quantile(0.95)),
        max_acc_std=("max_acc", "std"),
        peer_support_rate=("has_peer", "mean"),
    )

    peer_only = df2[df2["has_peer"] == 1]
    if len(peer_only) > 0:
        gp = peer_only.groupby("设备ID").agg(
            peer_signed_residual_mean=("max_acc", lambda s: 0.0),
            peer_residual_mean=("peer_abs_residual", "mean"),
            peer_residual_p95=("peer_abs_residual", lambda s: s.quantile(0.95)),
            peer_level_disagree_rate=("peer_level_disagree", "mean"),
        ).reset_index()
        # 有符号残差：区分系统偏大/偏小
        signed = (peer_only["max_acc"] - peer_only["peer_med_max_acc"]).rename("peer_signed_residual")
        signed_df = pd.concat([peer_only[["设备ID"]], signed], axis=1).groupby("设备ID", as_index=False).agg(
            peer_signed_residual_mean=("peer_signed_residual", "mean")
        )
        gp = gp.drop(columns=["peer_signed_residual_mean"]).merge(signed_df, on="设备ID", how="left")
        feat = feat.merge(gp, on="设备ID", how="left")
    else:
        feat["peer_signed_residual_mean"] = np.nan
        feat["peer_residual_mean"] = np.nan
        feat["peer_residual_p95"] = np.nan
        feat["peer_level_disagree_rate"] = np.nan

    # 孤立三级率：三级及以上且同组只有1条
    high = df2[df2["数据等级"] >= 3].copy()
    if len(high) > 0:
        high["isolated"] = (high["peer_count"] <= 1).astype(int)
        iso = high.groupby("设备ID").agg(
            l3plus_cnt=("isolated", "size"),
            isolated_l3plus_rate=("isolated", "mean"),
        ).reset_index()
    else:
        iso = pd.DataFrame(columns=["设备ID", "l3plus_cnt", "isolated_l3plus_rate"])
    feat = feat.merge(iso, on="设备ID", how="left")
    feat["l3plus_cnt"] = feat["l3plus_cnt"].fillna(0)
    feat["isolated_l3plus_rate"] = feat["isolated_l3plus_rate"].fillna(0.0)

    # 横纵向失衡特征
    ratio_df = df2.copy()
    eps = 1e-6
    ratio_df["vh_ratio"] = ratio_df["垂加_abs"] / (ratio_df["水加_abs"] + eps)
    ratio_df["vh_level_gap"] = (ratio_df["垂加等级"] - ratio_df["水加等级"]).abs()
    vh = ratio_df.groupby("设备ID").agg(
        vh_ratio_mean=("vh_ratio", "mean"),
        vh_ratio_std=("vh_ratio", "std"),
        vh_level_gap_rate=("vh_level_gap", lambda s: (s >= 2).mean()),
    ).reset_index()
    feat = feat.merge(vh, on="设备ID", how="left")

    # 周均值漂移斜率
    weekly = df2.copy()
    weekly["week"] = weekly["日期"].dt.to_period("W").astype(str)
    wk = weekly.groupby(["设备ID", "week"], as_index=False)["max_acc"].mean()

    slope_map = {}
    piecewise_gap_map = {}
    roll_max_slope_map = {}
    for dev, sub in wk.groupby("设备ID"):
        y = sub["max_acc"].to_numpy()
        x = np.arange(len(y), dtype=float)
        if len(y) < 2:
            slope_map[dev] = 0.0
            piecewise_gap_map[dev] = 0.0
            roll_max_slope_map[dev] = 0.0
        else:
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0:
                slope_map[dev] = 0.0
            else:
                slope_map[dev] = float(((x - x_mean) * (y - y_mean)).sum() / denom)
            # 分段漂移：后半均值 - 前半均值
            mid = len(y) // 2
            if mid == 0:
                piecewise_gap_map[dev] = 0.0
            else:
                piecewise_gap_map[dev] = float(y[mid:].mean() - y[:mid].mean())
            # 滚动窗口最大斜率（窗口=4周）
            window = 4
            if len(y) < window:
                roll_max_slope_map[dev] = 0.0
            else:
                local_slopes = []
                for i in range(len(y) - window + 1):
                    yy = y[i : i + window]
                    xx = np.arange(window, dtype=float)
                    xx_mean = xx.mean()
                    yy_mean = yy.mean()
                    dd = ((xx - xx_mean) ** 2).sum()
                    if dd == 0:
                        local_slopes.append(0.0)
                    else:
                        local_slopes.append(float(((xx - xx_mean) * (yy - yy_mean)).sum() / dd))
                roll_max_slope_map[dev] = float(np.max(np.abs(local_slopes))) if local_slopes else 0.0
    feat["weekly_mean_slope"] = feat["设备ID"].map(slope_map).fillna(0.0).abs()
    feat["piecewise_mean_gap"] = feat["设备ID"].map(piecewise_gap_map).fillna(0.0).abs()
    feat["rolling_max_slope"] = feat["设备ID"].map(roll_max_slope_map).fillna(0.0)

    # 缺失填补（通常是peer不足导致）
    fill_cols = [
        "peer_signed_residual_mean",
        "peer_residual_mean",
        "peer_residual_p95",
        "peer_level_disagree_rate",
        "max_acc_std",
        "vh_ratio_mean",
        "vh_ratio_std",
    ]
    for c in fill_cols:
        feat[c] = feat[c].fillna(feat[c].median())

    return feat


def select_dbscan_params(x: np.ndarray, min_samples: int) -> tuple[float, np.ndarray]:
    # 基于k-distance分位点网格搜索 eps，优先选择噪声率在目标区间内的方案
    nbrs = NearestNeighbors(n_neighbors=min_samples, metric="euclidean")
    nbrs.fit(x)
    distances, _ = nbrs.kneighbors(x)
    kth_dist = distances[:, -1]

    quantiles = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    candidates = []
    for q in quantiles:
        eps = float(np.quantile(kth_dist, q))
        eps = max(eps, 1e-4)
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(x)
        noise_ratio = float((labels == -1).mean())
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        largest_cluster = 0
        if cluster_count > 0:
            largest_cluster = int(pd.Series(labels[labels >= 0]).value_counts().iloc[0])

        # 评分规则：优先有簇、噪声率适中、主簇占比高
        has_cluster = 1 if cluster_count > 0 else 0
        noise_target_gap = abs(noise_ratio - 0.10)
        score = (has_cluster, -noise_target_gap, largest_cluster, -noise_ratio)
        candidates.append((score, eps, labels))

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_eps, best_labels = candidates[0]
    return best_eps, best_labels


def stratified_robust_scale(features: pd.DataFrame, cols: list[str], group_col: str, min_group_size: int = 10) -> np.ndarray:
    x = features[cols].copy()
    scaled = np.zeros_like(x.to_numpy(dtype=float))
    global_scaler = RobustScaler()
    global_scaled = global_scaler.fit_transform(x.to_numpy(dtype=float))

    groups = features[group_col].to_numpy()
    for g in pd.Series(groups).unique():
        idx = np.where(groups == g)[0]
        if len(idx) >= min_group_size:
            scaler = RobustScaler()
            scaled[idx, :] = scaler.fit_transform(x.iloc[idx].to_numpy(dtype=float))
        else:
            scaled[idx, :] = global_scaled[idx, :]
    return np.clip(scaled, -8.0, 8.0)


def score_devices(features: pd.DataFrame) -> pd.DataFrame:
    cols = [
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

    # 按主车型分层标准化，减少车型动力学差异干扰
    x = stratified_robust_scale(features, cols=cols, group_col="dominant_vehicle_type", min_group_size=10)

    min_samples = max(5, int(math.sqrt(len(features))))
    eps, labels = select_dbscan_params(x, min_samples=min_samples)

    out = features.copy()
    out["cluster_label"] = labels

    # 找最大簇作为“正常簇”
    valid_labels = out[out["cluster_label"] >= 0]["cluster_label"]
    if len(valid_labels) == 0:
        center = np.median(x, axis=0)
    else:
        main_cluster = valid_labels.value_counts().idxmax()
        center = x[out["cluster_label"].to_numpy() == main_cluster].mean(axis=0)

    distance = np.sqrt(((x - center) ** 2).sum(axis=1))
    out["distance_to_main_cluster"] = distance

    # 距离映射成可靠性分（高距离=低可靠）
    q95 = np.quantile(distance, 0.95)
    q05 = np.quantile(distance, 0.05)
    denom = max(q95 - q05, 1e-9)
    risk = np.clip((distance - q05) / denom, 0, 1)
    out["reliability_score"] = (100 * (1 - risk)).round(2)

    # 异常判定：DBSCAN噪声 或 可靠性分低于P10
    p10 = out["reliability_score"].quantile(0.10)
    out["is_noise"] = (out["cluster_label"] == -1).astype(int)

    # 双门槛：规则命中数（>=2）也可判异常
    rule_cols = [
        "peer_residual_p95",
        "isolated_l3plus_rate",
        "weekly_mean_slope",
        "rolling_max_slope",
        "vh_level_gap_rate",
        "max_acc_std",
    ]
    thresholds = {c: float(out[c].quantile(0.95)) for c in rule_cols}
    for c in rule_cols:
        out[f"rule_{c}"] = (out[c] >= thresholds[c]).astype(int)
    out["rule_hit_count"] = out[[f"rule_{c}" for c in rule_cols]].sum(axis=1)

    out["is_abnormal_device"] = (
        (out["is_noise"] == 1)
        | (out["reliability_score"] <= p10)
        | (out["rule_hit_count"] >= 2)
    ).astype(int)
    return out, {"eps": eps, "min_samples": min_samples, "p10_score": float(p10), "rule_thresholds": thresholds}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_cleaned(INPUT_DIR)
    features = build_device_features(df)
    scores, params = score_devices(features)

    feature_path = OUTPUT_DIR / "q1_device_features.csv"
    score_path = OUTPUT_DIR / "q1_device_scores_cluster.csv"
    abnormal_path = OUTPUT_DIR / "q1_abnormal_devices_cluster.csv"
    summary_path = OUTPUT_DIR / "q1_cluster_summary.txt"

    features.to_csv(feature_path, index=False, encoding="utf-8-sig")
    scores.sort_values("reliability_score", ascending=True).to_csv(score_path, index=False, encoding="utf-8-sig")
    scores[scores["is_abnormal_device"] == 1].sort_values("reliability_score", ascending=True).to_csv(
        abnormal_path, index=False, encoding="utf-8-sig"
    )

    summary = []
    summary.append("Q1 聚类建模摘要")
    summary.append(f"总设备数: {len(scores)}")
    summary.append(f"异常设备数: {int(scores['is_abnormal_device'].sum())}")
    summary.append(f"噪声设备数(DBSCAN -1): {int(scores['is_noise'].sum())}")
    summary.append(f"DBSCAN eps: {params['eps']:.4f}")
    summary.append(f"DBSCAN min_samples: {params['min_samples']}")
    summary.append(f"可靠性分P10阈值: {params['p10_score']:.2f}")
    summary.append(f"规则命中>=2设备数: {int((scores['rule_hit_count'] >= 2).sum())}")
    summary.append("")
    summary.append("最低分前10设备:")
    top10 = scores.sort_values("reliability_score", ascending=True).head(10)[
        ["设备ID", "reliability_score", "is_noise", "l3plus_rate", "isolated_l3plus_rate", "conflict_rate"]
    ]
    summary.extend(top10.to_string(index=False).splitlines())
    summary_text = "\n".join(summary)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"\n输出文件:\n- {feature_path}\n- {score_path}\n- {abnormal_path}\n- {summary_path}")


if __name__ == "__main__":
    main()
