from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "csv_cleaned"
Q1_SCORE_FILE = BASE_DIR / "outputs" / "q1" / "q1_device_scores_cluster.csv"
THRESHOLD_FILE = BASE_DIR / "outputs" / "q2" / "q2_vehicle_speed_threshold_reference.csv"
OUT_DIR = BASE_DIR / "outputs" / "q2"


def load_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_清洗后.csv"))
    if not files:
        raise FileNotFoundError(f"未找到清洗后的CSV文件: {INPUT_DIR}")
    frames = [pd.read_csv(path, parse_dates=["日期"]) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("日期").reset_index(drop=True)
    return df


def load_threshold_table() -> pd.DataFrame:
    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(f"缺少门限表文件: {THRESHOLD_FILE}")
    th = pd.read_csv(THRESHOLD_FILE)
    return th


def apply_vehicle_speed_threshold(df: pd.DataFrame, th: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    speed_bins = [40, 60, 80, 100, 120, 140, 200]
    speed_labels = ["40-60", "60-80", "80-100", "100-120", "120-140", "140-200"]
    out["speed_bin"] = pd.cut(out["车速"], bins=speed_bins, labels=speed_labels, right=False, include_lowest=True)
    out["speed_bin"] = out["speed_bin"].astype(str)
    th["speed_bin"] = th["speed_bin"].astype(str)

    merged = out.merge(th, on=["车型", "speed_bin"], how="left")

    # Fallback to global references if any threshold is missing.
    fallback = {
        "垂加_t1": 0.10,
        "垂加_t2": 0.15,
        "垂加_t3": 0.20,
        "垂加_t4": 0.25,
        "水加_t1": 0.06,
        "水加_t2": 0.10,
        "水加_t3": 0.15,
        "水加_t4": 0.20,
    }
    for col, val in fallback.items():
        merged[col] = merged[col].fillna(val)
    return merged


def add_q1_device_prior(df: pd.DataFrame) -> pd.DataFrame:
    if not Q1_SCORE_FILE.exists():
        df["q1_reliability_score"] = np.nan
        df["q1_device_status"] = "未知"
        return df
    q1 = pd.read_csv(Q1_SCORE_FILE)[["设备ID", "reliability_score", "device_status"]]
    q1 = q1.rename(columns={"reliability_score": "q1_reliability_score", "device_status": "q1_device_status"})
    merged = df.merge(q1, on="设备ID", how="left")
    merged["q1_device_status"] = merged["q1_device_status"].fillna("未知")
    return merged


def infer_event_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_type"] = "不适用"
    df["event_confidence"] = 0.0
    df["event_reason"] = ""
    df["prior_high_count_3d"] = 0
    df["prior_device_count_3d"] = 0
    df["trend_slope_3d"] = 0.0

    candidates = df[df["corrected_level"] >= 3].copy()
    high_pool = candidates[["日期", "线号", "行别", "里程", "设备ID", "max_acc"]].copy()

    for idx, row in candidates.iterrows():
        t0 = row["日期"]
        t1 = t0 - pd.Timedelta(days=3)
        line = row["线号"]
        direction = row["行别"]
        mile = row["里程"]
        current_device = row["设备ID"]

        mask = (
            (high_pool["线号"] == line)
            & (high_pool["行别"] == direction)
            & (high_pool["日期"] < t0)
            & (high_pool["日期"] >= t1)
            & ((high_pool["里程"] - mile).abs() <= 0.05)
        )
        prior = high_pool[mask]
        prior_high_count = int(len(prior))
        prior_device_count = int(prior["设备ID"].nunique())

        trend_points = prior[["日期", "max_acc"]].copy()
        trend_points = pd.concat([trend_points, pd.DataFrame({"日期": [t0], "max_acc": [row["max_acc"]]})], ignore_index=True)
        trend_points = trend_points.sort_values("日期")
        if len(trend_points) >= 2:
            x = (trend_points["日期"] - trend_points["日期"].min()).dt.total_seconds().to_numpy()
            y = trend_points["max_acc"].to_numpy()
            slope = 0.0 if np.var(x) == 0 else float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
        else:
            slope = 0.0

        is_line_issue = False
        is_device_issue = False
        reason = []

        if prior_device_count >= 2 and prior_high_count >= 3:
            is_line_issue = True
            reason.append("3天多设备复现")
        if prior_device_count >= 1 and prior_high_count >= 2:
            is_line_issue = True
            reason.append("3天窗口复现")
        if prior_high_count >= 2 and slope > 1e-6:
            is_line_issue = True
            reason.append("3天趋势上升")
        if prior_high_count == 0 and slope <= 1e-6:
            is_device_issue = True
            reason.append("孤立高值")
        if row["q1_device_status"] in ("abnormal", "low_evidence", "异常", "证据不足") and prior_device_count == 0:
            is_device_issue = True
            reason.append("设备先验可信度低")
        if row["设备ID"] == current_device and prior_high_count > 0 and prior_device_count <= 1:
            is_device_issue = True
            reason.append("仅同设备重复")

        if is_line_issue and not is_device_issue:
            event_type = "线路问题"
            confidence = 0.85 if prior_device_count >= 2 else 0.70
        elif is_device_issue and not is_line_issue:
            event_type = "设备或干扰"
            confidence = 0.85 if "设备先验可信度低" in reason else 0.70
        elif is_line_issue and is_device_issue:
            event_type = "不确定"
            confidence = 0.55
            reason.append("线路与设备信号冲突")
        else:
            event_type = "不确定"
            confidence = 0.50
            reason.append("证据不足")

        df.at[idx, "event_type"] = event_type
        df.at[idx, "event_confidence"] = round(confidence, 3)
        df.at[idx, "event_reason"] = ",".join(reason)
        df.at[idx, "prior_high_count_3d"] = prior_high_count
        df.at[idx, "prior_device_count_3d"] = prior_device_count
        df.at[idx, "trend_slope_3d"] = slope
    return df


def weak_validation(df: pd.DataFrame) -> dict:
    eval_df = df[df["corrected_level"] >= 3].copy()
    if len(eval_df) == 0:
        return {"n_eval": 0, "precision": np.nan, "recall": np.nan, "f1": np.nan}

    high = eval_df[["日期", "线号", "行别", "里程", "设备ID"]].copy()
    weak_label = []
    for _, row in eval_df.iterrows():
        t0 = row["日期"]
        t1 = t0 - pd.Timedelta(days=3)
        t2 = t0 + pd.Timedelta(days=3)
        mask = (
            (high["线号"] == row["线号"])
            & (high["行别"] == row["行别"])
            & (high["日期"] >= t1)
            & (high["日期"] <= t2)
            & ((high["里程"] - row["里程"]).abs() <= 0.05)
            & (high["设备ID"] != row["设备ID"])
        )
        weak_label.append(1 if high[mask]["设备ID"].nunique() >= 1 else 0)
    eval_df["weak_line_label"] = weak_label
    eval_df["pred_line"] = (eval_df["event_type"] == "线路问题").astype(int)

    tp = int(((eval_df["pred_line"] == 1) & (eval_df["weak_line_label"] == 1)).sum())
    fp = int(((eval_df["pred_line"] == 1) & (eval_df["weak_line_label"] == 0)).sum())
    fn = int(((eval_df["pred_line"] == 0) & (eval_df["weak_line_label"] == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"n_eval": int(len(eval_df)), "precision": precision, "recall": recall, "f1": f1}


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    threshold_table = load_threshold_table()
    df = apply_vehicle_speed_threshold(df, threshold_table)
    df = add_q1_device_prior(df)

    # Vehicle-speed adaptive thresholds
    df["v_level_ref"] = (
        (df["垂加_abs"] >= df["垂加_t1"]).astype(int)
        + (df["垂加_abs"] >= df["垂加_t2"]).astype(int)
        + (df["垂加_abs"] >= df["垂加_t3"]).astype(int)
        + (df["垂加_abs"] >= df["垂加_t4"]).astype(int)
    )
    df["h_level_ref"] = (
        (df["水加_abs"] >= df["水加_t1"]).astype(int)
        + (df["水加_abs"] >= df["水加_t2"]).astype(int)
        + (df["水加_abs"] >= df["水加_t3"]).astype(int)
        + (df["水加_abs"] >= df["水加_t4"]).astype(int)
    )
    df["corrected_level"] = df[["v_level_ref", "h_level_ref"]].max(axis=1)
    df["level_is_correct"] = (df["数据等级"] == df["corrected_level"]).astype(int)

    df = infer_event_type(df)

    # Raw high alarms but corrected level < 3 are treated as likely overcall/interference.
    mismatch_mask = (df["数据等级"] >= 3) & (df["corrected_level"] < 3)
    df.loc[mismatch_mask, "event_type"] = "设备或干扰"
    df.loc[mismatch_mask, "event_confidence"] = 0.90
    df.loc[mismatch_mask, "event_reason"] = "原始高等级但修正后低于3级"

    metrics = weak_validation(df)

    all_events = df[df["corrected_level"] >= 3].copy()
    l3_after = df[(df["日期"] >= pd.Timestamp("2024-03-01")) & (df["数据等级"] == 3)].copy()

    cols = [
        "日期",
        "线号",
        "行别",
        "里程",
        "设备ID",
        "车速",
        "数据等级",
        "corrected_level",
        "level_is_correct",
        "event_type",
        "event_confidence",
        "event_reason",
        "prior_high_count_3d",
        "prior_device_count_3d",
        "trend_slope_3d",
        "q1_reliability_score",
        "q1_device_status",
    ]
    all_events[cols].sort_values("日期").to_csv(OUT_DIR / "q2_high_event_judgement.csv", index=False, encoding="utf-8-sig")
    l3_after[cols].sort_values("日期").to_csv(OUT_DIR / "q2_l3_after_20240301_judgement.csv", index=False, encoding="utf-8-sig")

    summary = []
    summary.append("Q2 实时模型摘要")
    summary.append(f"总记录数: {len(df)}")
    summary.append(f"修正等级>=3事件数: {len(all_events)}")
    summary.append(f"2024-03-01之后三级报警数: {len(l3_after)}")
    summary.append(f"等级一致率: {df['level_is_correct'].mean():.4f}")
    summary.append("高值事件类型分布(修正等级>=3):")
    for key, val in all_events["event_type"].value_counts().to_dict().items():
        summary.append(f"  {key}: {val}")
    summary.append("弱监督检验(线路问题 vs ±3天多设备复现):")
    summary.append(f"  样本数: {metrics['n_eval']}")
    summary.append(f"  精确率: {metrics['precision']:.4f}")
    summary.append(f"  召回率: {metrics['recall']:.4f}")
    summary.append(f"  F1: {metrics['f1']:.4f}")
    (OUT_DIR / "q2_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print(
        f"\n输出文件:\n- {OUT_DIR / 'q2_high_event_judgement.csv'}\n- {OUT_DIR / 'q2_l3_after_20240301_judgement.csv'}\n- {OUT_DIR / 'q2_summary.txt'}"
    )


if __name__ == "__main__":
    run()
