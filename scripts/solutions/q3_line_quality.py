from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "csv_cleaned"
OUT_DIR = BASE_DIR / "outputs" / "q3"


def load_cleaned_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_清洗后.csv"))
    if not files:
        raise FileNotFoundError(f"未找到清洗后数据: {INPUT_DIR}")
    frames = [pd.read_csv(path, parse_dates=["日期"]) for path in files]
    data = pd.concat(frames, ignore_index=True)
    return data


def robust_z(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad <= 1e-9:
        std = series.std(ddof=0)
        if std <= 1e-9:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std
    return (series - median) / mad


def build_km_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["公里区间"] = np.floor(data["里程"]).astype(int)
    data["is_l3_plus"] = (data["数据等级"] >= 3).astype(int)

    grouped = (
        data.groupby(["线号", "行别", "公里区间"], as_index=False)
        .agg(
            样本量=("max_acc", "size"),
            max_acc均值=("max_acc", "mean"),
            max_accP95=("max_acc", lambda s: s.quantile(0.95)),
            max_accP99=("max_acc", lambda s: s.quantile(0.99)),
            三级及以上占比=("is_l3_plus", "mean"),
            车速均值=("车速", "mean"),
            覆盖天数=("日期", lambda s: s.dt.date.nunique()),
        )
        .sort_values(["线号", "行别", "公里区间"])
        .reset_index(drop=True)
    )

    scored = []
    for (_, _), sub in grouped.groupby(["线号", "行别"], sort=False):
        block = sub.copy()
        block["z_max_acc均值"] = robust_z(block["max_acc均值"])
        block["z_max_accP95"] = robust_z(block["max_accP95"])
        block["z_三级及以上占比"] = robust_z(block["三级及以上占比"])

        # 核心质量分：P95主导，辅以均值和高等级占比。
        block["原始风险分"] = (
            0.20 * block["z_max_acc均值"]
            + 0.55 * block["z_max_accP95"]
            + 0.25 * block["z_三级及以上占比"]
        )

        # 样本量不足时向组内中位风险收缩，降低偶然性。
        n_median = max(float(block["样本量"].median()), 1.0)
        weight = np.clip(block["样本量"] / n_median, 0.0, 1.0)
        block["样本权重"] = weight
        block["风险分"] = block["原始风险分"] * block["样本权重"]
        scored.append(block)

    out = pd.concat(scored, ignore_index=True)

    # 转成统一可读分：0最好，100最差（全体分位尺度）
    rank_pct = out["风险分"].rank(method="average", pct=True)
    out["质量风险指数"] = (rank_pct * 100).round(2)
    out["质量等级"] = pd.cut(
        out["质量风险指数"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["A(优)", "B(良)", "C(中)", "D(差)", "E(很差)"],
        include_lowest=True,
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cleaned_data()
    km_table = build_km_quality_table(df)

    best10 = km_table.sort_values("质量风险指数", ascending=True).head(10).copy()
    worst10 = km_table.sort_values("质量风险指数", ascending=False).head(10).copy()

    score_file = OUT_DIR / "q3_km_quality_scores.csv"
    best_file = OUT_DIR / "q3_best_10_sections.csv"
    worst_file = OUT_DIR / "q3_worst_10_sections.csv"
    summary_file = OUT_DIR / "q3_summary.txt"

    km_table.to_csv(score_file, index=False, encoding="utf-8-sig")
    best10.to_csv(best_file, index=False, encoding="utf-8-sig")
    worst10.to_csv(worst_file, index=False, encoding="utf-8-sig")

    summary = []
    summary.append("Q3 线路质量评估摘要")
    summary.append(f"总区间数: {len(km_table)}")
    summary.append(f"线路-行别组合数: {km_table[['线号', '行别']].drop_duplicates().shape[0]}")
    summary.append(f"质量风险指数范围: {km_table['质量风险指数'].min():.2f} ~ {km_table['质量风险指数'].max():.2f}")
    summary.append("")
    summary.append("最优10个区间(风险最低):")
    summary.append(
        best10[
            ["线号", "行别", "公里区间", "质量风险指数", "质量等级", "max_accP95", "三级及以上占比", "样本量"]
        ].to_string(index=False)
    )
    summary.append("")
    summary.append("最差10个区间(风险最高):")
    summary.append(
        worst10[
            ["线号", "行别", "公里区间", "质量风险指数", "质量等级", "max_accP95", "三级及以上占比", "样本量"]
        ].to_string(index=False)
    )
    summary_file.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary[:4]))
    print(f"\n输出文件:\n- {score_file}\n- {best_file}\n- {worst_file}\n- {summary_file}")


if __name__ == "__main__":
    main()
