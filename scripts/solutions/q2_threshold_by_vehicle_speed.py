from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "csv_cleaned"
OUT_DIR = BASE_DIR / "outputs" / "q2"


def estimate_thresholds(block: pd.DataFrame, value_col: str, level_col: str, min_pair_count: int = 30) -> list[float]:
    series_by_level = {level: block.loc[block[level_col] == level, value_col].dropna() for level in range(5)}
    thresholds = []
    for level in range(4):
        low = series_by_level[level]
        high = series_by_level[level + 1]
        if len(low) >= min_pair_count and len(high) >= min_pair_count:
            # Robust boundary between adjacent levels.
            t = float((low.quantile(0.95) + high.quantile(0.05)) / 2.0)
        elif len(low) > 0 and len(high) > 0:
            t = float((low.median() + high.median()) / 2.0)
        else:
            t = np.nan
        thresholds.append(t)

    # Enforce monotonic increase.
    arr = np.array(thresholds, dtype=float)
    for idx in range(1, 4):
        if np.isfinite(arr[idx - 1]) and np.isfinite(arr[idx]) and arr[idx] < arr[idx - 1]:
            arr[idx] = arr[idx - 1]
    return arr.tolist()


def load_cleaned() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_清洗后.csv"))
    if not files:
        raise FileNotFoundError(f"未找到清洗后的数据文件: {INPUT_DIR}")
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cleaned()

    speed_bins = [40, 60, 80, 100, 120, 140, 200]
    speed_labels = ["40-60", "60-80", "80-100", "100-120", "120-140", "140-200"]
    df["speed_bin"] = pd.cut(df["车速"], bins=speed_bins, labels=speed_labels, right=False, include_lowest=True)

    # Fallbacks: global and per-vehicle thresholds.
    global_v = estimate_thresholds(df, "垂加_abs", "垂加等级", min_pair_count=200)
    global_h = estimate_thresholds(df, "水加_abs", "水加等级", min_pair_count=200)

    vehicle_fallback = {}
    for vehicle_type, block in df.groupby("车型"):
        vehicle_fallback[int(vehicle_type)] = {
            "v": estimate_thresholds(block, "垂加_abs", "垂加等级", min_pair_count=80),
            "h": estimate_thresholds(block, "水加_abs", "水加等级", min_pair_count=80),
        }

    rows = []
    for (vehicle_type, speed_bin), block in df.groupby(["车型", "speed_bin"], observed=True):
        vehicle_type = int(vehicle_type)
        n = int(len(block))
        v_t = estimate_thresholds(block, "垂加_abs", "垂加等级", min_pair_count=30)
        h_t = estimate_thresholds(block, "水加_abs", "水加等级", min_pair_count=30)

        # Fill missing using vehicle fallback then global fallback.
        for idx in range(4):
            if not np.isfinite(v_t[idx]):
                vv = vehicle_fallback[vehicle_type]["v"][idx]
                v_t[idx] = vv if np.isfinite(vv) else global_v[idx]
            if not np.isfinite(h_t[idx]):
                hv = vehicle_fallback[vehicle_type]["h"][idx]
                h_t[idx] = hv if np.isfinite(hv) else global_h[idx]

        rows.append(
            {
                "车型": vehicle_type,
                "speed_bin": str(speed_bin),
                "样本量": n,
                "垂加_t1": v_t[0],
                "垂加_t2": v_t[1],
                "垂加_t3": v_t[2],
                "垂加_t4": v_t[3],
                "水加_t1": h_t[0],
                "水加_t2": h_t[1],
                "水加_t3": h_t[2],
                "水加_t4": h_t[3],
            }
        )

    out = pd.DataFrame(rows).sort_values(["车型", "speed_bin"])
    out_path = OUT_DIR / "q2_vehicle_speed_threshold_reference.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary_lines = [
        "Q2 车型-车速门限摘要",
        f"总行数: {len(out)}",
        f"车型数: {out['车型'].nunique()}",
        f"速度分箱: {', '.join(speed_labels)}",
        f"全局垂向门限: {[round(x, 4) for x in global_v]}",
        f"全局横向门限: {[round(x, 4) for x in global_h]}",
        "",
        "样本量最高的前12行:",
        out.sort_values("样本量", ascending=False)
        .head(12)
        .round(4)
        .to_string(index=False),
    ]
    summary_path = OUT_DIR / "q2_vehicle_speed_threshold_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"已生成: {out_path}")
    print(f"已生成: {summary_path}")


if __name__ == "__main__":
    main()
