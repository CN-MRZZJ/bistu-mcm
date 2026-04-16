# 铁道线路动检数据分析项目

## 项目概览
本项目围绕题目三问组织为一套可复现的数据分析工作区：

- `Q1`：评估动检仪设备可靠性，筛选异常设备。
- `Q2`：构建实时判定流程，修正报警等级并识别高值事件类型。
- `Q3`：按公里区间评价线路质量，给出最优和最差区间。

当前工作区已经包含原始数据、清洗脚本、三问建模脚本、结果输出和论文文档主稿。

## 目录结构
- `csv_raw/`：原始 CSV 数据。
- `csv_cleaned/`：清洗后的 CSV 数据。
- `xlsx/`：原始 Excel 数据。
- `scripts/tools/`：工具性脚本，负责格式转换、数据清洗、结果清理，作为长期维护入口。
- `scripts/solutions/`：Q1/Q2/Q3 题解脚本，负责建模、判定和可视化。
- `outputs/`：三问结果输出目录，按 `q1/`、`q2/`、`q3/` 分组。
- `docs/`：当前主文档入口，仅保留题干、三问 writeup 和总论文稿。
- `reports/`：清洗报告等数据处理产物。
- `archive/`：归档的历史草稿、中间稿和运行缓存。
- `.vscode/`：VSCode 运行、任务和测试配置。

## 快速开始
常用命令如下：

```bash
python scripts/tools/data_cleaning.py
python scripts/solutions/q1_cluster_model.py
python scripts/solutions/q1_cluster_visualize.py
python scripts/solutions/q2_threshold_by_vehicle_speed.py
python scripts/solutions/q2_realtime_model.py
python scripts/solutions/q3_line_quality.py
```

常用清理命令：

```bash
python scripts/tools/clean_results.py --dry-run --target q1
python scripts/tools/clean_results.py --dry-run --target q2
python scripts/tools/clean_results.py
```

## 输出说明
`outputs/q1/`
- `q1_device_features.csv`：设备级特征表。
- `q1_device_scores_cluster.csv`：设备可靠性评分与状态。
- `q1_abnormal_devices_cluster.csv`：异常设备清单。
- `q1_low_evidence_devices.csv`：证据不足设备清单。
- `q1_cluster_summary.txt`：Q1 摘要。

`outputs/q2/`
- `q2_vehicle_speed_threshold_reference.csv`：车型-车速门限表。
- `q2_high_event_judgement.csv`：高值事件判定结果。
- `q2_l3_after_20240301_judgement.csv`：2024-03-01 后三级报警判定结果。
- `q2_summary.txt`：Q2 摘要。

`outputs/q3/`
- `q3_km_quality_scores.csv`：每公里质量评分表。
- `q3_best_10_sections.csv`：最优 10 区间。
- `q3_worst_10_sections.csv`：最差 10 区间。
- `q3_summary.txt`：Q3 摘要。

## 文档入口
- [题干](</c:/Users/zhuzh/Desktop/B/附件数据/docs/题干.md>)
- [Q1 Writeup](</c:/Users/zhuzh/Desktop/B/附件数据/docs/Q1_writeup.md>)
- [Q2 Writeup](</c:/Users/zhuzh/Desktop/B/附件数据/docs/Q2_writeup.md>)
- [Q3 Writeup](</c:/Users/zhuzh/Desktop/B/附件数据/docs/Q3_writeup.md>)
- [MCM 论文总稿](</c:/Users/zhuzh/Desktop/B/附件数据/docs/MCM论文总稿.md>)

归档草稿和中间稿位于 `archive/docs/`。当前仓库内未保留可直接移动的原始 `题干.txt`，因此以 `docs/题干.md` 作为唯一题面入口。

## 清理与 VSCode
- `scripts/tools/clean_results.py`：仅清理 `outputs/` 下的结果文件，不会触碰 `docs/` 和 `archive/`。
- `.vscode/launch.json`：一键运行 Q1/Q2 脚本和清理命令。
- `.vscode/tasks.json`：任务面板快捷入口。
