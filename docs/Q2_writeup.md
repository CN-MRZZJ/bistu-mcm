# 第二问 Writeup：服务器实时处理算法与三级报警判定

## 1. 题目要求拆解
第二问包含三个交付目标：

1. 实时判断最新检测数据的可靠性，并判断报警等级是否正确。  
2. 对三级（含）以上大值，判定其属于“线路变差导致的正常检测”还是“设备故障/干扰导致的异常数据”。  
3. 给出 `2024-03-01` 之后三级报警数据的判定结果。  

## 2. 实时处理算法设计

### 2.1 实时处理流程
输入每 3 分钟上传的一包数据，按以下顺序处理：

1. 数据校验：时间、里程、设备ID、加速度字段完整性检查。  
2. 等级复核：根据题干参考阈值重算 `corrected_level`，并与原 `数据等级` 比较得到 `level_is_correct`。  
3. 事件判定（仅对 `corrected_level>=3`）：基于 3 天窗口、±50 米邻域、多设备复现和趋势特征进行判定。  
4. 设备先验融合：引入 Q1 输出的 `q1_reliability_score` 与 `q1_device_status`。  
5. 输出结构化结果：事件类型、置信度、触发原因、等级正确性。  

### 2.2 报警等级复核规则
使用题干参考门限：

- 垂向（垂加）阈值：`[0.10, 0.15, 0.20, 0.25]`
- 横向（水加）阈值：`[0.06, 0.10, 0.15, 0.20]`

分别计算 `v_level_ref` 与 `h_level_ref`，再取：

`corrected_level = max(v_level_ref, h_level_ref)`

并定义：

`level_is_correct = 1(数据等级 == corrected_level)`

### 2.3 三级及以上事件判定规则
以 `线号+行别+里程±0.05km` 为邻域、`3天`为时窗，构造：

- `prior_high_count_3d`：窗口内历史高值次数（`corrected_level>=3`）  
- `prior_device_count_3d`：窗口内历史高值涉及设备数  
- `trend_slope_3d`：窗口内 `max_acc` 随时间斜率  

判定逻辑：

1. `line_issue`（线路问题）  
- 多设备复现：`prior_device_count_3d>=2 且 prior_high_count_3d>=3`，或  
- 复现 + 上升趋势：`prior_device_count_3d>=1 且 prior_high_count_3d>=2`，并伴随趋势上升。  

2. `device_or_interference`（设备/干扰）  
- 孤立高值：无复现且无上升趋势；  
- Q1先验不可信（`abnormal`/`low_evidence`）且无多设备复现；  
- 仅同一设备重复而无跨设备复现。  

3. `uncertain`  
- 线路与设备信号冲突或证据不足。  

补充规则：若原始 `数据等级>=3` 但 `corrected_level<3`，直接判为 `device_or_interference`，原因为 `raw_high_but_corrected_below3`。

## 3. 算法准确性检验
无人工标注场景下，采用弱监督验证：

- 弱标签定义：同一事件在 `±3天`、`±50米` 范围内是否被其他设备复现。  
- 评估对象：`corrected_level>=3` 事件。  
- 指标：Precision、Recall、F1（针对 `line_issue`）。  

本次结果（见 `outputs/q2/q2_summary.txt`）：

- `precision = 1.0000`
- `recall = 0.3158`
- `f1 = 0.4800`

解释：当前策略偏保守，高精度、低召回，适合“先减少误报、再迭代召回”的上线策略。

## 4. 结果汇总

### 4.1 全量统计
- 总记录数：`369,328`
- `corrected_level>=3` 事件数：`447`
- 等级一致率（`level_is_correct`）：`0.1745`
- 高值事件类型分布：  
  - `device_or_interference`: `432`  
  - `uncertain`: `9`  
  - `line_issue`: `6`

### 4.2 2024-03-01 之后三级报警判定
按题目要求筛选 `日期>=2024-03-01 且 数据等级=3`，共 `82` 条，判定结果已输出：

- 文件：`outputs/q2/q2_l3_after_20240301_judgement.csv`
- 本次判定结果：`82` 条均为 `device_or_interference`

主要原因分布：

- `raw_high_but_corrected_below3`: `59`
- `isolated_high_value,device_low_trust_prior`: `19`
- `same_device_only_repeat`: `3`
- `isolated_high_value`: `1`

## 5. 交付文件
- `scripts/q2_realtime_model.py`：Q2 实时算法脚本  
- `outputs/q2/q2_high_event_judgement.csv`：所有高值事件判定  
- `outputs/q2/q2_l3_after_20240301_judgement.csv`：题目要求的三级结果  
- `outputs/q2/q2_summary.txt`：摘要与评估指标  

## 6. 可推广的工程化建议
1. 线上采用“规则优先 + 模型补充”双通路，保证低延迟与可解释性。  
2. 保留 `uncertain` 人工复核通道，复核结果回流持续更新阈值。  
3. 按线路/车型分层校准阈值，逐步提升 `line_issue` 召回率。  
