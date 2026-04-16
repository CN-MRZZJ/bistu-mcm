# 工具脚本说明

`tools/` 用来放可复用、可长期维护的基础脚本，不直接承载某一问的建模逻辑。

当前约定如下：

- `file_converter.py`：将 `xlsx/` 中的原始表格批量转为 `csv_raw/`。
- `data_cleaning.py`：执行通用数据清洗，输出到 `csv_cleaned/`，并生成 `reports/`。
- `clean_results.py`：按目标清理 `outputs/` 下的结果文件，不触碰 `docs/`、`archive/`、原始数据。

维护原则：

- 工具脚本优先保持输入输出路径稳定。
- 工具脚本应尽量使用项目根目录绝对定位，不依赖当前终端所在目录。
- 题解相关逻辑不要回流到 `tools/`，统一放在 `scripts/solutions/`。
