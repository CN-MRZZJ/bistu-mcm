import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
XLSX_DIR = BASE_DIR / "xlsx"
CSV_RAW_DIR = BASE_DIR / "csv_raw"

def xlsx_to_csv(input_folder, output_folder):
    """
    自动将input文件夹下所有xlsx文件转换为csv文件并保存到output文件夹
    """
    # 确保输出文件夹存在
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for file_path in input_folder.iterdir():
        # 检查文件是否为xlsx格式
        if file_path.is_file() and file_path.suffix.lower() in {'.xlsx', '.xls'}:
            # 构建完整的文件路径
            xlsx_path = file_path
            # 生成对应的csv文件名
            csv_filename = file_path.stem + '.csv'
            csv_path = output_folder / csv_filename
            
            try:
                # 读取xlsx文件
                df = pd.read_excel(xlsx_path)
                # 保存为csv文件，使用utf-8-sig编码以支持中文
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"成功转换: {file_path.name} -> {csv_filename}")
            except Exception as e:
                print(f"转换失败 {file_path.name}: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 指定输入和输出文件夹路径
    input_folder = XLSX_DIR
    output_folder = CSV_RAW_DIR
    xlsx_to_csv(input_folder, output_folder)
