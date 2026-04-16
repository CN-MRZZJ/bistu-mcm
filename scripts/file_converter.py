import os
import pandas as pd

def xlsx_to_csv(input_folder, output_folder):
    """
    自动将input文件夹下所有xlsx文件转换为csv文件并保存到output文件夹
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为xlsx格式
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            # 构建完整的文件路径
            xlsx_path = os.path.join(input_folder, filename)
            # 生成对应的csv文件名
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            csv_path = os.path.join(output_folder, csv_filename)
            
            try:
                # 读取xlsx文件
                df = pd.read_excel(xlsx_path)
                # 保存为csv文件，使用utf-8-sig编码以支持中文
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"成功转换: {filename} -> {csv_filename}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 指定输入和输出文件夹路径
    input_folder = "./xlsx"   # 输入文件夹
    output_folder = "./csv_raw" # 输出文件夹
    xlsx_to_csv(input_folder, output_folder)
