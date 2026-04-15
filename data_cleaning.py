import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """加载CSV数据"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"成功加载数据，共 {len(df)} 行，{len(df.columns)} 列")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def inspect_data(df):
    """检查数据基本信息"""
    print("\n=== 数据基本信息 ===")
    print(df.info())
    
    print("\n=== 数据前5行 ===")
    print(df.head())
    
    print("\n=== 统计描述 ===")
    print(df.describe())
    
    print("\n=== 缺失值统计 ===")
    print(df.isnull().sum())

def clean_data(df):
    """数据清洗主函数"""
    df_clean = df.copy()
    
    print("\n=== 开始数据清洗 ===")
    
    # 1. 去除首尾空格（针对字符串字段）
    string_cols = df_clean.select_dtypes(include=['object']).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].str.strip()
    print("✓ 去除字符串首尾空格")
    
    # 2. 检查并处理缺失值
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"发现 {missing_count} 个缺失值，进行删除处理")
        df_clean = df_clean.dropna()
    else:
        print("✓ 无缺失值")
    
    # 3. 里程保留三位小数并补齐
    df_clean['里程'] = df_clean['里程'].apply(lambda x: f"{x:.3f}")
    print("✓ 里程保留三位小数并补齐")
    
    # 4. 加速度取绝对值
    df_clean['垂加'] = df_clean['垂加'].abs()
    df_clean['水加'] = df_clean['水加'].abs()
    print("✓ 加速度取绝对值")
    
    # 5. 车速低于40的删除，不设上限
    speed_mask = df_clean['车速'] >= 40
    invalid_speed = len(df_clean) - speed_mask.sum()
    if invalid_speed > 0:
        print(f"发现 {invalid_speed} 条车速低于40的数据，已删除")
        df_clean = df_clean[speed_mask]
    
    print(f"\n清洗完成！剩余数据：{len(df_clean)} 行")
    
    return df_clean

def save_cleaned_data(df, output_path):
    """保存清洗后的数据"""
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ 清洗后的数据已保存至: {output_path}")
    except Exception as e:
        print(f"保存失败: {e}")

def generate_report(df, original_count, filename, report_folder='./reports'):
    """生成清洗报告并保存到指定文件夹"""
    # 确保报告文件夹存在
    os.makedirs(report_folder, exist_ok=True)
    
    cleaned_count = len(df)
    removed_count = original_count - cleaned_count
    removal_rate = (removed_count / original_count) * 100
    
    report = f"""
{'='*50}
         数据清洗报告
{'='*50}
文件名: {filename}
原始数据行数: {original_count}
清洗后行数:   {cleaned_count}
删除行数:     {removed_count}
删除比例:     {removal_rate:.2f}%

数据字段:
{', '.join(df.columns.tolist())}

数据时间范围:
最早日期: {df['日期'].min()}
最晚日期: {df['日期'].max()}

里程范围:
最小值: {df['里程'].min()}
最大值: {df['里程'].max()}

车速范围:
最小值: {df['车速'].min()}
最大值: {df['车速'].max()}
平均值: {df['车速'].mean():.1f}
{'='*50}
"""
    print(report)
    
    # 生成报告文件名（与数据文件同名，后缀为_report.txt）
    report_filename = os.path.splitext(filename)[0] + '_清洗报告.txt'
    report_path = os.path.join(report_folder, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ 清洗报告已保存至: {report_path}")

if __name__ == '__main__':
    # 输入和输出文件夹
    input_folder = './csv_raw'
    output_folder = './csv_cleaned'
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹 {input_folder} 不存在！")
        exit(1)
    
    # 遍历输入文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"警告：输入文件夹 {input_folder} 中没有CSV文件")
        exit(0)
    
    print(f"发现 {len(csv_files)} 个CSV文件待处理")
    
    for filename in csv_files:
        print(f"\n{'='*50}")
        print(f"正在处理: {filename}")
        print('='*50)
        
        # 构建输入输出路径
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '_清洗后.csv')
        
        # 加载数据
        df = load_data(input_file)
        if df is None:
            continue
        
        original_count = len(df)
        
        # 检查数据
        inspect_data(df)
        
        # 清洗数据
        df_clean = clean_data(df)
        
        # 保存清洗后的数据
        save_cleaned_data(df_clean, output_file)
        
        # 生成报告（传入文件名）
        generate_report(df_clean, original_count, filename)
    
    print("\n数据清洗流程完成！")