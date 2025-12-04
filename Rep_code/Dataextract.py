import pandas as pd
import numpy as np
import os

def extract_right_foot_data(csv_path, save_extracted=True, output_folder="subjectRepro1"):
    """
    从原始CSV文件中提取右脚数据用于论文模型训练
    
    参数:
        csv_path: 原始CSV文件路径
        save_extracted: 是否保存提取后的数据
        output_folder: 输出文件夹名称
    
    返回:
        df_extracted: 提取后的DataFrame
    """
    
    print(f"正在提取数据: {os.path.basename(csv_path)}")
    
    # 1. 读取CSV文件
    df = pd.read_csv(csv_path)
    
    print(f"原始数据形状: {df.shape}")
    print(f"总行数: {len(df)}")
    
    # 2. 只选择右脚数据 (ele_36 == 0)
    df_right = df[df['ele_36'] == 0].copy()
    print(f"右脚数据行数 (ele_36==0): {len(df_right)}")
    
    # 检查是否有右脚数据
    if len(df_right) == 0:
        print("警告: 没有找到右脚数据 (ele_36==0)")
        return None
    
    # 3. 创建重命名映射字典
    rename_dict = {
        # 时间戳
        'timestamp': 'timestamp',
        
        # 右脚CapSense (0-11)
        'ele_0': 'C0', 'ele_1': 'C1', 'ele_2': 'C2', 'ele_3': 'C3',
        'ele_4': 'C4', 'ele_5': 'C5', 'ele_6': 'C6', 'ele_7': 'C7',
        'ele_8': 'C8', 'ele_9': 'C9', 'ele_10': 'C10', 'ele_11': 'C11',
        
        # 加速度计 (18-20)
        'ele_18': 'Ax', 'ele_19': 'Ay', 'ele_20': 'Az',
        
        # 陀螺仪 (22-24) - 包括Yaw，但论文不用
        'ele_22': 'Gr',  # Roll
        'ele_23': 'Gp',  # Pitch
        'ele_24': 'Gy',  # Yaw (提取但不使用)
        
        # GRF标签 (已用体重标准化)
        'Fx_norm': 'Fx_norm',
        'Fy_norm': 'Fy_norm',
        'Fz_norm': 'Fz_norm'
    }
    
    # 4. 提取需要的列
    columns_to_extract = list(rename_dict.keys())
    df_extracted = df_right[columns_to_extract].copy()
    
    # 5. 重命名列
    df_extracted = df_extracted.rename(columns=rename_dict)
    
    # 6. 重置索引
    df_extracted = df_extracted.reset_index(drop=True)
    
    # 7. 检查缺失值
    missing_count = df_extracted.isnull().sum().sum()
    if missing_count > 0:
        print(f"警告: 发现 {missing_count} 个缺失值，将进行前向填充")
        df_extracted = df_extracted.fillna(method='ffill').fillna(method='bfill')
    
    # 8. 验证提取结果
    print(f"\n提取后的数据形状: {df_extracted.shape}")
    print(f"提取的列数: {len(df_extracted.columns)}")
    
    # 9. 显示数据范围（用于后续归一化参考）
    print("\n数据范围:")
    print("=" * 50)
    
    # CapSense范围
    capsense_cols = [f'C{i}' for i in range(12)]
    capsense_min = df_extracted[capsense_cols].min().min()
    capsense_max = df_extracted[capsense_cols].max().max()
    print(f"右脚CapSense范围: [{capsense_min:.1f}, {capsense_max:.1f}] (论文范围: 0-800)")
    
    # 加速度计范围
    acc_cols = ['Ax', 'Ay', 'Az']
    acc_min = df_extracted[acc_cols].min().min()
    acc_max = df_extracted[acc_cols].max().max()
    print(f"加速度计范围: [{acc_min:.2f}, {acc_max:.2f}] (论文范围: -1到1)")
    
    # 陀螺仪范围
    gyro_cols = ['Gr', 'Gp', 'Gy']
    gyro_min = df_extracted[gyro_cols].min().min()
    gyro_max = df_extracted[gyro_cols].max().max()
    print(f"陀螺仪范围: [{gyro_min:.1f}, {gyro_max:.1f}] (论文范围: -600到600)")
    
    # GRF范围
    grf_cols = ['Fx_norm', 'Fy_norm', 'Fz_norm']
    grf_min = df_extracted[grf_cols].min().min()
    grf_max = df_extracted[grf_cols].max().max()
    print(f"GRF(已除体重)范围: [{grf_min:.3f}, {grf_max:.3f}]")
    
    # 10. 保存提取后的数据
    if save_extracted:
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 生成输出文件名
        base_name = os.path.basename(csv_path).replace('.csv', '')
        output_filename = f"{base_name}_extracted_right_foot.csv"
        output_path = os.path.join(output_folder, output_filename)
        
        # 保存到CSV
        df_extracted.to_csv(output_path, index=False)
        print(f"\n提取的数据已保存到: {output_path}")
        
        # 也保存一个用于模型训练的版本（只包含17个特征+3个标签）
        model_cols = (
            ['timestamp'] + 
            [f'C{i}' for i in range(12)] +  # 12个CapSense
            ['Ax', 'Ay', 'Az'] +            # 3个加速度计
            ['Gr', 'Gp'] +                  # 2个陀螺仪（Roll和Pitch）
            ['Fx_norm', 'Fy_norm', 'Fz_norm']  # 3个GRF标签
        )
        
        df_model = df_extracted[model_cols].copy()
        model_filename = f"{base_name}_model_ready.csv"
        model_path = os.path.join(output_folder, model_filename)
        df_model.to_csv(model_path, index=False)
        print(f"模型训练数据已保存到: {model_path}")
        
        # 打印模型数据信息
        print(f"\n模型训练数据形状: {df_model.shape}")
        print(f"模型训练数据列: {df_model.columns.tolist()}")
    
    # 11. 返回提取的数据
    return df_extracted

def batch_extract_data(csv_folder_path, output_folder="subjectRepro1"):
    """
    批量提取文件夹中的所有CSV文件
    """
    if not os.path.exists(csv_folder_path):
        print(f"文件夹不存在: {csv_folder_path}")
        return
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    all_extracted = []
    all_model_ready = []
    
    for file_name in os.listdir(csv_folder_path):
        if file_name.endswith('.csv'):
            csv_path = os.path.join(csv_folder_path, file_name)
            print("\n" + "="*60)
            
            try:
                df_extracted = extract_right_foot_data(csv_path, save_extracted=True, output_folder=output_folder)
                
                if df_extracted is not None:
                    # 添加文件名标识列
                    df_extracted['source_file'] = file_name
                    all_extracted.append(df_extracted)
                    
                    # 创建模型训练数据
                    model_cols = (
                        ['timestamp'] + 
                        [f'C{i}' for i in range(12)] +
                        ['Ax', 'Ay', 'Az'] +
                        ['Gr', 'Gp'] +
                        ['Fx_norm', 'Fy_norm', 'Fz_norm'] +
                        ['source_file']
                    )
                    df_model = df_extracted[model_cols].copy()
                    all_model_ready.append(df_model)
                    
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")
    
    # 合并所有提取的数据
    if all_extracted:
        # 合并完整提取的数据
        combined_df = pd.concat(all_extracted, ignore_index=True)
        combined_path = os.path.join(output_folder, "all_extracted_data.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"\n所有提取的数据已合并保存到: {combined_path}")
        print(f"合并后的总样本数: {len(combined_df)}")
        
        # 合并模型训练数据
        if all_model_ready:
            combined_model_df = pd.concat(all_model_ready, ignore_index=True)
            combined_model_path = os.path.join(output_folder, "all_model_ready_data.csv")
            combined_model_df.to_csv(combined_model_path, index=False)
            print(f"所有模型训练数据已合并保存到: {combined_model_path}")
            print(f"模型训练数据总样本数: {len(combined_model_df)}")
            print(f"模型训练数据列数: {len(combined_model_df.columns)}")
            
            # 打印特征和标签信息
            feature_cols = [f'C{i}' for i in range(12)] + ['Ax', 'Ay', 'Az'] + ['Gr', 'Gp']
            label_cols = ['Fx_norm', 'Fy_norm', 'Fz_norm']
            print(f"\n特征列数: {len(feature_cols)}")
            print(f"标签列数: {len(label_cols)}")
        
        return combined_df
    
    return None

def create_model_ready_data(df_extracted):
    """
    从提取的数据创建模型训练所需的X和y
    """
    if df_extracted is None:
        print("没有提取的数据可用")
        return None, None, None, None
    
    # 论文使用的特征：右脚CapSense(12) + 加速度计(3) + 陀螺仪(2) = 17维
    # 注意：论文只用了陀螺仪的Roll和Pitch，不用Yaw
    
    # 特征列
    feature_cols = (
        [f'C{i}' for i in range(12)] +  # 右脚CapSense
        ['Ax', 'Ay', 'Az'] +            # 加速度计
        ['Gr', 'Gp']                    # 陀螺仪 (只用Roll和Pitch)
    )
    
    # 标签列 (GRF)
    label_cols = ['Fx_norm', 'Fy_norm', 'Fz_norm']
    
    # 提取特征和标签
    X = df_extracted[feature_cols].values
    y = df_extracted[label_cols].values
    
    print(f"\n模型训练数据准备完成:")
    print(f"特征X形状: {X.shape} (样本数×{len(feature_cols)})")
    print(f"标签y形状: {y.shape} (样本数×{len(label_cols)})")
    
    # 特征描述
    print(f"\n使用的特征 ({len(feature_cols)}个):")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")
    
    return X, y, feature_cols, label_cols

# ==================== 使用示例 ====================

# 示例1: 提取单个文件
if __name__ == "__main__":
    # 原始数据路径
    csv_file = "Smart_Insole_Database/subject_1/jogging_s1_merged.csv"
    
    # 输出文件夹的完整路径
    output_dir = r"D:\TG0\PublicData_Rep\Smart_Insole_Database\subjectRepro1"
    
    # 输出文件名
    output_filename = "jogging_s1_merged.csv"
    model_filename = "jogging_s1_model_ready.csv"
    
    # 完整的输出路径
    output_path = os.path.join(output_dir, output_filename)
    model_data_path = os.path.join(output_dir, model_filename)
    
    print("="*60)
    print(f"开始提取数据并保存到: {output_dir}")
    print("="*60)
    
    # 提取单个文件（不自动保存，我们手动保存）
    df_extracted = extract_right_foot_data(
        csv_file, 
        save_extracted=False,  # 关键：设为False，我们手动保存
        output_folder=output_dir
    )
    
    if df_extracted is not None:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存完整提取数据到指定路径
        df_extracted.to_csv(output_path, index=False)
        print(f"\n✅ 完整提取数据已保存到: {output_path}")
        
        # 2. 创建模型训练数据
        X, y, feature_cols, label_cols = create_model_ready_data(df_extracted)
        
        # 3. 创建模型训练数据DataFrame并保存
        model_cols = ['timestamp'] + feature_cols + label_cols
        df_model = df_extracted[model_cols].copy()
        df_model.to_csv(model_data_path, index=False)
        print(f"✅ 模型训练数据已保存到: {model_data_path}")
        
        # 4. 显示前几个样本
        print("\n" + "="*60)
        print("数据样本预览:")
        print("="*60)
        print("\n前3个样本的特征值 (前5个特征):")
        print(X[:3, :5])
        
        print("\n前3个样本的标签值:")
        print(y[:3])
        
        # 5. 显示文件保存信息
        print("\n" + "="*60)
        print("文件保存位置:")
        print("="*60)
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 完整提取数据: {output_filename} ({df_extracted.shape[0]}行×{df_extracted.shape[1]}列)")
        print(f"🤖 模型训练数据: {model_filename} ({df_model.shape[0]}行×{df_model.shape[1]}列)")
        
        print(f"\n📊 数据统计:")
        print(f"  总样本数: {df_extracted.shape[0]}")
        print(f"  特征数量: {len(feature_cols)}")
        print(f"  标签数量: {len(label_cols)}")
        print(f"  特征列: {feature_cols}")
        print(f"  标签列: {label_cols}")
    # 示例2: 批量提取整个文件夹
    # print("\n" + "="*60)
    # print("批量提取文件夹中所有CSV文件")
    # print("="*60)
    # csv_folder = "your_dataset_folder"  # 替换为你的文件夹路径
    # batch_extract_data(csv_folder, output_folder="subjectRepro1")