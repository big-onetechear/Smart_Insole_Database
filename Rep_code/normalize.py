"""
normalize_data.py
数据归一化代码 - 根据检查结果实现论文预处理
"""

import pandas as pd
import numpy as np
import json
import os

class DataNormalizer:
    """数据归一化器 - 实现论文3.4节预处理步骤"""
    
    def __init__(self, params_path=None):
        """
        初始化归一化器
        
        参数:
            params_path: 参数文件路径（json格式）
                        如果为None，使用硬编码参数
        """
        self.params = {}
        
        if params_path and os.path.exists(params_path):
            self.load_params(params_path)
        else:
            # 使用你检查结果的参数（硬编码版本）
            self.params = {
                "dataset": "jogging_s1_model_ready",
                "samples": 21280,
                "feature_columns": {
                    "capsense": [f'C{i}' for i in range(12)],
                    "accelerometer": ['Ax', 'Ay', 'Az'],
                    "gyroscope": ['Gr', 'Gp']
                },
                "label_columns": ['Fx_norm', 'Fy_norm', 'Fz_norm'],
                "normalization": {
                    "capsense": {
                        "paper_scale": 800.0,
                        "actual_range": [355.0, 749.0],
                        "method": "x / 800.0"
                    },
                    "accelerometer": {
                        "scale": 16.0,
                        "actual_range": [-16.0, 16.0],
                        "paper_range": [-1, 1],
                        "method": "x / 16.0"
                    },
                    "gyroscope": {
                        "scale": 653.66,
                        "actual_range": [-653.7, 447.4],
                        "paper_range": [-600, 600],
                        "method": "x / 653.66"
                    },
                    "grf": {
                        "Fx_norm": {
                            "min": -0.590,
                            "max": 4.779,
                            "range": [-0.590, 4.779]
                        },
                        "Fy_norm": {
                            "min": -2.177,
                            "max": 3.664,
                            "range": [-2.177, 3.664]
                        },
                        "Fz_norm": {
                            "min": -0.041,
                            "max": 23.456,
                            "range": [-0.041, 23.456]
                        }
                    }
                }
            }
        
        # 提取常用参数
        self.capsense_scale = self.params["normalization"]["capsense"]["paper_scale"]
        self.acc_scale = self.params["normalization"]["accelerometer"]["scale"]
        self.gyro_scale = self.params["normalization"]["gyroscope"]["scale"]
        
        self.grf_params = self.params["normalization"]["grf"]
        
        print(f"📊 归一化参数加载完成:")
        print(f"  • CapSense比例因子: {self.capsense_scale}")
        print(f"  • 加速度计比例因子: {self.acc_scale}")
        print(f"  • 陀螺仪比例因子: {self.gyro_scale:.2f}")
    
    def load_params(self, params_path):
        """从JSON文件加载参数"""
        with open(params_path, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        
        # 更新比例因子
        self.capsense_scale = self.params["normalization"]["capsense"]["paper_scale"]
        self.acc_scale = self.params["normalization"]["accelerometer"]["scale"]
        self.gyro_scale = self.params["normalization"]["gyroscope"]["scale"]
        self.grf_params = self.params["normalization"]["grf"]
        
        print(f"✅ 从 {os.path.basename(params_path)} 加载参数")
    
    def normalize_capsense(self, df):
        """归一化CapSense数据（论文方法）"""
        capsense_cols = self.params["feature_columns"]["capsense"]
        
        # 方法：除以800（论文附录A范围）
        df[capsense_cols] = df[capsense_cols] / self.capsense_scale
        
        print(f"  • CapSense归一化: 除以{self.capsense_scale}")
        print(f"    结果范围: [{df[capsense_cols].min().min():.3f}, {df[capsense_cols].max().max():.3f}]")
        
        return df
    
    def normalize_imu_two_step(self, df, sensor_type='accelerometer'):
        """两步归一化IMU数据（论文方法）"""
        if sensor_type == 'accelerometer':
            cols = self.params["feature_columns"]["accelerometer"]
            scale = self.acc_scale
            sensor_name = "加速度计"
        elif sensor_type == 'gyroscope':
            cols = self.params["feature_columns"]["gyroscope"]
            scale = self.gyro_scale
            sensor_name = "陀螺仪"
        else:
            raise ValueError(f"未知传感器类型: {sensor_type}")
        
        # 第一步：缩放到[-1, 1]范围
        df[cols] = df[cols] / scale
        
        # 第二步：转换到[0, 1]范围
        df[cols] = (df[cols] + 1) / 2
        
        print(f"  • {sensor_name}归一化: 先/{scale:.2f}到[-1,1]，再(+1)/2到[0,1]")
        
        # 检查范围
        for col in cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < -0.1 or col_max > 1.1:
                print(f"    ⚠️  {col}范围: [{col_min:.3f}, {col_max:.3f}] (应在[0,1]内)")
            else:
                print(f"    ✓ {col}范围: [{col_min:.3f}, {col_max:.3f}]")
        
        return df
    
    def normalize_grf(self, df):
        """归一化GRF标签（论文第二步：Min-Max到[0,1]）"""
        for col in self.params["label_columns"]:
            if col in self.grf_params and col in df.columns:
                col_min = self.grf_params[col]["min"]
                col_max = self.grf_params[col]["max"]
                
                # 避免除零
                if col_max - col_min < 1e-10:
                    print(f"⚠️  {col}范围过小，跳过归一化")
                    continue
                
                # Min-Max归一化
                df[col] = (df[col] - col_min) / (col_max - col_min)
                
                # 检查是否在[0,1]范围内
                result_min = df[col].min()
                result_max = df[col].max()
                
                print(f"  • {col}归一化: (x - {col_min:.3f}) / ({col_max:.3f} - {col_min:.3f})")
                print(f"    结果范围: [{result_min:.3f}, {result_max:.3f}]")
        
        return df
    
    def apply_moving_average(self, df, window_size=5):
        """应用移动平均滤波（论文提到）"""
        print(f"\n📈 应用移动平均滤波 (window={window_size})")
        
        # 所有特征列
        feature_cols = (
            self.params["feature_columns"]["capsense"] +
            self.params["feature_columns"]["accelerometer"] +
            self.params["feature_columns"]["gyroscope"]
        )
        
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
        
        print(f"  • 已对{len(feature_cols)}个特征列应用滤波")
        
        return df
    
    def normalize_all(self, df, apply_filter=True, window_size=5):
        """
        完整的归一化流程（论文3.4节）
        
        参数:
            df: 原始DataFrame
            apply_filter: 是否应用移动平均滤波
            window_size: 滤波窗口大小
            
        返回:
            df_normalized: 归一化后的DataFrame
        """
        print("=" * 60)
        print("🚀 开始数据归一化（按论文3.4节方法）")
        print("=" * 60)
        
        # 创建副本
        df_norm = df.copy()
        
        # 1. 移动平均滤波（论文提到）
        if apply_filter:
            df_norm = self.apply_moving_average(df_norm, window_size)
        
        # 2. CapSense归一化
        print(f"\n📊 传感器数据归一化:")
        df_norm = self.normalize_capsense(df_norm)
        
        # 3. 加速度计归一化
        df_norm = self.normalize_imu_two_step(df_norm, 'accelerometer')
        
        # 4. 陀螺仪归一化
        df_norm = self.normalize_imu_two_step(df_norm, 'gyroscope')
        
        # 5. GRF标签归一化（论文第二步）
        print(f"\n🏷️  GRF标签归一化:")
        df_norm = self.normalize_grf(df_norm)
        
        # 6. 验证归一化结果
        print(f"\n✅ 归一化完成！验证结果:")
        self._verify_normalization(df_norm)
        
        return df_norm
    
    def _verify_normalization(self, df):
        """验证归一化结果是否在[0,1]范围内"""
        print("  📋 范围验证:")
        
        # 检查特征列
        all_features = (
            self.params["feature_columns"]["capsense"] +
            self.params["feature_columns"]["accelerometer"] +
            self.params["feature_columns"]["gyroscope"]
        )
        
        feature_issues = []
        for col in all_features:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min < -0.01 or col_max > 1.01:  # 允许微小误差
                    feature_issues.append((col, col_min, col_max))
        
        if feature_issues:
            print(f"  ⚠️  以下特征超出[0,1]范围:")
            for col, cmin, cmax in feature_issues[:5]:  # 只显示前5个
                print(f"    {col}: [{cmin:.3f}, {cmax:.3f}]")
        else:
            print(f"  ✓ 所有特征都在[0,1]范围内")
        
        # 检查标签列
        label_issues = []
        for col in self.params["label_columns"]:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min < -0.01 or col_max > 1.01:
                    label_issues.append((col, col_min, col_max))
        
        if label_issues:
            print(f"  ⚠️  以下标签超出[0,1]范围:")
            for col, cmin, cmax in label_issues:
                print(f"    {col}: [{cmin:.3f}, {cmax:.3f}]")
        else:
            print(f"  ✓ 所有标签都在[0,1]范围内")
        
        # 统计信息
        print(f"\n  📊 统计信息:")
        print(f"    总样本数: {len(df):,}")
        print(f"    特征列数: {len(all_features)}")
        print(f"    标签列数: {len(self.params['label_columns'])}")
    
    def save_normalized_data(self, df, output_path, save_features_only=False):
        """保存归一化后的数据"""
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
           
        os.makedirs(output_dir, exist_ok=True)
        
        # 选择要保存的列
        if save_features_only:
            # 只保存特征和标签（用于模型训练）
            cols_to_save = (
                self.params["feature_columns"]["capsense"] +
                self.params["feature_columns"]["accelerometer"] +
                self.params["feature_columns"]["gyroscope"] +
                self.params["label_columns"]
            )
            df_to_save = df[cols_to_save]
        else:
            # 保存所有列（包括timestamp等）
            df_to_save = df
        
        # 保存为CSV
        df_to_save.to_csv(output_path, index=False)
        
        print(f"💾 归一化数据已保存到: {output_path}")
        print(f"   数据形状: {df_to_save.shape}")
        
        return output_path


# ==================== 使用示例 ====================

# ... 前面的代码保持不变 ...

def main():
    """主函数 - 使用示例"""
    print("=" * 60)
    print("📊 数据归一化工具")
    print("=" * 60)
    
    # ===== 配置部分 =====
    # 1. 输入文件（你的原始数据）
    input_csv = r"D:\TG0\PublicData_Rep\Smart_Insole_Database\subjectRepro1\jogging_s1_model_ready.csv"
    
    # 2. 参数文件（可选，如果不提供则使用硬编码参数）
    params_file = r"D:\TG0\PublicData_Rep\Smart_Insole_Database\subjectRepro1\Param\jogging_s1_preprocess_params\jogging_s1_model_ready_params.json"
    
    # 3. 输出文件路径
    output_csv = r"D:\TG0\PublicData_Rep\Smart_Insole_Database\subjectRepro1\norm\jogging_s1_normalized.csv"
    
    # 4. 是否应用移动平均滤波
    apply_filter = True
    window_size = 5  # 论文提到的滤波窗口
    
    # ===== 执行部分 =====
    print(f"📂 输入文件: {input_csv}")
    print(f"📄 参数文件: {params_file}")
    print(f"📁 输出文件: {output_csv}")
    print(f"🔧 滤波设置: {'是' if apply_filter else '否'} (window={window_size})")
    print("-" * 60)
    
    try:
        # 1. 加载数据
        print(f"📥 加载数据...")
        df_raw = pd.read_csv(input_csv)
        print(f"   原始数据: {df_raw.shape[0]}行 × {df_raw.shape[1]}列")
        
        # 2. 创建归一化器
        if os.path.exists(params_file):
            normalizer = DataNormalizer(params_file)
        else:
            print(f"⚠️  参数文件不存在，使用默认参数")
            normalizer = DataNormalizer()
        
        # 3. 执行归一化
        df_normalized = normalizer.normalize_all(
            df_raw, 
            apply_filter=apply_filter,
            window_size=window_size
        )
        
        # 4. 保存结果
        output_file = normalizer.save_normalized_data(
            df_normalized, 
            output_csv,
            save_features_only=True  # 只保存特征和标签，去掉timestamp等
        )
        
        print("\n" + "=" * 60)
        print("🎉 归一化完成！")
        print("=" * 60)
        print(f"\n📋 生成的文件:")
        print(f"  归一化数据: {output_file}")
        print(f"  数据形状: {df_normalized.shape}")
        print(f"  特征列: {len(normalizer.params['feature_columns']['capsense']) + 3 + 2}个")
        print(f"  标签列: {len(normalizer.params['label_columns'])}个")
        
        # 显示前几个样本
        print(f"\n👀 归一化后数据预览（前3行）:")
        print(df_normalized.head(3))
        
        # ===== 归一化完成后立即检查 =====
        print("\n" + "="*60)
        print("✅ 归一化结果快速检查")
        print("="*60)
        
        # 快速检查关键列
        key_columns = ['C0', 'C5', 'C10', 'Ax', 'Az', 'Gr', 'Fx_norm', 'Fz_norm']
        for col in key_columns:
            if col in df_normalized.columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                # 放宽检查条件，允许微小误差
                if col_min >= -0.05 and col_max <= 1.05:
                    status = "✓"
                else:
                    status = "❌"
                print(f"{status} {col}: [{col_min:.3f}, {col_max:.3f}]")
        
        print(f"\n📊 数据统计:")
        print(f"  样本数: {len(df_normalized):,}")
        print(f"  列数: {len(df_normalized.columns)}")
        print(f"  缺失值: {df_normalized.isnull().sum().sum()}个")
        
        # 额外检查：是否有负值或大于1的值（排除timestamp）
        # 只检查特征列和标签列，不检查timestamp
        check_columns = []
        # 添加特征列
        if 'feature_columns' in normalizer.params:
            check_columns.extend(normalizer.params['feature_columns']['capsense'])
            check_columns.extend(normalizer.params['feature_columns']['accelerometer'])  
            check_columns.extend(normalizer.params['feature_columns']['gyroscope'])
        # 添加标签列
        if 'label_columns' in normalizer.params:
            check_columns.extend(normalizer.params['label_columns'])
        
        # 过滤掉不在df_normalized中的列
        check_columns = [col for col in check_columns if col in df_normalized.columns]
        
        if check_columns:
            negative_count = ((df_normalized[check_columns] < -0.01).sum().sum())
            above_one_count = ((df_normalized[check_columns] > 1.01).sum().sum())
            
            if negative_count > 0:
                print(f"⚠️  发现 {negative_count} 个小于-0.01的值（在特征/标签列中）")
            else:
                print(f"✓ 没有发现小于-0.01的值（在特征/标签列中）")
                
            if above_one_count > 0:
                print(f"⚠️  发现 {above_one_count} 个大于1.01的值（在特征/标签列中）")
            else:
                print(f"✓ 没有发现大于1.01的值（在特征/标签列中）")
        else:
            print("⚠️  无法检查特征/标签列范围")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()