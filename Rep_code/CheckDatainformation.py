"""
CheckDatainformation.py
数据检查器 - 只关注维度和归一化参数
"""

import pandas as pd
import numpy as np
import os
import json

class DataChecker:
    """数据检查器 - 只检查维度和计算归一化参数"""
    
    def __init__(self, csv_path, output_dir=None):
        """
        参数:
            csv_path: 输入CSV文件路径
            output_dir: 输出目录（默认：输入文件目录）
        """
        self.csv_path = csv_path
        self.filename = os.path.basename(csv_path)
        
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.dirname(csv_path)
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df = None
        self.params = {}
    
    def run(self):
        """运行检查 - 主函数"""
        print(f"🔍 开始检查: {self.filename}")
        print(f"📁 输出目录: {self.output_dir}")
        
        # 1. 加载数据
        self._load_data()
        
        # 2. 检查基本维度
        self._check_dimensions()
        
        # 3. 计算归一化参数
        self._calculate_params()
        
        # 4. 保存参数文件
        param_file = self._save_params()
        
        print(f"✅ 检查完成！")
        print(f"📄 参数文件: {param_file}")
        
        return param_file
    
    def _load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"📊 数据大小: {len(self.df)}行 × {len(self.df.columns)}列")
        except Exception as e:
            raise Exception(f"数据加载失败: {e}")
    
    def _check_dimensions(self):
        """检查数据维度"""
        # 检查必要的列（去掉timestamp）
        required = [f'C{i}' for i in range(12)] + \
                  ['Ax', 'Ay', 'Az', 'Gr', 'Gp', 'Fx_norm', 'Fy_norm', 'Fz_norm']
        
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            print(f"⚠️  缺少列: {missing}")
        
        # 统计维度
        capsense_count = len([col for col in self.df.columns if col.startswith('C')])
        acc_count = len([col for col in ['Ax', 'Ay', 'Az'] if col in self.df.columns])
        gyro_count = len([col for col in ['Gr', 'Gp'] if col in self.df.columns])
        label_count = len([col for col in self.df.columns if '_norm' in col])
        
        total_features = capsense_count + acc_count + gyro_count
        
        print(f"📐 维度统计:")
        print(f"  • 总样本: {len(self.df):,}")
        print(f"  • 特征: {total_features}个 (CapSense:{capsense_count} + Acc:{acc_count} + Gyro:{gyro_count})")
        print(f"  • 标签: {label_count}个")
        
        # 检查是否符合论文要求
        expected_features = 12 + 3 + 2  # 论文要求
        if total_features != expected_features:
            print(f"⚠️  特征数量不符: 应有{expected_features}个，实际{total_features}个")
    
    def _calculate_params(self):
        """计算归一化参数"""
        print("\n🧮 计算归一化参数:")
        
        params = {
            "dataset": os.path.splitext(self.filename)[0],
            "samples": len(self.df),
            "feature_columns": {
                "capsense": [f'C{i}' for i in range(12)],
                "accelerometer": ['Ax', 'Ay', 'Az'],
                "gyroscope": ['Gr', 'Gp']
            },
            "label_columns": ['Fx_norm', 'Fy_norm', 'Fz_norm'],
            "normalization": {}
        }
        
        # 1. CapSense参数（论文固定值）
        if all(col in self.df.columns for col in [f'C{i}' for i in range(12)]):
            # 检查实际范围
            capsense_min = float(self.df[[f'C{i}' for i in range(12)]].min().min())
            capsense_max = float(self.df[[f'C{i}' for i in range(12)]].max().max())
            
            params["normalization"]["capsense"] = {
                "paper_scale": 800.0,
                "actual_range": [capsense_min, capsense_max],
                "method": "x / 800.0"
            }
            print(f"  • CapSense: /800.0")
            print(f"    实际范围: [{capsense_min:.1f}, {capsense_max:.1f}]")
        
        # 2. 加速度计参数
        if all(col in self.df.columns for col in ['Ax', 'Ay', 'Az']):
            # 找出最大绝对值作为比例因子
            acc_data = self.df[['Ax', 'Ay', 'Az']].values
            acc_scale = float(np.max(np.abs(acc_data)))
            acc_min = float(self.df[['Ax', 'Ay', 'Az']].min().min())
            acc_max = float(self.df[['Ax', 'Ay', 'Az']].max().max())
            
            params["normalization"]["accelerometer"] = {
                "scale": acc_scale,
                "actual_range": [acc_min, acc_max],
                "paper_range": [-1, 1],
                "method": f"x / {acc_scale:.4f}  # 到[-1,1]\n    x = (x + 1) / 2  # 到[0,1]"
            }
            print(f"  • 加速度计: /{acc_scale:.2f} → (+1)/2")
            print(f"    实际范围: [{acc_min:.2f}, {acc_max:.2f}]")
        
        # 3. 陀螺仪参数
        if all(col in self.df.columns for col in ['Gr', 'Gp']):
            gyro_data = self.df[['Gr', 'Gp']].values
            gyro_scale = float(np.max(np.abs(gyro_data)))
            gyro_min = float(self.df[['Gr', 'Gp']].min().min())
            gyro_max = float(self.df[['Gr', 'Gp']].max().max())
            
            params["normalization"]["gyroscope"] = {
                "scale": gyro_scale,
                "actual_range": [gyro_min, gyro_max],
                "paper_range": [-600, 600],
                "method": f"x / {gyro_scale:.4f}  # 到[-1,1]\n    x = (x + 1) / 2  # 到[0,1]"
            }
            print(f"  • 陀螺仪: /{gyro_scale:.2f} → (+1)/2")
            print(f"    实际范围: [{gyro_min:.1f}, {gyro_max:.1f}]")
        
        # 4. GRF参数（每个分量单独）
        grf_params = {}
        for col in ['Fx_norm', 'Fy_norm', 'Fz_norm']:
            if col in self.df.columns:
                col_min = float(self.df[col].min())
                col_max = float(self.df[col].max())
                grf_params[col] = {
                    "min": col_min,
                    "max": col_max,
                    "range": [col_min, col_max],
                    "method": f"(x - {col_min:.6f}) / ({col_max:.6f} - {col_min:.6f})"
                }
                print(f"  • {col}: Min-Max")
                print(f"    范围: [{col_min:.3f}, {col_max:.3f}]")
        
        if grf_params:
            params["normalization"]["grf"] = grf_params
        
        self.params = params
    
    def _save_params(self):
        """保存参数文件"""
        # 生成文件名
        base_name = os.path.splitext(self.filename)[0]
        param_file = os.path.join(self.output_dir, f"{base_name}_params.json")
        
        # 保存JSON
        with open(param_file, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, ensure_ascii=False, indent=2)
        
        return param_file
    


# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("📊 数据检查工具")
    print("=" * 50)
    
    # ===== 配置部分 =====
    # 1. 输入文件
    input_csv = r"D:\TG0\PublicData_Rep\Smart_Insole_Database\subjectRepro1\jogging_s1_merged.csv"
    
    # 2. 输出目录（可选）
    # output_dir = None  # 默认：与输入文件同目录
    output_dir = r"D:\TG0\PublicData_Rep\Smart_Insole_Database\subjectRepro1\Param\jogging_s1_preprocess_params"
    
    # ===== 执行部分 =====
    print(f"📂 输入文件: {input_csv}")
    if output_dir:
        print(f"📁 输出目录: {output_dir}")
    else:
        print(f"📁 输出目录: 自动（输入文件目录）")
    print("-" * 50)
    
    try:
        # 创建检查器
        checker = DataChecker(input_csv, output_dir=output_dir)
        
        # 运行检查
        param_file = checker.run()
        
        print("\n" + "=" * 50)
        print("🎉 完成！")
        print("=" * 50)
        print(f"\n📋 下一步:")
        print(f"1. 查看参数文件: {param_file}")
        print(f"2. 根据参数实现数据归一化")
        print(f"3. 准备模型训练")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()