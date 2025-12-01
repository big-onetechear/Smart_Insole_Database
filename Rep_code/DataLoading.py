# 输入：CSV文件（包含传感器数据）
# 处理：
# 提取18个压力传感器特征 (CapSense)
# 提取7个IMU特征（加速度计+陀螺仪）
# 提取3个地面反作用力标签
# 输出：标准化后的PyTorch张量
# DataLoading.py
# DataLoading.py - 修复版
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class SmartInsoleDataset(Dataset):
    """加载真实论文数据的Dataset类 - 修复NaN问题"""
    def __init__(self, csv_path):
        # 加载数据
        self.data = pd.read_csv(csv_path)
        
        print(f"📊 数据加载: {len(self.data)} 样本")
        
        # 检查并处理NaN
        self._handle_nan_values()
        
        # 提取特征和标签
        self._extract_features()
        
        # 可选：数据标准化
        self._normalize_features()
        
    def _handle_nan_values(self):
        """处理NaN值"""
        # 检查NaN数量
        nan_count = self.data.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  发现 {nan_count} 个NaN值")
            
            # 显示哪些列有NaN
            nan_columns = self.data.columns[self.data.isna().any()].tolist()
            print(f"   包含NaN的列: {nan_columns}")
            
            # 方法1：删除有NaN的行（如果数据量大）
            # self.data = self.data.dropna()
            
            # 方法2：用前一行的值填充（时间序列数据常用）
            self.data = self.data.fillna(method='ffill')  # 前向填充
            
            # 方法3：如果开头有NaN，再用后向填充
            self.data = self.data.fillna(method='bfill')
            
            print(f"✅ NaN值已处理，处理后样本数: {len(self.data)}")
        
        # 确保没有NaN
        assert not self.data.isna().any().any(), "数据中仍有NaN值"
    
    def _extract_features(self):
        """提取特征"""
        # CapSense特征 (0-17)
        capsense_cols = [f'ele_{i}' for i in range(18)]
        self.capsense_features = self.data[capsense_cols].values
        
        # IMU特征 (18-24: 加速度+陀螺仪)
        imu_cols = [f'ele_{i}' for i in range(18, 25)]
        self.imu_features = self.data[imu_cols].values
        
        # 标签 (Fx_norm, Fy_norm, Fz_norm)
        label_cols = ['Fx_norm', 'Fy_norm', 'Fz_norm']
        self.labels = self.data[label_cols].values
        
        print(f"   CapSense特征: {self.capsense_features.shape}")
        print(f"   IMU特征: {self.imu_features.shape}")
        print(f"   标签: {self.labels.shape}")
    
    def _normalize_features(self):
        """特征标准化（可选）"""
        # 记录原始数据范围
        self.capsense_mean = np.mean(self.capsense_features)
        self.capsense_std = np.std(self.capsense_features)
        self.imu_mean = np.mean(self.imu_features)
        self.imu_std = np.std(self.imu_features)
        self.labels_mean = np.mean(self.labels)
        self.labels_std = np.std(self.labels)
        
        # 标准化（如果数据范围差异很大）
        if self.capsense_std > 0:
            self.capsense_features = (self.capsense_features - self.capsense_mean) / self.capsense_std
        if self.imu_std > 0:
            self.imu_features = (self.imu_features - self.imu_mean) / self.imu_std
        if self.labels_std > 0:
            self.labels = (self.labels - self.labels_mean) / self.labels_std
        
        print(f"   CapSense标准化: 均值={self.capsense_mean:.2f}, 标准差={self.capsense_std:.2f}")
        print(f"   IMU标准化: 均值={self.imu_mean:.2f}, 标准差={self.imu_std:.2f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.capsense_features[idx]),
            torch.FloatTensor(self.imu_features[idx]), 
            torch.FloatTensor(self.labels[idx])
        )

# 测试代码
# if __name__ == "__main__":
#     dataset = SmartInsoleDataset("../subject_1/squatting_s1_merged.csv")
#     capsense, imu, labels = dataset[0]
#     print(f"\n✅ 数据加载测试:")
#     print(f"   样本 0 - CapSense范围: {capsense.min():.2f} ~ {capsense.max():.2f}")
#     print(f"   样本 0 - IMU范围: {imu.min():.2f} ~ {imu.max():.2f}")
#     print(f"   样本 0 - 标签范围: {labels.min():.2f} ~ {labels.max():.2f}")


# # 使用我们之前的基础模型
# dataset = SmartInsoleDataset("D:\TG0\PublicData_Rep\Smart_Insole_Database\subject_1\squatting_s1_merged.csv")
# print(f"数据量: {len(dataset)}")
# print(f"CapSense维度: {dataset[0][0].shape}")
# print(f"IMU维度: {dataset[0][1].shape}")
# print(f"标签维度: {dataset[0][2].shape}")
# 数据量: 20000
# CapSense维度: torch.Size([18])每个样本有18个压力传感器特征
# IMU维度: torch.Size([7])每个样本有7个IMU（加速度计+陀螺仪）特征
# 标签维度: torch.Size([3]) 每个样本有3个地面反作用力标签（Fx_norm, Fy_norm, Fz_norm）
