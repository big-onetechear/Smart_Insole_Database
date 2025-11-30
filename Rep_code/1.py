# 今日产出：真实数据训练管道
import pandas as pd
import torch
from torch.utils.data import Dataset

class SmartInsoleDataset(Dataset):
    """加载真实论文数据的Dataset类"""
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        # 提取CapSense特征 (0-17)
        self.capsense_features = self.data[[f'ele_{i}' for i in range(18)]].values
        # 提取IMU特征 (18-24: 加速度+陀螺仪)
        self.imu_features = self.data[[f'ele_{i}' for i in range(18, 25)]].values
        # 提取标签 (Fx_norm, Fy_norm, Fz_norm)
        self.labels = self.data[['Fx_norm', 'Fy_norm', 'Fz_norm']].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.capsense_features[idx]),
                torch.FloatTensor(self.imu_features[idx]), 
                torch.FloatTensor(self.labels[idx]))

# 使用我们之前的基础模型
dataset = SmartInsoleDataset("D:\TG0\PublicData_Rep\Smart_Insole_Database\subject_1\squatting_s1_merged.csv")
print(f"数据量: {len(dataset)}")
print(f"CapSense维度: {dataset[0][0].shape}")
print(f"IMU维度: {dataset[0][1].shape}")
print(f"标签维度: {dataset[0][2].shape}")