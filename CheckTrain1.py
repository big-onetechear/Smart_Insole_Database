# Check.py - 系统性问题排查工具
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
from torch.utils.data import Dataset

print("=" * 60)
print("🔍 智能鞋垫项目 - 系统问题排查工具")
print("=" * 60)

# ==================== 1. 环境检查 ====================
print("\n1. 🛠️ 环境检查")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")

# ==================== 2. 数据检查 ====================
print("\n2. 📊 数据完整性检查")

class DataChecker:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        
    def load_and_check(self):
        print(f"检查文件: {self.csv_path}")
        
        # 检查文件是否存在
        if not os.path.exists(self.csv_path):
            print(f"❌ 文件不存在: {self.csv_path}")
            return False
        
        try:
            # 加载数据
            self.data = pd.read_csv(self.csv_path)
            print(f"✅ 文件加载成功，形状: {self.data.shape}")
            
            # 检查基本列
            required_columns = ['Fx_norm', 'Fy_norm', 'Fz_norm']
            for col in required_columns:
                if col not in self.data.columns:
                    print(f"❌ 缺失必要列: {col}")
                    return False
            
            print(f"✅ 必要列检查通过")
            return True
            
        except Exception as e:
            print(f"❌ 文件读取失败: {e}")
            return False
    
    def analyze_data(self):
        if self.data is None:
            return
        
        print(f"\n  数据详情:")
        print(f"    总行数: {len(self.data)}")
        print(f"    总列数: {len(self.data.columns)}")
        
        # 检查CapSense列 (0-17)
        capsense_cols = [f'ele_{i}' for i in range(18)]
        capsense_missing = [col for col in capsense_cols if col not in self.data.columns]
        if capsense_missing:
            print(f"  ⚠️  缺失CapSense列: {capsense_missing[:5]}...")
        else:
            print(f"  ✅ CapSense列完整 (0-17)")
        
        # 检查IMU列 (18-24)
        imu_cols = [f'ele_{i}' for i in range(18, 25)]
        imu_missing = [col for col in imu_cols if col not in self.data.columns]
        if imu_missing:
            print(f"  ⚠️  缺失IMU列: {imu_missing}")
        else:
            print(f"  ✅ IMU列完整 (18-24)")
        
        # 检查NaN值
        nan_counts = self.data.isna().sum()
        total_nan = nan_counts.sum()
        print(f"  NaN值总数: {total_nan}")
        
        if total_nan > 0:
            print(f"  ⚠️  包含NaN的列:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"    {col}: {count}个NaN ({count/len(self.data)*100:.2f}%)")
        
        # 检查无穷值
        inf_cols = []
        for col in self.data.select_dtypes(include=[np.number]).columns:
            if np.any(np.isinf(self.data[col].values)):
                inf_cols.append(col)
        
        if inf_cols:
            print(f"  ❌ 包含无穷值的列: {inf_cols}")
        else:
            print(f"  ✅ 无无穷值")
        
        # 检查数据范围
        print(f"\n  数据范围检查:")
        
        # CapSense范围
        capsense_data = self.data[[f'ele_{i}' for i in range(18)]].values
        print(f"    CapSense范围: {capsense_data.min():.2f} ~ {capsense_data.max():.2f}")
        print(f"    CapSense均值: {capsense_data.mean():.2f} ± {capsense_data.std():.2f}")
        
        # IMU范围
        imu_data = self.data[[f'ele_{i}' for i in range(18, 25)]].values
        print(f"    IMU范围: {imu_data.min():.2f} ~ {imu_data.max():.2f}")
        print(f"    IMU均值: {imu_data.mean():.2f} ± {imu_data.std():.2f}")
        
        # 标签范围
        labels = self.data[['Fx_norm', 'Fy_norm', 'Fz_norm']].values
        print(f"    标签范围: {labels.min():.2f} ~ {labels.max():.2f}")
        print(f"    标签均值: {labels.mean():.2f} ± {labels.std():.2f}")

# 测试数据文件
test_file = "D:/TG0/PublicData_Rep/Smart_Insole_Database/subject_1/squatting_s1_merged.csv"
checker = DataChecker(test_file)

if checker.load_and_check():
    checker.analyze_data()
else:
    print("❌ 数据检查失败，请检查文件路径")
    sys.exit(1)

# ==================== 3. 模型组件检查 ====================
print("\n3. 🤖 模型组件检查")

def test_dense_block():
    print("  测试DenseBlock实现...")
    
    class SimpleDenseBlock(nn.Module):
        def __init__(self, input_dim, output_dim=32):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, output_dim)
            self.fc2 = nn.Linear(output_dim, output_dim)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return x
    
    # 测试
    block = SimpleDenseBlock(18, 32)
    test_input = torch.randn(32, 18)
    output = block(test_input)
    
    print(f"   输入: {test_input.shape}")
    print(f"   输出: {output.shape}")
    print(f"   输出范围: {output.min():.3f} ~ {output.max():.3f}")
    print(f"   ✅ DenseBlock测试通过")

def test_cross_attention():
    print("  测试CrossAttention实现...")
    
    class SimpleCrossAttention(nn.Module):
        def __init__(self, dim=32):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
        
        def forward(self, query, key, dim):
            Q = self.q_proj(query)
            K = self.k_proj(key)
            V = self.v_proj(value)
            
            attention = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5), dim=-1)
            output = torch.matmul(attention, V)
            return output
    
    # 测试
    attention = SimpleCrossAttention(32)
    query = torch.randn(32, 32)
    key = torch.randn(32, 32)
    value = torch.randn(32, 32)
    
    output = attention(query, key, value)
    
    print(f"   Query: {query.shape}")
    print(f"   Output: {output.shape}")
    print(f"   ✅ CrossAttention测试通过")

# 运行组件测试
test_dense_block()
test_cross_attention()

# ==================== 4. 训练流程检查 ====================
print("\n4. 🔄 训练流程检查")

def test_training_flow():
    print("  测试最小训练流程...")
    
    # 最小数据集
    class MiniDataset:
        def __init__(self):
            self.capsense = torch.randn(100, 18)
            self.imu = torch.randn(100, 7)
            self.labels = torch.randn(100, 3)
        
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return self.capsense[idx], self.imu[idx], self.labels[idx]
    
    # 最小模型
    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cap_fc = nn.Linear(18, 16)
            self.imu_fc = nn.Linear(7, 16)
            self.fusion = nn.Linear(32, 3)
        
        def forward(self, cap, imu):
            cap_out = torch.relu(self.cap_fc(cap))
            imu_out = torch.relu(self.imu_fc(imu))
            combined = torch.cat([cap_out, imu_out], dim=1)
            return self.fusion(combined)
    
    # 测试训练
    dataset = MiniDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MiniModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 一个epoch的训练
    model.train()
    total_loss = 0
    for batch_idx, (cap, imu, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(cap, imu)
        loss = criterion(outputs, labels)
        
        # 检查损失是否为nan
        if torch.isnan(loss):
            print(f"  ❌ 第{batch_idx}批次损失为nan!")
            break
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / (batch_idx + 1)
    print(f"  平均损失: {avg_loss:.6f}")
    
    if not torch.isnan(torch.tensor(avg_loss)):
        print(f"  ✅ 训练流程测试通过")
    else:
        print(f"  ❌ 训练流程存在问题")

test_training_flow()

# ==================== 5. 问题诊断建议 ====================
print("\n5. 💡 问题诊断建议")

print("""
根据你的训练输出，发现以下问题：

🔴 主要问题：损失值为nan

可能原因及解决方案：

1. 📊 数据问题
   - 检查数据中是否有NaN或无穷值
   - 检查数据范围是否异常
   - 确保数据已正确标准化

2. 🏗️ 模型架构问题
   - DenseBlock输出维度不一致（114 vs 115）
   - 交叉注意力维度不匹配
   - 梯度爆炸（数值过大）

3. ⚙️ 训练配置问题
   - 学习率可能过高
   - 没有梯度裁剪
   - 权重初始化问题

建议操作顺序：
1. 先运行本检查脚本确认数据质量
2. 使用简单模型确保基础流程能跑通
3. 逐步增加复杂度
4. 添加梯度监控和数值稳定性检查
""")

print("\n" + "=" * 60)
print("✅ 检查完成！请根据以上建议进行修复")
print("=" * 60)


