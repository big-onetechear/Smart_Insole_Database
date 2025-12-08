# test_split_data.py
import sys
import torch
sys.path.append('.')  # 添加当前目录到路径

from SmartInsoleDataset_batch import create_batch_data_loaders

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows：黑体；Mac：Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False
# 创建数据加载器
train_loader, val_loader, test_loader, _, _, _ = create_batch_data_loaders(
    base_path='D:/TG0/PublicData_Rep/Smart_Insole_Database',  # 修改为你的数据路径
    split_method='mixed',  # 使用混合划分
    batch_size=32,
    cache_dir='./data_cache',
    force_reload=True  # 强制重新加载，查看数据
)

print("\n" + "="*60)
print("🔍 检查拆分后的双流数据")
print("="*60)

# 获取第一个批次
batch = next(iter(train_loader))

# 1. 检查张量形状
print(f"\n📐 张量形状:")
print(f"  CapSense特征形状: {batch['cap_features'].shape}")  # 应为 [batch_size, 12]
print(f"  IMU特征形状: {batch['imu_features'].shape}")       # 应为 [batch_size, 5]
print(f"  标签形状: {batch['labels'].shape}")               # 应为 [batch_size, 3]

# 2. 检查数据类型
print(f"\n🔢 数据类型:")
print(f"  CapSense类型: {batch['cap_features'].dtype}")
print(f"  IMU类型: {batch['imu_features'].dtype}")
print(f"  标签类型: {batch['labels'].dtype}")

# 3. 检查数值范围
print(f"\n📈 数值范围:")
print(f"  CapSense范围: [{batch['cap_features'].min():.4f}, {batch['cap_features'].max():.4f}]")
print(f"  IMU范围: [{batch['imu_features'].min():.4f}, {batch['imu_features'].max():.4f}]")
print(f"  标签范围: [{batch['labels'].min():.4f}, {batch['labels'].max():.4f}]")

# 4. 检查单个样本
print(f"\n👣 单个样本示例 (第一个样本):")
print(f"  CapSense值 (12维):")
for i, val in enumerate(batch['cap_features'][0]):
    print(f"    C{i}: {val:.4f}" if i < 10 else f"    C{i}: {val:.4f}", end="\n" if (i+1) % 3 == 0 else "  ")

print(f"\n  IMU值 (5维):")
imu_names = ['Ax', 'Ay', 'Az', 'Gp', 'Gr']
for i, val in enumerate(batch['imu_features'][0]):
    print(f"    {imu_names[i]}: {val:.4f}")

print(f"\n  标签值 (3维):")
label_names = ['Fx_norm', 'Fy_norm', 'Fz_norm']
for i, val in enumerate(batch['labels'][0]):
    print(f"    {label_names[i]}: {val:.4f}")

# 5. 检查统计信息
print(f"\n📊 统计摘要:")
print(f"  CapSense均值: {batch['cap_features'].mean():.4f} ± {batch['cap_features'].std():.4f}")
print(f"  IMU均值: {batch['imu_features'].mean():.4f} ± {batch['imu_features'].std():.4f}")
print(f"  标签均值: {batch['labels'].mean():.4f} ± {batch['labels'].std():.4f}")

# 6. 检查是否有NaN或Inf值
print(f"\n⚠️ 数据完整性检查:")
print(f"  CapSense NaN数量: {torch.isnan(batch['cap_features']).sum().item()}")
print(f"  CapSense Inf数量: {torch.isinf(batch['cap_features']).sum().item()}")
print(f"  IMU NaN数量: {torch.isnan(batch['imu_features']).sum().item()}")
print(f"  IMU Inf数量: {torch.isinf(batch['imu_features']).sum().item()}")
print(f"  标签 NaN数量: {torch.isnan(batch['labels']).sum().item()}")
print(f"  标签 Inf数量: {torch.isinf(batch['labels']).sum().item()}")

# 7. 分布可视化 (可选)
print(f"\n🎨 数据分布 (可选 - 需要matplotlib):")
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # CapSense分布
    cap_data = batch['cap_features'].flatten().numpy()
    axes[0].hist(cap_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('CapSense 分布')
    axes[0].set_xlabel('值')
    axes[0].set_ylabel('频率')
    
    # IMU分布
    imu_data = batch['imu_features'].flatten().numpy()
    axes[1].hist(imu_data, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('IMU 分布')
    axes[1].set_xlabel('值')
    axes[1].set_ylabel('频率')
    
    # 标签分布
    label_data = batch['labels'].flatten().numpy()
    axes[2].hist(label_data, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[2].set_title('GRF标签 分布')
    axes[2].set_xlabel('值')
    axes[2].set_ylabel('频率')
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=100, bbox_inches='tight')
    print("  ✅ 分布图已保存为 data_distribution.png")
    plt.show()
    
except ImportError:
    print("  ⚠️  未安装matplotlib，跳过可视化")

print(f"\n{'='*60}")
print("✅ 数据拆分检查完成!")
print(f"{'='*60}")