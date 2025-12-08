# quick_test.py
from SmartInsoleDataset_batch import BatchSmartInsoleDataset
import glob
from pathlib import Path

def quick_test():
    """快速测试数据加载"""
    base_path = "D:/TG0/PublicData_Rep/Smart_Insole_Database"
    
    # 查找所有文件
    pattern = f"{base_path}/subjectRepro*/norm/*_normalized.csv"
    all_files = glob.glob(pattern)[:5]  # 只用前5个文件测试
    
    print(f"🔍 找到 {len(all_files)} 个测试文件")
    for f in all_files:
        print(f"  • {Path(f).name}")
    
    # 创建数据集
    dataset = BatchSmartInsoleDataset(
        all_files,
        cache_dir='./test_cache',
        force_reload=True
    )
    
    # 获取样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n✅ 样本信息:")
        print(f"  特征形状: {sample['features'].shape}")
        print(f"  标签形状: {sample['labels'].shape}")
        print(f"  特征值示例: {sample['features'][:5]}")  # 前5个值
        print(f"  标签值示例: {sample['labels']}")
        
        # 统计信息
        stats = dataset.get_statistics()
        print(f"\n📊 数据统计:")
        print(f"  总样本数: {stats['n_samples']}")
        print(f"  特征范围: {stats['feature_range']}")
        print(f"  标签范围: {stats['label_range']}")
    else:
        print("❌ 数据集为空，请检查数据文件")

if __name__ == "__main__":
    quick_test()