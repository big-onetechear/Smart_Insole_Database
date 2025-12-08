# SmartInsoleDataset_batch.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class BatchSmartInsoleDataset(Dataset):
    """批量处理的数据集类 - 按论文要求"""
    
    def __init__(self, file_paths, seq_length=1, cache_dir=None, force_reload=False):
        """
        参数:
            file_paths: 归一化CSV文件路径列表
            seq_length: 序列长度（论文中为1）
            cache_dir: 缓存目录，用于存储预处理数据
            force_reload: 是否强制重新加载数据
        """
        self.file_paths = file_paths
        self.seq_length = seq_length
        self.cache_dir = cache_dir
        
        # 特征列定义（按论文要求）
        self.feature_columns = self._get_feature_columns()
        self.label_columns = ['Fx_norm', 'Fy_norm', 'Fz_norm']
        
        # 加载数据
        self.data, self.labels = self._load_batch_data(file_paths, cache_dir, force_reload)
        
        print(f"✅ 数据集加载完成:")
        print(f"   总样本数: {len(self.data):,}")
        print(f"   特征维度: {self.data.shape[1]} (12 Cap + 3 Acc + 2 Gyro = 17)")
        print(f"   标签维度: {self.labels.shape[1]} (3 GRF分量)")
    
    def _get_feature_columns(self):
        """根据论文定义特征列 - 匹配实际数据表头"""
        # CapSense传感器: 实际列名为 C0, C1, ..., C11
        cap_cols = [f'C{i}' for i in range(12)]
        
        # 加速度计: Ax, Ay, Az
        acc_cols = ['Ax', 'Ay', 'Az']
        
        # 陀螺仪: Gp, Gr (pitch, roll)
        gyro_cols = ['Gp', 'Gr']
        
        # 共17个特征
        return cap_cols + acc_cols + gyro_cols
    
    def _load_single_file(self, file_path):
        """加载单个CSV文件"""
        try:
            # 使用更健壮的CSV读取方式
            try:
                df = pd.read_csv(file_path)
            except pd.errors.ParserError:
                # 尝试跳过错误行
                df = pd.read_csv(file_path, on_bad_lines='skip')
            except Exception as e:
                print(f"❌ 读取CSV失败 {Path(file_path).name}: {e}")
                return None, None
            
            # 检查必要的列
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            missing_labels = [col for col in self.label_columns if col not in df.columns]
            
            if missing_features:
                print(f"⚠️  文件 {Path(file_path).name} 缺少特征列: {missing_features}")
                return None, None
            
            if missing_labels:
                print(f"⚠️  文件 {Path(file_path).name} 缺少标签列: {missing_labels}")
                return None, None
            
            # 提取特征和标签
            features = df[self.feature_columns].values.astype(np.float32)
            labels = df[self.label_columns].values.astype(np.float32)
            
            return features, labels
            
        except Exception as e:
            print(f"❌ 处理文件失败 {Path(file_path).name}: {e}")
            return None, None
    
    def _load_batch_data(self, file_paths, cache_dir=None, force_reload=False):
        """批量加载所有文件数据"""
        all_features = []
        all_labels = []
        
        # 检查缓存
        cache_file = None
        if cache_dir and not force_reload:
            os.makedirs(cache_dir, exist_ok=True)
            file_hash = hash(tuple(sorted(file_paths)))
            cache_file = os.path.join(cache_dir, f"dataset_cache_{file_hash}.npz")
            
            if os.path.exists(cache_file):
                print(f"📦 从缓存加载数据: {cache_file}")
                npz_data = np.load(cache_file)
                all_features = npz_data['features']
                all_labels = npz_data['labels']
                return all_features, all_labels
        
        # 批量加载数据
        print(f"📥 批量加载 {len(file_paths)} 个文件...")
        
        successful_files = 0
        failed_files = 0
        
        for file_idx, file_path in enumerate(tqdm(file_paths, desc="加载文件")):
            features, labels = self._load_single_file(file_path)
            
            if features is not None and labels is not None:
                all_features.append(features)
                all_labels.append(labels)
                successful_files += 1
            else:
                failed_files += 1
            
            # 定期报告进度
            if (file_idx + 1) % 10 == 0:
                print(f"  已处理 {file_idx + 1}/{len(file_paths)} 文件，成功: {successful_files}, 失败: {failed_files}")
        
        print(f"✅ 文件加载完成: {successful_files} 成功, {failed_files} 失败")
        
        # 合并所有数据
        if all_features:
            all_features = np.vstack(all_features)
            all_labels = np.vstack(all_labels)
        else:
            all_features = np.array([]).reshape(0, len(self.feature_columns))
            all_labels = np.array([]).reshape(0, 3)
        
        # 保存缓存
        if cache_file and all_features.size > 0:
            print(f"💾 保存数据到缓存: {cache_file}")
            np.savez_compressed(cache_file, 
                               features=all_features, 
                               labels=all_labels,
                               feature_columns=self.feature_columns,
                               label_columns=self.label_columns)
        
        return all_features, all_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx]  # [17]
        
        # 拆分特征(双流)
        cap_features = features[:12]      # C0-C11
        imu_features = features[12:]      # Ax,Ay,Az,Gp,Gr
        
        labels = self.labels[idx]         # [3]
        
        return {
            'cap_features': torch.FloatTensor(cap_features),
            'imu_features': torch.FloatTensor(imu_features),
            'labels': torch.FloatTensor(labels)
        }
    
    def get_statistics(self):
        """获取数据统计信息"""
        stats = {
            'n_samples': len(self.data),
            'feature_shape': self.data.shape[1] if len(self.data) > 0 else 0,
            'label_shape': self.labels.shape[1] if len(self.labels) > 0 else 0,
            'feature_range': (
                float(self.data.min()) if len(self.data) > 0 else 0,
                float(self.data.max()) if len(self.data) > 0 else 1
            ),
            'label_range': (
                float(self.labels.min()) if len(self.labels) > 0 else 0,
                float(self.labels.max()) if len(self.labels) > 0 else 1
            )
        }
        return stats

def create_batch_data_loaders(base_path, split_method='subject', batch_size=32, 
                              cache_dir='./data_cache', force_reload=False):
    """
    批量创建数据加载器
    
    参数:
        base_path: 数据根目录
        split_method: 划分策略 ('subject', 'random', 'mixed')
        batch_size: 批量大小
        cache_dir: 缓存目录
        force_reload: 是否强制重新加载
    """
    
    print("="*80)
    print("🤖 批量数据加载器 - 智能数据划分与加载")
    print("="*80)
    
    # 1. 查找所有归一化文件
    pattern = f"{base_path}/subjectRepro*/norm/*_normalized.csv"
    all_files = glob.glob(pattern)
    
    print(f"🔍 在 {base_path} 中查找文件...")
    print(f"找到 {len(all_files)} 个归一化文件")
    
    if len(all_files) == 0:
        raise ValueError(f"未找到归一化文件，请检查路径: {pattern}")
    
    # 显示前几个文件
    print(f"\n📁 文件示例:")
    for i, f in enumerate(all_files[:3]):
        print(f"  {i+1}. {Path(f).name}")
    if len(all_files) > 3:
        print(f"  ... 还有 {len(all_files)-3} 个文件")
    
    # 2. 按subject分类
    print(f"\n📊 按subject分类...")
    files_by_subject = {}
    for file_path in all_files:
        # 从路径提取subject编号
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part.startswith('subjectRepro'):
                subject_id = part.replace('subjectRepro', '')
                if subject_id not in files_by_subject:
                    files_by_subject[subject_id] = []
                files_by_subject[subject_id].append(file_path)
                break
    
    print(f"  找到 {len(files_by_subject)} 个subjects")
    for subject_id, files in sorted(files_by_subject.items()):
        print(f"    subject_{subject_id}: {len(files)} 文件")
    
    # 3. 划分数据集
    print(f"\n📈 使用 '{split_method}' 策略划分数据集")
    
    if split_method == 'subject':
        # 留出一个subject做测试
        subject_ids = list(files_by_subject.keys())
        test_subject = subject_ids[-1]  # 用最后一个subject测试
        train_val_subjects = subject_ids[:-1]
        
        # 从训练验证集中再分一个验证subject
        val_subject = train_val_subjects[-1]
        train_subjects = train_val_subjects[:-1]
        
        # 收集文件
        train_files = []
        for subject in train_subjects:
            train_files.extend(files_by_subject[subject])
        
        val_files = files_by_subject[val_subject]
        test_files = files_by_subject[test_subject]
        
        print(f"  训练subjects: {train_subjects} ({len(train_files)} 文件)")
        print(f"  验证subject: {val_subject} ({len(val_files)} 文件)")
        print(f"  测试subject: {test_subject} ({len(test_files)} 文件)")
    
    elif split_method == 'random':
        # 随机划分
        from sklearn.model_selection import train_test_split
        train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        print(f"  随机划分: {len(train_files)}训练, {len(val_files)}验证, {len(test_files)}测试")
    
    elif split_method == 'mixed':
        # 混合划分：每个subject都有数据在三个集合中
        train_files, val_files, test_files = [], [], []
        
        for subject_id, files in files_by_subject.items():
            n_files = len(files)
            n_test = max(1, int(n_files * 0.15))  # 15%测试
            n_val = max(1, int(n_files * 0.15))   # 15%验证
            n_train = n_files - n_test - n_val     # 70%训练
            
            # 打乱文件
            import random
            random.shuffle(files)
            
            train_files.extend(files[:n_train])
            val_files.extend(files[n_train:n_train+n_val])
            test_files.extend(files[n_train+n_val:])
        
        print(f"  混合划分: {len(train_files)}训练, {len(val_files)}验证, {len(test_files)}测试")
    
    else:
        raise ValueError(f"未知划分方法: {split_method}")
    
    # 4. 创建数据集
    print(f"\n🔄 创建数据集...")
    
    train_dataset = BatchSmartInsoleDataset(
        train_files, 
        seq_length=1,
        cache_dir=os.path.join(cache_dir, 'train') if cache_dir else None,
        force_reload=force_reload
    )
    
    val_dataset = BatchSmartInsoleDataset(
        val_files,
        seq_length=1,
        cache_dir=os.path.join(cache_dir, 'val') if cache_dir else None,
        force_reload=force_reload
    )
    
    test_dataset = BatchSmartInsoleDataset(
        test_files,
        seq_length=1,
        cache_dir=os.path.join(cache_dir, 'test') if cache_dir else None,
        force_reload=force_reload
    )
    
    # 显示统计信息
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()
    test_stats = test_dataset.get_statistics()
    
    print(f"\n📊 数据集统计:")
    print(f"  训练集: {train_stats['n_samples']:,} 样本")
    print(f"  验证集: {val_stats['n_samples']:,} 样本")
    print(f"  测试集: {test_stats['n_samples']:,} 样本")
    print(f"  特征维度: {train_stats['feature_shape']} (应为17)")
    print(f"  标签维度: {train_stats['label_shape']} (应为3)")
    
    # 5. 创建数据加载器
    print(f"\n⚡ 创建数据加载器...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 可以设为CPU核心数，但注意内存使用
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    # 6. 验证第一个批次
    print(f"\n🧪 验证数据加载器...")
    if train_loader:
        batch = next(iter(train_loader))
    
        cap_features = batch['cap_features']
        imu_features = batch['imu_features']
        labels = batch['labels']
        
        print(f"  批量CapSense特征形状: {cap_features.shape}")
        print(f"  批量IMU特征形状: {imu_features.shape}")
        print(f"  批量标签形状: {labels.shape}")
        print(f"  CapSense范围: [{cap_features.min():.3f}, {cap_features.max():.3f}]")
        print(f"  IMU范围: [{imu_features.min():.3f}, {imu_features.max():.3f}]")
        print(f"  标签范围: [{labels.min():.3f}, {labels.max():.3f}]")
        
        if cap_features.shape[1] != 12:
            print(f"⚠️  警告: CapSense特征维度应为12，实际为{cap_features.shape[1]}")
        if imu_features.shape[1] != 5:
            print(f"⚠️  警告: IMU特征维度应为5，实际为{imu_features.shape[1]}")
        
        print(f"\n✅ 批量数据加载器创建完成!")
        
        return train_loader, val_loader, test_loader, train_files, val_files, test_files

def save_split_info(train_files, val_files, test_files, output_file='dataset_split_info.json'):
    """保存数据集划分信息"""
    split_info = {
        'train_files': [str(f) for f in train_files],
        'val_files': [str(f) for f in val_files],
        'test_files': [str(f) for f in test_files],
        'train_count': len(train_files),
        'val_count': len(val_files),
        'test_count': len(test_files),
        'total_count': len(train_files) + len(val_files) + len(test_files)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"💾 划分信息已保存到: {output_file}")
    return output_file