# train_model.py
"""
智能鞋垫模型训练脚本
基于论文《Estimation of Three-Dimensional Ground Reaction Forces Using Low-Cost Smart Insoles》
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from DataLoading import SmartInsoleDataset
from modelAssemb import PaperFusionModel
from LearnMethod import (
    TrainingConfig, 
    SmartTrainingManager,
    ModelEvaluator,
    create_default_config
)

def setup_data_paths(base_dir="."):
    """设置数据路径"""
    print("📁 设置数据路径...")
    
    subjects = [f"subject_{i}" for i in range(1, 6)]
    movements = [
        "squatting",
        "walking",  # 或 stepping_in_place
        "jogging",  # 或 running_in_place  
        "swaying",
        "jump_inplace",
        "jump_fb"   # forward_backward
    ]
    
    # 查找所有CSV文件
    data_files = []
    for subject in subjects:
        subject_dir = os.path.join(base_dir, subject)
        if os.path.exists(subject_dir):
            for file in os.listdir(subject_dir):
                if file.endswith('_merged.csv'):
                    movement = file.split('_')[0]
                    data_files.append({
                        'subject': subject,
                        'movement': movement,
                        'path': os.path.join(subject_dir, file),
                        'size': os.path.getsize(os.path.join(subject_dir, file))
                    })
    
    print(f"✅ 找到 {len(data_files)} 个数据文件")
    for i, file_info in enumerate(data_files[:5]):  # 显示前5个
        print(f"  {i+1}. {file_info['subject']}/{file_info['movement']}: {file_info['size']/1024/1024:.1f} MB")
    
    return data_files

def load_and_split_dataset(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """加载数据集并划分"""
    print(f"\n📊 加载数据集: {os.path.basename(csv_path)}")
    
    # 创建数据集
    dataset = SmartInsoleDataset(csv_path)
    
    print(f"   数据集大小: {len(dataset)}")
    print(f"   CapSense特征: {dataset[0][0].shape}")
    print(f"   IMU特征: {dataset[0][1].shape}")
    print(f"   标签: {dataset[0][2].shape}")
    
    # 计算划分大小
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # 随机划分
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )
    
    print(f"   训练集: {len(train_dataset)}")
    print(f"   验证集: {len(val_dataset)}")
    print(f"   测试集: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64):
    """创建数据加载器"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Windows上设为0避免问题
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

def train_single_subject(subject_data, config, save_dir="results"):
    """训练单个受试者的模型"""
    print(f"\n🎯 开始训练 {subject_data['subject']}...")
    
    # 加载数据
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(
        subject_data['path']
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=config.batch_size
    )
    
    # 创建模型
    model = PaperFusionModel()
    
    # 创建训练管理器
    manager = SmartTrainingManager(model, config)
    
    # 训练
    start_time = time.time()
    history = manager.fit(train_loader, val_loader)
    training_time = time.time() - start_time
    
    # 评估
    print(f"\n📈 评估模型...")
    eval_results = ModelEvaluator.evaluate(
        model, test_loader, config.device
    )
    
    # 保存结果
    subject_name = subject_data['subject']
    movement_name = subject_data['movement']
    
    results = {
        'subject': subject_name,
        'movement': movement_name,
        'model': model,
        'history': history,
        'eval_results': eval_results,
        'config': config.to_dict(),
        'training_time': training_time,
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    # 保存模型和结果
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"{subject_name}_{movement_name}_{timestamp}.pth")
    results_path = os.path.join(save_dir, f"results_{subject_name}_{movement_name}_{timestamp}.pkl")
    
    # 保存检查点
    manager.save_checkpoint(model_path)
    
    # 保存结果
    torch.save(results, results_path)
    
    print(f"\n💾 结果已保存:")
    print(f"   模型: {model_path}")
    print(f"   结果: {results_path}")
    
    return results

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 训练/验证损失
    axes[0, 0].plot(history['train_loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 学习率
    axes[0, 1].plot(history['learning_rate'], color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('学习率变化')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 训练时间
    axes[1, 0].plot(history['train_time'], label='训练时间')
    axes[1, 0].plot(history['val_time'], label='验证时间')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('时间 (秒)')
    axes[1, 0].set_title('每轮训练时间')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 总时间累积
    cumulative_time = np.cumsum(history['epoch_time'])
    axes[1, 1].plot(cumulative_time)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('累计时间 (秒)')
    axes[1, 1].set_title('累计训练时间')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 训练图已保存: {save_path}")
    
    plt.show()

def main():
    """主训练流程"""
    print("=" * 70)
    print("🚀 智能鞋垫 - 3D地面反作用力估计模型训练")
    print("=" * 70)
    
    # 1. 设置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"training_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📁 结果目录: {results_dir}")
    
    # 2. 查找数据文件
    data_files = setup_data_paths()
    
    if not data_files:
        print("❌ 未找到数据文件，请检查路径")
        return
    
    # 3. 选择要训练的数据
    print("\n📋 可用数据文件:")
    for i, file_info in enumerate(data_files):
        print(f"  [{i+1}] {file_info['subject']} - {file_info['movement']}")
    
    # 先测试一个文件
    print("\n🔧 先测试第一个文件...")
    test_file = data_files[0]  # subject_1 squatting
    
    # 4. 创建配置（使用论文参数）
    config = TrainingConfig(
        # 论文参数
        batch_size=64,
        learning_rate=0.0001,  # 论文使用0.0001
        weight_decay=1e-8,     # 论文使用1e-8
        epochs=50,             # 可以先少一些快速测试
        lr_scheduler_type='plateau',
        lr_patience=3,
        lr_factor=0.1,
        
        # 早停
        use_early_stopping=True,
        early_stop_patience=10,
        
        # 其他
        loss_function='mse',
        optimizer='adam',
        verbose=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n⚙️ 训练配置:")
    print(config)
    
    # 5. 训练
    print(f"\n🎬 开始训练: {test_file['subject']} - {test_file['movement']}")
    results = train_single_subject(test_file, config, results_dir)
    
    # 6. 显示结果
    print("\n" + "=" * 70)
    print("📊 训练结果摘要")
    print("=" * 70)
    
    print(f"受试者: {results['subject']}")
    print(f"运动类型: {results['movement']}")
    print(f"训练时间: {results['training_time']:.2f} 秒")
    print(f"模型参数量: {results['model_params']:,}")
    print(f"最佳验证损失: {min(results['history']['val_loss']):.6f}")
    
    print("\n📈 测试集评估:")
    ModelEvaluator.print_metrics(results['eval_results']['metrics'])
    
    # 7. 绘制图表
    plot_path = os.path.join(results_dir, f"training_plot_{test_file['subject']}.png")
    plot_training_history(results['history'], plot_path)
    
    # 8. 保存训练报告
    report_path = os.path.join(results_dir, "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("智能鞋垫模型训练报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"受试者: {results['subject']}\n")
        f.write(f"运动类型: {results['movement']}\n")
        f.write(f"数据文件: {test_file['path']}\n")
        f.write(f"训练时间: {results['training_time']:.2f} 秒\n")
        f.write(f"总轮数: {len(results['history']['train_loss'])}\n\n")
        
        f.write("模型配置:\n")
        for key, value in results['config'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n评估结果:\n")
        metrics = results['eval_results']['metrics']
        for key, value in metrics.items():
            if isinstance(value, list):
                f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {key}: {value:.6f}\n")
    
    print(f"\n📄 训练报告已保存: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()