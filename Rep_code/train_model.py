# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from datetime import datetime

# 导入模型和数据
from model_architecture import DualStreamAttentionModel
from SmartInsoleDataset_batch import create_batch_data_loaders

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1, 
            patience=3,
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
        print(f"🚀 训练器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  训练样本: {len(train_loader.dataset):,}")
        print(f"  验证样本: {len(val_loader.dataset):,}")
        print(f"  测试样本: {len(test_loader.dataset):,}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            cap_features = batch['cap_features'].to(self.device)
            imu_features = batch['imu_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(cap_features, imu_features)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
            
            # 每100个批次记录一次
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   批次 {batch_idx:4d}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.6f} | LR: {current_lr:.6f}")
        
        avg_train_loss = total_loss / num_batches
        self.train_losses.append(avg_train_loss)
        
        return avg_train_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="验证")
            for batch in pbar:
                cap_features = batch['cap_features'].to(self.device)
                imu_features = batch['imu_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(cap_features, imu_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'val_loss': f'{avg_loss:.6f}'})
        
        avg_val_loss = total_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def test(self):
        """测试"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # 存储预测和真实值用于分析
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="测试")
            for batch in pbar:
                cap_features = batch['cap_features'].to(self.device)
                imu_features = batch['imu_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(cap_features, imu_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 收集数据用于分析
                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())
                
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'test_loss': f'{avg_loss:.6f}'})
        
        avg_test_loss = total_loss / num_batches
        
        # 计算NRMSE（论文中的评价指标）
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算每个GRF分量的NRMSE
        mse = nn.MSELoss(reduction='none')(all_predictions, all_labels).mean(dim=0)
        rmse = torch.sqrt(mse)
        
        # 归一化（除以标签范围）
        label_range = all_labels.max(dim=0)[0] - all_labels.min(dim=0)[0]
        nrmse = (rmse / label_range) * 100  # 百分比
        
        return avg_test_loss, nrmse, all_predictions, all_labels
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config['save_dir'], 
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型: {best_path}")
    
    def plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失', marker='o', markersize=3)
        plt.plot(self.val_losses, label='验证损失', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        plot_path = os.path.join(self.config['save_dir'], 'loss_curve.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 损失曲线已保存: {plot_path}")
    
    def train(self):
        """主训练循环"""
        print(f"\n{'='*80}")
        print("🚀 开始训练!")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n📈 Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"  训练损失: {train_loss:.6f}")
            
            # 验证
            val_loss = self.validate()
            print(f"  验证损失: {val_loss:.6f}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"🎯 新的最佳验证损失: {val_loss:.6f}")
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, val_loss)
            
            # 提前终止检查
            if epoch >= 10 and val_loss > np.mean(self.val_losses[-5:]):
                print("⚠️  验证损失上升，考虑提前终止...")
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n✅ 训练完成! 用时: {training_time:.2f}秒")
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("🔁 已加载最佳模型")
        
        # 绘制损失曲线
        self.plot_losses()
        
        # 最终测试
        print(f"\n{'='*80}")
        print("🧪 最终测试")
        print(f"{'='*80}")
        
        test_loss, nrmse, predictions, labels = self.test()
        
        print(f"📊 测试结果:")
        print(f"  MSE Loss: {test_loss:.6f}")
        print(f"  NRMSE (Fx): {nrmse[0].item():.2f}%")
        print(f"  NRMSE (Fy): {nrmse[1].item():.2f}%")
        print(f"  NRMSE (Fz): {nrmse[2].item():.2f}%")
        print(f"  NRMSE 平均: {nrmse.mean().item():.2f}%")
        
        # 保存最终模型
        final_path = os.path.join(self.config['save_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'test_loss': test_loss,
            'nrmse': nrmse,
            'predictions': predictions,
            'labels': labels
        }, final_path)
        print(f"💾 最终模型已保存: {final_path}")
        
        return test_loss, nrmse

def main():
    """主函数"""
    # 配置
    config = {
        'learning_rate': 0.0001,      # 论文中使用的学习率
        'weight_decay': 1e-8,         # 论文中的权重衰减
        'num_epochs': 50,             # 训练轮数
        'batch_size': 32,             # 批量大小
        'save_dir': './checkpoints',  # 保存目录
        'save_interval': 5,           # 保存间隔
    }
    
    print("="*80)
    print("🤖 GRF预测模型训练")
    print("="*80)
    
    # 1. 加载数据
    print("\n📥 加载数据...")
    train_loader, val_loader, test_loader, _, _, _ = create_batch_data_loaders(
        base_path='D:/TG0/PublicData_Rep/Smart_Insole_Database',
        split_method='mixed',
        batch_size=config['batch_size'],
        cache_dir='./data_cache',
        force_reload=False  # 使用缓存加速
    )
    
    # 2. 创建模型
    print("\n🧠 创建模型...")
    model = DualStreamAttentionModel(
        cap_dim=12,
        imu_dim=5,
        hidden_dim=32,
        output_dim=3
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 创建训练器
    trainer = Trainer(model, train_loader, val_loader, test_loader, config)
    
    # 4. 训练
    test_loss, nrmse = trainer.train()
    
    # 5. 与论文结果对比
    print(f"\n{'='*80}")
    print("📊 与论文结果对比")
    print(f"{'='*80}")
    print(f"我们的结果:")
    print(f"  NRMSE 平均: {nrmse.mean().item():.2f}%")
    print(f"  NRMSE (Fx, Fy, Fz): {nrmse[0].item():.2f}%, {nrmse[1].item():.2f}%, {nrmse[2].item():.2f}%")
    print(f"\n论文结果 (Table 1):")
    print(f"  Best NRMSE: 4.16%")
    print(f"  其他方法: 8.46%-20%")
    
    if nrmse.mean().item() <= 5.0:
        print(f"\n🎉 成功复现论文结果!")
    else:
        print(f"\n⚠️  与论文结果有差距，可能需要调整超参数或模型结构")

if __name__ == "__main__":
    main()