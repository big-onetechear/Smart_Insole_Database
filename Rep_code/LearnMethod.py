"""
LearnMethod.py
学习机制模块：提供灵活配置的训练组件 - 修复版
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
import time

# ==================== 1. 配置管理类 ====================
class TrainingConfig:
    """训练配置管理类 - 提供灵活的配置接口"""
    
    def __init__(self, **kwargs):
        """初始化配置，支持多种参数设置方式"""
        
        # ===== 先设置默认值 =====
        self._set_defaults()
        
        # ===== 再更新用户提供的配置 =====
        self._update_config(kwargs)
        
        # ===== 最后处理依赖关系（如设备检测） =====
        self._post_process()
    
    def _set_defaults(self):
        """设置所有默认值"""
        # 基础训练参数
        self.batch_size = 64
        self.learning_rate = 0.001
        self.weight_decay = 1e-8
        self.epochs = 100
        
        # 学习率调度
        self.use_lr_scheduler = True
        self.lr_scheduler_type = 'plateau'  # 'plateau', 'step', 'cosine'
        self.lr_patience = 3
        self.lr_factor = 0.1
        self.lr_min = 1e-6
        
        # 早停机制
        self.use_early_stopping = True
        self.early_stop_patience = 10
        self.early_stop_delta = 1e-4
        
        # 梯度处理
        self.use_gradient_clip = True
        self.grad_clip_norm = 1.0
        
        # 损失函数
        self.loss_function = 'mse'  # 'mse', 'mae', 'huber'
        self.huber_delta = 1.0
        
        # 优化器
        self.optimizer = 'adam'  # 'adam', 'sgd', 'adamw'
        self.sgd_momentum = 0.9
        
        # 设备
        self.device = 'auto'  # 'auto', 'cuda', 'cpu'
        
        # 数据划分
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # 日志和保存
        self.save_checkpoints = True
        self.checkpoint_freq = 10
        self.verbose = True
    
    def _update_config(self, user_config: Dict):
        """更新配置参数"""
        for key, value in user_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"警告: 未知配置参数 '{key}'，将被忽略")
    
    def _post_process(self):
        """配置后处理"""
        # 自动检测设备
        if self.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)
        # 如果已经是torch.device对象，保持不变
    
    def update(self, **kwargs):
        """动态更新配置"""
        self._update_config(kwargs)
        self._post_process()  # 重新后处理
        return self
    
    def to_dict(self) -> Dict:
        """将配置转换为字典"""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                # 如果是torch.device对象，转换为字符串
                if isinstance(attr_value, torch.device):
                    config_dict[attr_name] = str(attr_value)
                else:
                    config_dict[attr_name] = attr_value
        return config_dict
    
    def __str__(self) -> str:
        """友好的配置显示"""
        config_str = "=== 训练配置 ===\n"
        
        # 定义配置分类
        categories = {
            '基础训练': ['batch_size', 'learning_rate', 'weight_decay', 'epochs'],
            '学习率调度': ['use_lr_scheduler', 'lr_scheduler_type', 'lr_patience', 'lr_factor', 'lr_min'],
            '早停机制': ['use_early_stopping', 'early_stop_patience', 'early_stop_delta'],
            '梯度处理': ['use_gradient_clip', 'grad_clip_norm'],
            '损失函数': ['loss_function', 'huber_delta'],
            '优化器': ['optimizer', 'sgd_momentum'],
            '设备': ['device'],
            '数据划分': ['train_ratio', 'val_ratio', 'test_ratio'],
            '日志保存': ['save_checkpoints', 'checkpoint_freq', 'verbose']
        }
        
        for category, params in categories.items():
            config_str += f"\n【{category}】\n"
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    if isinstance(value, torch.device):
                        value = str(value)
                    config_str += f"  {param}: {value}\n"
        
        return config_str

# ==================== 2. 工厂模式创建训练组件 ====================
class TrainingComponentFactory:
    """训练组件工厂 - 根据配置创建各种组件"""
    
    @staticmethod
    def create_loss_function(config: TrainingConfig) -> nn.Module:
        """创建损失函数"""
        loss_type = config.loss_function.lower()
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss(delta=config.huber_delta)
        else:
            raise ValueError(f"不支持的损失函数: {loss_type}")
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = config.optimizer.lower()
        
        if optimizer_type == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.sgd_momentum,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")
    
    @staticmethod
    def create_lr_scheduler(optimizer: optim.Optimizer, config: TrainingConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if not config.use_lr_scheduler:
            return None
        
        scheduler_type = config.lr_scheduler_type.lower()
        
        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.lr_factor,
                patience=config.lr_patience,
                min_lr=config.lr_min,
                verbose=config.verbose
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=20,
                gamma=0.1
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")

# ==================== 3. 智能训练管理器 ====================
class SmartTrainingManager:
    """智能训练管理器 - 自动管理训练流程"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig = None):
        """初始化训练管理器"""
        self.model = model
        self.config = config if config else TrainingConfig()
        
        # 创建所有训练组件
        self._create_components()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stop_counter = 0
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'train_time': [],
            'val_time': []
        }
        
        if self.config.verbose:
            print("✅ 智能训练管理器初始化完成")
            print(self.config)
    
    def _create_components(self):
        """创建所有训练组件"""
        factory = TrainingComponentFactory()
        
        # 损失函数
        self.criterion = factory.create_loss_function(self.config)
        
        # 优化器
        self.optimizer = factory.create_optimizer(self.model, self.config)
        
        # 学习率调度器
        self.lr_scheduler = factory.create_lr_scheduler(self.optimizer, self.config)
        
        # 移动模型到设备
        self.model.to(self.config.device)
        
        if self.config.verbose:
            print("\n📦 训练组件详情:")
            print(f"   损失函数: {self.config.loss_function.upper()}")
            print(f"   优化器: {self.config.optimizer.upper()}")
            if self.lr_scheduler:
                print(f"   学习率调度器: {self.config.lr_scheduler_type.upper()}")
            else:
                print(f"   学习率调度器: 无")
            print(f"   设备: {self.config.device}")
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        epoch_start_time = time.time()
        
        for batch_idx, (capsense, imu, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # 移动到设备
            capsense = capsense.to(self.config.device)
            imu = imu.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(capsense, imu)
            loss = self.criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.use_gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 进度显示
            if self.config.verbose and batch_idx % 10 == 0:
                batch_time = time.time() - batch_start_time
                print(f"   批次 {batch_idx:4d}/{len(train_loader)} | "
                      f"损失: {loss.item():.6f} | "
                      f"时间: {batch_time:.3f}s")
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        return avg_loss, epoch_time
    
    def validate(self, val_loader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        val_start_time = time.time()
        
        with torch.no_grad():
            for capsense, imu, labels in val_loader:
                capsense = capsense.to(self.config.device)
                imu = imu.to(self.config.device)
                labels = labels.to(self.config.device)
                
                predictions = self.model(capsense, imu)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                batch_count += 1
        
        val_time = time.time() - val_start_time
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        return avg_loss, val_time
    
    def fit(self, train_loader, val_loader) -> Dict:
        """完整的训练流程"""
        print("🚀 开始训练...")
        print(f"训练轮数: {self.config.epochs}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        print("-" * 60)
        
        total_start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch + 1
            
            if self.config.verbose:
                print(f"\n📊 Epoch {self.current_epoch}/{self.config.epochs}")
                print("-" * 40)
            
            # 训练
            train_loss, train_time = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_time = self.validate(val_loader)
            
            # 更新学习率
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 保存最佳模型
            if val_loss < self.best_val_loss - self.config.early_stop_delta:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.early_stop_counter = 0
                improved = True
            else:
                self.early_stop_counter += 1
                improved = False
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(train_time + val_time)
            self.history['train_time'].append(train_time)
            self.history['val_time'].append(val_time)
            
            # 打印信息
            if self.config.verbose:
                print(f"  训练损失: {train_loss:.6f} ({train_time:.2f}s)")
                print(f"  验证损失: {val_loss:.6f} ({val_time:.2f}s)")
                print(f"  学习率: {current_lr:.6f}")
                print(f"  最佳验证损失: {self.best_val_loss:.6f}")
                if improved:
                    print("  🎯 模型改进!")
                else:
                    print(f"  ⏳ 无改进计数: {self.early_stop_counter}/{self.config.early_stop_patience}")
            
            # 保存检查点
            if (self.config.save_checkpoints and 
                self.current_epoch % self.config.checkpoint_freq == 0):
                checkpoint_path = f"checkpoint_epoch_{self.current_epoch}.pth"
                self.save_checkpoint(checkpoint_path)
            
            # 早停检查
            if (self.config.use_early_stopping and 
                self.early_stop_counter >= self.config.early_stop_patience):
                if self.config.verbose:
                    print(f"\n⚠️ 早停触发! 连续{self.early_stop_counter}轮无改进")
                break
        
        total_time = time.time() - total_start_time
        
        # 加载最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("✅ 训练完成!")
            print(f"   总训练时间: {total_time:.2f}s ({total_time/60:.2f}分钟)")
            print(f"   最佳验证损失: {self.best_val_loss:.6f}")
            print(f"   最终验证损失: {self.history['val_loss'][-1]:.6f}")
            print(f"   总训练轮数: {self.current_epoch}")
            print("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, filepath)
        
        if self.config.verbose:
            print(f"💾 检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        # 更新配置
        for key, value in checkpoint['config'].items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        if self.config.verbose:
            print(f"📂 检查点已加载: {filepath}")
            print(f"   从第 {self.current_epoch} 轮继续训练")

# ==================== 4. 评估工具 ====================
class ModelEvaluator:
    """模型评估工具"""
    
    @staticmethod
    def evaluate(model: nn.Module, test_loader, device) -> Dict:
        """全面评估模型"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        eval_start_time = time.time()
        
        with torch.no_grad():
            for capsense, imu, labels in test_loader:
                capsense = capsense.to(device)
                imu = imu.to(device)
                labels = labels.to(device)
                
                predictions = model(capsense, imu)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        eval_time = time.time() - eval_start_time
        
        # 合并结果
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # 计算各种指标
        metrics = ModelEvaluator.calculate_all_metrics(predictions, labels)
        
        print(f"✅ 评估完成 ({eval_time:.2f}s)")
        print(f"   样本数量: {len(predictions)}")
        
        return {
            'predictions': predictions,
            'labels': labels,
            'metrics': metrics,
            'eval_time': eval_time
        }
    
    @staticmethod
    def calculate_all_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict:
        """计算所有评估指标"""
        
        def safe_divide(a, b):
            return a / b if b != 0 else 0
        
        # MSE
        mse = nn.MSELoss()(predictions, labels).item()
        
        # MAE
        mae = nn.L1Loss()(predictions, labels).item()
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # NRMSE (论文主要指标)
        label_range = labels.max() - labels.min()
        nrmse = safe_divide(rmse, label_range) * 100
        
        # R²
        ss_res = torch.sum((labels - predictions) ** 2)
        ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
        r2 = 1 - safe_divide(ss_res, ss_tot)
        
        # 相关系数
        corr_coeffs = []
        for i in range(predictions.shape[1]):
            pred_i = predictions[:, i]
            label_i = labels[:, i]
            corr = np.corrcoef(pred_i.numpy(), label_i.numpy())[0, 1]
            corr_coeffs.append(corr)
        
        # 平均相关系数
        avg_corr = np.mean(corr_coeffs) if corr_coeffs else 0
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'NRMSE (%)': nrmse,
            'R²': r2.item(),
            'Correlation (Fx,Fy,Fz)': corr_coeffs,
            'Avg Correlation': avg_corr
        }
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """格式化打印评估指标"""
        print("\n📊 评估结果:")
        print("-" * 40)
        print(f"  MSE:          {metrics['MSE']:.6f}")
        print(f"  MAE:          {metrics['MAE']:.6f}")
        print(f"  RMSE:         {metrics['RMSE']:.6f}")
        print(f"  NRMSE:        {metrics['NRMSE (%)']:.2f}%")
        print(f"  R²:           {metrics['R²']:.4f}")
        print(f"  平均相关系数:  {metrics['Avg Correlation']:.4f}")
        print(f"  分量相关系数:  Fx={metrics['Correlation (Fx,Fy,Fz)'][0]:.4f}, "
              f"Fy={metrics['Correlation (Fx,Fy,Fz)'][1]:.4f}, "
              f"Fz={metrics['Correlation (Fx,Fy,Fz)'][2]:.4f}")

# ==================== 5. 使用示例 ====================
def create_default_config() -> TrainingConfig:
    """创建默认配置（论文参数）"""
    config = TrainingConfig(
        # 论文使用的参数
        learning_rate=0.0001,  # 论文: 0.0001
        weight_decay=1e-8,     # 论文: 1e-8
        lr_scheduler_type='plateau',  # 论文: ReduceLROnPlateau
        lr_patience=3,         # 论文: 3
        lr_factor=0.1,         # 论文: 0.1
        
        # 训练参数
        batch_size=64,
        epochs=100,
        
        # 早停
        use_early_stopping=True,
        early_stop_patience=10,
        
        # 其他
        loss_function='mse',
        optimizer='adam',
        verbose=True
    )
    return config

def create_fast_config() -> TrainingConfig:
    """创建快速测试配置"""
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        epochs=5,
        use_lr_scheduler=False,
        use_early_stopping=False,
        verbose=True
    )
    return config

# ==================== 6. 测试代码 ====================
if __name__ == "__main__":
    print("🧪 测试 LearnMethod.py 模块")
    print("=" * 60)
    
    # 测试配置管理
    print("1. 测试配置管理...")
    config = TrainingConfig(
        batch_size=128,
        learning_rate=0.01,
        epochs=50
    )
    print(config)
    
    # 测试动态更新
    config.update(loss_function='huber', huber_delta=0.5)
    print("\n更新后的配置:")
    print(config)
    
    # 测试配置字典
    print("\n配置字典:")
    print(config.to_dict())
    
    # 测试预设配置
    print("\n2. 测试预设配置...")
    paper_config = create_default_config()
    print("论文配置:")
    print(paper_config)
    
    fast_config = create_fast_config()
    print("\n快速配置:")
    print(fast_config)
    
    print("\n✅ LearnMethod.py 模块测试完成！")