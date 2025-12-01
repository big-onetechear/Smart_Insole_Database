import torch
import torch.nn as nn
import torch.nn.functional as F
from modelArch import DenseBlock, PaperCrossAttention
# modelAssemb.py
# 可以这样理解它的工作流程：
# 输入 → [CapSense流] → [IMU流] → [交叉注意力] → [融合] → 输出
#       ↑           ↑           ↑
#       DenseBlock  DenseBlock  交叉注意力模块



# modelAssemb.py - 完整修复版
import torch
import torch.nn as nn
from modelArch import DenseBlock, PaperCrossAttention

class PaperFusionModel(nn.Module):
    """论文中的融合模型"""
    def __init__(self):
        super().__init__()
        
        # ========== 1. 定义所有配置参数 ==========
        # 输入维度
        self.cap_input_channels = 18  # CapSense特征数
        self.imu_input_channels = 7   # IMU特征数
        
        # DenseBlock配置
        self.cap_dense_growth_rate = 32
        self.imu_dense_growth_rate = 36
        self.cap_dense_num_layers = 3
        self.imu_dense_num_layers = 3
        
        # CrossAttention配置
        self.cross_attention_dim = 114  # 18 + 32×3 = 114
        
        # ========== 2. 计算维度并验证 ==========
        # 计算DenseBlock输出维度
        cap_output = self.cap_input_channels + self.cap_dense_growth_rate * self.cap_dense_num_layers
        imu_output = self.imu_input_channels + self.imu_dense_growth_rate * self.imu_dense_num_layers
        
        print(f"[模型配置] CapSense DenseBlock输出: {cap_output}")
        print(f"[模型配置] IMU DenseBlock输出: {imu_output}")
        print(f"[模型配置] CrossAttention维度: {self.cross_attention_dim}")
        
        # ========== 3. 构建模型组件 ==========
        # DenseBlock
        self.cap_dense = DenseBlock(
            self.cap_input_channels,
            growth_rate=self.cap_dense_growth_rate,
            num_layers=self.cap_dense_num_layers
        )
        
        self.imu_dense = DenseBlock(
            self.imu_input_channels,
            growth_rate=self.imu_dense_growth_rate,
            num_layers=self.imu_dense_num_layers
        )
        
        # IMU适配层（将103维适配到114维）
        self.imu_adapter = nn.Linear(imu_output, self.cross_attention_dim)
        
        # 交叉注意力层
        self.cross_attention1 = PaperCrossAttention(dim=self.cross_attention_dim)
        self.cross_attention2 = PaperCrossAttention(dim=self.cross_attention_dim)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.cross_attention_dim * 2, 64),  # 输入：114×2=228
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 输出：3个地面反作用力
        )
    
    def forward(self, capsense, imu):
        """前向传播"""
        # 特征提取
        cap_features = self.cap_dense(capsense)      # [batch, 114]
        imu_features_raw = self.imu_dense(imu)       # [batch, 103]
        
        # 适配IMU维度
        imu_features = self.imu_adapter(imu_features_raw)  # [batch, 114]
        
        # 交叉注意力
        attended_cap = self.cross_attention1(cap_features, imu_features, imu_features)
        attended_imu = self.cross_attention2(imu_features, cap_features, cap_features)
        
        # 融合
        fused = torch.cat([attended_cap, attended_imu], dim=1)  # [batch, 228]
        
        # 回归
        output = self.fc(fused)  # [batch, 3]
        
        return output

if __name__ == "__main__":
    # 测试代码
    model = PaperFusionModel()
    print(f"✅ 模型创建成功")
    
    # 测试前向传播
    test_capsense = torch.randn(4, 18)
    test_imu = torch.randn(4, 7)
    output = model(test_capsense, test_imu)
    
    print(f"✅ 前向传播测试成功")
    print(f"输入: capsense={test_capsense.shape}, imu={test_imu.shape}")
    print(f"输出: {output.shape}")
    
    # 打印模型结构
    print(f"\n模型结构:")
    print(model)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
      