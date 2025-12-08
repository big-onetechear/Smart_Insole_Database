# model_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PaperCrossAttention(nn.Module):
    """论文中的交叉注意力机制"""
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim) 
        self.v_proj = nn.Linear(dim, dim)
    
    def forward(self, query, key, value):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # 注意力计算
        attention = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5), 
            dim=-1
        )
        out = torch.matmul(attention, V)
        return out


class DenseBlock(nn.Module):
    """论文中的Dense Block - 真正的密集连接"""
    def __init__(self, input_dim, output_dim, growth_rate=32, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # 第一个层：输入到growth_rate
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, growth_rate),
            nn.ReLU()
        )
        
        # 中间层：每个层的输入是前面所有层的输出拼接
        self.layers = nn.ModuleList()
        for i in range(1, num_layers):
            input_size = input_dim + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_size, growth_rate),
                    nn.ReLU()
                )
            )
        
        # 输出层：将所有特征映射到output_dim
        total_features = input_dim + num_layers * growth_rate
        self.output_layer = nn.Linear(total_features, output_dim)
    
    def forward(self, x):
        features = [x]
        
        # 第一层
        out = self.first_layer(x)
        features.append(out)
        
        # 中间层
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        # 输出层
        final_output = self.output_layer(torch.cat(features, dim=1))
        return final_output


class DualStreamAttentionModel(nn.Module):
    """完整的双流交叉注意力模型"""
    def __init__(self, cap_dim=12, imu_dim=5, hidden_dim=32, output_dim=3):
        super().__init__()
        
        # Dense Block 1: CapSense流 (12 -> 32)
        self.cap_dense = DenseBlock(
            input_dim=cap_dim, 
            output_dim=hidden_dim,
            growth_rate=32,
            num_layers=3
        )
        
        # Dense Block 2: IMU流 (5 -> 32)
        self.imu_dense = DenseBlock(
            input_dim=imu_dim,
            output_dim=hidden_dim,
            growth_rate=32,
            num_layers=3
        )
        
        # 交叉注意力机制
        self.cap_to_imu_attention = PaperCrossAttention(dim=hidden_dim)
        self.imu_to_cap_attention = PaperCrossAttention(dim=hidden_dim)
        
        # Dense Block 3: 特征融合 (64 -> 3)
        self.fusion_dense = DenseBlock(
            input_dim=hidden_dim * 2,  # 两个注意力输出拼接
            output_dim=output_dim,
            growth_rate=32,
            num_layers=3
        )
        
    def forward(self, cap_features, imu_features):
        # 1. 分别处理两个流
        cap_encoded = self.cap_dense(cap_features)  # [batch, 32]
        imu_encoded = self.imu_dense(imu_features)  # [batch, 32]
        
        # 2. 交叉注意力
        # CapSense作为Query，IMU作为Key/Value
        cap_attended = self.cap_to_imu_attention(
            query=cap_encoded,
            key=imu_encoded,
            value=imu_encoded
        )
        
        # IMU作为Query，CapSense作为Key/Value
        imu_attended = self.imu_to_cap_attention(
            query=imu_encoded,
            key=cap_encoded,
            value=cap_encoded
        )
        
        # 3. 特征拼接
        fused_features = torch.cat([cap_attended, imu_attended], dim=1)  # [batch, 64]
        
        # 4. 最终回归
        output = self.fusion_dense(fused_features)  # [batch, 3]
        
        return output


# 测试模型
def test_model():
    """测试模型是否能正常工作"""
    batch_size = 32
    
    # 创建模拟数据
    cap_features = torch.randn(batch_size, 12)  # CapSense数据
    imu_features = torch.randn(batch_size, 5)   # IMU数据
    
    # 初始化模型
    model = DualStreamAttentionModel()
    
    # 前向传播
    output = model(cap_features, imu_features)
    
    print("✅ 模型测试通过!")
    print(f"输入形状: Cap={cap_features.shape}, IMU={imu_features.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    return model


if __name__ == "__main__":
    model = test_model()
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")