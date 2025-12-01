import torch
import torch.nn as nn
# modelArch.py
class PaperCrossAttention(nn.Module):
    """论文中的交叉注意力机制"""

    # 输入 → 线性投影 → Q/K/V → 注意力计算 → 输出
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
        attention = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5), dim=-1)
        out = torch.matmul(attention, V)
        return out
    
# 输入 → 特征拼接 → 线性层 →  ReLU(激活函数）→ 特征累积 → 最终拼接
class DenseBlock(nn.Module):
    """论文中的Dense Block"""
    def __init__(self, input_dim, growth_rate=32, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim + i * growth_rate, growth_rate),
                nn.ReLU()
            ) for i in range(num_layers)
        ])
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)