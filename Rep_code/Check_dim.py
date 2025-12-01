# 快速检查维度
# check_dimensions.py
import torch
import modelAssemb

print("🔍 检查模型维度一致性")
print("=" * 60)

model = modelAssemb.PaperFusionModel()

# 获取模型配置
print("模型配置:")
print(f"  cap_input_channels: {model.cap_input_channels}")
print(f"  imu_input_channels: {model.imu_input_channels}")
print(f"  cap_dense_growth_rate: {model.cap_dense_growth_rate}")
print(f"  imu_dense_growth_rate: {model.imu_dense_growth_rate}")
print(f"  cap_dense_num_layers: {model.cap_dense_num_layers}")
print(f"  imu_dense_num_layers: {model.imu_dense_num_layers}")
print(f"  cross_attention_dim: {model.cross_attention_dim}")

# 计算输出维度
cap_output = model.cap_input_channels + model.cap_dense_growth_rate * model.cap_dense_num_layers
imu_output = model.imu_input_channels + model.imu_dense_growth_rate * model.imu_dense_num_layers

print(f"\n维度计算:")
print(f"  CapSense DenseBlock输出: {cap_output}")
print(f"  IMU DenseBlock输出: {imu_output}")

print(f"\n维度匹配检查:")
print(f"  CapSense输出 == CrossAttention输入: {cap_output == model.cross_attention_dim}")
print(f"  IMU输出 == CrossAttention输入: {imu_output == model.cross_attention_dim}")

# 测试前向传播
print(f"\n🧪 测试前向传播...")
test_cap = torch.randn(4, model.cap_input_channels)
test_imu = torch.randn(4, model.imu_input_channels)

try:
    output = model(test_cap, test_imu)
    print(f"✅ 前向传播成功!")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: {output.min():.6f} ~ {output.max():.6f}")
    
    # 检查中间层
    cap_features = model.cap_dense(test_cap)
    imu_features = model.imu_dense(test_imu)
    
    print(f"\n中间层检查:")
    print(f"  cap_dense输出: {cap_features.shape} | 范围: {cap_features.min():.6f}~{cap_features.max():.6f}")
    print(f"  imu_dense输出: {imu_features.shape} | 范围: {imu_features.min():.6f}~{imu_features.max():.6f}")
    
    # 检查是否有适配层
    if hasattr(model, 'imu_adapter'):
        imu_features = model.imu_adapter(imu_features)
        print(f"  imu_adapter输出: {imu_features.shape} | 范围: {imu_features.min():.6f}~{imu_features.max():.6f}")
    
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()