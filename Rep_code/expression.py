# verify_fix.py
"""
验证维度修复是否有效
"""
import torch
import modelAssemb

print("🔧 验证维度修复")
print("=" * 60)

# 创建模型
model = modelAssemb.PaperFusionModel()
print("✅ 模型创建成功")

# 检查维度
print(f"\n模型配置:")
print(f"  cap_input_channels: {model.cap_input_channels}")
print(f"  imu_input_channels: {model.imu_input_channels}")
print(f"  cap_dense_growth_rate: {model.cap_dense_growth_rate}")
print(f"  imu_dense_growth_rate: {model.imu_dense_growth_rate}")
print(f"  cap_dense_num_layers: {model.cap_dense_num_layers}")
print(f"  imu_dense_num_layers: {model.imu_dense_num_layers}")
print(f"  cross_attention_dim: {model.cross_attention_dim}")

# 计算维度
cap_output = model.cap_input_channels + model.cap_dense_growth_rate * model.cap_dense_num_layers
imu_output = model.imu_input_channels + model.imu_dense_growth_rate * model.imu_dense_num_layers

print(f"\n维度计算:")
print(f"  CapSense DenseBlock输出: {cap_output}")
print(f"  IMU DenseBlock输出: {imu_output}")
print(f"  CrossAttention期望输入: {model.cross_attention_dim}")

print(f"\n维度匹配:")
print(f"  CapSense匹配: {cap_output == model.cross_attention_dim}")
print(f"  IMU匹配: {imu_output == model.cross_attention_dim}")

# 测试前向传播
print(f"\n测试前向传播...")
batch_size = 4
capsense_input = torch.randn(batch_size, model.cap_input_channels)
imu_input = torch.randn(batch_size, model.imu_input_channels)

try:
    output = model(capsense_input, imu_input)
    print(f"✅ 前向传播成功!")
    print(f"  输入形状: CapSense={capsense_input.shape}, IMU={imu_input.shape}")
    print(f"  输出形状: {output.shape}")
    
    # 检查各层输出
    print(f"\n各层输出维度:")
    cap_features = model.cap_dense(capsense_input)
    imu_features = model.imu_dense(imu_input)
    print(f"  CapSense DenseBlock输出: {cap_features.shape}")
    print(f"  IMU DenseBlock输出: {imu_features.shape}")
    
    # 测试CrossAttention
    attended_cap = model.cross_attention1(cap_features, imu_features, imu_features)
    attended_imu = model.cross_attention2(imu_features, cap_features, cap_features)
    print(f"  CrossAttention1输出: {attended_cap.shape}")
    print(f"  CrossAttention2输出: {attended_imu.shape}")
    
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("验证完成!")
print("=" * 60)