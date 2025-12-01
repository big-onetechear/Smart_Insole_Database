# diagnose_dimensions.py
"""
模型维度诊断脚本
用于检查 modelArch.py 和 modelAssemb.py 中的维度配置是否匹配
"""
import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_model_arch():
    """检查 modelArch.py 中的维度"""
    print("🔍 检查 modelArch.py 维度配置")
    print("=" * 60)
    
    try:
        # 导入 modelArch
        try:
            import modelArch
            print("✅ 成功导入 modelArch")
        except Exception as e:
            print(f"❌ 导入 modelArch 失败: {e}")
            return False
        
        # 检查 DenseBlock 类
        print("\n1. 检查 DenseBlock 类:")
        if hasattr(modelArch, 'DenseBlock'):
            print("   ✅ 找到 DenseBlock 类")
            
            # 测试维度计算
            input_channels = 18
            growth_rate = 32
            num_layers = 3
            
            # 创建 DenseBlock
            dense = modelArch.DenseBlock(input_channels, growth_rate, num_layers)
            
            # 测试前向传播
            test_input = torch.randn(4, input_channels)
            output = dense(test_input)
            
            # 计算理论输出维度
            expected_output = input_channels + growth_rate * num_layers
            actual_output = output.shape[1]
            
            print(f"   输入维度: {input_channels}")
            print(f"   growth_rate: {growth_rate}")
            print(f"   num_layers: {num_layers}")
            print(f"   理论输出维度: {expected_output}")
            print(f"   实际输出维度: {actual_output}")
            print(f"   是否匹配: {expected_output == actual_output}")
            
            if expected_output != actual_output:
                print("   ⚠️  维度不匹配！检查 DenseBlock 实现")
        else:
            print("   ❌ 未找到 DenseBlock 类")
        
        # 检查 PaperCrossAttention 类
        print("\n2. 检查 PaperCrossAttention 类:")
        if hasattr(modelArch, 'PaperCrossAttention'):
            print("   ✅ 找到 PaperCrossAttention 类")
            
            # 测试不同维度
            test_dims = [32, 64, 128]
            for dim in test_dims:
                try:
                    attention = modelArch.PaperCrossAttention(dim=dim)
                    query = torch.randn(4, dim)
                    key = torch.randn(4, dim)
                    value = torch.randn(4, dim)
                    output = attention(query, key, value)
                    
                    print(f"      dim={dim}: 输入 {query.shape} -> 输出 {output.shape}")
                except Exception as e:
                    print(f"      dim={dim}: 失败 - {e}")
        else:
            print("   ❌ 未找到 PaperCrossAttention 类")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查 modelArch.py 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_assemb():
    """检查 modelAssemb.py 中的维度"""
    print("\n🔍 检查 modelAssemb.py 维度配置")
    print("=" * 60)
    
    try:
        # 导入 modelAssemb
        try:
            import modelAssemb
            print("✅ 成功导入 modelAssemb")
        except Exception as e:
            print(f"❌ 导入 modelAssemb 失败: {e}")
            return False
        
        # 检查 PaperFusionModel 类
        print("\n1. 检查 PaperFusionModel 类:")
        if hasattr(modelAssemb, 'PaperFusionModel'):
            print("   ✅ 找到 PaperFusionModel 类")
            
            # 创建模型实例
            model = modelAssemb.PaperFusionModel()
            
            # 获取模型配置
            print("   模型配置:")
            config_attrs = [
                'cap_input_channels', 'imu_input_channels',
                'cap_dense_growth_rate', 'imu_dense_growth_rate',
                'cap_dense_num_layers', 'imu_dense_num_layers',
                'cross_attention_dim'
            ]
            
            for attr in config_attrs:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    print(f"     {attr}: {value}")
                else:
                    print(f"     {attr}: ❌ 未找到")
            
            # 计算维度
            print("\n   维度计算:")
            
            # CapSense 维度计算
            cap_input = getattr(model, 'cap_input_channels', 18)
            cap_growth = getattr(model, 'cap_dense_growth_rate', 32)
            cap_layers = getattr(model, 'cap_dense_num_layers', 3)
            cap_output = cap_input + cap_growth * cap_layers
            
            # IMU 维度计算
            imu_input = getattr(model, 'imu_input_channels', 7)
            imu_growth = getattr(model, 'imu_dense_growth_rate', 32)
            imu_layers = getattr(model, 'imu_dense_num_layers', 3)
            imu_output = imu_input + imu_growth * imu_layers
            
            # CrossAttention 维度
            cross_dim = getattr(model, 'cross_attention_dim', 32)
            
            print(f"     CapSense输入: {cap_input}")
            print(f"     CapSense DenseBlock 输出: {cap_input} + {cap_growth}×{cap_layers} = {cap_output}")
            print(f"     IMU输入: {imu_input}")
            print(f"     IMU DenseBlock 输出: {imu_input} + {imu_growth}×{imu_layers} = {imu_output}")
            print(f"     CrossAttention 输入维度: {cross_dim}")
            
            # 检查匹配
            print(f"\n   维度匹配检查:")
            print(f"     CapSense输出 == CrossAttention输入: {cap_output == cross_dim} ({cap_output} == {cross_dim})")
            print(f"     IMU输出 == CrossAttention输入: {imu_output == cross_dim} ({imu_output} == {cross_dim})")
            
            if cap_output != cross_dim or imu_output != cross_dim:
                print(f"\n   ⚠️  维度不匹配！需要调整以下参数之一:")
                print(f"       1. 调整 growth_rate")
                print(f"       2. 调整 num_layers")
                print(f"       3. 调整 cross_attention_dim")
                
                # 建议的修复方案
                print(f"\n   建议修复方案:")
                if cap_output != cross_dim:
                    needed_growth = (cross_dim - cap_input) / cap_layers
                    print(f"       CapSense: 设置 growth_rate = {needed_growth:.1f}")
                
                if imu_output != cross_dim:
                    needed_growth = (cross_dim - imu_input) / imu_layers
                    print(f"       IMU: 设置 growth_rate = {needed_growth:.1f}")
            
            # 测试前向传播
            print("\n   测试前向传播...")
            try:
                batch_size = 4
                capsense_input = torch.randn(batch_size, cap_input)
                imu_input = torch.randn(batch_size, imu_input)
                
                output = model(capsense_input, imu_input)
                print(f"   ✅ 前向传播成功!")
                print(f"      输入: CapSense={capsense_input.shape}, IMU={imu_input.shape}")
                print(f"      输出: {output.shape}")
            except Exception as e:
                print(f"   ❌ 前向传播失败: {e}")
                
        else:
            print("   ❌ 未找到 PaperFusionModel 类")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查 modelAssemb.py 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dimension_matching():
    """检查两个模块之间的维度匹配"""
    print("\n🔍 检查 modelArch 和 modelAssemb 之间的维度匹配")
    print("=" * 60)
    
    try:
        # 导入两个模块
        import modelArch
        import modelAssemb
        
        # 获取模型配置
        model = modelAssemb.PaperFusionModel()
        
        # 从 modelAssemb 获取配置
        cap_input = getattr(model, 'cap_input_channels', 18)
        imu_input = getattr(model, 'imu_input_channels', 7)
        cap_growth = getattr(model, 'cap_dense_growth_rate', 32)
        imu_growth = getattr(model, 'imu_dense_growth_rate', 32)
        cap_layers = getattr(model, 'cap_dense_num_layers', 3)
        imu_layers = getattr(model, 'imu_dense_num_layers', 3)
        cross_dim = getattr(model, 'cross_attention_dim', 32)
        
        # 创建对应的 DenseBlock
        cap_dense = modelArch.DenseBlock(cap_input, cap_growth, cap_layers)
        imu_dense = modelArch.DenseBlock(imu_input, imu_growth, imu_layers)
        
        # 创建 CrossAttention
        attention = modelArch.PaperCrossAttention(dim=cross_dim)
        
        print("   组件创建:")
        print(f"     CapSense DenseBlock: {cap_dense}")
        print(f"     IMU DenseBlock: {imu_dense}")
        print(f"     CrossAttention: {attention}")
        
        # 测试维度流
        print("\n   维度流测试:")
        
        # CapSense 路径
        cap_test = torch.randn(2, cap_input)
        cap_features = cap_dense(cap_test)
        print(f"     CapSense: {cap_input} -> DenseBlock -> {cap_features.shape[1]}")
        
        # IMU 路径
        imu_test = torch.randn(2, imu_input)
        imu_features = imu_dense(imu_test)
        print(f"     IMU: {imu_input} -> DenseBlock -> {imu_features.shape[1]}")
        
        # CrossAttention 输入
        print(f"     CrossAttention 期望输入维度: {cross_dim}")
        
        # 检查是否可以直接连接
        if cap_features.shape[1] == cross_dim and imu_features.shape[1] == cross_dim:
            print("\n   ✅ 维度完全匹配！")
            
            # 测试连接
            query = cap_features
            key = imu_features
            value = imu_features
            
            try:
                attended = attention(query, key, value)
                print(f"   ✅ CrossAttention 连接成功！输出: {attended.shape}")
                return True
            except Exception as e:
                print(f"   ❌ CrossAttention 连接失败: {e}")
                return False
        else:
            print("\n   ⚠️  维度不匹配！")
            print(f"       需要 CapSense输出={cap_features.shape[1]} == {cross_dim}")
            print(f"       需要 IMU输出={imu_features.shape[1]} == {cross_dim}")
            return False
        
    except Exception as e:
        print(f"❌ 维度匹配检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_fix_suggestions():
    """提供修复建议"""
    print("\n💡 修复建议")
    print("=" * 60)
    
    # 常见问题解决方案
    suggestions = [
        "1. 如果 DenseBlock 输出维度不等于 CrossAttention 的 dim 参数:",
        "   - 修改 modelAssemb.py 中的 cross_attention_dim 参数",
        "   - 或修改 DenseBlock 的 growth_rate 或 num_layers",
        "",
        "2. 计算公式:",
        "   DenseBlock输出 = 输入通道数 + growth_rate × num_layers",
        "   CrossAttention输入 = dim 参数",
        "",
        "3. 示例匹配配置:",
        "   对于 CapSense输入=18, IMU输入=7:",
        "   - 方案A: growth_rate=32, num_layers=3, cross_attention_dim=114",
        "   - 方案B: growth_rate=?, num_layers=3, cross_attention_dim=32",
        "     计算: growth_rate = (32 - 输入通道数) / 3",
        "           CapSense: (32-18)/3≈4.67 → 取5",
        "           IMU: (32-7)/3≈8.33 → 取8",
        "",
        "4. 快速修复:",
        "   在 modelAssemb.py 的 PaperFusionModel.__init__() 中:",
        "   self.cross_attention_dim = 114  # 匹配当前DenseBlock输出",
        "   或",
        "   self.cap_dense_growth_rate = 5",
        "   self.imu_dense_growth_rate = 8",
        "   self.cross_attention_dim = 32",
    ]
    
    for line in suggestions:
        print(line)

def main():
    """主函数"""
    print("🧪 模型维度诊断工具")
    print("=" * 60)
    print(f"工作目录: {os.getcwd()}")
    print(f"脚本目录: {os.path.dirname(os.path.abspath(__file__))}")
    print()
    
    # 运行检查
    arch_ok = check_model_arch()
    assemb_ok = check_model_assemb()
    
    if arch_ok and assemb_ok:
        match_ok = check_dimension_matching()
    
    # 提供修复建议
    get_fix_suggestions()
    
    print("\n" + "=" * 60)
    print("诊断完成！请根据上述建议修改配置文件")
    print("=" * 60)

if __name__ == "__main__":
    main()
    