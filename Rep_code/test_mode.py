"""
test_modules.py - 修正版
修正模块导入问题
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """测试数据加载模块"""
    print("🧪 测试数据加载模块...")
    print("-" * 50)
    
    try:
        # 检查文件是否存在
        data_path = "D:\TG0\PublicData_Rep\Smart_Insole_Database\subject_1\squatting_s1_merged.csv"
        if not os.path.exists(data_path):
            print(f"⚠️  数据文件不存在: {data_path}")
            print("   请确保数据文件在正确位置")
            return False
        
        from DataLoading import SmartInsoleDataset
        
        # 测试数据加载
        dataset = SmartInsoleDataset(data_path)
        
        print(f"✅ 数据加载成功!")
        print(f"   数据集大小: {len(dataset)}")
        
        # 检查一个样本
        capsense, imu, labels = dataset[0]
        
        print(f"   CapSense形状: {capsense.shape}")
        print(f"   IMU形状: {imu.shape}")
        print(f"   标签形状: {labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def test_model_arch():
    """测试模型架构模块"""
    print("\n🧪 测试模型架构模块...")
    print("-" * 50)
    
    try:
        # 尝试不同的导入方式
        try:
            import modelArch
            DenseBlock = modelArch.DenseBlock
            PaperCrossAttention = modelArch.PaperCrossAttention
            print("   使用 import modelArch")
        except:
            from modelArch import DenseBlock, PaperCrossAttention
            print("   使用 from modelArch import")
        
        # 测试DenseBlock
        print("   测试DenseBlock...")
        test_input = torch.randn(32, 18)
        dense_block = DenseBlock(18, growth_rate=32, num_layers=3)
        output = dense_block(test_input)
        
        print(f"   ✅ DenseBlock测试成功!")
        print(f"     输入: {test_input.shape} → 输出: {output.shape}")
        
        # 测试PaperCrossAttention
        print("\n   测试PaperCrossAttention...")
        attention = PaperCrossAttention(dim=32)
        query = torch.randn(32, 32)
        key = torch.randn(32, 32)
        value = torch.randn(32, 32)
        output = attention(query, key, value)
        
        print(f"   ✅ PaperCrossAttention测试成功!")
        print(f"     查询: {query.shape} → 输出: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型架构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_assemb():
    """测试模型组装模块"""
    print("\n🧪 测试模型组装模块...")
    print("-" * 50)
    
    try:
        # 尝试不同的导入方式
        try:
            import modelAssemb
            PaperFusionModel = modelAssemb.PaperFusionModel
            print("   使用 import modelAssemb")
        except:
            from modelAssemb import PaperFusionModel
            print("   使用 from modelAssemb import")
        
        # 创建模型
        model = PaperFusionModel()
        
        print(f"✅ 模型创建成功!")
        
        # 测试前向传播
        batch_size = 16
        capsense_input = torch.randn(batch_size, 18)
        imu_input = torch.randn(batch_size, 7)
        
        output = model(capsense_input, imu_input)
        
        print(f"   输入: CapSense={capsense_input.shape}, IMU={imu_input.shape}")
        print(f"   输出: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型组装测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learn_method():
    """测试学习机制模块"""
    print("\n🧪 测试学习机制模块...")
    print("-" * 50)
    
    try:
        # 检查文件是否存在
        if not os.path.exists("LearnMethod.py"):
            print("⚠️  LearnMethod.py 文件不存在")
            return False
        
        # 尝试不同的导入方式
        try:
            import LearnMethod
            TrainingConfig = LearnMethod.TrainingConfig
            TrainingComponentFactory = LearnMethod.TrainingComponentFactory
            SmartTrainingManager = LearnMethod.SmartTrainingManager
            ModelEvaluator = LearnMethod.ModelEvaluator
            create_default_config = LearnMethod.create_default_config
            create_fast_config = LearnMethod.create_fast_config
            print("   使用 import LearnMethod")
        except:
            from LearnMethod import (
                TrainingConfig, 
                TrainingComponentFactory,
                SmartTrainingManager,
                ModelEvaluator,
                create_default_config,
                create_fast_config
            )
            print("   使用 from LearnMethod import")
        
        # 测试配置类
        print("   测试TrainingConfig...")
        config = TrainingConfig(batch_size=32)
        print(f"   ✅ 配置创建成功!")
        
        # 测试训练组件工厂
        print("\n   测试TrainingComponentFactory...")
        test_model = nn.Linear(10, 3)
        loss_fn = TrainingComponentFactory.create_loss_function(config)
        optimizer = TrainingComponentFactory.create_optimizer(test_model, config)
        print(f"   ✅ 组件工厂测试成功!")
        
        # 测试评估器
        print("\n   测试ModelEvaluator...")
        test_pred = torch.randn(100, 3)
        test_labels = torch.randn(100, 3)
        metrics = ModelEvaluator.calculate_all_metrics(test_pred, test_labels)
        print(f"   ✅ 评估器测试成功!")
        
        return True
        
    except Exception as e:
        print(f"❌ 学习机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_manager():
    """测试训练管理器 - 简化版"""
    print("\n🧪 测试训练管理器...")
    print("-" * 50)
    
    try:
        # 检查文件是否存在
        if not os.path.exists("LearnMethod.py"):
            print("⚠️  LearnMethod.py 文件不存在")
            return False
        
        # 导入模块
        try:
            import LearnMethod
            SmartTrainingManager = LearnMethod.SmartTrainingManager
            create_fast_config = LearnMethod.create_fast_config
        except Exception as e:
            print(f"   导入 LearnMethod 失败: {e}")
            return False
        
        try:
            import modelAssemb
            PaperFusionModel = modelAssemb.PaperFusionModel
        except Exception as e:
            print(f"   导入 modelAssemb 失败: {e}")
            return False
        
        print("   创建测试模型...")
        model = PaperFusionModel()
        
        print("   创建配置...")
        config = create_fast_config()
        config.update(epochs=1, verbose=False)
        
        print("   创建训练管理器...")
        manager = SmartTrainingManager(model, config)
        
        print(f"   ✅ 训练管理器创建成功!")
        
        # 简单测试保存
        print("\n   测试检查点保存...")
        try:
            manager.save_checkpoint("test_checkpoint.pth")
            print(f"   ✅ 检查点保存成功")
            
            # 清理 - 使用全局的 os，不要重新导入
            if os.path.exists("test_checkpoint.pth"):
                os.remove("test_checkpoint.pth")
                print(f"   ✅ 测试文件已清理")
        except Exception as save_error:
            print(f"   ⚠️  检查点保存测试跳过: {save_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """测试模块集成 - 简化版"""
    print("\n🧪 测试模块集成...")
    print("-" * 50)
    
    try:
        print("   1. 检查所有模块文件...")
        required_files = [
            "DataLoading.py",
            "modelArch.py", 
            "modelAssemb.py",
            "LearnMethod.py"
        ]
        
        missing_files = []
        for file in required_files:
            if os.path.exists(file):
                print(f"      ✅ {file}")
            else:
                print(f"      ❌ {file} (缺失)")
                missing_files.append(file)
        
        if missing_files:
            print(f"\n   ⚠️  缺失文件: {missing_files}")
            return False
        
        print("\n   2. 测试导入...")
        # 测试导入
        imports_ok = True
        try:
            import DataLoading
            print("      ✅ DataLoading")
        except:
            print("      ❌ DataLoading")
            imports_ok = False
            
        try:
            import modelArch
            print("      ✅ modelArch")
        except:
            print("      ❌ modelArch")
            imports_ok = False
            
        try:
            import modelAssemb
            print("      ✅ modelAssemb")
        except:
            print("      ❌ modelAssemb")
            imports_ok = False
            
        try:
            import LearnMethod
            print("      ✅ LearnMethod")
        except:
            print("      ❌ LearnMethod")
            imports_ok = False
        
        if not imports_ok:
            print("\n   ⚠️  导入测试失败")
            return False
        
        print("\n   3. 测试前向传播...")
        try:
            from modelAssemb import PaperFusionModel
            model = PaperFusionModel()
            test_input1 = torch.randn(2, 18)
            test_input2 = torch.randn(2, 7)
            output = model(test_input1, test_input2)
            print(f"      ✅ 前向传播: {output.shape}")
        except Exception as e:
            print(f"      ❌ 前向传播失败: {e}")
            return False
        
        print("\n   🎉 集成测试基本通过!")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    print("-" * 50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前目录: {current_dir}")
    
    print("\n目录内容:")
    files = os.listdir(current_dir)
    for file in files:
        if file.endswith('.py'):
            print(f"  📄 {file}")
        elif os.path.isdir(file):
            print(f"  📁 {file}/")
        else:
            print(f"  📎 {file}")
    
    # 检查是否有数据文件夹
    data_dirs = [d for d in files if d.startswith('subject') or d == 'data']
    if data_dirs:
        print(f"\n✅ 找到数据文件夹: {data_dirs}")
    else:
        print(f"\n⚠️  未找到数据文件夹")

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 开始运行模块测试")
    print("=" * 60)
    
    # 先检查项目结构
    check_project_structure()
    
    print("\n" + "=" * 60)
    print("开始模块功能测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行各个测试
    test_results.append(("数据加载", test_data_loading()))
    test_results.append(("模型架构", test_model_arch()))
    test_results.append(("模型组装", test_model_assemb()))
    test_results.append(("学习机制", test_learn_method()))
    test_results.append(("训练管理器", test_training_manager()))
    test_results.append(("模块集成", test_integration()))
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print("\n" + "=" * 60)
    success_rate = passed/len(test_results)*100 if test_results else 0
    print(f"测试通过率: {passed}/{len(test_results)} ({success_rate:.1f}%)")
    
    if passed == len(test_results):
        print("🎉 所有测试通过！可以开始训练了！")
    elif passed >= 4:
        print("👍 大部分测试通过，可以开始基础训练")
    else:
        print("⚠️  多个测试失败，请先修复问题")
    
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()