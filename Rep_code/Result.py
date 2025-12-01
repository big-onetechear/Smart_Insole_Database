import torch
import matplotlib.pyplot as plt
import os
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)  # 切换到项目根目录
# 设置英文字体，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载结果文件
results = torch.load('training_results_20251201_201559/results_subject_1_jogging_20251201_201658.pkl')

# ===== 新增：从results中提取history =====
history = results['history']  # 这行必须加上！

print("=" * 50)
print("Training Results Summary")
print("=" * 50)

# 基本信息
subject = results['subject']
movement = results['movement']
print(f"Subject: {subject}")
print(f"Movement: {movement}")
print(f"Training Time: {results['training_time']:.2f} seconds")
print(f"Model Parameters: {results['model_params']:,}")

# 评估结果
print("\n📊 Test Set Evaluation Results:")
eval_results = results['eval_results']['metrics']
for key, value in eval_results.items():
    if isinstance(value, list):
        if key in ['predictions', 'targets']:
            # 太长，只显示摘要
            print(f"  {key}: list of {len(value)} elements")
            print(f"    Mean: {np.mean(value):.4f}, Std: {np.std(value):.4f}")
        else:
            print(f"  {key}: {[f'{v:.4f}' for v in value]}")
    else:
        print(f"  {key}: {value:.6f}")

# ============ 创建Visual文件夹 ============
visual_dir = "D:/TG0/PublicData_Rep/Smart_Insole_Database/Visual"
os.makedirs(visual_dir, exist_ok=True)
print(f"\n📁 结果将保存到: {visual_dir}")

# 显示训练历史图表
fig = plt.figure(figsize=(16, 5))

# 主标题
fig.suptitle(f'{subject} - {movement}', fontsize=18, fontweight='bold', y=1.05)

# 子图1: 训练和验证损失
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
# 添加最佳epoch标记
best_epoch = np.argmin(history['val_loss'])
plt.scatter(best_epoch, history['val_loss'][best_epoch], color='red', s=100, zorder=5, label=f'Best (Epoch {best_epoch+1})')
plt.legend()

# 子图2: 学习率变化
plt.subplot(1, 3, 2)
plt.plot(history['learning_rate'], color='darkgreen', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 对数尺度

# 子图3: 预测 vs 真实值
plt.subplot(1, 3, 3)
# 检查是否有预测数据
if 'predictions' in eval_results and 'targets' in eval_results:
    predictions = eval_results['predictions']
    targets = eval_results['targets']
    
    # 如果是3D输出，只显示第一个维度
    if len(predictions.shape) > 1 and predictions.shape[1] == 3:
        pred = predictions[:100, 0]  # 只显示Fx
        tar = targets[:100, 0]
        ylabel = 'Force (Fx)'
    else:
        pred = predictions[:100]
        tar = targets[:100]
        ylabel = 'Force'
    
    plt.plot(pred, label='Predictions', alpha=0.8, linewidth=1.5)
    plt.plot(tar, label='Ground Truth', alpha=0.8, linewidth=1.5)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title('Predictions vs Ground Truth\n(First 100 Samples)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
else:
    # 如果没有预测数据，显示训练时间
    train_times = history.get('train_time', [])
    val_times = history.get('val_time', [])
    if train_times and val_times:
        plt.plot(train_times, label='Train Time', linewidth=2)
        plt.plot(val_times, label='Val Time', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Time (s)', fontsize=12)
        plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

plt.tight_layout()

# ============ 保存图片 ============
# 生成文件名：subject_1_jogging_results.png
filename = f"{subject}_{movement}_results.png"
save_path = os.path.join(visual_dir, filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"📸 结果图已保存: {save_path}")

plt.show()

# 额外：打印性能总结
print("\n" + "=" * 50)
print("Performance Summary")
print("=" * 50)
print(f"Best Val Loss: {min(history['val_loss']):.6f}")
print(f"Final Val Loss: {history['val_loss'][-1]:.6f}")
print(f"R² Score: {eval_results.get('r2', 0):.4f}")
print(f"RMSE: {eval_results.get('rmse', 0):.4f}")
print(f"Training Epochs: {len(history['train_loss'])}")
print(f"Total Training Time: {results['training_time']:.2f}s")

# 保存文本报告
report_filename = f"{subject}_{movement}_report.txt"
report_path = os.path.join(visual_dir, report_filename)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 50 + "\n")
    f.write(f"Training Report: {subject} - {movement}\n")
    f.write("=" * 50 + "\n\n")
    
    f.write(f"Subject: {subject}\n")
    f.write(f"Movement: {movement}\n")
    f.write(f"Training Time: {results['training_time']:.2f} seconds\n")
    f.write(f"Model Parameters: {results['model_params']:,}\n\n")
    
    f.write("Performance Metrics:\n")
    f.write(f"  Best Val Loss: {min(history['val_loss']):.6f}\n")
    f.write(f"  Final Val Loss: {history['val_loss'][-1]:.6f}\n")
    f.write(f"  R² Score: {eval_results.get('r2', 0):.4f}\n")
    f.write(f"  RMSE: {eval_results.get('rmse', 0):.4f}\n")
    f.write(f"  MSE: {eval_results.get('mse', 0):.6f}\n")
    f.write(f"  MAE: {eval_results.get('mae', 0):.6f}\n")
    
    f.write("\nCorrelation Coefficients:\n")
    if 'corr_coefs' in eval_results:
        corr = eval_results['corr_coefs']
        if isinstance(corr, list) and len(corr) == 3:
            f.write(f"  Fx: {corr[0]:.4f}\n")
            f.write(f"  Fy: {corr[1]:.4f}\n")
            f.write(f"  Fz: {corr[2]:.4f}\n")

print(f"📝 文本报告已保存: {report_path}")