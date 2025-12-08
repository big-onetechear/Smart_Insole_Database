# check_all_normalized.py

import glob
import json
from pathlib import Path
# check_normalized_data.py
import pandas as pd
import numpy as np
import os
from pathlib import Path

def validate_normalized_data(data_path):
    """验证归一化数据是否符合模型要求"""
    
    print("="*60)
    print("🔍 归一化数据验证")
    print("="*60)
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    print(f"📊 数据基本信息:")
    print(f"  文件: {Path(data_path).name}")
    print(f"  样本数: {len(df):,}")
    print(f"  特征数: {len(df.columns)}")
    
    # 1. 检查数据范围 [0, 1]
    print(f"\n📈 数据范围检查:")
    violations = []
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # 允许微小误差
        if col_min < -0.05 or col_max > 1.05:
            violations.append((col, col_min, col_max))
        else:
            print(f"  ✓ {col}: [{col_min:.3f}, {col_max:.3f}]")
    
    if violations:
        print(f"  ⚠️  以下列超出[0,1]范围:")
        for col, cmin, cmax in violations:
            print(f"    {col}: [{cmin:.3f}, {cmax:.3f}]")
    
    # 2. 检查缺失值
    print(f"\n❓ 缺失值检查:")
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  ⚠️  发现 {missing_count} 个缺失值")
        missing_cols = df.columns[df.isnull().any()].tolist()
        print(f"    有缺失的列: {missing_cols}")
    else:
        print(f"  ✓ 无缺失值")
    
    # 3. 检查特征和标签
    print(f"\n🏷️  特征/标签识别:")
    
    # 根据你的列名模式自动识别
    feature_cols = []
    label_cols = []
    
    for col in df.columns:
        if col.startswith('C') and col[1:].isdigit():  # C0-C11
            feature_cols.append(col)
        elif col in ['Ax', 'Ay', 'Az', 'Gr', 'Gp']:
            feature_cols.append(col)
        elif '_norm' in col:
            label_cols.append(col)
    
    print(f"  特征列 ({len(feature_cols)}个): {feature_cols}")
    print(f"  标签列 ({len(label_cols)}个): {label_cols}")
    
    # 4. 检查数据分布
    print(f"\n📊 数据分布统计:")
    print("  特征统计:")
    for col in feature_cols[:5]:  # 只显示前5个特征
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"    {col}: 均值={mean_val:.3f}, 标准差={std_val:.3f}")
    
    # 5. 保存验证报告
    report = {
        "file": str(data_path),
        "samples": len(df),
        "features": len(feature_cols),
        "labels": len(label_cols),
        "range_violations": len(violations),
        "missing_values": missing_count,
        "feature_columns": feature_cols,
        "label_columns": label_cols,
        "feature_stats": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
            for col in feature_cols[:10]  # 只保存前10个特征的详细统计
        }
    }
    
    return report

def check_all_normalized_files(base_path="Smart_Insole_Database"):
    """检查所有归一化文件"""
    
    all_reports = {}
    issues = []
    
    # 查找所有归一化文件
    pattern = f"{base_path}/subjectRepro*/norm/*_normalized.csv"
    normalized_files = glob.glob(pattern)
    
   
    print(f"找到 {len(normalized_files)} 个归一化文件")
    print("\n📂 找到的文件列表:")
    for i, file_path in enumerate(sorted(normalized_files), 1):
        file_name = Path(file_path).name
        folder = Path(file_path).parent.name  # norm文件夹
        subject_folder = Path(file_path).parent.parent.name  # subjectReproX文件夹
        
        print(f"  {i:2d}. {subject_folder}/{folder}/{file_name}")
    for file_path in normalized_files:
        print(f"\n{'='*60}")
        print(f"检查: {Path(file_path).name}")
        
        try:
            report = validate_normalized_data(file_path)
            all_reports[file_path] = report
            
            # 记录问题
            if report['range_violations'] > 0 or report['missing_values'] > 0:
                issues.append({
                    'file': file_path,
                    'range_violations': report['range_violations'],
                    'missing_values': report['missing_values']
                })
                
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            issues.append({
                'file': file_path,
                'error': str(e)
            })
    
    # 生成总结报告
    print(f"\n{'='*60}")
    print("📊 总体检查报告")
    print(f"{'='*60}")
    
    total_files = len(normalized_files)
    files_with_issues = len(issues)
    
    print(f"总文件数: {total_files}")
    print(f"有问题的文件: {files_with_issues}")
    print(f"通过率: {(total_files - files_with_issues)/total_files*100:.1f}%")
    
    if issues:
        print(f"\n❌ 问题文件:")
        for issue in issues:
            print(f"  • {Path(issue['file']).name}")
            if 'error' in issue:
                print(f"    错误: {issue['error']}")
            else:
                if issue['range_violations'] > 0:
                    print(f"    范围违规: {issue['range_violations']}列")
                if issue['missing_values'] > 0:
                    print(f"    缺失值: {issue['missing_values']}个")
    
    # 保存详细报告
    report_file = "normalized_data_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    print(f"\n📄 详细报告已保存到: {report_file}")
    
    return all_reports, issues

# 在文件末尾添加
if __name__ == "__main__":
    # 检查所有文件
    reports, issues = check_all_normalized_files()
    print("检查完成！")