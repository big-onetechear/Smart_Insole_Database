# check_file_count.py
import glob
from pathlib import Path

base_path = r"D:\TG0\PublicData_Rep\Smart_Insole_Database"
pattern = f"{base_path}/subjectRepro*/norm/*_normalized.csv"
files = glob.glob(pattern)

print(f"📊 文件统计:")
print(f"找到 {len(files)} 个文件")
print(f"应该有: 5个subject × 6个动作 = 30个文件")
print(f"多出: {len(files) - 30} 个文件")

print("\n📁 详细列表:")
file_counts = {}
for file_path in sorted(files):
    subject = Path(file_path).parent.parent.name
    file_name = Path(file_path).name
    
    if subject not in file_counts:
        file_counts[subject] = []
    file_counts[subject].append(file_name)
    
    print(f"  • {subject}/norm/{file_name}")

print("\n📈 各subject文件数:")
for subject, files_list in sorted(file_counts.items()):
    print(f"  {subject}: {len(files_list)} 个文件")
    # 显示文件列表
    for f in sorted(files_list):
        print(f"    - {f}")

# 找出重复或异常的文件名
print("\n🔍 查找异常文件:")
all_filenames = [Path(f).name for f in files]
from collections import Counter
filename_counts = Counter(all_filenames)

for filename, count in filename_counts.items():
    if count > 1:
        print(f"⚠️  {filename} 出现 {count} 次（可能有重复）")