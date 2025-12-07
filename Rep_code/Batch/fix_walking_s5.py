# fix_walking_s5_encoding.py
import pandas as pd
import chardet
import os

file_path = r"D:\TG0\PublicData_Rep\Smart_Insole_Database\subject_5\walking_s5_merged.csv"

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节检测编码
        result = chardet.detect(raw_data)
        return result['encoding'], result['confidence']

def fix_csv_encoding(file_path):
    """修复CSV文件编码和格式问题"""
    print(f"正在修复文件: {os.path.basename(file_path)}")
    
    # 1. 检测编码
    encoding, confidence = detect_encoding(file_path)
    print(f"检测到编码: {encoding} (置信度: {confidence:.2%})")
    
    # 2. 尝试用不同编码读取
    encodings_to_try = [
        'utf-8', 'gbk', 'gb2312', 'gb18030', 
        'latin1', 'iso-8859-1', 'cp1252'
    ]
    
    if encoding and encoding.lower() not in [e.lower() for e in encodings_to_try]:
        encodings_to_try.insert(0, encoding)
    
    df = None
    successful_encoding = None
    
    for enc in encodings_to_try:
        try:
            print(f"尝试使用 {enc} 编码读取...")
            df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
            print(f"✓ 使用 {enc} 编码成功读取")
            print(f"  数据形状: {df.shape}")
            print(f"  列数: {len(df.columns)}")
            successful_encoding = enc
            break
        except Exception as e:
            print(f"  ✗ {enc} 失败: {str(e)[:100]}")
    
    if df is None:
        print("❌ 所有编码都失败了，尝试原始二进制读取...")
        try:
            # 使用更低级的读取方式
            with open(file_path, 'rb') as f:
                lines = f.readlines()
            
            # 转换为utf-8，忽略错误
            utf8_lines = []
            for i, line in enumerate(lines):
                try:
                    utf8_lines.append(line.decode('utf-8'))
                except:
                    try:
                        utf8_lines.append(line.decode('gbk', errors='ignore'))
                    except:
                        # 如果还是失败，跳过这一行
                        print(f"跳过第{i+1}行（无法解码）")
            
            # 写入临时文件
            temp_file = file_path.replace('.csv', '_temp_fixed.csv')
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.writelines(utf8_lines)
            
            # 重新读取
            df = pd.read_csv(temp_file, on_bad_lines='skip')
            os.remove(temp_file)  # 清理临时文件
            
        except Exception as e:
            print(f"❌ 二进制读取也失败: {e}")
            return None
    
    # 3. 检查数据质量
    print("\n📊 数据质量检查:")
    print(f"  总行数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")
    print(f"  缺失值总数: {df.isnull().sum().sum()}")
    
    # 显示列名
    print(f"\n📋 列名列表:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 4. 检查是否有 ele_36 列
    if 'ele_36' in df.columns:
        print(f"\n✅ 找到 ele_36 列（右脚数据标识）")
        right_foot_count = (df['ele_36'] == 0).sum()
        print(f"  右脚数据行数 (ele_36==0): {right_foot_count:,}")
    else:
        print(f"\n⚠️  警告: 没有找到 ele_36 列")
        # 尝试查找类似的列名
        potential_cols = [col for col in df.columns if '36' in str(col) or 'ele' in str(col).lower()]
        if potential_cols:
            print(f"  可能的替代列: {potential_cols}")
    
    # 5. 保存修复后的文件
    fixed_file = file_path.replace('.csv', '_fixed.csv')
    try:
        df.to_csv(fixed_file, index=False, encoding='utf-8')
        print(f"\n💾 修复后的文件已保存到: {fixed_file}")
        
        # 验证保存的文件
        try:
            df_check = pd.read_csv(fixed_file)
            print(f"✅ 验证通过: 修复文件可以正常读取，形状: {df_check.shape}")
            return fixed_file
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return None
            
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None
    
    return df

def process_with_dataextract(fixed_file):
    """使用你的Dataextract.py处理修复后的文件"""
    print("\n🔄 使用Dataextract.py处理修复后的文件...")
    
    try:
        from Dataextract import extract_right_foot_data
        
        output_folder = r"D:\TG0\PublicData_Rep\Smart_Insole_Database\subjectRepro5"
        
        result = extract_right_foot_data(
            csv_path=fixed_file,
            save_extracted=True,
            output_folder=output_folder
        )
        
        if result is not None:
            print("✅ Dataextract处理成功！")
            return True
        else:
            print("❌ Dataextract返回None")
            return False
            
    except Exception as e:
        print(f"❌ Dataextract处理失败: {e}")
        return False

def main():
    print("="*60)
    print("🛠️  CSV文件修复工具 - walking_s5_merged.csv")
    print("="*60)
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 修复文件
    fixed_file = fix_csv_encoding(file_path)
    
    if fixed_file:
        print("\n" + "="*60)
        print("🎯 现在可以:")
        print("1. 用修复后的文件重新运行批处理")
        print(f"2. 或者直接处理: {fixed_file}")
        print("="*60)
        
        # 询问是否继续处理
        response = input("\n是否用Dataextract.py处理修复后的文件? (y/n): ")
        if response.lower() == 'y':
            process_with_dataextract(fixed_file)
    else:
        print("\n❌ 文件修复失败，可能需要手动检查文件内容。")

if __name__ == "__main__":
    main()