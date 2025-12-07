# run_batch.py
"""
简单批处理运行脚本
"""

from BatchDataProcessor import BatchDataProcessor, BatchConfig, ProcessingMode
import sys

def main():
    print("🚀 开始批量处理所有受试者数据")
    
    # 简单配置
    config = BatchConfig(
        base_path=r"D:\TG0\PublicData_Rep\Smart_Insole_Database",  # 修改为你的路径
        subjects=[1, 2, 3, 4, 5],
        processing_mode=ProcessingMode.AUTO,
        skip_existing=True,
        continue_on_error=True
    )
    
    # 创建处理器
    processor = BatchDataProcessor(config)
    
    # 扫描文件
    print("🔍 扫描文件...")
    files = processor.scan_files()
    
    if not files:
        print("❌ 没有找到可处理的文件")
        return
    
    print(f"📁 找到 {len(files)} 个文件")
    
    # 确认
    response = input(f"开始处理 {len(files)} 个文件？(y/n): ")
    if response.lower() != 'y':
        print("取消处理")
        return
    
    # 开始处理
    print("\n" + "="*60)
    print("🚀 开始处理...")
    print("="*60)
    
    results = processor.process_all()
    
    # 显示结果
    processor.print_summary()
    
    # 保存报告
    processor.save_report("processing_report.json")
    
    print("\n🎉 处理完成！")

if __name__ == "__main__":
    main()