"""
BatchDataProcessor.py
智能鞋垫数据批处理器 - 完整修正版
已修复文件名匹配问题：jogging_s1_merged_model_ready.csv 而不是 jogging_s1_model_ready.csv
"""

import os
import re
import yaml
import time
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import traceback
from dataclasses import dataclass
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings('ignore')

# ==================== 导入你的现有函数 ====================
try:
    # 假设你的函数在同一个目录或可以导入
    from Dataextract import extract_right_foot_data
    from CheckDatainformation import DataChecker
    from normalize_data import DataNormalizer
    print("✅ 成功导入所有处理函数")
except ImportError as e:
    print(f"⚠️  导入函数时警告: {e}")
    print("请确保你的三个函数文件在可导入路径中")
    
    # 定义占位函数以避免错误
    def extract_right_foot_data(csv_path, save_extracted=True, output_folder="subjectRepro1"):
        """占位函数"""
        print(f"占位: 提取数据 {csv_path}")
        return None
    
    class DataChecker:
        """占位类"""
        def __init__(self, csv_path, output_dir=None):
            self.csv_path = csv_path
            
        def run(self):
            print(f"占位: 检查数据 {self.csv_path}")
            return "placeholder_params.json"
    
    class DataNormalizer:
        """占位类"""
        def __init__(self, params_path=None):
            self.params = {}
            
        def normalize_all(self, df, apply_filter=True, window_size=5):
            print("占位: 归一化数据")
            return df
            
        def save_normalized_data(self, df, output_path, save_features_only=False):
            print(f"占位: 保存到 {output_path}")
            return output_path

# ==================== 配置类 ====================

class ProcessingMode(Enum):
    """处理模式"""
    AUTO = "auto"           # 自动检测，存在则跳过
    FORCE = "force_all"     # 强制重新处理所有
    MISSING = "missing_only" # 只处理缺失的文件


@dataclass
class BatchConfig:
    """批处理配置"""
    # 基础路径
    base_path: str = r"D:\TG0\PublicData_Rep\Smart_Insole_Database"
    
    # 要处理的受试者
    subjects: List[int] = None
    
    # 要处理的动作（按文件名前缀）
    activities: List[str] = None
    
    # 处理模式
    processing_mode: ProcessingMode = ProcessingMode.AUTO
    
    # 是否跳过已存在的文件
    skip_existing: bool = True
    
    # 出错时是否继续
    continue_on_error: bool = True
    
    # 最大重试次数
    max_retries: int = 3
    
    # 日志级别
    log_level: str = "INFO"
    
    # 是否保存详细日志
    save_detailed_log: bool = True
    
    # 是否显示进度条
    show_progress: bool = True
    
    # 各阶段配置
    stage1_config: Dict = None
    stage2_config: Dict = None
    stage3_config: Dict = None
    
    def __post_init__(self):
        if self.subjects is None:
            self.subjects = [1, 2, 3, 4, 5]
        if self.activities is None:
            self.activities = [
                "jogging",
                "jump_fb", 
                "jump_inplace",
                "squatting",
                "swaying",
                "walking"
            ]
        if self.stage1_config is None:
            self.stage1_config = {
                "output_folder_pattern": "subjectRepro{subject_id}",
                "save_extracted": True
            }
        if self.stage2_config is None:
            self.stage2_config = {
                "params_dir_pattern": "{output_folder}/Param/{activity}_s{subject_id}_merged_preprocess_params",
                "params_filename_pattern": "{activity}_s{subject_id}_merged_model_ready_params.json"
            }
        if self.stage3_config is None:
            self.stage3_config = {
                "norm_dir_pattern": "{output_folder}/norm",
                "norm_filename_pattern": "{activity}_s{subject_id}_merged_normalized.csv",
                "apply_filter": True,
                "window_size": 5
            }


# ==================== 文件路径管理器 ====================

class PathManager:
    """管理所有文件路径的创建和解析"""
    
    @staticmethod
    def parse_filename(filename: str) -> Dict[str, str]:
        """
        解析原始文件名，提取动作和受试者ID
        
        参数:
            filename: 文件名，如 "jogging_s1_merged.csv"
            
        返回:
            包含动作和受试者ID的字典
        """
        # 匹配模式: {动作}_s{数字}_merged.csv
        pattern = r"^(?P<activity>[a-zA-Z_]+)_s(?P<subject_id>\d+)_merged\.csv$"
        match = re.match(pattern, filename)
        
        if not match:
            # 尝试其他可能的模式
            patterns = [
                r"^(?P<activity>[a-zA-Z_]+)_(?P<subject_id>\d+)\.csv$",
                r"^(?P<activity>[a-zA-Z_]+)_subject(?P<subject_id>\d+)\.csv$",
            ]
            
            for pat in patterns:
                match = re.match(pat, filename)
                if match:
                    break
        
        if not match:
            raise ValueError(f"无法解析文件名: {filename}")
        
        return {
            "activity": match.group("activity"),
            "subject_id": match.group("subject_id")
        }
    
    @staticmethod
    def build_all_paths(config: BatchConfig, raw_file_path: str) -> Dict[str, str]:
        """
        构建所有相关文件路径 - 修正版本（匹配实际文件名）
        
        参数:
            config: 配置对象
            raw_file_path: 原始文件路径
            
        返回:
            包含所有路径的字典
        """
        # 提取文件名和目录信息
        raw_file = Path(raw_file_path)
        filename = raw_file.name  # 例如: jogging_s1_merged.csv
        
        # 解析动作和受试者ID
        info = PathManager.parse_filename(filename)
        activity = info["activity"]
        subject_id = info["subject_id"]
        
        # 基础名称（去掉.csv）
        base_name = filename.replace('.csv', '')  # jogging_s1_merged
        
        # 构建基础路径
        base_path = Path(config.base_path)
        
        # ===== 阶段1路径 =====
        # subjectRepro1/
        stage1_folder = base_path / f"subjectRepro{subject_id}"
        
        # 注意：Dataextract.py保存的文件名是带有"_extracted_right_foot"和"_model_ready"的
        stage1_file1 = stage1_folder / f"{base_name}_extracted_right_foot.csv"  # 提取数据
        stage1_file2 = stage1_folder / f"{base_name}_model_ready.csv"          # 模型训练数据
        
        # ===== 阶段2路径 =====
        # subjectRepro1/Param/jogging_s1_merged_preprocess_params/
        params_dir = stage1_folder / "Param" / f"{base_name}_preprocess_params"
        
        # 参数文件: subjectRepro1/Param/jogging_s1_merged_preprocess_params/jogging_s1_merged_model_ready_params.json
        params_file = params_dir / f"{base_name}_model_ready_params.json"
        
        # ===== 阶段3路径 =====
        # subjectRepro1/norm/
        norm_dir = stage1_folder / "norm"
        
        # 归一化文件: subjectRepro1/norm/jogging_s1_merged_normalized.csv
        norm_file = norm_dir / f"{base_name}_normalized.csv"
        
        return {
            "raw_file": str(raw_file_path),
            "subject_id": subject_id,
            "activity": activity,
            "base_name": base_name,  # jogging_s1_merged
            
            # 阶段1
            "stage1_folder": str(stage1_folder),
            "stage1_file1": str(stage1_file1),  # 提取数据 (带_extracted_right_foot)
            "stage1_file2": str(stage1_file2),  # 模型训练数据 (带_model_ready)
            
            # 阶段2
            "stage2_params_dir": str(params_dir),
            "stage2_params_file": str(params_file),
            
            # 阶段3
            "stage3_norm_dir": str(norm_dir),
            "stage3_norm_file": str(norm_file),
            
            # 状态文件
            "status_file": str(stage1_folder / f"status_{base_name}.json")
        }


# ==================== 文件扫描器 ====================

class FileScanner:
    """扫描原始文件"""
    
    @staticmethod
    def scan_raw_files(config: BatchConfig) -> List[Dict[str, Any]]:
        """
        扫描所有原始文件
        
        返回:
            文件信息列表
        """
        base_path = Path(config.base_path)
        files_info = []
        
        for subject_id in config.subjects:
            subject_folder = base_path / f"subject_{subject_id}"
            
            if not subject_folder.exists():
                print(f"警告: 文件夹不存在: {subject_folder}")
                continue
            
            # 扫描该文件夹下的所有CSV文件
            for csv_file in subject_folder.glob("*.csv"):
                filename = csv_file.name
                
                try:
                    # 解析文件名
                    info = PathManager.parse_filename(filename)
                    activity = info["activity"]
                    
                    # 检查是否在要处理的动作列表中
                    if activity in config.activities:
                        # 构建所有路径
                        paths = PathManager.build_all_paths(config, str(csv_file))
                        
                        files_info.append({
                            "raw_file": str(csv_file),
                            "subject_id": subject_id,
                            "activity": activity,
                            "paths": paths,
                            "status": "pending",  # pending, processing, success, failed, skipped
                            "stage1_status": "pending",
                            "stage2_status": "pending", 
                            "stage3_status": "pending",
                            "error_messages": [],
                            "start_time": None,
                            "end_time": None,
                            "processing_time": None
                        })
                        
                except ValueError as e:
                    print(f"警告: 跳过文件 {filename}: {e}")
                except Exception as e:
                    print(f"警告: 处理文件 {filename} 时出错: {e}")
        
        return files_info


# ==================== 处理流水线 ====================

class ProcessingPipeline:
    """处理流水线，按顺序执行三个阶段"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("BatchProcessor")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果启用）
        if self.config.save_detailed_log:
            log_file = f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def check_skip_stage(self, file_info: Dict, stage: str) -> Tuple[bool, str]:
        """
        检查是否应该跳过某个阶段 - 修正版本
        
        返回:
            (是否跳过, 原因)
        """
        paths = file_info["paths"]
        
        if self.config.processing_mode == ProcessingMode.FORCE:
            return False, "强制重新处理模式"
        
        # 检查阶段特定的文件是否存在 - 使用正确的路径
        if stage == "stage1":
            # paths["stage1_file2"] 已经是正确的路径: jogging_s1_merged_model_ready.csv
            if Path(paths["stage1_file2"]).exists() and self.config.skip_existing:
                return True, f"文件已存在: {paths['stage1_file2']}"
                
        elif stage == "stage2":
            if Path(paths["stage2_params_file"]).exists() and self.config.skip_existing:
                return True, f"文件已存在: {paths['stage2_params_file']}"
                
        elif stage == "stage3":
            if Path(paths["stage3_norm_file"]).exists() and self.config.skip_existing:
                return True, f"文件已存在: {paths['stage3_norm_file']}"
        
        return False, "需要处理"
    
    def run_stage1(self, file_info: Dict) -> Tuple[bool, str]:
        """运行阶段1：提取右脚数据"""
        paths = file_info["paths"]
        raw_file = paths["raw_file"]
        output_folder = paths["stage1_folder"]
        
        try:
            self.logger.info(f"阶段1开始: {raw_file}")
            
            # 创建输出文件夹
            os.makedirs(output_folder, exist_ok=True)
            
            # 调用你的提取函数
            result = extract_right_foot_data(
                csv_path=raw_file,
                save_extracted=True,
                output_folder=output_folder
            )
            
            if result is None:
                return False, "提取函数返回None"
            
            # 检查输出文件是否存在 - 使用正确的文件名
            if not Path(paths["stage1_file2"]).exists():
                # 再检查一下其他可能的文件名
                alt_file = paths["stage1_file2"].replace("_merged_model_ready.csv", "_model_ready.csv")
                if Path(alt_file).exists():
                    self.logger.warning(f"找到文件但文件名不同: {alt_file}")
                    # 重命名为标准文件名
                    Path(alt_file).rename(paths["stage1_file2"])
                else:
                    return False, f"输出文件不存在: {paths['stage1_file2']}"
            
            self.logger.info(f"阶段1完成: {paths['stage1_file2']}")
            return True, "成功"
            
        except Exception as e:
            error_msg = f"阶段1失败: {str(e)}"
            self.logger.error(error_msg)
            traceback.print_exc()
            return False, error_msg
    
    def run_stage2(self, file_info: Dict) -> Tuple[bool, str]:
        """运行阶段2：检查数据并生成参数"""
        paths = file_info["paths"]
        model_ready_file = paths["stage1_file2"]
        params_dir = paths["stage2_params_dir"]
        
        try:
            self.logger.info(f"阶段2开始: {model_ready_file}")
            
            # 创建参数文件夹
            os.makedirs(params_dir, exist_ok=True)
            
            # 检查输入文件是否存在
            if not Path(model_ready_file).exists():
                return False, f"输入文件不存在: {model_ready_file}"
            
            # 调用你的检查函数
            checker = DataChecker(
                csv_path=model_ready_file,
                output_dir=params_dir
            )
            
            # 运行检查
            param_file = checker.run()
            
            # 检查参数文件是否存在
            if not Path(param_file).exists():
                return False, f"参数文件不存在: {param_file}"
            
            self.logger.info(f"阶段2完成: {param_file}")
            return True, "成功"
            
        except Exception as e:
            error_msg = f"阶段2失败: {str(e)}"
            self.logger.error(error_msg)
            traceback.print_exc()
            return False, error_msg
    
    def run_stage3(self, file_info: Dict) -> Tuple[bool, str]:
        """运行阶段3：归一化处理"""
        paths = file_info["paths"]
        model_ready_file = paths["stage1_file2"]
        params_file = paths["stage2_params_file"]
        norm_dir = paths["stage3_norm_dir"]
        norm_file = paths["stage3_norm_file"]
        
        try:
            self.logger.info(f"阶段3开始: {model_ready_file}")
            
            # 创建归一化文件夹
            os.makedirs(norm_dir, exist_ok=True)
            
            # 检查输入文件是否存在
            if not Path(model_ready_file).exists():
                return False, f"输入文件不存在: {model_ready_file}"
            
            if not Path(params_file).exists():
                return False, f"参数文件不存在: {params_file}"
            
            # 调用你的归一化函数
            normalizer = DataNormalizer(params_path=params_file)
            
            # 加载数据
            df_raw = pd.read_csv(model_ready_file)
            
            # 应用归一化
            df_norm = normalizer.normalize_all(
                df_raw,
                apply_filter=self.config.stage3_config["apply_filter"],
                window_size=self.config.stage3_config["window_size"]
            )
            
            # 保存归一化数据
            normalizer.save_normalized_data(
                df_norm,
                norm_file,
                save_features_only=True
            )
            
            # 检查输出文件是否存在
            if not Path(norm_file).exists():
                return False, f"输出文件不存在: {norm_file}"
            
            self.logger.info(f"阶段3完成: {norm_file}")
            return True, "成功"
            
        except Exception as e:
            error_msg = f"阶段3失败: {str(e)}"
            self.logger.error(error_msg)
            traceback.print_exc()
            return False, error_msg
    
    def process_file(self, file_info: Dict, retry_count: int = 0) -> Dict:
        """
        处理单个文件（三个阶段）
        
        参数:
            file_info: 文件信息
            retry_count: 当前重试次数
            
        返回:
            更新后的文件信息
        """
        file_info["start_time"] = datetime.now()
        paths = file_info["paths"]
        
        try:
            # ===== 阶段1 =====
            skip_stage1, reason1 = self.check_skip_stage(file_info, "stage1")
            if skip_stage1:
                file_info["stage1_status"] = "skipped"
                file_info["stage1_reason"] = reason1
                self.logger.info(f"跳过阶段1: {reason1}")
            else:
                success1, message1 = self.run_stage1(file_info)
                file_info["stage1_status"] = "success" if success1 else "failed"
                file_info["stage1_message"] = message1
                
                if not success1:
                    file_info["status"] = "failed"
                    file_info["error_messages"].append(f"阶段1: {message1}")
                    return file_info
            
            # ===== 阶段2 =====
            skip_stage2, reason2 = self.check_skip_stage(file_info, "stage2")
            if skip_stage2:
                file_info["stage2_status"] = "skipped"
                file_info["stage2_reason"] = reason2
                self.logger.info(f"跳过阶段2: {reason2}")
            else:
                success2, message2 = self.run_stage2(file_info)
                file_info["stage2_status"] = "success" if success2 else "failed"
                file_info["stage2_message"] = message2
                
                if not success2:
                    file_info["status"] = "failed"
                    file_info["error_messages"].append(f"阶段2: {message2}")
                    return file_info
            
            # ===== 阶段3 =====
            skip_stage3, reason3 = self.check_skip_stage(file_info, "stage3")
            if skip_stage3:
                file_info["stage3_status"] = "skipped"
                file_info["stage3_reason"] = reason3
                self.logger.info(f"跳过阶段3: {reason3}")
            else:
                success3, message3 = self.run_stage3(file_info)
                file_info["stage3_status"] = "success" if success3 else "failed"
                file_info["stage3_message"] = message3
                
                if not success3:
                    file_info["status"] = "failed"
                    file_info["error_messages"].append(f"阶段3: {message3}")
                    return file_info
            
            # 所有阶段成功
            file_info["status"] = "success"
            
        except Exception as e:
            file_info["status"] = "failed"
            error_msg = f"处理过程中发生异常: {str(e)}"
            file_info["error_messages"].append(error_msg)
            self.logger.error(error_msg)
            traceback.print_exc()
            
            # 重试逻辑
            if retry_count < self.config.max_retries:
                self.logger.warning(f"准备重试 (第{retry_count+1}次): {paths['raw_file']}")
                time.sleep(2 ** retry_count)  # 指数退避
                return self.process_file(file_info, retry_count + 1)
        
        finally:
            file_info["end_time"] = datetime.now()
            if file_info["start_time"] and file_info["end_time"]:
                file_info["processing_time"] = (file_info["end_time"] - file_info["start_time"]).total_seconds()
            
            # 保存状态文件
            self.save_file_status(file_info)
        
        return file_info
    
    def save_file_status(self, file_info: Dict):
        """保存文件处理状态"""
        try:
            status_file = file_info["paths"]["status_file"]
            status_dir = os.path.dirname(status_file)
            os.makedirs(status_dir, exist_ok=True)
            
            # 准备状态信息
            status_info = {
                "subject_id": file_info["subject_id"],
                "activity": file_info["activity"],
                "raw_file": file_info["raw_file"],
                "status": file_info["status"],
                "stage1_status": file_info.get("stage1_status", "pending"),
                "stage2_status": file_info.get("stage2_status", "pending"),
                "stage3_status": file_info.get("stage3_status", "pending"),
                "error_messages": file_info.get("error_messages", []),
                "processing_time": file_info.get("processing_time"),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status_info, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.warning(f"无法保存状态文件: {e}")


# ==================== 进度跟踪器 ====================

class ProgressTracker:
    """跟踪处理进度"""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = datetime.now()
        
        # 按阶段统计
        self.stage_stats = {
            "stage1": {"success": 0, "failed": 0, "skipped": 0},
            "stage2": {"success": 0, "failed": 0, "skipped": 0},
            "stage3": {"success": 0, "failed": 0, "skipped": 0},
        }
        
        # 按受试者统计
        self.subject_stats = {}
        
        # 按动作统计
        self.activity_stats = {}
    
    def update(self, file_info: Dict):
        """更新统计信息"""
        self.processed_files += 1
        
        # 更新总体状态
        status = file_info.get("status", "unknown")
        if status == "success":
            self.successful += 1
        elif status == "failed":
            self.failed += 1
        elif status == "skipped":
            self.skipped += 1
        
        # 更新阶段统计
        for stage in ["stage1", "stage2", "stage3"]:
            stage_status = file_info.get(f"{stage}_status", "unknown")
            if stage_status in ["success", "failed", "skipped"]:
                self.stage_stats[stage][stage_status] += 1
        
        # 更新受试者统计
        subject_id = file_info.get("subject_id")
        if subject_id:
            if subject_id not in self.subject_stats:
                self.subject_stats[subject_id] = {"success": 0, "failed": 0, "skipped": 0}
            
            if status in ["success", "failed", "skipped"]:
                self.subject_stats[subject_id][status] += 1
        
        # 更新动作统计
        activity = file_info.get("activity")
        if activity:
            if activity not in self.activity_stats:
                self.activity_stats[activity] = {"success": 0, "failed": 0, "skipped": 0}
            
            if status in ["success", "failed", "skipped"]:
                self.activity_stats[activity][status] += 1
    
    def get_progress(self) -> Dict[str, Any]:
        """获取进度信息"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.processed_files > 0:
            avg_time_per_file = elapsed / self.processed_files
            remaining_files = self.total_files - self.processed_files
            estimated_remaining = avg_time_per_file * remaining_files if remaining_files > 0 else 0
        else:
            estimated_remaining = 0
        
        # 计算进度百分比
        progress_percent = (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0
        
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "progress_percent": progress_percent,
            "elapsed_time": elapsed,
            "estimated_remaining": estimated_remaining,
            "stage_stats": self.stage_stats,
            "subject_stats": self.subject_stats,
            "activity_stats": self.activity_stats
        }
    
    def display_progress(self, show_detailed: bool = True):
        """显示进度信息"""
        progress = self.get_progress()
        
        # 进度条
        bar_length = 50
        filled_length = int(bar_length * progress["progress_percent"] / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\n📊 处理进度: [{bar}] {progress['progress_percent']:.1f}%")
        print(f"📁 文件: {progress['processed_files']}/{progress['total_files']}")
        print(f"✅ 成功: {progress['successful']} | ⏭️ 跳过: {progress['skipped']} | ❌ 失败: {progress['failed']}")
        
        # 时间信息
        elapsed_str = self._format_time(progress["elapsed_time"])
        remaining_str = self._format_time(progress["estimated_remaining"])
        print(f"⏱️  已用: {elapsed_str} | 预计剩余: {remaining_str}")
        
        if show_detailed and progress["processed_files"] > 0:
            print("\n📈 详细统计:")
            
            # 阶段统计
            print("  阶段统计:")
            for stage, stats in progress["stage_stats"].items():
                stage_name = stage.replace("stage", "阶段")
                print(f"    {stage_name}: ✅{stats['success']} ⏭️{stats['skipped']} ❌{stats['failed']}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}分{seconds%60:.0f}秒"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}时{minutes:.0f}分"


# ==================== 主批处理器 ====================

class BatchDataProcessor:
    """主批处理器"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.pipeline = ProcessingPipeline(self.config)
        self.file_scanner = FileScanner()
        self.progress_tracker = None
        self.files_info = []
        self.results = []
    
    def load_config_from_yaml(self, config_file: str):
        """从YAML文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # 更新配置
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    if key == "processing_mode":
                        value = ProcessingMode(value)
                    setattr(self.config, key, value)
            
            self.pipeline = ProcessingPipeline(self.config)
            print(f"✅ 从 {config_file} 加载配置")
            
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")
    
    def scan_files(self) -> List[Dict]:
        """扫描所有文件"""
        print(f"\n🔍 扫描文件...")
        self.files_info = self.file_scanner.scan_raw_files(self.config)
        
        if not self.files_info:
            print("❌ 没有找到可处理的文件")
            return []
        
        print(f"✅ 找到 {len(self.files_info)} 个原始文件")
        
        # 按受试者和动作排序
        self.files_info.sort(key=lambda x: (int(x["subject_id"]), x["activity"]))
        
        # 显示文件列表
        print("\n📋 文件列表:")
        for i, file_info in enumerate(self.files_info[:10], 1):  # 只显示前10个
            print(f"  {i:2d}. {file_info['raw_file']}")
        
        if len(self.files_info) > 10:
            print(f"  ... 和 {len(self.files_info) - 10} 个其他文件")
        
        return self.files_info
    
    def process_all(self) -> List[Dict]:
        """处理所有文件"""
        if not self.files_info:
            print("❌ 没有可处理的文件，请先运行 scan_files()")
            return []
        
        # 初始化进度跟踪器
        self.progress_tracker = ProgressTracker(len(self.files_info))
        
        print(f"\n🚀 开始批量处理...")
        print(f"📊 总文件数: {len(self.files_info)}")
        print(f"⚙️  处理模式: {self.config.processing_mode.value}")
        print(f"⏭️  跳过已存在: {self.config.skip_existing}")
        print(f"🔁 出错继续: {self.config.continue_on_error}")
        print("=" * 60)
        
        self.results = []
        
        for i, file_info in enumerate(self.files_info, 1):
            print(f"\n📄 处理文件 {i}/{len(self.files_info)}: {file_info['raw_file']}")
            
            try:
                # 处理单个文件
                result = self.pipeline.process_file(file_info)
                self.results.append(result)
                
                # 更新进度
                self.progress_tracker.update(result)
                
                # 显示进度
                if self.config.show_progress and i % 5 == 0:  # 每5个文件显示一次
                    self.progress_tracker.display_progress(show_detailed=False)
                
                # 显示处理结果
                status = result.get("status", "unknown")
                if status == "success":
                    print(f"  ✅ 完成: {result['paths']['stage3_norm_file']}")
                elif status == "failed":
                    print(f"  ❌ 失败: {result.get('error_messages', ['未知错误'])[0]}")
                elif status == "skipped":
                    print(f"  ⏭️ 跳过: {result.get('stage1_reason', '已存在')}")
                
            except Exception as e:
                print(f"  ❌ 处理过程中发生异常: {e}")
                file_info["status"] = "failed"
                file_info["error_messages"].append(f"处理异常: {str(e)}")
                self.results.append(file_info)
                self.progress_tracker.update(file_info)
                
                if not self.config.continue_on_error:
                    print("❌ 由于错误而停止处理")
                    break
        
        # 显示最终进度
        if self.config.show_progress:
            self.progress_tracker.display_progress(show_detailed=True)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """生成处理报告"""
        if not self.results:
            return {"error": "没有处理结果"}
        
        # 收集统计信息
        total_files = len(self.results)
        successful = sum(1 for r in self.results if r.get("status") == "success")
        failed = sum(1 for r in self.results if r.get("status") == "failed")
        skipped = sum(1 for r in self.results if r.get("status") == "skipped")
        
        # 按阶段统计
        stage_stats = {"stage1": {}, "stage2": {}, "stage3": {}}
        for stage in stage_stats.keys():
            stage_stats[stage]["success"] = sum(1 for r in self.results if r.get(f"{stage}_status") == "success")
            stage_stats[stage]["failed"] = sum(1 for r in self.results if r.get(f"{stage}_status") == "failed")
            stage_stats[stage]["skipped"] = sum(1 for r in self.results if r.get(f"{stage}_status") == "skipped")
        
        # 按受试者统计
        subject_stats = {}
        for result in self.results:
            subject_id = result.get("subject_id")
            if subject_id not in subject_stats:
                subject_stats[subject_id] = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
            
            subject_stats[subject_id]["total"] += 1
            status = result.get("status", "unknown")
            if status in ["success", "failed", "skipped"]:
                subject_stats[subject_id][status] += 1
        
        # 按动作统计
        activity_stats = {}
        for result in self.results:
            activity = result.get("activity")
            if activity not in activity_stats:
                activity_stats[activity] = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
            
            activity_stats[activity]["total"] += 1
            status = result.get("status", "unknown")
            if status in ["success", "failed", "skipped"]:
                activity_stats[activity][status] += 1
        
        # 错误汇总
        errors = []
        for result in self.results:
            if result.get("status") == "failed":
                errors.append({
                    "file": result.get("raw_file"),
                    "errors": result.get("error_messages", []),
                    "subject_id": result.get("subject_id"),
                    "activity": result.get("activity")
                })
        
        # 时间统计
        processing_times = [r.get("processing_time", 0) for r in self.results if r.get("processing_time")]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        total_time = sum(processing_times)
        
        report = {
            "summary": {
                "total_files": total_files,
                "successful": successful,
                "failed": failed,
                "skipped": skipped,
                "success_rate": successful / total_files * 100 if total_files > 0 else 0,
                "total_processing_time": total_time,
                "average_time_per_file": avg_time
            },
            "stage_stats": stage_stats,
            "subject_stats": subject_stats,
            "activity_stats": activity_stats,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_path": self.config.base_path,
                "subjects": self.config.subjects,
                "activities": self.config.activities,
                "processing_mode": self.config.processing_mode.value,
                "skip_existing": self.config.skip_existing
            }
        }
        
        return report
    
    def save_report(self, report_file: str = None):
        """保存处理报告"""
        if not report_file:
            report_file = f"batch_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_report()
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 处理报告已保存到: {report_file}")
            return report_file
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
            return None
    
    def print_summary(self):
        """打印处理摘要"""
        if not self.results:
            print("没有处理结果")
            return
        
        report = self.generate_report()
        summary = report["summary"]
        
        print("\n" + "=" * 60)
        print("📊 批处理完成摘要")
        print("=" * 60)
        print(f"📁 总文件数: {summary['total_files']}")
        print(f"✅ 成功处理: {summary['successful']} ({summary['success_rate']:.1f}%)")
        print(f"⏭️  跳过: {summary['skipped']}")
        print(f"❌ 失败: {summary['failed']}")
        print(f"⏱️  总处理时间: {summary['total_processing_time']:.1f}秒")
        print(f"📈 平均每文件: {summary['average_time_per_file']:.1f}秒")
        
        # 显示失败的详细信息
        if report["errors"]:
            print(f"\n❌ 失败文件 ({len(report['errors'])}个):")
            for error in report["errors"][:5]:  # 只显示前5个
                print(f"  • {error['file']}")
                for err_msg in error["errors"][:2]:  # 只显示前2个错误
                    print(f"    - {err_msg}")
            
            if len(report["errors"]) > 5:
                print(f"    ... 和 {len(report['errors']) - 5} 个其他失败")


# ==================== 主函数和命令行接口 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("🏗️  智能鞋垫数据批处理器 - 修正版")
    print("已修复文件名匹配问题")
    print("=" * 60)
    
    # 创建配置
    config = BatchConfig(
        base_path=r"D:\TG0\PublicData_Rep\Smart_Insole_Database",
        subjects=[1, 2, 3, 4, 5],
        processing_mode=ProcessingMode.AUTO,
        skip_existing=True,
        continue_on_error=True,
        log_level="INFO",
        show_progress=True
    )
    
    # 创建批处理器
    processor = BatchDataProcessor(config)
    
    # 扫描文件
    files = processor.scan_files()
    if not files:
        return
    
    # 处理所有文件
    results = processor.process_all()
    
    # 生成报告
    processor.print_summary()
    
    # 保存报告
    report_file = processor.save_report()
    
    print("\n" + "=" * 60)
    print("🎉 批处理完成！")
    print("=" * 60)
    
    # 显示下一步建议
    print("\n📋 下一步建议:")
    print("1. 检查失败的文件，查看错误日志")
    print("2. 查看处理报告:", report_file)
    print("3. 归一化数据保存在各 subjectReproX/norm/ 文件夹下")
    print("4. 可以修改配置重新运行失败的文件")


def create_sample_config():
    """创建示例配置文件"""
    config = {
        "base_path": r"D:\TG0\PublicData_Rep\Smart_Insole_Database",
        "subjects": [1, 2, 3, 4, 5],
        "activities": [
            "jogging",
            "jump_fb",
            "jump_inplace",
            "squatting",
            "swaying",
            "walking"
        ],
        "processing_mode": "auto",  # auto, force_all, missing_only
        "skip_existing": True,
        "continue_on_error": True,
        "max_retries": 3,
        "log_level": "INFO",
        "save_detailed_log": True,
        "show_progress": True,
        "stage1_config": {
            "output_folder_pattern": "subjectRepro{subject_id}",
            "save_extracted": True
        },
        "stage2_config": {
            "params_dir_pattern": "{output_folder}/Param/{activity}_s{subject_id}_merged_preprocess_params",
            "params_filename_pattern": "{activity}_s{subject_id}_merged_model_ready_params.json"
        },
        "stage3_config": {
            "norm_dir_pattern": "{output_folder}/norm",
            "norm_filename_pattern": "{activity}_s{subject_id}_merged_normalized.csv",
            "apply_filter": True,
            "window_size": 5
        }
    }
    
    with open("batch_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print("✅ 示例配置文件已创建: batch_config.yaml")


# ==================== 命令行接口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="智能鞋垫数据批处理器")
    parser.add_argument("--base_path", type=str, help="基础数据路径")
    parser.add_argument("--subjects", type=str, help="要处理的受试者，如 '1,2,3,4,5'")
    parser.add_argument("--mode", type=str, choices=["auto", "force_all", "missing_only"], default="auto", help="处理模式")
    parser.add_argument("--skip_existing", type=lambda x: x.lower() == "true", default=True, help="是否跳过已存在的文件")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--create_config", action="store_true", help="创建示例配置文件")
    parser.add_argument("--scan_only", action="store_true", help="只扫描文件，不处理")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        sys.exit(0)
    
    # 创建配置
    config = BatchConfig()
    
    # 应用命令行参数
    if args.base_path:
        config.base_path = args.base_path
    
    if args.subjects:
        config.subjects = [int(s.strip()) for s in args.subjects.split(",")]
    
    config.processing_mode = ProcessingMode(args.mode)
    config.skip_existing = args.skip_existing
    
    # 创建处理器
    processor = BatchDataProcessor(config)
    
    # 加载配置文件（如果指定）
    if args.config and os.path.exists(args.config):
        processor.load_config_from_yaml(args.config)
    
    # 扫描文件
    files = processor.scan_files()
    
    if args.scan_only or not files:
        sys.exit(0)
    
    # 处理文件
    results = processor.process_all()
    
    # 生成报告
    processor.print_summary()
    processor.save_report()