import os
import json
import logging
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datasets import Dataset, DatasetDict

# 添加項目根目錄到 Python 路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.helpers import load_config, setup_logger

class DataProcessor:
    """數據預處理類，用於處理原始數據並準備用於微調的數據集"""
    
    def __init__(self, config_path="config/config.yaml"):
        """初始化數據處理器"""
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config["output"].get("logging_dir", "logs"))
        
        # 創建必要的目錄
        Path(self.config["data"]["processed_path"]).mkdir(parents=True, exist_ok=True)
        raw_dir = Path(os.path.dirname(self.config["data"]["train_path"]))
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("數據處理器初始化完成")
    
    def load_json_data(self, file_path: str) -> List[Dict]:
        """從JSON文件載入數據"""
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"文件不存在: {file_path}")
                return []
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"已從 {file_path} 載入 {len(data)} 條記錄")
            return data
        except Exception as e:
            self.logger.error(f"載入JSON數據錯誤: {e}")
            raise
    
    def format_instruction_dataset(self, 
                                  examples: List[Dict], 
                                  instruction_key: str = "instruction",
                                  input_key: str = "input",
                                  output_key: str = "output") -> List[Dict]:
        """將數據格式化為指令微調格式"""
        formatted_data = []
        
        for example in examples:
            instruction = example.get(instruction_key, "")
            input_text = example.get(input_key, "")
            output = example.get(output_key, "")
            
            if not instruction and not input_text:
                continue
                
            # 構建提示模板
            if input_text:
                prompt = f"### 指令:\n{instruction}\n\n### 輸入:\n{input_text}\n\n### 回應:\n"
            else:
                prompt = f"### 指令:\n{instruction}\n\n### 回應:\n"
                
            formatted_data.append({
                "prompt": prompt,
                "completion": output,
                "text": prompt + output,  # 完整文本，用於某些訓練方法
                "instruction": instruction,
                "input": input_text,
                "output": output
            })
        
        self.logger.info(f"已格式化 {len(formatted_data)} 條訓練數據")
        return formatted_data
    
    def split_train_val(self, 
                       data: List[Dict], 
                       val_size: float = None) -> Tuple[List[Dict], List[Dict]]:
        """將數據分割為訓練集和驗證集"""
        if val_size is None:
            val_size = self.config["training"].get("val_set_size", 0.05)
            
        val_size = min(max(0.0, val_size), 0.5)  # 確保驗證集大小在合理範圍內
        
        # 計算驗證集大小
        val_count = max(1, int(len(data) * val_size))
        train_data = data[val_count:]
        val_data = data[:val_count]
        
        self.logger.info(f"數據分割完成: 訓練集 {len(train_data)} 條, 驗證集 {len(val_data)} 條")
        return train_data, val_data
    
    def convert_to_dataset(self, data: List[Dict]) -> Dataset:
        """將數據轉換為Hugging Face Dataset格式"""
        return Dataset.from_pandas(pd.DataFrame(data))
    
    def save_processed_data(self, 
                           train_dataset: Dataset, 
                           val_dataset: Optional[Dataset] = None):
        """保存處理後的數據集"""
        output_path = self.config["data"]["processed_path"]
        
        # 創建數據集字典
        datasets_dict = {"train": train_dataset}
        if val_dataset is not None:
            datasets_dict["validation"] = val_dataset
        
        dataset_dict = DatasetDict(datasets_dict)
        dataset_dict.save_to_disk(output_path)
        self.logger.info(f"已將處理後的數據保存到 {output_path}")
    
    def process(self):
        """執行整個數據處理流程"""
        # 載入原始數據
        train_data = self.load_json_data(self.config["data"]["train_path"])
        eval_data = self.load_json_data(self.config["data"]["eval_path"])
        
        if not train_data:
            self.logger.error("找不到訓練數據，請確保資料準備妥當")
            return
        
        # 格式化數據
        formatted_train = self.format_instruction_dataset(train_data)
        
        # 如果沒有指定評估數據，從訓練數據中分割
        if not eval_data:
            train_data_split, val_data_split = self.split_train_val(formatted_train)
            train_dataset = self.convert_to_dataset(train_data_split)
            val_dataset = self.convert_to_dataset(val_data_split)
        else:
            formatted_eval = self.format_instruction_dataset(eval_data)
            train_dataset = self.convert_to_dataset(formatted_train)
            val_dataset = self.convert_to_dataset(formatted_eval)
        
        # 保存處理後的數據
        self.save_processed_data(train_dataset, val_dataset)
        
        return train_dataset, val_dataset

def main():
    """主函數"""
    processor = DataProcessor()
    processor.process()

if __name__ == "__main__":
    main()