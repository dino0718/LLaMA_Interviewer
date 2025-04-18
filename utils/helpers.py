import os
import yaml
import logging
import torch
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_path="config/config.yaml"):
    """載入配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            logging.error(f"載入配置檔案錯誤: {e}")
            raise

def setup_logger(log_dir=None, level=logging.INFO):
    """設置日誌記錄器"""
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_dir, "train.log")
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )
    return logging.getLogger()

def check_gpu_availability():
    """檢查 GPU 可用性"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        return True, device_count, device_names
    else:
        return False, 0, []

def create_output_dir(output_dir):
    """創建輸出目錄"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def print_trainable_parameters(model):
    """計算並打印可訓練參數數量"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"訓練參數量: {trainable_params:,d} ({100 * trainable_params / all_param:.2f}% 的總參數)"
    )
    return trainable_params

def format_metrics(metrics):
    """格式化指標以便於顯示"""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = value
    return formatted

# 載入環境變數
def load_env_vars():
    """載入環境變數，特別是 Hugging Face 令牌"""
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    return {"HUGGINGFACE_TOKEN": hf_token}