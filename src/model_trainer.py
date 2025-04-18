import os
import torch
import sys
from pathlib import Path
from datasets import load_from_disk
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# 添加項目根目錄到 Python 路徑，確保可以導入其他模組
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.helpers import (
    load_config,
    setup_logger,
    print_trainable_parameters,
    check_gpu_availability,
    load_env_vars,
)

# 判斷是否為 Apple Silicon 晶片的函數
def is_apple_silicon():
    import platform
    return platform.processor() == 'arm' and platform.system() == 'Darwin'

# 檢查 MPS (Metal Performance Shaders) 是否可用的函數
def check_mps_availability():
    if not hasattr(torch, 'backends') or not hasattr(torch.backends, 'mps'):
        return False, "PyTorch 未編譯支援 MPS"
    if not torch.backends.mps.is_available():
        return False, "MPS 不可用"
    if not torch.backends.mps.is_built():
        return False, "PyTorch 未構建 MPS 支援"
    return True, "MPS 可用"

class ModelTrainer:
    """LLaMA 模型微調器，使用 LoRA 方法進行高效微調"""

    def __init__(self, config_path="config/config.yaml"):
        # 初始化：載入配置檔案與設定日誌
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config["output"].get("logging_dir", "logs"))

        # 智能偵測可用的運算裝置 (GPU, MPS, CPU)
        self.device = "cpu"
        gpu_available, device_count, device_names = check_gpu_availability()
        if gpu_available:
            self.logger.info(f"找到 {device_count} 個 GPU 裝置: {', '.join(device_names)}")
            self.device = "cuda"
        elif is_apple_silicon():
            ok, msg = check_mps_availability()
            if ok:
                self.logger.info(f"在 Apple Silicon 上啟用 MPS: {msg}")
                self.device = "mps"
            else:
                self.logger.warning(f"Apple Silicon 無法啟用 MPS: {msg}，將使用 CPU")
        else:
            self.logger.warning("未找到 GPU/MPS，將使用 CPU")

        # 建立輸出目錄，確保存在
        Path(self.config["output"]["output_dir"]).mkdir(parents=True, exist_ok=True)
        self.logger.info("模型訓練器初始化完成")

    def load_datasets(self):
        # 從磁碟載入預處理過的數據集
        path = self.config["data"]["processed_path"]
        self.logger.info(f"正在從 {path} 載入數據集")
        return load_from_disk(path)

    def load_tokenizer(self):
        # 載入模型對應的分詞器
        base = self.config["model"]["base_model"]
        self.logger.info(f"從 {base} 載入分詞器")
        hf_token = load_env_vars().get("HUGGINGFACE_TOKEN")
        if hf_token:
            self.logger.info("使用 Hugging Face 令牌")
        else:
            self.logger.warning("未找到 Hugging Face 令牌")
        tok = AutoTokenizer.from_pretrained(
            base,
            padding_side="right",  # 設定填充在右側
            use_fast=True,         # 使用快速分詞器
            token=hf_token,
        )
        # 確保有填充標記，如果沒有則使用結束標記代替
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        return tok

    def load_model(self):
        # 載入預訓練模型並套用 LoRA 參數高效微調設定
        base = self.config["model"]["base_model"]
        self.logger.info(f"載入模型: {base}")
        hf_token = load_env_vars().get("HUGGINGFACE_TOKEN")
        use_4bit = (self.device == "cuda")

        if use_4bit:
            # GPU 上使用 4bit 量化加速且減少記憶體用量
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base,
                device_map="auto",  # 自動分配模型到可用裝置
                quantization_config=bnb,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=hf_token,
            )
        else:
            # CPU 或 MPS 上載入模型
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.logger.info(f"使用 {dtype} 在 {self.device} 載入")
            model = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=dtype,
                trust_remote_code=True,
                token=hf_token,
            )
            if self.device == "mps":
                model = model.to("mps")

        if use_4bit:
            # 為量化模型準備訓練
            model = prepare_model_for_kbit_training(model)

        # 設定 LoRA 配置，只微調部分參數以節省資源
        lora_cfg = LoraConfig(
            r=self.config["model"]["lora_r"],                      # LoRA 秩
            lora_alpha=self.config["model"]["lora_alpha"],         # LoRA 縮放參數
            lora_dropout=self.config["model"]["lora_dropout"],     # LoRA Dropout 值
            target_modules=self.config["model"]["target_modules"], # 要微調的目標模組
            bias="none",                                           # 不微調偏差項
            task_type="CAUSAL_LM",                                 # 因果語言模型任務
        )
        # 將模型轉換為 PEFT 模型
        model = get_peft_model(model, lora_cfg)
        # 顯示可訓練參數資訊
        print_trainable_parameters(model)
        return model

    def get_sft_config(self) -> SFTConfig:
        # 建立監督式微調配置
        tp = self.config["training"]
        op = self.config["output"]

        # 型別轉換，確保參數為正確型別
        num_epochs = int(tp["num_epochs"])
        batch_size = int(tp["micro_batch_size"])
        grad_accum = int(tp["gradient_accumulation_steps"])
        save_steps = int(tp["save_steps"])
        warmup_steps = int(tp["warmup_steps"])
        lr = float(tp["learning_rate"])
        wd = float(tp["weight_decay"])
        max_grad_norm = float(tp["max_grad_norm"])
        cutoff = int(tp["cutoff_len"])

        # 只有 CUDA 支援 fp16 半精度訓練
        use_fp16 = (self.device == "cuda")

        # 返回監督式微調配置
        return SFTConfig(
            # 一般訓練參數
            output_dir=op["output_dir"],                   # 輸出目錄
            num_train_epochs=num_epochs,                   # 訓練輪數
            per_device_train_batch_size=batch_size,        # 每個裝置的批次大小
            gradient_accumulation_steps=grad_accum,        # 梯度累積步數
            optim=tp["optimizer"],                         # 優化器
            learning_rate=lr,                              # 學習率
            weight_decay=wd,                               # 權重衰減
            lr_scheduler_type="cosine",                    # 餘弦學習率調度
            save_strategy="steps",                         # 按步數保存
            save_steps=save_steps,                         # 每多少步保存一次
            logging_steps=50,                              # 每50步記錄一次
            fp16=use_fp16,                                 # 是否使用半精度
            bf16=False,                                    # 不使用 bfloat16
            max_grad_norm=max_grad_norm,                   # 梯度裁剪最大範數
            max_steps=-1,                                  # 最大步數 (-1表示無限制)
            warmup_steps=warmup_steps,                     # 預熱步數
            group_by_length=True,                          # 按長度分組，提高效率
            report_to=op["report_to"],                     # 報告工具 (如 wandb)
            # SFT 特有參數
            packing=False,                                 # 不使用序列打包
            max_seq_length=cutoff,                         # 最大序列長度
            dataset_text_field="text",                     # 數據集文本欄位名
        )

    def train(self):
        # 訓練主流程
        # 載入數據集、分詞器和模型
        self.dataset = self.load_datasets()
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        sft_config = self.get_sft_config()

        # 創建SFT訓練器
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=self.dataset["train"],                # 訓練集
            eval_dataset=self.dataset.get("validation", None),  # 驗證集
            processing_class=tokenizer,                         # 使用的分詞器
        )

        # 執行訓練
        self.logger.info(f"開始在 {self.device} 上訓練...")
        trainer.train()
        self.logger.info("訓練完成，保存模型…")
        trainer.save_model(self.config["output"]["output_dir"])
        self.logger.info(f"已保存至 {self.config['output']['output_dir']}")
        return model, tokenizer

def main():
    # 主函數，創建訓練器並執行訓練
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
