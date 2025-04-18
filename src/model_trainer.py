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

# 添加項目根目錄到 Python 路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.helpers import (
    load_config,
    setup_logger,
    print_trainable_parameters,
    check_gpu_availability,
    load_env_vars,
)

def is_apple_silicon():
    import platform
    return platform.processor() == 'arm' and platform.system() == 'Darwin'

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
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config["output"].get("logging_dir", "logs"))

        # 偵測裝置
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

        # 建立輸出目錄
        Path(self.config["output"]["output_dir"]).mkdir(parents=True, exist_ok=True)
        self.logger.info("模型訓練器初始化完成")

    def load_datasets(self):
        path = self.config["data"]["processed_path"]
        self.logger.info(f"正在從 {path} 載入數據集")
        return load_from_disk(path)

    def load_tokenizer(self):
        base = self.config["model"]["base_model"]
        self.logger.info(f"從 {base} 載入分詞器")
        hf_token = load_env_vars().get("HUGGINGFACE_TOKEN")
        if hf_token:
            self.logger.info("使用 Hugging Face 令牌")
        else:
            self.logger.warning("未找到 Hugging Face 令牌")
        tok = AutoTokenizer.from_pretrained(
            base,
            padding_side="right",
            use_fast=True,
            token=hf_token,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        return tok

    def load_model(self):
        base = self.config["model"]["base_model"]
        self.logger.info(f"載入模型: {base}")
        hf_token = load_env_vars().get("HUGGINGFACE_TOKEN")
        use_4bit = (self.device == "cuda")

        if use_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base,
                device_map="auto",
                quantization_config=bnb,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=hf_token,
            )
        else:
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
            model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=self.config["model"]["lora_r"],
            lora_alpha=self.config["model"]["lora_alpha"],
            lora_dropout=self.config["model"]["lora_dropout"],
            target_modules=self.config["model"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        print_trainable_parameters(model)
        return model

    def get_sft_config(self) -> SFTConfig:
        tp = self.config["training"]
        op = self.config["output"]

        # 型別轉換
        num_epochs = int(tp["num_epochs"])
        batch_size = int(tp["micro_batch_size"])
        grad_accum = int(tp["gradient_accumulation_steps"])
        save_steps = int(tp["save_steps"])
        warmup_steps = int(tp["warmup_steps"])
        lr = float(tp["learning_rate"])
        wd = float(tp["weight_decay"])
        max_grad_norm = float(tp["max_grad_norm"])
        cutoff = int(tp["cutoff_len"])

        # 只有 CUDA 支援 fp16
        use_fp16 = (self.device == "cuda")

        return SFTConfig(
            # HF Trainer args
            output_dir=op["output_dir"],
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            optim=tp["optimizer"],
            learning_rate=lr,
            weight_decay=wd,
            lr_scheduler_type="cosine",
            save_strategy="steps",
            save_steps=save_steps,
            logging_steps=50,
            fp16=use_fp16,
            bf16=False,
            max_grad_norm=max_grad_norm,
            max_steps=-1,
            warmup_steps=warmup_steps,
            group_by_length=True,
            report_to=op["report_to"],
            # SFT 特有
            packing=False,
            max_seq_length=cutoff,
            dataset_text_field="text",
        )

    def train(self):
        self.dataset = self.load_datasets()
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        sft_config = self.get_sft_config()

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset.get("validation", None),
            processing_class=tokenizer,
        )

        self.logger.info(f"開始在 {self.device} 上訓練...")
        trainer.train()
        self.logger.info("訓練完成，保存模型…")
        trainer.save_model(self.config["output"]["output_dir"])
        self.logger.info(f"已保存至 {self.config['output']['output_dir']}")
        return model, tokenizer

def main():
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
