import os
import json
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 要使用 rouge_score 和 sacrebleu 進行指標計算，請先安裝：
# pip install rouge-score sacrebleu
from rouge_score import rouge_scorer, scoring
import sacrebleu

# 將專案根目錄加入 Python 路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import load_config, setup_logger, format_metrics, check_gpu_availability

def is_apple_silicon() -> bool:
    import platform
    return platform.processor() == 'arm' and platform.system() == 'Darwin'

def check_mps_availability() -> bool:
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

class ModelEvaluator:
    """模型評估器，用於評估經 LoRA 微調後模型的效能"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config["output"].get("logging_dir", "logs"))

        # 裝置偵測：優先 CUDA，其次 MPS，其次 CPU
        gpu_ok, gpu_count, gpu_names = check_gpu_availability()
        if gpu_ok:
            self.device = "cuda"
            self.logger.info(f"使用 CUDA 裝置: {', '.join(gpu_names)}")
        elif is_apple_silicon() and check_mps_availability():
            self.device = "mps"
            self.logger.info("在 Apple Silicon 上啟用 MPS")
        else:
            self.device = "cpu"
            self.logger.warning("僅使用 CPU 進行評估")

        # 模型路徑
        self.model_path = self.config["output"]["output_dir"]
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路徑不存在: {self.model_path}")

        # 評估結果存放目錄
        self.eval_result_path = os.path.join(self.model_path, "evaluation_results")
        Path(self.eval_result_path).mkdir(parents=True, exist_ok=True)
        self.logger.info("模型評估器初始化完成")

    def load_base_model_and_tokenizer(self):
        """根據裝置選擇精度載入基礎模型與分詞器"""
        base_model = self.config["model"]["base_model"]
        self.logger.info(f"從 {base_model} 載入基礎模型 (device={self.device})")

        # 非 CUDA 環境：MPS 用 float16，CPU 用 float32
        dtype = torch.float16 if self.device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if self.device in ("mps", "cpu"):
            model = model.to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            padding_side="right",
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer

    def load_finetuned_model(self):
        """加載經 LoRA 微調後的模型"""
        self.logger.info(f"從 {self.model_path} 加載微調模型")
        base_model, tokenizer = self.load_base_model_and_tokenizer()
        finetuned = PeftModel.from_pretrained(base_model, self.model_path)
        finetuned = finetuned.to(self.device)
        return finetuned, tokenizer

    def load_validation_data(self):
        """載入並回傳驗證集 Dataset"""
        dataset_path = self.config["data"]["processed_path"]
        self.logger.info(f"從 {dataset_path} 載入驗證數據集")
        ds = load_from_disk(dataset_path)
        val = ds.get("validation", None)
        if val is None:
            self.logger.error("找不到 validation split")
        return val

    def generate_responses(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        validation_data,
        num_samples: Optional[int] = None
    ) -> List[Dict]:
        """手動使用 model.generate 為每個樣本產生回應"""
        if validation_data is None:
            return []

        total = len(validation_data)
        limit = min(total, num_samples) if num_samples else min(total, 50)
        self.logger.info(f"為前 {limit} 條樣本生成回應")

        results = []
        for sample in tqdm(validation_data.select(range(limit))):
            prompt = sample["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            results.append({
                "prompt": prompt,
                "true_completion": sample["completion"],
                "generated_completion": gen_text,
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", "")
            })
        return results

    def calculate_metrics(self, evaluation_results: List[Dict]) -> Dict:
        """計算 ROUGE & BLEU 指標（使用 rouge_score 與 sacrebleu）"""
        self.logger.info("計算評估指標…")

        preds = [r["generated_completion"] for r in evaluation_results]
        refs  = [r["true_completion"] for r in evaluation_results]

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        for ref, pred in zip(refs, preds):
            scores = scorer.score(ref, pred)
            aggregator.add_scores(scores)
        result = aggregator.aggregate()
        rouge1 = result['rouge1'].mid.fmeasure
        rouge2 = result['rouge2'].mid.fmeasure
        rougeL = result['rougeL'].mid.fmeasure

        # BLEU
        # sacrebleu expects list of references lists
        bleu = sacrebleu.corpus_bleu(preds, [refs]).score

        metrics = {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "bleu": bleu
        }
        self.logger.info(f"指標：{format_metrics(metrics)}")
        return metrics

    def save_evaluation_results(self, results: List[Dict], metrics: Dict):
        """將生成結果與指標保存到檔案"""
        with open(os.path.join(self.eval_result_path, "generated_responses.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.eval_result_path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        self.logger.info(f"評估結果已保存至 {self.eval_result_path}")

    def evaluate(self, num_samples: Optional[int] = None) -> Dict:
        finetuned, tokenizer = self.load_finetuned_model()
        val_data = self.load_validation_data()
        if val_data is None:
            raise RuntimeError("無驗證數據，無法評估")

        results = self.generate_responses(finetuned, tokenizer, val_data, num_samples)
        metrics = self.calculate_metrics(results)
        self.save_evaluation_results(results, metrics)
        return metrics

def main():
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate()
    print("\n評估結果：")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
