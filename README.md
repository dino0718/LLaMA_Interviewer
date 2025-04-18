# LLaMA_Interviewer 模型微調專案

## 專案概述

LLaMA_Interviewer 是一個針對面試場景的語言模型微調專案。該專案利用 TinyLlama-1.1B-Chat-v1.0 作為基礎模型，透過參數高效微調技術（PEFT），特別是 LoRA（Low-Rank Adaptation）方法，建立一個能夠進行面試互動的智能助手。

## 技術選擇與原理

### 0. 為什麼要微調模型

**微調的必要性：**
- **專業領域適應**：預訓練模型缺乏特定領域（如面試）的專業知識和互動模式
- **語言風格調整**：面試場景需要專業、友善且具建設性的回答風格
- **任務特定優化**：基礎模型需要調整以理解面試問題結構和產生符合期待的回答格式
- **減少幻覺**：針對面試情境的微調可以降低生成虛假或不準確信息的可能性

**推薦微調的情境：**
1. **專業領域應用**：當需要模型在特定專業領域（如技術面試、HR面試）表現出專業知識
2. **特定風格需求**：需要模型體現特定語言風格、專業性或人格特質
3. **資源受限環境**：在計算資源有限情況下（如 Apple Silicon M系列）需要高效運行大型語言模型
4. **自定義互動流程**：需要模型遵循特定互動模式或問答結構
5. **數據隱私考量**：希望在本地處理敏感面試資料，避免雲端API調用的隱私風險

### 1. 基礎模型：TinyLlama-1.1B-Chat-v1.0

**選擇原因：**
- **資源效率**：相較於完整的 LLaMA-2 模型，TinyLlama 僅有 1.1B 參數，能在有限計算資源（如 M4 晶片）上運行
- **開源可用**：TinyLlama 無需授權即可使用，解決了獲取 LLaMA-2 需要申請的限制
- **聊天優化**：該版本已經針對對話場景進行過預訓練，更適合面試互動任務

### 2. 微調技術：LoRA（Low-Rank Adaptation）

**選擇原因：**
- **記憶體效率**：LoRA 只訓練低秩矩陣，而不是完整模型參數，大幅降低了記憶體需求（降低約 10000 倍）
- **訓練速度**：使用 LoRA 能加速微調過程，減少訓練時間
- **適應性強**：透過添加適應層修改特定模塊的行為，保持基礎模型其他部分不變
- **部署方便**：最終只需將小型 LoRA 權重與基礎模型結合，無需存儲完整微調模型

### 3. PEFT（Parameter-Efficient Fine-Tuning）框架

**選擇原因：**
- **簡化實施**：提供標準化的 LoRA 實施方法，減少代碼複雜度
- **整合優勢**：與 Hugging Face 生態系統無縫集成
- **靈活配置**：允許精細控制哪些層需要微調

## 微調參數設定

### 模型參數
- **基礎模型**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **LoRA 秩 (r)**: 16（決定了低秩矩陣的維度，較大的值提供更高的模型容量）
- **LoRA alpha (α)**: 32（縮放參數，影響 LoRA 更新的權重）
- **LoRA dropout**: 0.05（防止過擬合的正則化技術）
- **目標模組**: `["q_proj", "k_proj", "v_proj", "o_proj"]`（注意力機制的核心組件）

### 訓練參數
- **批次大小**: 2（適應有限 GPU 記憶體）
- **微批次大小**: 1（支援梯度累積）
- **訓練輪數**: 3
- **學習率**: 2e-4
- **最大序列長度**: 512
- **驗證集比例**: 0.05
- **預熱步數**: 50
- **保存檢查點頻率**: 每 100 步
- **梯度累積步數**: 2
- **梯度檢查點**: 啟用（節省記憶體）
- **最大梯度範數**: 0.3（防止梯度爆炸）
- **權重衰減**: 0.001
- **優化器**: `adamw_torch`

## 訓練配置說明

以下參數定義了模型的訓練行為，請根據硬體資源和任務需求調整：

- `batch_size`: 2  
  整體訓練批次大小。一次送入模型計算的樣本數。較小的批次佔用較少記憶體，但可能影響訓練穩定性。

- `micro_batch_size`: 1  
  微批次大小，用於梯度累積。將大批次拆成多個微批次計算，可在有限顯存下模擬更大 batch。

- `num_epochs`: 5  
  訓練輪數。模型將完整迭代訓練資料集的次數，輪數越多擬合越充分，但過多可能過擬合。

- `learning_rate`: 2e-4  
  學習率，控制每次參數更新的步長。過大易發散，過小收斂緩慢。

- `cutoff_len`: 512  
  最大序列長度。訓練中截斷或填充的最大令牌數，影響記憶體佔用與上下文容量。

- `val_set_size`: 0.1  
  驗證集比例。若未提供獨立驗證資料，將從訓練集中按此比例隨機拆分。

- `warmup_steps`: 50  
  預熱步數，初始階段線性增大学習率以穩定訓練，預熱結束後再進入正常調度。

- `save_steps`: 100  
  每隔多少訓練步保存一次檢查點，方便中斷後繼續或回滾到歷史版本。

- `gradient_accumulation_steps`: 2  
  梯度累積步數，將多次微批次梯度累積後再做一次參數更新，有助於穩定訓練。

- `gradient_checkpointing`: true  
  是否啟用梯度檢查點，將部分中間激活值丟棄後再逐層重計算，以降低顯存佔用。

- `max_grad_norm`: 0.3  
  梯度裁剪閾值，當梯度範數超過此值時進行裁剪，有助於防止梯度爆炸。

- `weight_decay`: 0.001  
  權重衰減係數，作為正則化手段，抑制模型過度擬合。

- `optimizer`: "adamw_torch"  
  優化器選擇，此處使用 AdamW（PyTorch 實現），結合權重衰減和動量加速收斂。

## 評估結果分析

```
評估結果：
rouge1: 0.72
rouge2: 0.3333333333333333
rougeL: 0.6933333333333334
bleu: 9.878765474230743
```

### 結果解讀

1. **ROUGE-1 分數 (0.72)**：
   - 表示模型生成的單詞與參考回答有 72% 的重疊率
   - 這是一個相對良好的結果，表明模型能捕捉關鍵詞彙

2. **ROUGE-2 分數 (0.33)**：
   - 表示模型生成的二元組（連續兩個詞）與參考回答有 33% 的重疊
   - 較低的 ROUGE-2 表明模型在保持短語連貫性方面有改進空間

3. **ROUGE-L 分數 (0.69)**：
   - 衡量最長公共子序列，反映生成文本與參考文本的結構相似性
   - 接近 ROUGE-1 的分數表明模型能較好地保持整體結構

4. **BLEU 分數 (9.88)**：
   - 屬於較低水平（BLEU 滿分為 100）
   - 表明生成的回答在精確匹配參考答案的表達方式方面有限

### 改進建議

1. **數據質量與數量提升**：
   - **擴充訓練數據**：收集更多高質量的面試問答對
   - **數據多樣化**：確保涵蓋不同類型的面試問題與風格
   - **數據質量提升**：進行人工審核，確保訓練數據的準確性與專業性

2. **模型優化**：
   - **增加 LoRA 秩**：考慮將 r 從 16 增加到 32 或 64，增強模型容量
   - **擴展目標模組**：將 LoRA 應用到更多層，如中間層的前饋網路部分
   - **調整 alpha 值**：實驗不同的 LoRA 縮放參數，如將 alpha 增加到 64

3. **訓練策略改進**：
   - **增加訓練輪數**：從 3 輪提高到 5-10 輪
   - **學習率調整**：嘗試更小的學習率（如 1e-4）和更複雜的學習率調度
   - **批次大小優化**：如果硬體允許，增加批次大小至 4 或 8

4. **評估與測試改進**：
   - **人類評估**：引入人工評估機制，對模型回答的專業性和適用性進行打分
   - **多樣化評估指標**：增加語義相似度等更現代的評估指標
   - **A/B 測試**：與其他商業面試助手進行對比評估

5. **應用優化**：
   - **提示詞工程**：優化模型的輸入提示，引導生成更高質量的回答
   - **後處理策略**：實施回答的後處理機制，確保專業性與一致性
   - **混合策略**：考慮將檢索增強生成（RAG）與微調模型結合

## 技術實施指南

### 環境設置

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # 在 Linux/Mac 上
# venv\Scripts\activate   # 在 Windows 上

# 安裝依賴
pip install -r requirements.txt
```

### 訓練流程

```bash
# 數據預處理
python src/data_processor.py

# 開始訓練
python src/train.py

# 模型評估
python src/evaluation.py
```

### 模型使用

#### Python API 範例

````python
# 載入分詞器與模型
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "models/finetune-tinyllama")

# 定義生成函數
def generate_answer(question: str,
                    max_new_tokens: int = 128,
                    temperature: float = 0.7,
                    top_p: float = 0.9) -> str:
    prompt = f"面試官：{question}\n應聘者："
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 呼叫範例
if __name__ == "__main__":
    q = "請簡單介紹一下自己？"
    print(generate_answer(q))
````

#### 命令列快速測試

```bash
python - <<'PYCODE'
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "models/finetune-tinyllama")

prompt = "面試官：你為什麼想加入我們公司？\n應聘者："
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=True)
echo=$(python - <<PYCODE
print(tokenizer.decode(out[0], skip_special_tokens=True))
PYCODE
)
echo
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 載入基礎模型和分詞器
base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 載入 LoRA 權重
adapter_path = "models/finetune-tinyllama"
model = PeftModel.from_pretrained(model, adapter_path)

# 生成回答
def generate_answer(question):
    input_text = f"面試官：{question}\n應聘者："
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"], 
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 結論

LLaMA_Interviewer 專案展示了如何在資源有限的環境中使用 PEFT 技術有效微調語言模型。當前版本顯示出中等水平的表現，通過實施上述改進建議，特別是數據擴充和訓練優化，模型性能有望顯著提升。下一步開發將專注於提高生成文本的連貫性和專業性，使其更好地適應實際面試場景的需求。
