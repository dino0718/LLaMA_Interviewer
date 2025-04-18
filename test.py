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
    q = "請說明 RAG（Retrieval-Augmented Generation）的基本流程。"
    print(generate_answer(q))