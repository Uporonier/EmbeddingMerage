from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import math

# 设置模型路径
model_paths = {
    "merged": Path(r"D:/desktop/merage/models/meraged/llama1Bmerged").resolve(),
    "merged_train": Path(r"D:/desktop/merage/models/meraged/merged_llama_train").resolve(),
    "original_1b": Path(r"D:/desktop/merage/models/llama-1b").resolve(),
    "original_3b": Path(r"D:/desktop/merage/models/llama-3b").resolve(),
    "WeightSum37": Path(r"D:/desktop/merage/models/meraged/llamaWeightSum").resolve(),
    "WeightSum73": Path(r"D:/desktop/merage/models/meraged/llamaWeightSum73").resolve(),
    "WeightSum82": Path(r"D:/desktop/merage/models/meraged/llamaWeightSum82").resolve(),
    "LinearTransformations": Path(r"D:/desktop/merage/models/meraged/LinearTransformations").resolve(),
    "merged_llama_train_3Bto1B": Path(r"D:/desktop/merage/models/meraged/merged_llama_train_3Bto1B").resolve(),
    "merged_llama_train_adapter": Path(r"D:/desktop/merage/models/meraged/merged_llama_train_adapter").resolve(),
    "merged_llama_nn_combined": Path(r"D:/desktop/merage/models/meraged/merged_llama_nn_combined").resolve(),
    "merged_llama_contrastive": Path(r"D:/desktop/merage/models/meraged/merged_llama_contrastive").resolve()
}

# 手动选择要运行的模型
model_name = "merged_llama_contrastive"  # 可选："merged", "original_1b", "original_3b"

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载测试数据集（wikitext-2）
dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")

# 计算困惑度（PPL）
def compute_perplexity(model, tokenizer, dataset, print_interval=100):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_samples = len(dataset)

    with torch.no_grad():
        for i, example in enumerate(dataset):
            input_text = example["text"]
            if not input_text.strip():  # 跳过空白行
                continue

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
            labels = inputs["input_ids"]

            # 确保 input_ids 为 LongTensor
            inputs = {key: value.to(torch.long) if key == "input_ids" else value for key, value in inputs.items()}

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss  # 交叉熵损失
            total_loss += loss.item() * labels.size(1)  # 乘以 token 数
            total_tokens += labels.size(1)

            # 每处理一定数量的样本就打印一次进度
            if (i + 1) % print_interval == 0 or i + 1 == total_samples:
                print(f"Progress: {i + 1}/{total_samples} samples processed...")

    if total_tokens > 0:
        perplexity = math.exp(total_loss / total_tokens)
        return perplexity
    else:
        return float("nan")  # 如果没有有效的 tokens，返回 NaN

# 只加载用户指定的模型
if model_name in model_paths:
    print(f"Loading {model_name} model from {model_paths[model_name]}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_paths[model_name]),
            device_map="auto",
            torch_dtype=torch.float32,
            local_files_only=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_paths[model_name]),
            local_files_only=True
        )

        print(f"{model_name} model loaded successfully.\n")

        # 评估模型
        print(f"Evaluating {model_name} model...")
        ppl = compute_perplexity(model, tokenizer, dataset)
        print(f"Model: {model_name}, Perplexity (PPL): {ppl:.2f}\n")

        # 释放显存
        del model
        del tokenizer
        torch.cuda.empty_cache()
        print(f"Released GPU memory for {model_name}.\n")

    except Exception as e:
        print(f"Failed to load {model_name} model: {str(e)}\n")
else:
    print(f"Model {model_name} not found. Check your model_paths dictionary.")
