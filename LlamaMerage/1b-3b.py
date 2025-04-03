from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

# Step 1: 加载模型（统一使用 float16）
model_A_path = 'D:\\desktop\\merage\\models\\llama-1b'
model_B_path = 'D:\\desktop\\merage\\models\\llama-3b'
torch_dtype = torch.float16  # 半精度以节省内存

model_A = AutoModelForCausalLM.from_pretrained(
    model_A_path,
    device_map="auto",
    torch_dtype=torch_dtype,
    offload_folder="offload"
)
model_B = AutoModelForCausalLM.from_pretrained(
    model_B_path,
    device_map="auto",
    torch_dtype=torch_dtype,
    offload_folder="offload"
)

# Step 2: 提取 Embedding 层并确保设备/数据类型一致
embeddings_A = model_A.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)
embeddings_B = model_B.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)

# Step 3: 定义投影层（将 3B 的 3072 维降投影到 1B 的 2048 维）
dim_1B = embeddings_A.size(1)  # 2048
dim_3B = embeddings_B.size(1)  # 3072

# 仅对 3B 的 Embedding 做降维投影
if dim_3B > dim_1B:
    projector = nn.Linear(dim_3B, dim_1B).to("cuda").to(torch_dtype)
    embeddings_B_projected = projector(embeddings_B)
else:
    embeddings_B_projected = embeddings_B  # 如果维度相同或更小，无需投影

# Step 4: 融合 Embedding（直接按相同维度相加平均）
combined_embeddings = (embeddings_A + embeddings_B_projected) / 2

# Step 5: 替换并保存
model_A.model.embed_tokens.weight = nn.Parameter(combined_embeddings)
output_path = 'D:\\desktop\\merage\\models\\meraged\\llama1Bmerged'
model_A.save_pretrained(output_path)
AutoTokenizer.from_pretrained(model_A_path).save_pretrained(output_path)

print(f"融合后的模型（保持 2048 维）已保存至 {output_path}")