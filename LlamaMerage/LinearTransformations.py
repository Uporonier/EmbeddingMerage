from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

# Step 1: 加载模型（使用 float16 以节省内存）
model_A_path = 'D:\\desktop\\merage\\models\\llama-1b'
model_B_path = 'D:\\desktop\\merage\\models\\llama-3b'
torch_dtype = torch.float16  # 半精度模式

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

# Step 2: 提取 Embedding 层
embeddings_A = model_A.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)  # (V, 2048)
embeddings_B = model_B.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)  # (V, 3072)

dim_1B = embeddings_A.size(1)  # 2048
dim_3B = embeddings_B.size(1)  # 3072

# Step 3: 定义降维投影（Linear Transformation）
class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProjection, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

# 投影 3B -> 1B 维度
projector = LinearProjection(dim_3B, dim_1B).to("cuda").to(torch_dtype)
embeddings_B_projected = projector(embeddings_B)

# Step 4: 进一步学习融合方式（Linear Combination）
class LinearCombiner(nn.Module):
    def __init__(self, embedding_dim):
        super(LinearCombiner, self).__init__()
        self.transform = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, emb1, emb2):
        combined = torch.cat((emb1, emb2), dim=-1)  # 先拼接
        return self.transform(combined)  # 通过线性变换融合

combiner = LinearCombiner(dim_1B).to("cuda").to(torch_dtype)
combined_embeddings = combiner(embeddings_A, embeddings_B_projected)

# Step 5: 替换模型的嵌入层
model_A.model.embed_tokens.weight = nn.Parameter(combined_embeddings)

# Step 6: 保存融合后的模型
output_path = 'D:\\desktop\\merage\\models\\meraged\\LinearTransformations'
model_A.save_pretrained(output_path)
AutoTokenizer.from_pretrained(model_A_path).save_pretrained(output_path)

print(f"融合后的模型（使用线性变换）已保存至 {output_path}")
