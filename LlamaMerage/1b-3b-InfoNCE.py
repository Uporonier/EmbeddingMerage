from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Step 1: 加载模型（使用 float32 以确保一致性）
model_A_path = 'D:\\desktop\\merage\\models\\llama-1b'
model_B_path = 'D:\\desktop\\merage\\models\\llama-3b'
torch_dtype = torch.float32

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

# Step 2: 提取 Embedding 层并处理
embeddings_A = model_A.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)
embeddings_B = model_B.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)

# 处理 NaN 值
embeddings_A = torch.nan_to_num(embeddings_A)
embeddings_B = torch.nan_to_num(embeddings_B)

dim_1B = embeddings_A.size(1)
dim_3B = embeddings_B.size(1)


# Step 3: 定义投影层
class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias) if self.projection.bias is not None else None

    def forward(self, x):
        return self.projection(x)


projector = LinearProjection(dim_3B, dim_1B).to("cuda").to(torch_dtype)

# Step 4: 使用对比损失训练投影层
optimizer = torch.optim.Adam(projector.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵对比损失
temperature = 0.07  # 温度参数

num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 投影并归一化
    embeddings_B_projected = projector(embeddings_B)
    embeddings_B_norm = F.normalize(embeddings_B_projected, p=2, dim=1)
    embeddings_A_norm = F.normalize(embeddings_A, p=2, dim=1)

    # 计算相似度矩阵
    logits = embeddings_B_norm @ embeddings_A_norm.T / temperature

    # 创建标签（每个位置i对应自己的索引）
    labels = torch.arange(embeddings_B.size(0), device=embeddings_B.device)

    loss = loss_fn(logits, labels)

    if torch.isnan(loss):
        print("⚠️ NaN损失，终止训练")
        break

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Step 5: 融合嵌入层
with torch.no_grad():
    final_projection = projector(embeddings_B)
    combined_embeddings = (embeddings_A + final_projection) / 2

# 替换并保存模型
model_A.model.embed_tokens.weight = nn.Parameter(combined_embeddings)
output_path = 'D:\\desktop\\merage\\models\\meraged\\1b-3b-InfoNCE'
model_A.save_pretrained(output_path)
AutoTokenizer.from_pretrained(model_A_path).save_pretrained(output_path)

print(f"模型已保存至 {output_path}")