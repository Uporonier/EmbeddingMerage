from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

# Step 1: 加载模型（使用 float32 以确保一致性）
model_A_path = 'D:\\desktop\\merage\\models\\llama-1b'
model_B_path = 'D:\\desktop\\merage\\models\\llama-3b'
torch_dtype = torch.float32  # 使用 float32 进行训练

model_A = AutoModelForCausalLM.from_pretrained(
    model_A_path,
    device_map="auto",
    torch_dtype=torch_dtype,  # 保证模型是使用 float32
    offload_folder="offload"
)
model_B = AutoModelForCausalLM.from_pretrained(
    model_B_path,
    device_map="auto",
    torch_dtype=torch_dtype,  # 保证模型是使用 float32
    offload_folder="offload"
)

# Step 2: 提取 Embedding 层并确保一致的数据类型
embeddings_A = model_A.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)  # (V, 2048)
embeddings_B = model_B.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)  # (V, 3072)

# 检查 NaN 值并处理
if torch.isnan(embeddings_A).any() or torch.isnan(embeddings_B).any():
    print("⚠️ Found NaN values in embeddings, replacing with zeros.")
    embeddings_A = torch.nan_to_num(embeddings_A)
    embeddings_B = torch.nan_to_num(embeddings_B)

dim_1B = embeddings_A.size(1)  # 2048
dim_3B = embeddings_B.size(1)  # 3072

# Step 3: 定义降维投影（Linear Transformation）
class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProjection, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.projection.weight)  # 使用 Xavier 初始化
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        return self.projection(x)

# 创建投影层，确保投影层使用 float32 进行训练
projector = LinearProjection(dim_3B, dim_1B).to("cuda").to(torch_dtype)  # 使用 float32 训练

# Step 4: 训练 `projector` 让 embeddings_B_projected 逼近 embeddings_A
optimizer = torch.optim.Adam(projector.parameters(), lr=1e-5)  # 降低学习率
loss_fn = nn.MSELoss()  # 让投影后的 `embeddings_B_projected` 逼近 `embeddings_A`

num_epochs = 10  # 训练 10 轮
for epoch in range(num_epochs):
    optimizer.zero_grad()

    embeddings_B_projected = projector(embeddings_B)

    loss = loss_fn(embeddings_B_projected, embeddings_A)  # 目标：让 B 投影后更接近 A
    if torch.isnan(loss).any():  # 检查 NaN
        print("⚠️ Detected NaN in loss, terminating training.")
        break

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Step 5: 融合 Embedding（线性变换后求平均）
combined_embeddings = (embeddings_A + embeddings_B_projected) / 2

# Step 6: 替换模型的嵌入层
model_A.model.embed_tokens.weight = nn.Parameter(combined_embeddings)

# Step 7: 保存融合后的模型
output_path = 'D:\\desktop\\merage\\models\\meraged\\merged_llama_train_3Bto1B'
model_A.save_pretrained(output_path)
AutoTokenizer.from_pretrained(model_A_path).save_pretrained(output_path)

print(f"融合后的模型（使用训练过的 Linear Transformation）已保存至 {output_path}")
