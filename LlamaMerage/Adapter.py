from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: 加载 LLaMA-1B 和 LLaMA-3B 模型
model_A_path = 'D:\\desktop\\merage\\models\\llama-1b'
model_B_path = 'D:\\desktop\\merage\\models\\llama-3b'
torch_dtype = torch.float32  # 使用 float32 进行训练

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

# Step 2: 提取 LLaMA-1B 和 LLaMA-3B 的嵌入层
embeddings_A = model_A.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)  # (V, 2048)
embeddings_B = model_B.model.embed_tokens.weight.data.to("cuda").to(torch_dtype)  # (V, 3072)


# Step 3: 定义 Adapter 层
class EmbeddingAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingAdapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 3072 -> 2048
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 2048 -> 2048

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建 Adapter 实例
input_dim = embeddings_B.size(1)  # 3072
hidden_dim = 2048  # 中间层维度
output_dim = embeddings_A.size(1)  # 2048
adapter = EmbeddingAdapter(input_dim, hidden_dim, output_dim).to("cuda")

# Step 4: 定义优化器和损失函数
optimizer = optim.Adam(adapter.parameters(), lr=1e-5)  # 设定较小的学习率
loss_fn = nn.MSELoss()  # 让投影后的 `embeddings_B` 更接近 `embeddings_A`

# Step 5: 微调 Adapter（训练）
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 将 embeddings_B 通过 Adapter 映射到 2048 维
    embeddings_B_projected = adapter(embeddings_B)

    # 计算损失
    loss = loss_fn(embeddings_B_projected, embeddings_A)

    if torch.isnan(loss).any():  # 检查 NaN
        print("⚠️ Detected NaN in loss, terminating training.")
        break

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Step 6: 更新 LLaMA-1B 模型的嵌入层
# 使用训练后的 Adapter 进行嵌入层转换
embeddings_A_updated = (embeddings_A + embeddings_B_projected) / 2  # 融合结果（也可以采用其他方式）

# Step 7: 替换 LLaMA-1B 模型的嵌入层
model_A.model.embed_tokens.weight = nn.Parameter(embeddings_A_updated)

# Step 8: 保存训练后的模型
output_path = 'D:\\desktop\\merage\\models\\meraged\\merged_llama_train_adapter'
model_A.save_pretrained(output_path)
AutoTokenizer.from_pretrained(model_A_path).save_pretrained(output_path)

print(f"融合后的模型（使用训练过的 Adapter）已保存至 {output_path}")
