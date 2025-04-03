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


# Step 3: 定义嵌入组合器神经网络
class EmbeddingCombiner(nn.Module):
    def __init__(self, embedding1_dim, embedding2_dim):
        super(EmbeddingCombiner, self).__init__()
        # 首先将两个嵌入投影到相同维度（这里选择投影到embedding1_dim）
        self.proj_B = nn.Linear(embedding2_dim, embedding1_dim)
        # 然后定义组合网络
        self.fc1 = nn.Linear(embedding1_dim * 2, embedding1_dim)
        self.fc2 = nn.Linear(embedding1_dim, embedding1_dim)
        self.relu = nn.ReLU()

        # 初始化权重
        nn.init.xavier_uniform_(self.proj_B.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.proj_B.bias is not None:
            nn.init.zeros_(self.proj_B.bias)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, emb1, emb2):
        # 首先将emb2投影到与emb1相同的维度
        emb2_proj = self.proj_B(emb2)
        # 然后组合两个嵌入
        combined = torch.cat((emb1, emb2_proj), dim=-1)
        combined = self.relu(self.fc1(combined))
        return self.fc2(combined)


# 创建组合器，确保使用 float32 进行训练
combiner = EmbeddingCombiner(dim_1B, dim_3B).to("cuda").to(torch_dtype)

# Step 4: 训练 combiner 让组合后的嵌入逼近 embeddings_A
optimizer = torch.optim.Adam(combiner.parameters(), lr=1e-5)  # 降低学习率
loss_fn = nn.MSELoss()  # 让组合后的嵌入逼近 embeddings_A

num_epochs = 10  # 训练 10 轮
batch_size = 1024  # 设定 batch size
num_batches = embeddings_A.size(0) // batch_size + 1  # 计算总 batch 数

for epoch in range(num_epochs):
    epoch_loss = 0  # 记录损失
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, embeddings_A.size(0))

        emb_A_batch = embeddings_A[start:end]  # 取当前 batch
        emb_B_batch = embeddings_B[start:end]

        optimizer.zero_grad()
        combined_embeddings = combiner(emb_A_batch, emb_B_batch)

        loss = loss_fn(combined_embeddings, emb_A_batch)
        if torch.isnan(loss).any():
            print("⚠️ Detected NaN in loss, terminating training.")
            break

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {epoch_loss / num_batches}")


# Step 5: 获取最终组合嵌入
with torch.no_grad():
    final_combined_embeddings = combiner(embeddings_A, embeddings_B)

# Step 6: 替换模型的嵌入层
model_A.model.embed_tokens.weight = nn.Parameter(final_combined_embeddings)

# Step 7: 保存融合后的模型
output_path = 'D:\\desktop\\merage\\models\\meraged\\merged_llama_nn_combined'
model_A.save_pretrained(output_path)
AutoTokenizer.from_pretrained(model_A_path).save_pretrained(output_path)

print(f"融合后的模型（使用神经网络组合）已保存至 {output_path}")