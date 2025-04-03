from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import warnings

# 禁用不必要的警告
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 配置参数（保持原始路径，优化其他参数）
class Config:
    model_A_path = '/content/EmbeddingMerage/models/llama-1b'  # 保持原始1B模型路径
    model_B_path = '/content/EmbeddingMerage/models/llama-3b'  # 保持原始3B模型路径
    output_path = '/content/EmbeddingMerage/models/meraged/1b-3b-optimized'  # 保持原始输出路径
    
    # 优化后的训练参数
    batch_size = 512  # 减小batch size防止显存溢出
    learning_rate = 1e-5  # 更保守的学习率
    num_epochs = 50
    temperature = 0.2
    sim_weight = 1.0  # 正样本损失权重
    dis_weight = 0.5  # 负样本损失权重
    
    # 技术参数
    eps = 1e-8
    grad_clip = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型（保持原始加载方式）
def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16,  # 保持原始精度
        offload_folder="offload"
    )

print("加载模型中...")
model_A = load_model(Config.model_A_path)
model_B = load_model(Config.model_B_path)

# 提取embedding（增强数值稳定性）
def get_embeddings(model):
    emb = model.model.embed_tokens.weight.data.clone()
    emb = emb.to(device).to(torch.float32)  # 训练时转为float32
    emb = torch.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0)
    return F.normalize(emb, p=2, dim=1, eps=Config.eps)

print("准备embedding...")
embeddings_A = get_embeddings(model_A)
embeddings_B = get_embeddings(model_B)

assert embeddings_A.shape[0] == embeddings_B.shape[0], "词表大小不匹配!"

# 改进的投影层（保持原始维度）
class TokenProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )
        # 初始化
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1, eps=Config.eps)

projector = TokenProjector(embeddings_B.size(1), embeddings_A.size(1)).to(device)

# 专用损失函数（针对token对齐优化）
def token_alignment_loss(projected_B, embeddings_A, target_indices):
    # 正样本相似度（最大化）
    pos_sim = F.cosine_similarity(projected_B, embeddings_A[target_indices], dim=-1)
    
    # 负样本相似度（最小化，使用负采样）
    neg_samples = min(1024, len(embeddings_A))  # 动态负采样数量
    rand_indices = torch.randint(0, len(embeddings_A), (len(projected_B), neg_samples), device=device)
    neg_emb = embeddings_A[rand_indices]
    neg_sim = F.cosine_similarity(projected_B.unsqueeze(1), neg_emb, dim=-1)
    
    # 组合损失
    loss = -Config.sim_weight * pos_sim.mean() + Config.dis_weight * neg_sim.mean()
    
    # 数值稳定性
    return torch.nan_to_num(loss, nan=0.0)

# 优化器设置
optimizer = torch.optim.AdamW(projector.parameters(), lr=Config.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 训练循环（添加更多监控）
print("开始训练...")
best_loss = float('inf')
for epoch in range(Config.num_epochs):
    projector.train()
    epoch_loss = 0
    
    # 随机shuffle
    indices = torch.randperm(len(embeddings_B), device=device)
    
    progress_bar = tqdm(range(0, len(indices), Config.batch_size), 
                       desc=f"Epoch {epoch+1}/{Config.num_epochs}")
    
    for i in progress_bar:
        batch_idx = indices[i:i+Config.batch_size]
        batch_B = embeddings_B[batch_idx]
        
        optimizer.zero_grad()
        projected = projector(batch_B)
        loss = token_alignment_loss(projected, embeddings_A, batch_idx)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), Config.grad_clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    # 学习率调整
    avg_loss = epoch_loss / len(progress_bar)
    scheduler.step(avg_loss)
    
    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(projector.state_dict(), "best_projector.pt")
        print(f"保存最佳模型，loss: {best_loss:.4f}")

# 加载最佳模型
print("加载最佳模型...")
projector.load_state_dict(torch.load("best_projector.pt"))

# 模型融合（保持原始合并方式）
print("融合embedding...")
with torch.no_grad():
    projector.eval()
    projected_B = projector(embeddings_B.to(torch.float32))
    
    # 加权平均（可调整alpha）
    alpha = 0.5  # 保持原始比例
    combined_embeddings = alpha * embeddings_A + (1-alpha) * projected_B
    combined_embeddings = F.normalize(combined_embeddings, p=2, dim=1, eps=Config.eps)
    
    # 检查数值
    assert not torch.isnan(combined_embeddings).any(), "融合后出现NaN!"
    model_A.model.embed_tokens.weight = nn.Parameter(combined_embeddings.to(torch.float16))  # 恢复原始精度

# 保存模型（保持原始保存方式）
print("保存模型...")
model_A.save_pretrained(Config.output_path)
AutoTokenizer.from_pretrained(Config.model_A_path).save_pretrained(Config.output_path)

print(f"模型已保存到 {Config.output_path}")