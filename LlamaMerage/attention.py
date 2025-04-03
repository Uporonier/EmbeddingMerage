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

class Config:
    # 模型路径配置
    model_A_path = '/content/EmbeddingMerage/models/llama-1b'
    model_B_path = '/content/EmbeddingMerage/models/llama-3b'
    output_path = '/content/EmbeddingMerage/models/merged/1b-3b-attention-fusion'
    projector_path = '/content/EmbeddingMerage/best_projector.pt'  # 预训练投影层路径
    
    # 训练参数
    batch_size = 512
    fusion_batch_size = 256
    learning_rate = 1e-5
    fusion_lr = 3e-5
    num_epochs = 50
    fusion_epochs = 30
    
    # 技术参数
    eps = 1e-8
    grad_clip = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 模型加载函数
def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder="offload"
    )

# 2. 嵌入提取函数
def get_embeddings(model):
    emb = model.model.embed_tokens.weight.data.clone()
    emb = emb.to(device).to(torch.float32)
    emb = torch.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0)
    return F.normalize(emb, p=2, dim=1, eps=Config.eps)

# 3. 投影层定义
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

# 4. 嵌入融合网络（使用您指定的结构）
class EmbeddingCombiner(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        
        # 初始化权重
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, emb1, emb2):
        combined = torch.cat((emb1, emb2), dim=-1)
        combined = self.relu(self.fc1(combined))
        return F.normalize(self.fc2(combined), dim=-1)

# 5. 对比损失函数
def contrastive_loss(fused, original):
    # 正样本相似度（最大化）
    pos_sim = F.cosine_similarity(fused, original, dim=-1)
    
    # 负样本相似度（最小化）
    neg_idx = torch.randint(0, len(original), (len(fused), 512), device=device)
    neg_sim = F.cosine_similarity(fused.unsqueeze(1), original[neg_idx], dim=-1)
    
    return (1 - pos_sim.mean()) + 0.3 * neg_sim.mean()

# 主流程
def main():
    # 加载模型
    print("加载模型中...")
    model_A = load_model(Config.model_A_path)
    model_B = load_model(Config.model_B_path)
    
    # 提取嵌入
    print("准备嵌入...")
    embeddings_A = get_embeddings(model_A)
    embeddings_B = get_embeddings(model_B)
    assert embeddings_A.shape[0] == embeddings_B.shape[0], "词表大小不匹配!"
    
    # 初始化并加载投影层
    projector = TokenProjector(embeddings_B.size(1), embeddings_A.size(1)).to(device)
    if os.path.exists(Config.projector_path):
        print(f"加载预训练投影层: {Config.projector_path}")
        projector.load_state_dict(torch.load(Config.projector_path))
    else:
        raise FileNotFoundError(f"投影层权重文件不存在: {Config.projector_path}")
    
    # 投影嵌入B
    with torch.no_grad():
        projector.eval()
        projected_B = projector(embeddings_B)
    
    # 初始化融合网络
    combiner = EmbeddingCombiner(embeddings_A.size(1)).to(device)
    optimizer = torch.optim.AdamW(combiner.parameters(), lr=Config.fusion_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.fusion_epochs)
    
    # 训练融合网络
    print("\n训练融合网络...")
    best_loss = float('inf')
    for epoch in range(Config.fusion_epochs):
        combiner.train()
        indices = torch.randperm(len(embeddings_A), device=device)
        epoch_loss = 0
        
        for i in tqdm(range(0, len(indices), Config.fusion_batch_size), 
                     desc=f"Epoch {epoch+1}/{Config.fusion_epochs}"):
            batch_idx = indices[i:i+Config.fusion_batch_size]
            batch_A = embeddings_A[batch_idx]
            batch_B = projected_B[batch_idx]
            
            optimizer.zero_grad()
            fused = combiner(batch_A, batch_B)
            loss = contrastive_loss(fused, batch_A)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), Config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(indices) // Config.fusion_batch_size)
        scheduler.step()
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(combiner.state_dict(), "best_combiner.pt")
            print(f"保存最佳融合网络，loss: {best_loss:.4f}")
    
    # 加载最佳融合网络
    combiner.load_state_dict(torch.load("best_combiner.pt"))
    
    # 生成最终融合嵌入
    print("\n生成融合嵌入...")
    with torch.no_grad():
        combiner.eval()
        combined_embeddings = combiner(embeddings_A, projected_B)
        combined_embeddings = F.normalize(combined_embeddings, p=2, dim=1)
        
        # 检查并保存
        assert not torch.isnan(combined_embeddings).any(), "融合后出现NaN!"
        model_A.model.embed_tokens.weight = nn.Parameter(combined_embeddings.to(torch.float16))
    
    # 保存最终模型
    print("\n保存模型...")
    model_A.save_pretrained(Config.output_path)
    AutoTokenizer.from_pretrained(Config.model_A_path).save_pretrained(Config.output_path)
    print(f"模型已保存到 {Config.output_path}")

if __name__ == "__main__":
    main()