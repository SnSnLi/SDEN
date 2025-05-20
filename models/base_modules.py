import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import CLIPTextModel, CLIPVisionModel

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, temperature=0.05):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature  # 温度参数

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        # 计算原始注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 调整温度以增强差异性
        scores = scores / self.temperature  # 控制温度，避免过度平滑

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 特征重要性门控
        feat_importance = torch.sigmoid(q.mean(dim=-1, keepdim=True))  # [B, H, L, 1]
        v = v * feat_importance  # 重要性加权

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.proj(out), attn


class EmergenceCore(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, 8)
        self.critical_mlp = nn.Sequential(
            nn.Linear(dim, dim*4),  
            nn.GELU(),
            nn.Linear(dim*4, dim*2),  # 4096 -> 2048
            nn.GELU(),
            nn.Linear(dim*2, dim)  # 2048 -> 1024
        )
        
    def forward(self, x, context=None):
        x_permuted = x.permute(1, 0, 2)
        attn_out, _ = self.self_attention(x_permuted, x_permuted, x_permuted)
        attn_out = attn_out.permute(1, 0, 2)
        return self.critical_mlp(attn_out) + x

class BidirectionalEmergenceCore(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.text_emergence = EmergenceCore(dim)
        self.image_emergence = EmergenceCore(dim)
        self.cross_modal = CrossModalAttention(dim)
        self.fusion = nn.Sequential(
            nn.Linear(dim*2, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, text_feat, image_feat):
        text_emerged = self.text_emergence(text_feat, image_feat)
        image_emerged = self.image_emergence(image_feat, text_feat)
        text_context = self.cross_modal(text_emerged, image_emerged)
        image_context = self.cross_modal(image_emerged, text_emerged)
        text_final = self.fusion(torch.cat([text_emerged, text_context], dim=-1))
        image_final = self.fusion(torch.cat([image_emerged, image_context], dim=-1))
        return text_final, image_final

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.query_transform = nn.Linear(dim, dim)
        self.key_transform = nn.Linear(dim, dim)
        self.value_transform = nn.Linear(dim, dim)
        self.final_linear = nn.Linear(dim, dim)
        
    def forward(self, text_feat, image_feat):
        batch_size, seq_len, _ = text_feat.size()
        _, num_regions, _ = image_feat.size()

        text_query = self.query_transform(text_feat).view(batch_size, seq_len, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        image_key = self.key_transform(image_feat).view(batch_size, num_regions, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        image_value = self.value_transform(image_feat).view(batch_size, num_regions, self.num_heads, self.dim // self.num_heads).transpose(1, 2)

        attention_scores = torch.matmul(text_query, image_key.transpose(-2, -1)) / (self.dim // self.num_heads) ** 0.5
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, image_value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.final_linear(weighted_values)
        return output + text_feat

class SimpleAttention(nn.Module):
    """简单的注意力机制，用于特征加权
    
    这个模块为输入的特征计算注意力权重，可用于加权池化
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, features):
        """
        Args:
            features: 输入特征 [batch_size, seq_len, feature_dim]
            
        Returns:
            attention_weights: 注意力权重 [batch_size, seq_len, 1]
        """
        # 计算注意力分数
        attention_scores = self.attn(features)
        
        # 对seq_len维度进行softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights
        
    def apply(self, features):
        """应用注意力权重到特征上
        
        Args:
            features: 输入特征 [batch_size, seq_len, feature_dim]
            
        Returns:
            weighted_features: 加权后的特征 [batch_size, feature_dim]
        """
        # 计算注意力权重
        attention_weights = self.forward(features)
        
        # 应用权重到特征上
        weighted_features = (features * attention_weights).sum(dim=1)
        
        return weighted_features

class EnhancedSimilarityModule(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=256):
        super().__init__()
        self.temperature = 0.5  # 小温度参数
        
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, features1, features2, negative_features=None):
        # 处理输入形状
        if features1.dim() > 2:
            b, s, d = features1.size()
            features1 = features1.reshape(-1, d)
        if features2.dim() > 2:
            b, s, d = features2.size()
            features2 = features2.reshape(-1, d)
            
        # 正样本对
        pos_sim = self.compute_similarity(features1, features2)
        
        # 负样本对
        if negative_features is not None:
            neg_sims = []
            for neg_feat in negative_features:
                if neg_feat.dim() > 2:
                    b, s, d = neg_feat.size()
                    neg_feat = neg_feat.reshape(-1, d)
                neg_sim = self.compute_similarity(features1, neg_feat)
                neg_sims.append(neg_sim)
            
            # InfoNCE loss思想
            if len(neg_sims) > 0:
                all_sims = torch.cat([pos_sim.unsqueeze(0), torch.stack(neg_sims)])
                all_sims = all_sims / self.temperature
                
                # 打印相似度值
                print(f"正样本相似度: {pos_sim.mean().item():.4f}, 负样本相似度均值: {torch.stack(neg_sims).mean().item():.4f}")
                
                # 正样本应该比负样本相似度高
                sim = F.softmax(all_sims, dim=0)[0]
                print(f"对比学习后相似度: {sim.mean().item():.4f}")
                return sim
        
        # 如果没有负样本，映射到0-1范围
        final_sim = torch.sigmoid(pos_sim / self.temperature)
        print(f"单样本相似度: {final_sim.mean().item():.4f}")
        return final_sim

    def compute_similarity(self, f1, f2):
        # 应用轻量级变换并添加残差连接
        f1 = self.transform(f1) * 0.1 + f1  # 残差连接
        f2 = self.transform(f2) * 0.1 + f2
        
        # 归一化
        f1 = F.normalize(f1, p=2, dim=-1)
        f2 = F.normalize(f2, p=2, dim=-1)
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(f1, f2)
        
        return cos_sim
