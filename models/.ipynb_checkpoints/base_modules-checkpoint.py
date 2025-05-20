import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # 计算原始注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 增强稀疏性和多样性
        if seq_len > 1:
            # 增强稀疏性设置
            k_local = max(1, min(seq_len // 8, 3))  # 更严格的局部稀疏
            k_global = max(1, min(seq_len // 4, 5))  # 更严格的全局稀疏
            
            # 动态窗口注意力
            window_size = min(max(seq_len//2, 2), 4)  # 更小的窗口
            temperature = 0.05  # 更低的温度增强区分度
            local_mask = torch.ones_like(scores)
            for i in range(0, seq_len, max(1, window_size//2)):
                j = min(i+window_size, seq_len)
                if j - i < k_local:  # 窗口内元素不足k_local则跳过
                    continue
                local_scores = scores[..., i:j, i:j]
                topk_local, _ = torch.topk(local_scores, 
                                         k=min(k_local, j-i), 
                                         dim=-1)
                if topk_local.size(-1) > 0:  # 确保topk结果不为空
                    mask = (local_scores >= topk_local[..., -1:]).float()
                    local_mask[..., i:j, i:j] = mask
            
            # 全局稀疏
            topk_global, _ = torch.topk(scores, k=k_global, dim=-1)
            global_mask = (scores >= topk_global[..., -1:]).float()
            
            # 组合掩码
            combined_mask = (local_mask * global_mask)
            scores = scores.masked_fill(combined_mask == 0, -1e9)
        
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
