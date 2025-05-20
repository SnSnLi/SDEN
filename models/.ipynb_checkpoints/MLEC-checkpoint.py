import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_modules import MultiHeadAttention, EmergenceCore, BidirectionalEmergenceCore, CrossModalAttention
from .topology import DynamicTopologyCoupler

############################
# 各子模块定义保持不变
############################

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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.proj(out), attn

class EntropyController(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.entropy_estimator = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 1)
        )
        self.target_entropy = nn.Parameter(torch.ones(1) * math.log(dim))
        self.temperature = nn.Parameter(torch.ones(1))
        
    def compute_entropy(self, features):
        batch_size, seq_len, feat_dim = features.shape
        features_2d = features.view(-1, feat_dim)
        kernel = torch.exp(-torch.cdist(features_2d, features_2d) ** 2 / (2 * self.temperature ** 2))
        probs = kernel / kernel.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
        entropy = entropy.view(batch_size, seq_len)
        return entropy
    
    def forward(self, features):
        current_entropy = self.compute_entropy(features)
        estimated_entropy = self.entropy_estimator(features).squeeze(-1)
        entropy_diff = current_entropy - self.target_entropy
        control_signal = torch.sigmoid(-entropy_diff / self.temperature)
        controlled_features = features * control_signal.unsqueeze(-1)
        return controlled_features, control_signal

class PhaseTransitionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(dim*2, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, 1)
        )
        self.temperature = nn.Parameter(torch.tensor(0.1))
        self.entropy_controller = EntropyController(dim)
        
    def forward(self, text_feat, image_feat):
        joint_feat = torch.cat([text_feat, image_feat], dim=-1)
        energy = self.energy_net(joint_feat) / self.temperature.view(1, 1, -1)
        phase = torch.sigmoid(energy)
        text_feat, text_entropy_weights = self.entropy_controller(text_feat)
        image_feat, image_entropy_weights = self.entropy_controller(image_feat)
        return phase, text_feat, image_feat, (text_entropy_weights + image_entropy_weights) / 2

class LocalFeatureAligner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.phase_transition = PhaseTransitionLayer(dim)
        self.cross_attention = CrossModalAttention(dim)
        
    def forward(self, text_feat, image_feat):
        phase, text_feat_perturbed, image_feat_perturbed, entropy_weights = self.phase_transition(text_feat, image_feat)
        aligned_feat = self.cross_attention(text_feat_perturbed, image_feat_perturbed)
        return aligned_feat * phase.expand_as(aligned_feat), entropy_weights

class DynamicTopoNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.graph_gen = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        self.entropy_controller = EntropyController(dim)
    
    def forward(self, x):
        x_permuted = x.permute(1, 0, 2)
        attn_out, _ = self.attention(x_permuted, x_permuted, x_permuted)
        attn_out = attn_out.permute(1, 0, 2)
        attn_out, entropy_weights = self.entropy_controller(attn_out)
        gen_out = self.graph_gen(attn_out)  # [B, seq_len, dim]
        return gen_out, entropy_weights

class EntropyGateLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1)
        )
        
    def forward(self, graph):
        entropy = -torch.sum(graph * torch.log(torch.clamp(graph, min=1e-15, max=1.0)), dim=-1, keepdim=True)
        gate = torch.sigmoid(self.mlp(entropy))
        return graph * gate

class SemanticGraphBuilder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dynamic_topology = DynamicTopoNet(dim)
    
    def forward(self, local_feat):
        # local_feat: [B, 1, dim]
        features, entropy_weights = self.dynamic_topology(local_feat)
        return features, entropy_weights

class CriticalTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, 8)
        self.critical_mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, x):
        x_permuted = x.permute(1, 0, 2)
        attn_out, _ = self.self_attention(x_permuted, x_permuted, x_permuted)
        attn_out = attn_out.permute(1, 0, 2)
        return self.critical_mlp(attn_out) + x

class EmergencePredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, critical_state):
        return self.predictor(critical_state)

class EmergenceCore(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.phase_transition = PhaseTransitionLayer(dim)
        self.critical_transformer = CriticalTransformer(dim)
        self.emergence_predictor = EmergencePredictor(dim)
        self.cross_modal = CrossModalAttention(dim)
        
    def forward(self, x, context=None):
        if context is not None:
            phase_state, x_perturbed, context_perturbed, _ = self.phase_transition(x, context)
            x = x_perturbed * phase_state.expand_as(x)
            context = context_perturbed
        critical_state = self.critical_transformer(x)
        if context is not None:
            critical_state = self.cross_modal(critical_state, context)
        emerged = self.emergence_predictor(critical_state)
        return emerged

class GlobalEmergenceLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.critical_transformer = CriticalTransformer(dim)
        self.emergence_predictor = EmergencePredictor(dim)
        
    def forward(self, semantic_graph):
        critical_state = self.critical_transformer(semantic_graph)
        emerged_feat = self.emergence_predictor(critical_state)
        return emerged_feat

class ScaleInteractionModule(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sum(dims), sum(dims)),  # 例如3072 -> 3072
            nn.GELU()
        )
        
    def forward(self, features):
        concat_feats = torch.cat(features, dim=-1)
        return self.mlp(concat_feats)

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
        self.proj_final = nn.Identity()
        
    def forward(self, text_feat, image_feat):
        text_emerged = self.text_emergence(text_feat, image_feat)
        image_emerged = self.image_emergence(image_feat, text_feat)
        text_context = self.cross_modal(text_emerged, image_emerged)
        image_context = self.cross_modal(image_emerged, text_emerged)
        text_final = self.fusion(torch.cat([text_emerged, text_context], dim=-1))
        image_final = self.fusion(torch.cat([image_emerged, image_context], dim=-1))
        text_final = self.proj_final(text_final)
        image_final = self.proj_final(image_final)
        return text_final, image_final

#####################################
# 修改 MultiScaleEmergenceModule
#####################################
class MultiScaleEmergenceModule(nn.Module):
    def __init__(self, base_dim=1024):
        super().__init__()
        self.micro_layer = LocalFeatureAligner(base_dim)
        self.micro_to_meso = nn.Linear(base_dim, base_dim)
        self.meso_layer = SemanticGraphBuilder(base_dim)
        self.macro_layer = GlobalEmergenceLayer(base_dim)
        self.bidirectional = BidirectionalEmergenceCore(base_dim)
        self.scale_interaction = ScaleInteractionModule([base_dim, base_dim, base_dim])
    
    def forward(self, text_feat, image_feat):
        """
        返回 5 个值：
          1) final_text
          2) final_image
          3) global_emerged
          4) entropy_weights
          5) semantic_graph
        """
        local_feat, micro_entropy_weights = self.micro_layer(text_feat, image_feat)
        local_feat = self.micro_to_meso(local_feat)
        semantic_graph, meso_entropy_weights = self.meso_layer(local_feat)
        global_emerged = self.macro_layer(semantic_graph)
        text_emerged, image_emerged = self.bidirectional(text_feat, image_feat)
        final_text = self.scale_interaction([local_feat, semantic_graph, text_emerged])
        final_image = self.scale_interaction([local_feat, semantic_graph, image_emerged])
        entropy_weights = (micro_entropy_weights + meso_entropy_weights) / 2
        return final_text, final_image, global_emerged, entropy_weights, semantic_graph

#####################################
# EmergenceModel：统一返回9元组
#####################################
class EmergenceModel(nn.Module):
    def __init__(self, dim=1024, text_input_dim=768, image_input_dim=768, num_classes=None, modality_dims=None):
        super().__init__()
        self.is_training = True
        modality_dims = {'text': text_input_dim, 'image': image_input_dim}
        self.projections = nn.ModuleDict({k: nn.Linear(v, dim) for k, v in modality_dims.items()})
        self.multi_scale = MultiScaleEmergenceModule(base_dim=dim)
        self.emergence_core = EmergenceCore(dim)
        self.num_classes = num_classes
        if num_classes:
            self.classifier = nn.Linear(dim*4, num_classes)
        self.contrastive_temp = nn.Parameter(torch.ones(1) * 0.07)
    
    def contrastive_loss(self, text_feat, image_feat, entropy_weights):
        print(f"contrastive_loss input shapes - Text: {text_feat.shape}, Image: {image_feat.shape}")
        
        # 确保输入是2D [batch, features]
        text_flat = text_feat.view(-1, text_feat.size(-1))
        image_flat = image_feat.view(-1, image_feat.size(-1))
        
        # 归一化特征
        text_norm = F.normalize(text_flat, dim=-1)
        image_norm = F.normalize(image_flat, dim=-1)
        
        print(f"Normalized shapes - Text: {text_norm.shape}, Image: {image_norm.shape}")
        
        # 调整温度参数范围
        temp = torch.clamp(self.contrastive_temp, min=0.01, max=1.0)
        
        # 计算相似度矩阵 [batch, batch]
        sim_matrix = torch.matmul(text_norm, image_norm.transpose(-2, -1)) / temp
        
        # 创建标签 [0, 1, ..., batch-1]
        batch_size = text_norm.size(0)
        labels = torch.arange(batch_size).to(text_norm.device)
        
        # 计算对比损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        # 添加熵权重调节
        weighted_loss = loss * torch.sigmoid(entropy_weights.mean())
        
        return weighted_loss
    
    def forward(self, text_feat=None, image_feat=None, labels=None):
        """
        统一返回9元组：
        (final_text, final_image, global_emerged, logits, total_loss,
         semantic_graph, emerged_raw, adjacency, entropy_ranking)
        """
        if text_feat is None and image_feat is None:
            raise ValueError("至少需要提供文本或图像特征之一")
        # 保证输入是3D
        if text_feat is not None and text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if image_feat is not None and image_feat.dim() == 2:
            image_feat = image_feat.unsqueeze(1)
        
        if self.is_training:
            # 训练模式，需要 text_feat 和 image_feat 都不为 None
            if text_feat is None or image_feat is None:
                raise ValueError("训练模式需要同时提供文本和图像特征")
            
            # 投影
            proj_text = self.projections['text'](text_feat)
            proj_image = self.projections['image'](image_feat)
            # multi_scale 返回 5 个值
            final_text, final_image, global_emerged, entropy_weights, semantic_graph = self.multi_scale(proj_text, proj_image)
            
            # 计算损失
            consistency_loss = -F.cosine_similarity(final_text.mean(dim=1), final_image.mean(dim=1)).mean()
            contrastive_loss = self.contrastive_loss(final_text, final_image, entropy_weights)
            total_loss = consistency_loss + contrastive_loss
        
        else:
            # 推理模式
            if text_feat is not None and image_feat is not None:
                proj_text = self.projections['text'](text_feat)
                proj_image = self.projections['image'](image_feat)
                final_text, final_image, global_emerged, entropy_weights, semantic_graph = self.multi_scale(proj_text, proj_image)
                total_loss = None
            elif text_feat is not None:
                proj_text = self.projections['text'](text_feat)
                final_text = self.emergence_core(proj_text)
                final_image = None
                global_emerged = final_text
                entropy_weights = None
                semantic_graph = None
                total_loss = None
            else:  # image_feat is not None
                proj_image = self.projections['image'](image_feat)
                final_image = self.emergence_core(proj_image)
                final_text = None
                global_emerged = final_image
                entropy_weights = None
                semantic_graph = None
                total_loss = None
        
        # 计算 emerged_raw, adjacency, entropy_ranking
        if semantic_graph is not None:
            if (final_text is not None) and (final_image is not None):
                emerged_raw = torch.cat([final_text, final_image], dim=-1)
            else:
                emerged_raw = global_emerged  # 单模态时可以直接用 global_emerged
            normalized_semantic = F.normalize(semantic_graph, dim=-1)
            adjacency = torch.matmul(normalized_semantic, normalized_semantic.transpose(-2, -1))
            if entropy_weights is not None:
                entropy_ranking = torch.argsort(entropy_weights, dim=-1)
            else:
                entropy_ranking = None
        else:
            emerged_raw = None
            adjacency = None
            entropy_ranking = None
        
        # 如果有分类器，则根据 global_emerged 计算 logits
        if (self.num_classes and global_emerged is not None):
            logits = self.classifier(global_emerged.mean(dim=1))
        else:
            logits = None
        
        # 最终返回9元组
        return (
            final_text,         # 1
            final_image,        # 2
            global_emerged,     # 3
            logits,             # 4
            total_loss,         # 5
            semantic_graph,     # 6
            emerged_raw,        # 7
            adjacency,          # 8
            entropy_ranking     # 9
        )
    
    def _forward_train(self, text_feat, image_feat, **modalities):
        # 直接调用 forward
        return self.forward(text_feat, image_feat, labels=modalities.get('labels', None))
    
    def forward_image(self, image_feat):
        feat = self.projections['image'](image_feat)
        emerged = self.emergence_core(feat)
        return emerged
    
    def consistency_loss(self, final_text, final_image):
        return -F.cosine_similarity(final_text.mean(dim=1), final_image.mean(dim=1)).mean()
