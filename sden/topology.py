import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_modules import MultiHeadAttention, EmergenceCore, BidirectionalEmergenceCore

class PhaseMapper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        
    def forward(self, x):
        return self.mapper(x)

class CriticalDynamicsController(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.order_param = nn.Parameter(torch.ones(1))
        self.controller = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
    def forward(self, x):
        critical_state = self.controller(x)
        return critical_state * torch.sigmoid(self.order_param)

class AdaptiveGraphGenerator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.LayerNorm(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.threshold = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        node_feats = self.node_encoder(x)
        batch_size, num_nodes, _ = node_feats.shape
        node_pairs = torch.cat([
            node_feats.unsqueeze(2).expand(-1, -1, num_nodes, -1),
            node_feats.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        ], dim=-1)
        edge_probs = F.gumbel_softmax(self.edge_predictor(node_pairs), tau=1, hard=False)
        adj_matrix = (edge_probs > self.threshold).float()
        return adj_matrix, node_feats

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
        kernel = torch.exp(-torch.cdist(features, features) ** 2 / (2 * self.temperature ** 2))
        probs = kernel / kernel.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
        return entropy
    
    def forward(self, features, adj_matrix=None):
        current_entropy = self.compute_entropy(features)
        estimated_entropy = self.entropy_estimator(features)
        entropy_diff = current_entropy - self.target_entropy
        control_signal = torch.sigmoid(-entropy_diff / self.temperature)
        controlled_features = features * control_signal.unsqueeze(-1)
        if adj_matrix is not None:
            controlled_adj = adj_matrix * control_signal.unsqueeze(-1).unsqueeze(-1)
            return controlled_features, control_signal, controlled_adj
        return controlled_features, control_signal

class AsymmetricFlowController(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 模态特定的特征转换
        self.text_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.image_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # 动态流动权重预测器
        self.flow_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 2),  # 2个权重：text->image和image->text
            nn.Sigmoid()
        )
        
        # 层级自适应门控
        self.level_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    # 修复AsymmetricFlowController的forward方法中的维度问题
    def forward(self, text_feat, image_feat, entropy_weights=None):
        # 模态特定特征提取
        text_transformed = self.text_transform(text_feat)
        image_transformed = self.image_transform(image_feat)
        
        # 预测双向流动权重
        combined = torch.cat([text_transformed, image_transformed], dim=-1)
        flow_weights = self.flow_predictor(combined)  # [B, N, 2]
        text_to_image_weight = flow_weights[..., 0:1]
        image_to_text_weight = flow_weights[..., 1:2]
        
        # 非对称信息流动
        # 确保权重维度与特征维度匹配
        text_flow = text_transformed * text_to_image_weight.expand_as(text_transformed)
        image_flow = image_transformed * image_to_text_weight.expand_as(image_transformed)
        
        # 层级自适应融合
        if entropy_weights is not None:
            # 确保entropy_weights的维度与text_flow和image_flow兼容
            if entropy_weights.dim() == 2:  # [B,N]
                entropy_weights = entropy_weights.unsqueeze(-1)  # [B,N,1]
            elif entropy_weights.dim() == 3 and entropy_weights.size(-1) == 1:  # [B,N,1]
                pass  # 已经是正确形状
            else:
                raise ValueError(f"Unexpected entropy_weights shape: {entropy_weights.shape}")
            
            # 确保所有张量维度匹配
            if entropy_weights.size(1) != text_flow.size(1):
                min_len = min(entropy_weights.size(1), text_flow.size(1))
                entropy_weights = entropy_weights[:, :min_len, :]
                text_flow = text_flow[:, :min_len, :]
                image_flow = image_flow[:, :min_len, :]
            
            # 直接使用扩展后的权重
            entropy_weighted_sum = entropy_weights * (text_flow + image_flow)
            
            level_context = torch.cat([
                text_flow,
                image_flow,
                entropy_weighted_sum
            ], dim=-1)
            
            level_importance = self.level_gate(level_context)
            text_output = text_transformed + level_importance * image_flow
            image_output = image_transformed + level_importance * text_flow
        else:
            text_output = text_transformed + image_flow
            image_output = image_transformed + text_flow
            
        return text_output, image_output

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.asymmetric_controller = AsymmetricFlowController(dim)
        
        # 拓扑感知的特征变换
        self.topo_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # 熵感知的特征增强
        self.entropy_aware = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # 动态融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, entropy_weights=None, modality_type=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        dim = x.size(2)
        
        # 将特征分成两半，分别处理文本和图像部分
        split_point = seq_len // 2
        if seq_len % 2 != 0:
            split_point = (seq_len + 1) // 2
        text_part = x[:, :split_point, :]
        image_part = x[:, split_point:, :]
        
        # 保存原始特征
        original_text = text_part.clone()
        original_image = image_part.clone()
        
        # 调整大小一致性
        if text_part.size(1) != image_part.size(1):
            min_len = min(text_part.size(1), image_part.size(1))
            text_part = text_part[:, :min_len, :]
            image_part = image_part[:, :min_len, :]
            original_text = original_text[:, :min_len, :]
            original_image = original_image[:, :min_len, :]
        
        # 非对称信息流动
        text_enhanced, image_enhanced = self.asymmetric_controller(
            text_part, image_part, entropy_weights
        )
        
        # 特征保留 - 混合原始特征(保留40%)
        text_preserved = 0.6 * text_enhanced + 0.4 * original_text
        image_preserved = 0.6 * image_enhanced + 0.4 * original_image
        
        # 重新组合特征
        x = torch.cat([text_preserved, image_preserved], dim=1)
        
        # 添加随机噪声增加多样性
        noise = torch.rand_like(x) * 0.2 - 0.1
        x = x + noise
        
        # 拓扑感知特征
        topo_features = self.topo_transform(x)
        
        # 熵感知特征增强
        if entropy_weights is not None:
            # 添加随机性到熵权重
            rand_weights = entropy_weights * (1.0 + (torch.rand_like(entropy_weights) * 0.3 - 0.15))
            entropy_context = rand_weights.unsqueeze(-1) * x
            enhanced = self.entropy_aware(torch.cat([x, entropy_context], dim=-1))
        else:
            enhanced = x
        
        # 动态门控融合
        gate_input = torch.cat([topo_features, enhanced], dim=-1)
        gate = self.fusion_gate(gate_input)
        # 添加随机扰动到门控
        gate = torch.clamp(gate + torch.rand_like(gate) * 0.2 - 0.1, 0.1, 0.9)
        
        output = gate * topo_features + (1 - gate) * enhanced
        
        return output

class DynamicTopologyCoupler(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.phase_mapper = PhaseMapper(dim)
        self.graph_gen = AdaptiveGraphGenerator(dim)
        self.entropy_ctrl = EntropyController(dim)
        self.feature_extractor = BidirectionalEmergenceCore(dim)
        self.mha = MultiHeadAttention(dim, num_heads)
        self.critical_controller = CriticalDynamicsController(dim)
        self.feature_fusion = AdaptiveFeatureFusion(dim)

    def forward(self, text_feat=None, image_feat=None):
        # 单模态或多模态处理
        if text_feat is not None and image_feat is not None:
            # 多模态联合涌现 (保留原始特征)
            text_final, image_final = self.feature_extractor(text_feat, image_feat)
            x = torch.cat([
                text_final + 0.5 * text_feat.mean(dim=1, keepdim=True),  # 保留30%原始特征
                image_final + 0.5 * image_feat.mean(dim=1, keepdim=True)
            ], dim=1)
            modality_type = 'multimodal'
        elif text_feat is not None:
            # 单模态（文本）
            x = text_feat
            modality_type = 'text'
        elif image_feat is not None:
            # 单模态（图像）
            x = image_feat
            modality_type = 'image'
        else:
            raise ValueError("At least one of text_feat or image_feat must be provided")

        # 相空间映射
        phase_features = self.phase_mapper(x)
        
        # 动态图生成
        adj_matrix, node_features = self.graph_gen(phase_features)
        
        # 熵控制
        controlled_features, entropy_weights, controlled_adj = self.entropy_ctrl(node_features, adj_matrix)
        
        # 注意力机制
        mha_output, _ = self.mha(controlled_features)
        
        # 特征融合（考虑熵权重和模态类型）
        fused_features = self.feature_fusion(mha_output, entropy_weights, modality_type)
        
        # 临界动力学控制
        critical_state = self.critical_controller(fused_features)
        
        return {
            'output': critical_state,
            'text_features': text_final if text_feat is not None and image_feat is not None else None,
            'image_features': image_final if text_feat is not None and image_feat is not None else None,
            'adj_matrix': controlled_adj,
            'entropy_weights': entropy_weights
        }

def build_topology_network(dim=256, num_heads=8):
    return DynamicTopologyCoupler(dim, num_heads)
