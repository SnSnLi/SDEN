import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_modules import (
    MultiHeadAttention,
    EmergenceCore,
    BidirectionalEmergenceCore,
    CrossModalAttention
)
from .emergence import MultiScaleEmergenceModule
from .topology import (
    DynamicTopologyCoupler,
    EntropyController,
    CriticalDynamicsController
)
from .dual import DualEmergenceOptimizer

class SymmetricDynamicEmergenceNetwork(nn.Module):
    def __init__(self, dim=1024, num_heads=8, num_layers=4, temperature=0.1):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        
        # 多尺度涌现模块（内部返回5个值）
        self.emergence_module = MultiScaleEmergenceModule(base_dim=dim)
        self.bidirectional_core = BidirectionalEmergenceCore(dim=dim)
        self.cross_modal = CrossModalAttention(dim=dim)
        
        self.topology_coupler = DynamicTopologyCoupler(dim=dim, num_heads=num_heads)
        self.entropy_controller = EntropyController(dim=dim)
        self.critical_controller = CriticalDynamicsController(dim=dim)
        
        self.dual_optimizer = DualEmergenceOptimizer(dim=dim, feature_dim=dim, temperature=temperature)
        
        self.classifier = nn.Linear(dim, dim)
        self.contrastive_temp = nn.Parameter(torch.ones(1) * 0.07)
        self.is_training = True

    def contrastive_loss(self, text_feat, image_feat, entropy_weights):
        # 假设输入已经为 [B, D]
        text_norm = F.normalize(text_feat, dim=-1)
        image_norm = F.normalize(image_feat, dim=-1)
        logits = torch.matmul(text_norm, image_norm.transpose(-2, -1)) / self.contrastive_temp
        labels = torch.arange(text_norm.size(0)).to(text_norm.device)
        loss = F.cross_entropy(logits, labels)
        return loss * entropy_weights.mean()

    def emergence_forward(self, text_features, image_features):
        # 双向涌现，得到各模态融合后的特征
        text_emerged, image_emerged = self.bidirectional_core(text_features, image_features)
        final_text, final_image, global_emerged, entropy_weights = self.emergence_module(text_emerged, image_emerged)
        fused_features = self.cross_modal(final_text, final_image)
        return fused_features, final_text, final_image, entropy_weights

    def topology_forward(self, emerged_features, entropy_weights):
        topo_output = self.topology_coupler(emerged_features)
        controlled_features, topo_entropy_weights = self.entropy_controller(topo_output['output'])
        combined_entropy_weights = (entropy_weights + topo_entropy_weights) / 2
        critical_features = self.critical_controller(controlled_features)
        return {
            'features': critical_features,
            'entropy_weights': combined_entropy_weights,
            'adj_matrix': topo_output['adj_matrix']
        }

    def dual_forward(self, topo_features, labels=None):
        optimized_features = self.dual_optimizer(topo_features['features'], labels=labels if self.is_training else None)
        logits = self.classifier(optimized_features)
        return logits, optimized_features

    def forward(self, text_features=None, image_features=None, labels=None):
        """
        统一返回9元组：
         (final_text, final_image, global_emerged, logits, total_loss,
          semantic_graph, emerged_raw, adjacency, entropy_ranking)
        输入假设为3D张量 [B, seq_len, dim]；若为2D则自动 unsqueeze。
        """
        # 检查输入
        if text_features is None and image_features is None:
            raise ValueError("至少需要提供文本或图像特征之一")
        
        # 保证输入为3D
        if text_features is not None and text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        if image_features is not None and image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        
        if text_features is not None and image_features is not None:
            # 多模态情况
            fused_features, final_text, final_image, entropy_weights = self.emergence_forward(text_features, image_features)
            topo_features = self.topology_forward(fused_features, entropy_weights)
            entropy_ranking = torch.argsort(topo_features['entropy_weights'], dim=-1)
            logits, final_features = self.dual_forward(topo_features, labels)
            if self.is_training:
                text_emerged = self.forward_text(text_features)
                image_emerged = self.forward_image(image_features)
                consistency_loss = -F.cosine_similarity(text_emerged.mean(dim=1), image_emerged.mean(dim=1)).mean()
                contrastive_loss = self.contrastive_loss(final_text, final_image, topo_features['entropy_weights'])
                total_loss = consistency_loss + contrastive_loss
            else:
                total_loss = None
            emerged_raw = fused_features  # 使用融合特征作为原始特征
            global_emerged = fused_features
            adjacency = topo_features['adj_matrix']
            semantic_graph = topo_features.get('features', None)
        elif text_features is not None:
            # 单模态（文本）
            text_emerged = self.forward_text(text_features)
            topo_features = self.topology_forward(text_emerged, torch.ones_like(text_emerged[..., 0]))
            entropy_ranking = torch.argsort(topo_features['entropy_weights'], dim=-1)
            logits, final_features = self.dual_forward(topo_features, labels)
            total_loss = None
            final_text = text_emerged
            final_image = None
            global_emerged = text_emerged
            emerged_raw = text_emerged
            semantic_graph = None
            adjacency = topo_features['adj_matrix']
        elif image_features is not None:
            # 单模态（图像）
            image_emerged = self.forward_image(image_features)
            topo_features = self.topology_forward(image_emerged, torch.ones_like(image_emerged[..., 0]))
            entropy_ranking = torch.argsort(topo_features['entropy_weights'], dim=-1)
            logits, final_features = self.dual_forward(topo_features, labels)
            total_loss = None
            final_text = None
            final_image = image_emerged
            global_emerged = image_emerged
            emerged_raw = image_emerged
            semantic_graph = None
            adjacency = topo_features['adj_matrix']
        else:
            raise ValueError("At least one of text_features or image_features must be provided")
        
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

    def forward_text(self, text_features):
        text_emerged = self.bidirectional_core.text_emergence(text_features)
        return text_emerged

    def forward_image(self, image_features):
        image_emerged = self.bidirectional_core.image_emergence(image_features)
        return image_emerged

    @torch.no_grad()
    def extract_features(self, text_features, image_features):
        self.eval()
        output = self.forward(text_features, image_features)
        return output[2]  # global_emerged

    def set_inference_mode(self, mode=True):
        self.dual_optimizer.inference_mode = mode

class SDENModel(nn.Module):
    def __init__(self, feature_dim=1024, temperature=0.1):
        super().__init__()
        self.dual_optimizer = DualEmergenceOptimizer(dim=feature_dim, feature_dim=feature_dim, temperature=temperature)
        self.emergence = SymmetricDynamicEmergenceNetwork(dim=feature_dim)
        self.is_training = True

    def set_training_mode(self, mode=True):
        self.is_training = mode
        self.dual_optimizer.is_training = mode
        self.emergence.is_training = mode

    def forward(self, x):
        self.set_training_mode(self.is_training)
        return self.dual_optimizer(x)
