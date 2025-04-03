import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_modules import (
    EmergenceCore,
    CrossModalAttention,
    BidirectionalEmergenceCore
)
from .emergence import (
    MultiScaleEmergenceModule,
    ScaleInteractionModule
)

class DualEmergenceOptimizer(nn.Module):
    def __init__(self, dim, feature_dim, temperature=0.1, alpha=0.5, consistency_threshold=0.05, 
                 patience=3):
        super().__init__()
        self.is_training = True  # 添加训练标志
        self.temperature = temperature
        self.alpha = alpha
        self.below_threshold_count = 0  
        self.should_stop = False
        self.feature_dim = feature_dim
        self.emergence_core = EmergenceCore(feature_dim)
        self.cross_modal_attention = CrossModalAttention(feature_dim)
        self.scale_interaction = ScaleInteractionModule([feature_dim, feature_dim * 2])
        self.bidirectional_core = BidirectionalEmergenceCore(feature_dim)
        
        self.distribution_estimator = CriticalDistributionEstimator(
            feature_dim, self.emergence_core, self.cross_modal_attention
        )
        self.parameter_optimizer = AdaptiveParameterOptimizer(
            feature_dim, self.scale_interaction
        )
        self.consistency_loss = SymmetricConsistencyLoss()

    def check_consistency(self, loss_value): 
        if loss_value < self.consistency_threshold:
            self.below_threshold_count += 1
            if self.below_threshold_count >= self.patience:
                self.should_stop = True
        else:
            self.below_threshold_count = 0
        return self.should_stop

    def forward(self, text_features=None, image_features=None):
        if not self.is_training:
            # 推理时的单向处理
            if text_features is not None:
                return self.emergence_core(text_features), None, None, None, False
            elif image_features is not None:
                return None, self.emergence_core(image_features), None, None, False
            else:
                raise ValueError("At least one of text_features or image_features must be provided")
        else:
            # 训练时的双向处理
            return self._forward_train(text_features, image_features)
            
    def _forward_train(self, text_features, image_features):
        if text_features is None or image_features is None:
            raise ValueError("Both text_features and image_features must be provided in training mode")
            
        distribution, emergence_state = self.distribution_estimator(text_features, image_features)
        current_params = torch.cat([text_features, image_features], dim=-1)
        optimized_params, scale_weights = self.parameter_optimizer(distribution, current_params)
        split_size = text_features.size(-1)
        text_params, image_params = torch.split(optimized_params, [split_size, split_size], dim=-1)
        text_emerged, image_emerged = self.bidirectional_core(text_params, image_params)
        consistency_loss = self.consistency_loss(text_emerged, image_emerged, distribution)
        emergence_state = torch.sigmoid(emergence_state)
        emergence_weighted_loss = consistency_loss * emergence_state
        should_stop = self.check_consistency(consistency_loss.item())
        return text_emerged, image_emerged, emergence_weighted_loss, consistency_loss, should_stop

class CriticalDistributionEstimator(nn.Module):
    def __init__(self, feature_dim, emergence_core, cross_attention):
        super().__init__()
        self.emergence_core = emergence_core
        self.cross_attention = cross_attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, text_features, image_features):
        emergence_state = self.emergence_core(torch.cat([text_features, image_features], dim=-1))
        attn_output = self.cross_attention(text_features, image_features)
        temp = torch.clamp(self.temperature, min=1e-3)
        distribution = F.softmax(attn_output / (temp * emergence_state), dim=-1)
        return distribution, emergence_state

class AdaptiveParameterOptimizer(nn.Module):
    def __init__(self, feature_dim, scale_interaction):
        super().__init__()
        self.scale_interaction = scale_interaction
        self.param_predictor = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, distribution, current_params):
        scale_weights = self.scale_interaction([distribution, current_params])
        param_update = self.param_predictor(torch.cat([distribution, current_params], dim=-1))
        optimized_params = current_params + scale_weights * param_update
        return optimized_params, scale_weights

class SymmetricConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, text_features, image_features, distribution):
        text_dist = F.softmax(text_features, dim=-1)
        img_dist = F.softmax(image_features, dim=-1)
        m = 0.5 * (text_dist + img_dist)
        jsd = 0.5 * (F.kl_div(text_dist.log(), m, reduction='batchmean') +
                     F.kl_div(img_dist.log(), m, reduction='batchmean'))
        return jsd

