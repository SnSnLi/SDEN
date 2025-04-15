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
        self.is_training = True  
        self.temperature = temperature
        self.alpha = alpha
        self.consistency_threshold = consistency_threshold  # 保存为实例变量
        self.patience = patience  # 保存为实例变量
        self.below_threshold_count = 0  
        self.should_stop = False
        self.feature_dim = feature_dim
        
        # 使用正确维度的EmergenceCore
        self.emergence_core = EmergenceCore(feature_dim)
        # 修改CrossModalAttention，确保它能处理降维后的特征
        self.cross_modal_attention = CrossModalAttention(feature_dim)
        self.scale_interaction = ScaleInteractionModule([feature_dim, feature_dim])
        self.bidirectional_core = BidirectionalEmergenceCore(feature_dim)
        
        self.distribution_estimator = CriticalDistributionEstimator(
            feature_dim, self.emergence_core, self.cross_modal_attention
        )
        self.parameter_optimizer = AdaptiveParameterOptimizer(
            feature_dim, self.scale_interaction
        )
        self.consistency_loss = SymmetricConsistencyLoss()
        
        
        self.fc = nn.Linear(1536 * 2, feature_dim)  # 将拼接后的3072维降为512
        
        # 添加投影层，将分割后的256维特征映射回512维
        self.text_projection = nn.Linear(feature_dim // 2, feature_dim)
        self.image_projection = nn.Linear(feature_dim // 2, feature_dim)
        
        # 添加残差连接层
        self.residual_fc = nn.Linear(feature_dim, feature_dim)

    
    def check_consistency(self, loss_value): 
        """检查一致性损失是否低于阈值，决定是否应该停止训练"""
        if loss_value < self.consistency_threshold:
            self.below_threshold_count += 1
            if self.below_threshold_count >= self.patience:
                self.should_stop = True
        else:
            self.below_threshold_count = 0
        return self.should_stop

    def forward(self, text_features=None, image_features=None):
        # 打印输入特征的形状，用于调试
        if text_features is not None:
            print(f"Text features shape in forward: {text_features.shape}")
        if image_features is not None:
            print(f"Image features shape in forward: {image_features.shape}")
            
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
        
        # 确保维度一致性
        if text_features.dim() > 2:
            batch_size = text_features.size(0)
            text_features_flat = text_features.reshape(batch_size, -1)
            image_features_flat = image_features.reshape(batch_size, -1)

            
            # 拼接特征
            concatenated_features = torch.cat([text_features_flat, image_features_flat], dim=-1)

        else:
            # 如果已经是2D的，直接拼接
            concatenated_features = torch.cat([text_features, image_features], dim=-1)
            
        # 降维处理 - 这里将3072维降为512
        reduced_features = self.fc(concatenated_features)
        
        # 应用残差连接
        residual = self.residual_fc(reduced_features)
        reduced_features = reduced_features + residual
        
        # 为后续处理重新整形，确保形状是 [batch_size, 1, feature_dim]
        if reduced_features.dim() == 2:
            reduced_features = reduced_features.unsqueeze(1)

        # 分布估计
        distribution, emergence_state = self.distribution_estimator(reduced_features)
        
        # 参数优化
        optimized_params, scale_weights = self.parameter_optimizer(distribution, reduced_features)
        
        # 特征分割 - 将512维分成两个256维
        split_size = self.feature_dim // 2
        text_params, image_params = torch.split(optimized_params, [split_size, split_size], dim=-1)
        
        # 投影回512维 - 解决嵌入维度不匹配问题
        text_params_projected = self.text_projection(text_params)
        image_params_projected = self.image_projection(image_params)
        
        # 双向处理 - 使用投影后的512维特征
        text_emerged, image_emerged = self.bidirectional_core(text_params_projected, image_params_projected)
        
        # 一致性损失
        consistency_loss = self.consistency_loss(text_emerged, image_emerged, distribution)
        
        # 处理emergence_state
        emergence_state = torch.sigmoid(emergence_state)
        emergence_weighted_loss = consistency_loss * emergence_state
        
        # 使用check_consistency方法检查是否应该停止训练
        should_stop = self.check_consistency(consistency_loss.item())
        
        return text_emerged, image_emerged, emergence_weighted_loss, consistency_loss, should_stop


class AdaptiveParameterOptimizer(nn.Module):
    def __init__(self, feature_dim, scale_interaction):
        super().__init__()
        self.scale_interaction = scale_interaction
        self.feature_dim = feature_dim
        
        # 输入维度为feature_dim*2，输出维度为feature_dim
        self.param_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 添加一个投影层，将1024维压缩到512维
        self.scale_projection = nn.Linear(1024, feature_dim)
        
    def forward(self, distribution, current_params):
        # 打印输入的形状，用于调试
        
        # 使用scale_interaction生成scale_weights（形状为[1, 1, 1024]）
        try:
            scale_weights = self.scale_interaction([distribution, current_params])

            # 如果scale_weights是1024维，使用投影层将其降为512维
            if scale_weights.size(-1) == 1024 and current_params.size(-1) == 512:
                # 投影到与current_params相同的维度
                scale_weights = self.scale_projection(scale_weights)
            
        except Exception as e:
            print(f"Error with scale_interaction: {e}")
            # 创建形状与current_params相同的全1张量作为fallback
            scale_weights = torch.ones_like(current_params)
            print(f"Using fallback scale weights: {scale_weights.shape}")
            
        # 准备concat_input用于param_predictor
        # 如果distribution和current_params的最后一维不同，调整distribution
        if distribution.size(-1) != current_params.size(-1):
            # 创建一个临时线性层，动态地处理不同维度
            temp_projection = nn.Linear(
                distribution.size(-1), current_params.size(-1)
            ).to(distribution.device)
            distribution_adjusted = temp_projection(distribution)
        else:
            distribution_adjusted = distribution
            
        # 拼接调整后的distribution与current_params
        concat_input = torch.cat([distribution_adjusted, current_params], dim=-1)
       
        # 使用param_predictor生成更新
        param_update = self.param_predictor(concat_input)


        # 计算优化后的参数
        optimized_params = current_params + scale_weights * param_update
        
        return optimized_params, scale_weights


class CriticalDistributionEstimator(nn.Module):
    def __init__(self, feature_dim, emergence_core, cross_attention):
        super().__init__()
        self.emergence_core = emergence_core
        self.cross_attention = cross_attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, combined_features):
        # 直接使用组合特征
        emergence_state = self.emergence_core(combined_features)
        # 使用self-attention处理
        attn_output = self.cross_attention(combined_features, combined_features)
        temp = torch.clamp(self.temperature, min=1e-3)
        distribution = F.softmax(attn_output / (temp * emergence_state), dim=-1)
        return distribution, emergence_state


class SymmetricConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, text_features, image_features, distribution):
        # 确保输入特征具有正确的形状
        if text_features.dim() > 2:
            text_features = text_features.mean(dim=1)
        if image_features.dim() > 2:
            image_features = image_features.mean(dim=1)
            
        text_dist = F.softmax(text_features, dim=-1)
        img_dist = F.softmax(image_features, dim=-1)
        m = 0.5 * (text_dist + img_dist)
        
        # 使用sum reduction，然后自己计算平均值，以避免维度不匹配
        jsd = 0.5 * (F.kl_div(text_dist.log(), m, reduction='sum') +
                     F.kl_div(img_dist.log(), m, reduction='sum'))
        
        # 归一化为每个样本的平均损失
        jsd = jsd / text_features.size(0)
        return jsd
