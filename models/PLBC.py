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
    ScaleInteractionModule,
    EntropyController,
    MultiLevelEntropyModule
)
from encoder.clip import CLIPWrapperModel


class AdaptiveParameterOptimizer(nn.Module):
    def __init__(self, feature_dim, scale_interaction):
        super().__init__()
        self.scale_interaction = scale_interaction
        self.feature_dim = feature_dim
        
        # Predictor for parameter updates
        self.param_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        
        self.scale_projection = nn.Linear(512, feature_dim)
        
    def forward(self, distribution, current_params):
        # Adjust dimensions if necessary using scale_interaction
        try:
            scale_weights = self.scale_interaction([distribution, current_params])
            if scale_weights.size(-1) == 512 and current_params.size(-1) == self.feature_dim:
                scale_weights = self.scale_projection(scale_weights)
        except Exception as e:
            print(f"Error with scale_interaction: {e}")
            scale_weights = torch.ones_like(current_params)
            print(f"Using fallback scale weights: {scale_weights.shape}")
            
        # Adjust distribution dimension if mismatched
        if distribution.size(-1) != current_params.size(-1):
            temp_projection = nn.Linear(distribution.size(-1), current_params.size(-1)).to(distribution.device)
            distribution_adjusted = temp_projection(distribution)
        else:
            distribution_adjusted = distribution
            
        # Concatenate and predict parameter updates
        concat_input = torch.cat([distribution_adjusted, current_params], dim=-1)
        param_update = self.param_predictor(concat_input)
        optimized_params = current_params + scale_weights * param_update
        
        return optimized_params, scale_weights


class CriticalDistributionEstimator(nn.Module):
    def __init__(self, feature_dim, emergence_core, cross_attention):
        super().__init__()
        self.emergence_core = emergence_core
        self.cross_attention = cross_attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, combined_features):
        # Compute emergence state and apply cross-attention
        emergence_state = self.emergence_core(combined_features)
        attn_output = self.cross_attention(combined_features, combined_features)
        temp = torch.clamp(self.temperature, min=1e-3)
        distribution = F.softmax(attn_output / (temp * emergence_state), dim=-1)
        return distribution, emergence_state


class SymmetricConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        

    def forward(self, text_features, image_features, distribution=None, cosine_sim=None):
       
        if text_features.dim() > 2:
            text_features = text_features.mean(dim=1)
        if image_features.dim() > 2:
            image_features = image_features.mean(dim=1)
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        cosine = (text_features * image_features).sum(dim=-1)
        loss = 1 - cosine.mean()
        return loss


class PLBC(nn.Module):
    """
    Parameter-Level Bidirectional Coupling:
    θ_i ← θ_i - η (L_i + λ_{j→i} L_j.detach())
    """
    def __init__(self, feature_dim: int, beta: float = 0.1, gamma: float = 0.1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma  # Weight for consistency loss
        self.entropy_ctrl = EntropyController(feature_dim)
        self.multi_entropy = MultiLevelEntropyModule(feature_dim)

    def gradient_alignment(self, grads_text: dict, grads_image: dict):
        # Compute cosine similarity between text and image gradients
        grad_text_sum = sum(g.view(-1) for g in grads_text.values())
        grad_image_sum = sum(g.view(-1) for g in grads_image.values())
        grad_inner_product = (grad_text_sum * grad_image_sum).sum()
        norm_text = grad_text_sum.norm()
        norm_image = grad_image_sum.norm()
        cosine_sim = grad_inner_product / (norm_text * norm_image + 1e-6)

        # Adjust coupling strength (beta) based on gradient alignment
        if cosine_sim < 0:
            self.beta *= 0.9
        elif cosine_sim > 0.5:
            self.beta *= 1.1
        return cosine_sim

    def apply(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        text_loss: torch.Tensor,
        image_loss: torch.Tensor,
        grads_text: dict,
        grads_image: dict,
        text_loss_weight: float = 1.0,  # LLM-optimized text loss weight
        use_llm: bool = False  # Flag to indicate if LLM optimization is applied
    ) -> (torch.Tensor, torch.Tensor, float):
        entropies = self.multi_entropy(text_feats, image_feats)
        combined_t, combined_i = entropies["combined_entropy"]
       
        # Compute entropy difference for coupling coefficients
        lam_i2t = torch.tanh(self.beta * F.relu(combined_t - combined_i))
        lam_t2i = torch.tanh(self.beta * F.relu(combined_i - combined_t))
        
    
        cosine_sim = self.gradient_alignment(grads_text, grads_image)
        consistency_loss = self.gamma * (1 - cosine_sim)
    
   
        if use_llm:
            text_loss_c = text_loss_weight * text_loss + lam_i2t * image_loss.detach()
            image_loss_c = image_loss + lam_t2i * text_loss.detach()
        else:
            text_loss_c = text_loss + lam_i2t * image_loss.detach()
            image_loss_c = image_loss + lam_t2i * text_loss.detach()
    
   
        text_loss_c += consistency_loss
        image_loss_c += consistency_loss
        
        return text_loss_c, image_loss_c, cosine_sim
       


class DualEmergenceOptimizer(nn.Module):
    def __init__(self, dim, feature_dim, temperature=0.1, alpha=0.5,
                 consistency_threshold=0.05, patience=3, beta=0.1,
                 clip_model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.is_training = True
        self.temperature = temperature
        self.alpha = alpha
        self.consistency_threshold = consistency_threshold
        self.patience = patience
        self.below_threshold_count = 0
        self.should_stop = False
        self.feature_dim = feature_dim
        
        # CLIP encoder
        self.clip_encoder = CLIPWrapperModel(model_name=clip_model_name)
        clip_output_dim = self.clip_encoder.model.config.projection_dim

        # Parameter-Level Bidirectional Coupling module
        self.plbc = PLBC(feature_dim, beta=beta)

        # Emergence components
        self.emergence_core = EmergenceCore(feature_dim)
        self.cross_modal_attention = CrossModalAttention(feature_dim)
        self.scale_interaction = ScaleInteractionModule([feature_dim, feature_dim])
        self.bidirectional_core = BidirectionalEmergenceCore(feature_dim)
        
        self.distribution_estimator = CriticalDistributionEstimator(
            feature_dim, self.emergence_core, self.cross_modal_attention
        )
        self.parameter_optimizer = AdaptiveParameterOptimizer(
            feature_dim, self.scale_interaction
        )
        self.consistency_loss = SymmetricConsistencyLoss(temperature=temperature)
        
        # Projection layers
        self.fc = nn.Linear(clip_output_dim * 2, feature_dim)
        self.text_projection = nn.Linear(feature_dim // 2, feature_dim)
        self.image_projection = nn.Linear(feature_dim // 2, feature_dim)
        self.residual_fc = nn.Linear(feature_dim, feature_dim)

    def check_consistency(self, loss_value):
        """Check if consistency loss is below threshold for early stopping."""
        if loss_value < self.consistency_threshold:
            self.below_threshold_count += 1
            if self.below_threshold_count >= self.patience:
                self.should_stop = True
        else:
            self.below_threshold_count = 0
        return self.should_stop

    def forward(self, raw_text_input=None, raw_image_input=None):
        """Inference mode: Process single modality input (text or image)."""
        if not self.is_training:
            clip_embedding = None
            if raw_text_input is not None:
                clip_embedding = self.clip_encoder.encode_text(raw_text_input)
            elif raw_image_input is not None:
                clip_embedding = self.clip_encoder.encode_image(raw_image_input)
            else:
                return None
            
            if clip_embedding.dim() == 1:
                clip_embedding = clip_embedding.unsqueeze(0)
            if clip_embedding.dim() == 2:
                clip_embedding = clip_embedding.unsqueeze(1)

            processed_embedding = self.emergence_core(clip_embedding.to(next(self.emergence_core.parameters()).device))
            
            if processed_embedding.dim() == 3:
                if processed_embedding.size(1) == 1:
                    final_embedding = processed_embedding.squeeze(1)
                else:
                    final_embedding = processed_embedding.mean(dim=1)
            elif processed_embedding.dim() == 2:
                final_embedding = processed_embedding
            else:
                print(f"Warning: Unexpected embedding dimension from emergence_core: {processed_embedding.dim()}")
                final_embedding = processed_embedding

            return final_embedding

        # Training mode requires both inputs
        if raw_text_input is None or raw_image_input is None:
            raise ValueError("Training mode requires both raw_text_input and raw_image_input.")

        t_emb, i_emb, loss_em, loss_cons, stop = self._forward_train(raw_text_input, raw_image_input)
        return t_emb, i_emb, loss_em, loss_cons, stop

    def _forward_train(self, raw_text_input, raw_image_input, gamma=0.1):
        """Forward pass for training: Compute embeddings and losses."""
        # Encode inputs using CLIP
        text_features = self.clip_encoder.encode_text(raw_text_input).to(self.fc.weight.device)
        image_features = self.clip_encoder.encode_image(raw_image_input).to(self.fc.weight.device)

        # Process embeddings
        B = text_features.size(0)
        tf = text_features.view(B, -1)
        im = image_features.view(B, -1)
        
        concat = torch.cat([tf, im], dim=-1)
        reduced = self.fc(concat)
        reduced = reduced + self.residual_fc(reduced)
        if reduced.dim() == 2:
            reduced = reduced.unsqueeze(1)

        dist, state = self.distribution_estimator(reduced)
        opt_params, _ = self.parameter_optimizer(dist, reduced)
        split = self.feature_dim // 2
        t_p, i_p = torch.split(opt_params, [split, split], dim=-1)
        t_proj = self.text_projection(t_p)
        i_proj = self.image_projection(i_p)
        t_emb, i_emb = self.bidirectional_core(t_proj, i_proj)
        
        
        jsd_infonce_loss = self.consistency_loss(t_emb, i_emb, dist)
    
        cosine_loss = 1 - F.cosine_similarity(t_emb, i_emb, dim=-1).mean()
     
        total_loss = jsd_infonce_loss + gamma * cosine_loss
    
        stop = self.check_consistency(cosine_loss.item())
        return t_emb, i_emb, total_loss, jsd_infonce_loss, cosine_loss, stop

    def train_step(self, raw_text_input, raw_image_input, optimizer, text_loss_weight=1.0, use_llm=False, gamma=0.1):
       
        # Forward pass
        t_emb, i_emb, total_loss, jsd_infonce_loss, cosine_loss, stop = self._forward_train(raw_text_input, raw_image_input, gamma=gamma)


        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute similarity for ranking
        similarity = torch.matmul(F.normalize(t_emb), F.normalize(i_emb).T)

        return t_emb, i_emb, total_loss, jsd_infonce_loss, cosine_loss, stop, similarity
