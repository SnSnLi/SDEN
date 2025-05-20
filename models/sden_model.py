import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_modules import (
    MultiHeadAttention,
    EmergenceCore,
    BidirectionalEmergenceCore,
    CrossModalAttention,
    SimpleAttention
)
from clip import CLIPWrapperModel          
from .emergence import MultiLevelEntropyModule

from .topology import (
    DynamicTopologyCoupler,
    EntropyController as TopologyEntropyController,
    CriticalDynamicsController
)
from .dual import DualEmergenceOptimizer

class SymmetricDynamicEmergenceNetwork(nn.Module):
    def __init__(self, dim=1024, num_heads=8, num_layers=4, temperature=0.1,
                 eta_critical_topo=0.01, alpha_feedback_topo=0.05,
                 H_feat_diff_critical=0.1, gamma_feedback_H_sync=0.05,
                 lambda_graph_entropy_reg_topo=0.01,
                 T0_graph_entropy_softmax_topo=2.0,
                 k_decay_graph_entropy_softmax_topo=0.01):
        super().__init__()
        self.dim = dim
        
        self.emergence_module = MultiScaleEmergenceModule(base_dim=dim)
        self.bidirectional_core = BidirectionalEmergenceCore(dim=dim)
        self.cross_modal = CrossModalAttention(dim=dim)
        
        self.topology_coupler = DynamicTopologyCoupler(dim=dim, num_heads=num_heads,
                                                       eta_critical=eta_critical_topo,
                                                       alpha_feedback=alpha_feedback_topo,
                                                       lambda_graph_entropy_reg=lambda_graph_entropy_reg_topo,
                                                       T0_graph_entropy_softmax=T0_graph_entropy_softmax_topo,
                                                       k_decay_graph_entropy_softmax=k_decay_graph_entropy_softmax_topo)
        
        self.sden_entropy_controller = TopologyEntropyController(dim=dim)
        self.critical_controller = CriticalDynamicsController(dim=dim)
        
        self.dual_optimizer = DualEmergenceOptimizer(dim=dim, feature_dim=dim, temperature=temperature)
        
        self.classifier = nn.Linear(dim, dim)
        self.contrastive_temp = nn.Parameter(torch.ones(1) * 0.07)

        self.H_feat_diff_critical = H_feat_diff_critical
        self.gamma_feedback_H_sync = gamma_feedback_H_sync

    @staticmethod
    def contrastive_loss_calc(text_feats, image_feats, temperature=0.07):
        batch_size = text_feats.shape[0]
        if text_feats.dim() > 2:
            text_feats = text_feats.mean(dim=1)
        if image_feats.dim() > 2:
            image_feats = image_feats.mean(dim=1)
            
        text_feats_norm = F.normalize(text_feats, dim=-1)
        image_feats_norm = F.normalize(image_feats, dim=-1)
        
        logits = torch.matmul(text_feats_norm, image_feats_norm.t()) / temperature
        labels = torch.arange(batch_size, device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss, logits, text_feats_norm, image_feats_norm

    def emergence_forward(self, text_features, image_features):
        text_emerged_bc, image_emerged_bc = self.bidirectional_core(text_features, image_features)
        
        ms_final_text, ms_final_image, ms_global_emerged, ms_entropy_weights, ms_semantic_graph = \
            self.emergence_module(text_emerged_bc, image_emerged_bc)
        
        fused_features_after_ms = self.cross_modal(ms_final_text, ms_final_image) 

        return fused_features_after_ms, ms_final_text, ms_final_image, ms_entropy_weights, ms_semantic_graph

    def topology_forward(self, emerged_features_input_to_topo, ptci_total_current_adjustment=None, current_step_for_annealing=None):
        if self.training and ptci_total_current_adjustment is not None and ptci_total_current_adjustment.abs().item() > 1e-9:
            try:
                ptci_micro_entropy_ctrl = self.emergence_module.micro_layer.phase_transition.entropy_controller
                
                original_target = ptci_micro_entropy_ctrl.target_entropy.data.clone()
                ptci_micro_entropy_ctrl.target_entropy.data += ptci_total_current_adjustment
            except AttributeError as e:
                print(f"Warning: Could not apply PTCI feedback to micro_layer. Attribute error: {e}")
            except Exception as e:
                print(f"Warning: Error applying PTCI feedback: {e}")
        
        topo_output_dict = self.topology_coupler(text_feat=emerged_features_input_to_topo, image_feat=None, current_step=current_step_for_annealing)
        
        controlled_features_after_sden_ec, sden_topo_entropy_weights = self.sden_entropy_controller(topo_output_dict['output'])
        
        next_eta_based_ptci_feedback_adjustment = topo_output_dict['delta_H_target_adjustment']
        loss_graph_entropy_from_topo = topo_output_dict['loss_graph_entropy']

        critical_features_final_stage = self.critical_controller(controlled_features_after_sden_ec)
        
        return {
            'features': critical_features_final_stage,
            'entropy_weights_sden': sden_topo_entropy_weights,
            'adj_matrix': topo_output_dict['adj_matrix'],
            'next_eta_based_ptci_feedback_adjustment': next_eta_based_ptci_feedback_adjustment,
            'loss_graph_entropy': loss_graph_entropy_from_topo
        }

    def dual_forward(self, topo_stage_output_dict, labels=None):
        self.dual_optimizer.is_training = self.training 
        
        optimized_features_from_dual = self.dual_optimizer(topo_stage_output_dict['features'])
        
        logits = self.classifier(optimized_features_from_dual.mean(dim=1) if optimized_features_from_dual.dim() > 2 else optimized_features_from_dual)
        return logits, optimized_features_from_dual

    def forward(self, text_features=None, image_features=None, labels=None, 
                prev_eta_based_feedback_from_loop=None, current_step_for_annealing_in_topo=None):
        
        if not (text_features is not None or image_features is not None):
            raise ValueError("At least one of text_features or image_features must be provided")
        
        current_device = text_features.device if text_features is not None else image_features.device

        if text_features is not None and text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        if image_features is not None and image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        
        final_text_output, final_image_output = None, None
        global_emerged_output, logits_output, total_loss_output = None, None, None
        semantic_graph_output, emerged_raw_output, adjacency_output, entropy_ranking_output = None, None, None, None
        next_eta_feedback_for_loop = torch.tensor(0.0, device=current_device)
        current_graph_entropy_loss = torch.tensor(0.0, device=current_device)
        
        if text_features is not None and image_features is not None:
            fused_features_from_emergence, ms_final_text, ms_final_image, ms_entropy_weights, semantic_graph_output = \
                self.emergence_forward(text_features, image_features)
            final_text_output, final_image_output = ms_final_text, ms_final_image
            emerged_raw_output = fused_features_from_emergence 

            adjustment_from_H_sync = torch.tensor(0.0, device=current_device)
            if self.training and ms_final_text is not None and ms_final_image is not None and \
               ms_final_text.numel() > 0 and ms_final_image.numel() > 0:
                entropy_text = self.sden_entropy_controller.compute_entropy(ms_final_text).mean()
                entropy_image = self.sden_entropy_controller.compute_entropy(ms_final_image).mean()
                
                abs_H_feat_diff = torch.abs(entropy_text - entropy_image)
                
                adjustment_from_H_sync = -self.gamma_feedback_H_sync * torch.tanh(abs_H_feat_diff - self.H_feat_diff_critical)
            
            current_total_ptci_adjustment = adjustment_from_H_sync + \
                                            (prev_eta_based_feedback_from_loop if prev_eta_based_feedback_from_loop is not None else 0.0)

            topo_results_dict = self.topology_forward(fused_features_from_emergence, 
                                                      ptci_total_current_adjustment=current_total_ptci_adjustment,
                                                      current_step_for_annealing=current_step_for_annealing_in_topo)
            next_eta_feedback_for_loop = topo_results_dict['next_eta_based_ptci_feedback_adjustment']
            current_graph_entropy_loss = topo_results_dict['loss_graph_entropy']
            adjacency_output = topo_results_dict['adj_matrix']
            if topo_results_dict['entropy_weights_sden'] is not None:
                 entropy_ranking_output = torch.argsort(topo_results_dict['entropy_weights_sden'], dim=-1)

            logits_output, final_features_from_dual = self.dual_forward(topo_results_dict, labels)
            global_emerged_output = final_features_from_dual

            if self.training:
                text_emerged_single = self.forward_text(text_features) 
                image_emerged_single = self.forward_image(image_features)
                consistency_loss_val = -F.cosine_similarity(text_emerged_single.mean(dim=1), 
                                                           image_emerged_single.mean(dim=1)).mean()
                
                contrastive_loss_val, _, _, _ = self.contrastive_loss_calc(ms_final_text, ms_final_image, 
                                                                           temperature=self.contrastive_temp)
                total_loss_output = consistency_loss_val + contrastive_loss_val + current_graph_entropy_loss
            
        elif text_features is not None:
            text_emerged_single = self.forward_text(text_features)
            final_text_output = text_emerged_single
            emerged_raw_output = text_emerged_single

            adjustment_from_H_sync = torch.tensor(0.0, device=current_device)
            current_total_ptci_adjustment = adjustment_from_H_sync + \
                                            (prev_eta_based_feedback_from_loop if prev_eta_based_feedback_from_loop is not None else 0.0)

            topo_results_dict = self.topology_forward(text_emerged_single, 
                                                      ptci_total_current_adjustment=current_total_ptci_adjustment,
                                                      current_step_for_annealing=current_step_for_annealing_in_topo)
            next_eta_feedback_for_loop = topo_results_dict['next_eta_based_ptci_feedback_adjustment']
            current_graph_entropy_loss = topo_results_dict['loss_graph_entropy']
            adjacency_output = topo_results_dict['adj_matrix']
            if topo_results_dict['entropy_weights_sden'] is not None:
                entropy_ranking_output = torch.argsort(topo_results_dict['entropy_weights_sden'], dim=-1)

            logits_output, final_features_from_dual = self.dual_forward(topo_results_dict, labels)
            global_emerged_output = final_features_from_dual
            total_loss_output = None

        elif image_features is not None:
            image_emerged_single = self.forward_image(image_features)
            final_image_output = image_emerged_single
            emerged_raw_output = image_emerged_single

            adjustment_from_H_sync = torch.tensor(0.0, device=current_device)
            current_total_ptci_adjustment = adjustment_from_H_sync + \
                                            (prev_eta_based_feedback_from_loop if prev_eta_based_feedback_from_loop is not None else 0.0)

            topo_results_dict = self.topology_forward(image_emerged_single, 
                                                      ptci_total_current_adjustment=current_total_ptci_adjustment,
                                                      current_step_for_annealing=current_step_for_annealing_in_topo)
            next_eta_feedback_for_loop = topo_results_dict['next_eta_based_ptci_feedback_adjustment']
            current_graph_entropy_loss = topo_results_dict['loss_graph_entropy']
            adjacency_output = topo_results_dict['adj_matrix']
            if topo_results_dict['entropy_weights_sden'] is not None:
                entropy_ranking_output = torch.argsort(topo_results_dict['entropy_weights_sden'], dim=-1)

            logits_output, final_features_from_dual = self.dual_forward(topo_results_dict, labels)
            global_emerged_output = final_features_from_dual
            total_loss_output = None
        
        return (
            final_text_output, final_image_output, global_emerged_output, logits_output, total_loss_output,
            semantic_graph_output, emerged_raw_output, adjacency_output, entropy_ranking_output,
            next_eta_feedback_for_loop 
        )

    def forward_text(self, text_features):
        if hasattr(self.bidirectional_core.text_emergence, '__call__'):
            import inspect
            sig = inspect.signature(self.bidirectional_core.text_emergence.forward if isinstance(self.bidirectional_core.text_emergence, nn.Module) else self.bidirectional_core.text_emergence)
            if 'context' in sig.parameters:
                return self.bidirectional_core.text_emergence(text_features, context=None)
            else:
                return self.bidirectional_core.text_emergence(text_features)
        else:
            temp_core = EmergenceCore(self.dim).to(text_features.device)
            return temp_core(text_features)

    def forward_image(self, image_features):
        if hasattr(self.bidirectional_core.image_emergence, '__call__'):
            import inspect
            sig = inspect.signature(self.bidirectional_core.image_emergence.forward if isinstance(self.bidirectional_core.image_emergence, nn.Module) else self.bidirectional_core.image_emergence)
            if 'context' in sig.parameters:
                return self.bidirectional_core.image_emergence(image_features, context=None)
            else:
                return self.bidirectional_core.image_emergence(image_features)
        else:
            temp_core = EmergenceCore(self.dim).to(image_features.device)
            return temp_core(image_features)

class SDENModel(nn.Module):
    def __init__(self, feature_dim=1024, temperature=0.1, eta_critical_topo=0.01, alpha_feedback_topo=0.05,
                 H_feat_diff_critical=0.1, gamma_feedback_H_sync=0.05,
                 lambda_graph_entropy_reg_topo=0.01,
                 T0_graph_entropy_softmax_topo=2.0,
                 k_decay_graph_entropy_softmax_topo=0.01):
        super().__init__()
        self.dual_optimizer = DualEmergenceOptimizer(dim=feature_dim, feature_dim=feature_dim, temperature=temperature)
        self.emergence = SymmetricDynamicEmergenceNetwork(dim=feature_dim, 
                                                            eta_critical_topo=eta_critical_topo, 
                                                            alpha_feedback_topo=alpha_feedback_topo,
                                                            H_feat_diff_critical=H_feat_diff_critical,
                                                            gamma_feedback_H_sync=gamma_feedback_H_sync,
                                                            lambda_graph_entropy_reg_topo=lambda_graph_entropy_reg_topo,
                                                            T0_graph_entropy_softmax_topo=T0_graph_entropy_softmax_topo,
                                                            k_decay_graph_entropy_softmax_topo=k_decay_graph_entropy_softmax_topo)
        self.is_training = True
        
        self.clip_dim = 768
        self.clip_projection_text = nn.Linear(self.clip_dim, feature_dim)
        self.clip_projection_image = nn.Linear(self.clip_dim, feature_dim)
        
        self.text_attention = SimpleAttention(feature_dim)
        self.image_attention = SimpleAttention(feature_dim)
        
        self.temperature = 0.5
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.current_step = 0

    def set_training_mode(self, mode=True):
        if mode:
            self.train()
        else:
            self.eval()

    def forward(self, text_features=None, image_features=None, labels=None):
        if text_features is not None or image_features is not None:
            if self.training:
                self.current_step += 1
            
            prev_eta_based_feedback_from_loop = getattr(self, '_prev_eta_feedback', torch.tensor(0.0, device=self.device))
            current_step_for_annealing_in_topo = self.current_step if self.training else None
            
            return self.emergence(text_features, image_features, labels, prev_eta_based_feedback_from_loop, current_step_for_annealing_in_topo)
        
        else:
            self.set_training_mode(self.training)
            if text_features is not None or image_features is not None:
                return self.emergence(text_features, image_features, labels, None, self.current_step if self.training else None)
            else:
                raise ValueError("Invalid inputs to SDENModel.forward")
        
    @torch.no_grad()
    def handle_clip_features(self, text_features, image_features, prev_eta_based_feedback_for_ptci=None):
        self.eval()
        original_training_state_emergence = self.emergence.training
        self.emergence.eval()
        
        eval_step_for_annealing = self.current_step
        
        if text_features is not None:
            print(f"输入文本特征维度: {text_features.shape}")
        if image_features is not None:
            print(f"输入图像特征维度: {image_features.shape}")
        
        if text_features is not None and text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        if image_features is not None and image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
            
        if text_features is not None:
            text_features = text_features.to(self.device)
        if image_features is not None:
            image_features = image_features.to(self.device)
            
        try:
            if text_features is not None and text_features.size(-1) != self.emergence.dim:
                text_features = self.clip_projection_text(text_features.view(-1, text_features.size(-1))).view(text_features.size(0), text_features.size(1), -1)
            if image_features is not None and image_features.size(-1) != self.emergence.dim:
                image_features = self.clip_projection_image(image_features.view(-1, image_features.size(-1))).view(image_features.size(0), image_features.size(1), -1)
        except Exception as e:
            print(f"投影特征维度时出错: {e}"); self.emergence.training = original_training_state_emergence; return self._fallback_features(text_features, image_features)
        
        try:
            feedback_for_eval = prev_eta_based_feedback_for_ptci if prev_eta_based_feedback_for_ptci is not None else torch.tensor(0.0, device=self.device)
            
            output_tuple = self.emergence(text_features, image_features, None, 
                                          feedback_for_eval, 
                                          eval_step_for_annealing)
            self.emergence.training = original_training_state_emergence
            
            enhanced_features = output_tuple[2]
            
            if enhanced_features is not None and enhanced_features.size(1) > 1:
                if text_features is not None and image_features is None:
                    weighted_features = self.text_attention.apply(enhanced_features)
                elif image_features is not None and text_features is None:
                    weighted_features = self.image_attention.apply(enhanced_features)
                else:
                    weighted_features = self.text_attention.apply(enhanced_features)
                return weighted_features.unsqueeze(1) if weighted_features.dim() == 2 else weighted_features
            return enhanced_features
        
        except RuntimeError as e:
            self.emergence.training = original_training_state_emergence
            print(f"SDEN处理特征时出错，使用回退方案: {e}")
            return self._fallback_features(text_features, image_features)
    
    def _fallback_features(self, text_features, image_features):
        if text_features is not None and image_features is not None:
            if text_features.dim() > 2:
                text_mean = text_features.mean(dim=1, keepdim=True)
            else:
                text_mean = text_features.unsqueeze(1)
                
            if image_features.dim() > 2:
                image_mean = image_features.mean(dim=1, keepdim=True)
            else:
                image_mean = image_features.unsqueeze(1)
                
            feature_dim = self.emergence.dim
            batch_size = text_features.size(0)
            
            fallback = torch.randn(batch_size, 1, feature_dim, device=self.device)
            fallback = fallback * 0.1 + 0.5
            
            print(f"已生成回退特征，形状: {fallback.shape}")
            return fallback
        elif text_features is not None:
            if text_features.dim() <= 2:
                text_features = text_features.unsqueeze(1) if text_features.dim() == 2 else text_features
            return text_features
        else:
            if image_features.dim() <= 2:
                image_features = image_features.unsqueeze(1) if image_features.dim() == 2 else image_features
            return image_features
    
    @torch.no_grad()
    def extract_features(self, text_features, image_features):
        return self.handle_clip_features(text_features, image_features)

    def compute_similarity(self, features1, features2):
        try:
            print(f"原始输入形状: features1={features1.shape}, features2={features2.shape}")
            
            if features1.dim() > 2 and features2.dim() > 2:
                if features1.size(1) != features2.size(1):
                    if features1.size(1) == 1:
                        features1 = features1.expand(-1, features2.size(1), -1)
                    elif features2.size(1) == 1:
                        features2 = features2.expand(-1, features1.size(1), -1)
                    else:
                        min_seq_len = min(features1.size(1), features2.size(1))
                        features1 = features1[:, :min_seq_len, :]
                        features2 = features2[:, :min_seq_len, :]
                    
                    print(f"调整后形状: features1={features1.shape}, features2={features2.shape}")
            
            if features1.size(-1) != features2.size(-1):
                print(f"特征维度不匹配: {features1.size(-1)} vs {features2.size(-1)}")
                if features1.size(-1) == self.clip_dim and features2.size(-1) == self.emergence.dim:
                    if features1.dim() == 3:
                        b, s, _ = features1.size()
                        features1 = features1.view(b*s, -1)
                        features1 = self.clip_projection_text(features1)
                        features1 = features1.view(b, s, -1)
                    else:
                        features1 = self.clip_projection_text(features1)
                    print(f"已将features1从{self.clip_dim}投影到{self.emergence.dim}")
                elif features2.size(-1) == self.clip_dim and features1.size(-1) == self.emergence.dim:
                    if features2.dim() == 3:
                        b, s, _ = features2.size()
                        features2 = features2.view(b*s, -1)
                        features2 = self.clip_projection_image(features2)
                        features2 = features2.view(b, s, -1)
                    else:
                        features2 = self.clip_projection_image(features2)
                    print(f"已将features2从{self.clip_dim}投影到{self.emergence.dim}")
                else:
                    min_dim = min(features1.size(-1), features2.size(-1))
                    features1 = features1[..., :min_dim]
                    features2 = features2[..., :min_dim]
                    print(f"截断到相同维度: {min_dim}")
            
            if features1.dim() == 3:
                if features1.size(1) == 1:
                    features1 = features1.squeeze(1)
                else:
                    features1 = features1.mean(dim=1)
                    
            if features2.dim() == 3:
                if features2.size(1) == 1:
                    features2 = features2.squeeze(1)
                else:
                    features2 = features2.mean(dim=1)
            
            if features1.dim() == 1:
                features1 = features1.unsqueeze(0)
            if features2.dim() == 1:
                features2 = features2.unsqueeze(0)
            
            print(f"处理后形状: features1={features1.shape}, features2={features2.shape}")
            
            features1 = F.normalize(features1, p=2, dim=-1)
            features2 = F.normalize(features2, p=2, dim=-1)
            
            if features1.size(0) != features2.size(0):
                print(f"批次大小不匹配: {features1.size(0)} vs {features2.size(0)}")
                if features1.size(0) == 1 and features2.size(0) > 1:
                    features1 = features1.expand(features2.size(0), -1)
                elif features2.size(0) == 1 and features1.size(0) > 1:
                    features2 = features2.expand(features1.size(0), -1)
                else:
                    min_batch = min(features1.size(0), features2.size(0))
                    features1 = features1[:min_batch]
                    features2 = features2[:min_batch]
                
                print(f"调整后批次大小: features1={features1.shape}, features2={features2.shape}")
            
            cos_sim = torch.sum(features1 * features2, dim=-1)
            print(f"原始余弦相似度: {cos_sim.mean().item():.4f}")
            
            scaled_sim = cos_sim / self.temperature
            
            final_sim = torch.sigmoid(scaled_sim)
            
            print(f"调整后相似度: {final_sim.mean().item():.4f}, temperature={self.temperature:.4f}")
            
            return final_sim.mean().item()
        
        except Exception as e:
            print(f"相似度计算发生错误: {e}")
            try:
                if features1.dim() > 2:
                    features1 = features1.view(features1.size(0), -1)
                if features2.dim() > 2:
                    features2 = features2.view(features2.size(0), -1)
                    
                min_dim = min(features1.size(-1), features2.size(-1))
                features1 = features1[..., :min_dim]
                features2 = features2[..., :min_dim]
                
                features1 = F.normalize(features1, p=2, dim=-1)
                features2 = F.normalize(features2, p=2, dim=-1)
                
                if features1.size(0) != features2.size(0):
                    min_batch = min(features1.size(0), features2.size(0))
                    features1 = features1[:min_batch]
                    features2 = features2[:min_batch]
                
                cos_sim = F.cosine_similarity(features1, features2).mean().item()
                final_sim = 1.0 / (1.0 + math.exp(-cos_sim / 0.5))
                print(f"使用回退方法计算相似度: {final_sim:.4f}, temperature=0.5")
                return final_sim
            except Exception as fallback_error:
                print(f"回退计算也失败: {fallback_error}，返回默认值0.5")
                return 0.5
                
clip_model = CLIPWrapperModel()

dim = clip_model.model.config.projection_dim  
entropy_module = MultiLevelEntropyModule(dim)

batch_outputs = {'text_embeds': [], 'image_embeds': []}
for t, img in zip(texts, images):
    out = clip_model.encode_text_and_image(t, img)
    batch_outputs['text_embeds'].append(out['text_embeds'])
    batch_outputs['image_embeds'].append(out['image_embeds'])


text_batch = torch.cat(batch_outputs['text_embeds'], dim=0)
image_batch = torch.cat(batch_outputs['image_embeds'], dim=0)

entropy_dict = entropy_module(text_batch, image_batch)
print(entropy_dict)