
from transformers import CLIPTextModel, CLIPVisionModel, CLIPProcessor
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EntropyController(nn.Module):
    def __init__(self, dim):
        super(EntropyController, self).__init__()
        self.entropy_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
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
        return current_entropy.mean()

class CrossModalEntropyController(nn.Module):
    def __init__(self, dim):
        super(CrossModalEntropyController, self).__init__()

    def compute_cross_entropy(self, text_emb, image_emb):
      
        text_sim = torch.matmul(text_emb, image_emb.T)
        text_probs = text_sim / text_sim.sum(dim=-1, keepdim=True)
        text_to_image_entropy = -torch.sum(text_probs * torch.log(text_probs + 1e-6), dim=-1).mean()

    
        image_sim = torch.matmul(image_emb, text_emb.T)
        image_probs = image_sim / image_sim.sum(dim=-1, keepdim=True)
        image_to_text_entropy = -torch.sum(image_probs * torch.log(image_probs + 1e-6), dim=-1).mean()

        return text_to_image_entropy, image_to_text_entropy

    def forward(self, text_emb, image_emb):
 
        cross_entropy_text2image, cross_entropy_image2text = self.compute_cross_entropy(text_emb, image_emb)
        return cross_entropy_text2image, cross_entropy_image2text

class MultiLevelEntropyModule(nn.Module):
    def __init__(self, dim, alpha=0.5):
        super(MultiLevelEntropyModule, self).__init__()
        self.initial_entropy_ctrl = EntropyController(dim)
        self.cross_modal_entropy_ctrl = CrossModalEntropyController(dim)
        self.alpha = alpha 

    def forward(self, text_emb, image_emb):
     
        text_e = self.initial_entropy_ctrl(text_emb)
        image_e = self.initial_entropy_ctrl(image_emb)
       
        cross_t2i, cross_i2t = self.cross_modal_entropy_ctrl(text_emb, image_emb)
    
     
        combined_e = self.alpha * base_e + (1 - self.alpha) * cross_e

        return {
            "base_entropy": text_e, image_e
            "cross_entropy": cross_t2i, cross_i2t
            "combined_entropy": combined_t, combined_i
        }


