# sden/model.py
import torch
import torch.nn as nn
from sden_model import SymmetricDynamicEmergenceNetwork
from encoder.blip2_adapter import BLIP2Adapter 

class SDENWrapperModel(nn.Module):
    def __init__(self, dim=1024, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SymmetricDynamicEmergenceNetwork(dim=dim).to(self.device)
        self.model.eval()
        self.dim = dim

     
        self.encoder = BLIP2Adapter(device=self.device)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        使用 BLIP2 将文本 → [B, seq_len, 1024] → 喂给 SDEN → 输出 embedding 向量 [1024]
        """
        text_features = self.encoder.encode_text(text)  # [1, seq_len, 1024]
        outputs = self.model.forward(text_features=text_features)
        global_emerged = outputs[2]  # [1, seq_len, 1024]
        embedding = global_emerged.mean(dim=1).squeeze(0)  # [1024]
        return embedding

    @torch.no_grad()
    def encode_pair(self, text: str, image_path: str) -> torch.Tensor:
        """
        文本 + 图像 联合嵌入（可用于多模态问答场景）
        """
        text_feat = self.encoder.encode_text(text)
        image_feat = self.encoder.encode_image(image_path)
        outputs = self.model.forward(text_features=text_feat, image_features=image_feat)
        global_emerged = outputs[2]  # [1, seq_len, 1024]
        embedding = global_emerged.mean(dim=1).squeeze(0)  # [1024]
        return embedding
