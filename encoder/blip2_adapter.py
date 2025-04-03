# encoder/blip2_adapter.py

import torch
from transformers import Blip2Processor, Blip2Model
from PIL import Image
from typing import Union

class BLIP2Adapter:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # 投影到 SDEN 所需的维度
        hidden_dim = self.model.language_model.get_input_embeddings().embedding_dim
        vision_dim = self.model.vision_model.config.hidden_size
        self.text_proj = torch.nn.Linear(hidden_dim, 1024).to(self.device)
        self.vision_proj = torch.nn.Linear(vision_dim, 1024).to(self.device)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        将文本 → [B, seq_len, 1024]，输入 SDEN 的 text_features
        """
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        embeddings = self.model.language_model.get_input_embeddings()(inputs['input_ids'])  # [B, seq, D]
        return self.text_proj(embeddings)  # ➜ [B, seq, 1024]

    @torch.no_grad()
    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        将图像路径或 PIL Image → [B, seq_len, 1024]，输入 SDEN 的 image_features
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.vision_model(**inputs)
        return self.vision_proj(outputs.last_hidden_state)  # ➜ [B, seq, 1024]
