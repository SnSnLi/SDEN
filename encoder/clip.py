from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


class CLIPWrapperModel:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("CLIP model loaded successfully")
    
    @torch.no_grad()
    def encode_text_and_image(self, text, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model(**inputs)
        
        # 返回包含文本和图像嵌入的字典
        return {
            'text_embeds': outputs.text_embeds.to(self.device),
            'image_embeds': outputs.image_embeds.to(self.device)
        }
    
    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.processor(text=[text], images=None, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model.get_text_features(input_ids=inputs.input_ids)
        # 确保返回的张量在正确的设备上
        return outputs.squeeze(0).to(self.device)
    
    @torch.no_grad()
    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=None, images=image, return_tensors="pt").to(self.device)
        
        outputs = self.model.get_image_features(pixel_values=inputs.pixel_values)
        # 确保返回的张量在正确的设备上
        return outputs.squeeze(0).to(self.device)
