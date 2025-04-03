# sden/embedder.py

from typing import List
from langchain.embeddings.base import Embeddings
from sden.model import SDENWrapperModel  # 你刚封装好的
import torch

class SDENEmbedding(Embeddings):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SDENWrapperModel(device=self.device)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain 用来索引多个文档时调用（批量）
        """
        return [self.model.encode_text(text).tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        LangChain 查询时调用（单条）
        """
        return self.model.encode_text(text).tolist()
