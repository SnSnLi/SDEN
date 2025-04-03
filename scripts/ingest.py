# scripts/ingest.py

import json
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sden.embedder import SDENEmbedding
import os

def load_documents(json_path: str):
    """
    加载 data/documents.json 格式为：
    [
        {"id": "doc1", "text": "This is document one."},
        {"id": "doc2", "text": "Another document."}
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=entry["text"], metadata={"id": entry["id"]}) for entry in data]

def main():
    # 1. 加载语料
    docs = load_documents("data/documents.json")

    # 2. 可选：对长文进行切片
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # 3. 初始化 SDEN embedding
    embedding_model = SDENEmbedding()

    # 4. 构建 FAISS 向量库
    vectorstore = FAISS.from_documents(split_docs, embedding=embedding_model)

    # 5. 保存向量索引
    if not os.path.exists("data/faiss_index"):
        os.makedirs("data/faiss_index")
    vectorstore.save_local("data/faiss_index")
    print("✅ 向量库已保存到 data/faiss_index")

if __name__ == "__main__":
    main()
