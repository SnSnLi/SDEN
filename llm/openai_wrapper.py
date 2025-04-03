# llm/openai_wrapper.py

import os
from langchain.chat_models import ChatOpenAI


def load_openai_llm(model_name="gpt-4o", temperature=0.2, api_key=None):
    """
    加载 OpenAI LLM 模型（用于问答链等用途）
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OpenAI API key is not set. Use env var or pass explicitly.")

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=api_key,
    )
    return llm
