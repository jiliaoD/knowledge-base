# config/reranker_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RerankerConfig:
    text_rerank_api_key: str # DashScope API Key
    text_rerank_model: str # 模型名称
    text_rerank_instruct: str # 是否使用指令

reranker_config = RerankerConfig(
    text_rerank_api_key=os.getenv("OPENAI_API_KEY_DASHSCOPE"),
    text_rerank_model=os.getenv("TEXT_RERANK_MODEL"),
    text_rerank_instruct=os.getenv("TEXT_RERANK_INSTRUCT")
)
