# config/milvus_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MilvusConfig:
    milvus_url: str
    chunks_collection: str
    item_name_collection: str

# 实例化Milvus配置对象（和其他配置对象命名风格统一）
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)