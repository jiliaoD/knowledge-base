# knowledge/processor/import_process/config.py

"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Set, Optional, Tuple
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(override=True)


@dataclass
class ImportConfig:
    """导入流程配置"""
    # ==================== MinerU 配置 ====================
    mineru_api_token: str = field(
        default_factory=lambda: os.getenv("MINERU_API_TOKEN", "")
    )
    mineru_base_url: str = field(
        default_factory=lambda: os.getenv("MINERU_BASE_URL", "")
    )

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000      # 切片最大长度
    img_content_length: int = 200       # 图片上下文最大长度
    min_content_length: int = 500       # 合并短内容的最小长度
    overlap_sentences: int = 1          # 句子级切分时重叠句数
    item_name_chunk_k: int = 3          # 商品名识别时使用的切片数量
    item_name_chunk_size: int = 2500    # 商品名识别时使用的切片内容长度

    # 支持的图片扩展名
    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.7"))
    )
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )


    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "")
    )


    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secret=False

    minio_image_dir: str = field(
        default_factory=lambda: os.getenv("MINIO_IMAGE_DIR", "")
    )

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )

    embedding_batch_size: int = 8

    # ==================== 速率限制 ====================
    # 图片总结（多模态模型）每分钟最大请求数，可用 .env 的 VL_REQUESTS_PER_MINUTE 覆盖
    requests_per_minute: int = field(
        default_factory=lambda: int(os.getenv("VL_REQUESTS_PER_MINUTE", "60"))
    )

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    def get_minio_base_url(self) -> str:
        """获取 MinIO 基础 URL"""
        protocol = "https://" if self.minio_secure else "http://"
        return protocol + f"{self.minio_endpoint}"


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
