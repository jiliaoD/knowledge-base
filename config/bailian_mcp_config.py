# config/bailian_mcp_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class BailianMcpConfig:
    mcp_base_url: str   # 百炼 MCP 服务地址（Streamable HTTP）
    api_key: str        # 调用 MCP 服务用的百炼 API Key


# 实例化配置对象（和其他配置对象命名风格统一）
# Key 优先取 BAILIAN_API_KEY，没配就回退到 DashScope/OpenAI 兼容的那把 Key
mcp_config = BailianMcpConfig(
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL", ""),
    api_key=(
        os.getenv("BAILIAN_API_KEY", "")
        or os.getenv("DASHSCOPE_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    ),
)
