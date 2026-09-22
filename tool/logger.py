# 默认名字是：root
import logging
import os

import colorlog

# 初始化全局日志对象
logger = logging.getLogger()
# 设置日志的默认的级别（默认 INFO；排查细节时可设环境变量 LOG_LEVEL=DEBUG）
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# 这些第三方库在 DEBUG 级别下会输出巨量日志（数据库/网络连接的每一步细节），统一压到 WARNING
for _noisy_logger in (
    "pymongo", "httpx", "httpx2", "httpcore", "httpcore2", "urllib3",
    "mcp", "openai", "asyncio", "grpc", "FlagEmbedding",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# 加载彩色日志处理器
handler = colorlog.StreamHandler()
# 定义日志输出的格式
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',  # INFO 显示为绿色
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

# 避免重复添加 handler（模块被多次导入时）
logger.handlers.clear()

# 应用日志配置信息
logger.addHandler(handler)
