# utils/sse_utils.py
"""
SSE（Server-Sent Events）会话队列工具

作用：把"要推给前端的消息"按 session_id 存进内存队列，
      由 Web 层的 SSE 接口从队列里取出来，按 SSE 协议写给浏览器。
      本模块只负责入队/出队，不关心 HTTP 细节。
"""

import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号


# 全局 SSE 会话队列存储
# Key: session_id, Value: queue.Queue
_session_stream: Dict[str, "queue.Queue"] = {}


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _session_stream[session_id] = q
    return q


def remove_sse_queue(session_id: str):
    """移除指定 session 的队列"""
    _session_stream.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    通过 session_id 推送事件

    注意：如果这个 session 还没有注册队列（例如没走 Web 层的 SSE 接口，
    或是在命令行里单独跑节点），这里会静默跳过，不会报错。
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


async def sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # 如果没有对应的队列，直接结束
        return

    loop = asyncio.get_running_loop()
    try:
        # 发送连接建立信号
        yield _sse_pack("ready", {})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            # 特殊关闭事件
            if event == "__close__":
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        # 生成器被取消/对端断开：静默退出
        return
    finally:
        # 清理资源
        remove_sse_queue(session_id)
