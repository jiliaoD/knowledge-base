"""
查询流程的接口定义
"""
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse

from processor.query_processor.main_graph import KBQueryWorkflowV2
from tool.logger import logger
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.sse_utils import create_sse_queue, SSEEvent, push_to_session, sse_generator
from utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_task_result,
    update_task_status,
)

# 1. 创建应用
app = FastAPI(
    title="产品手册智能问答-查询API",
    description="此文档是产品手册智能问答查询流程的API接口说明"
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许的源
    allow_credentials=True,     # 允许携带cookie
    allow_methods=["*"],        # 允许的请求方法
    allow_headers=["*"],        # 允许的请求头
)


# 3. 静态页面路由
@app.get("/chat.html")  # 对外访问地址
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "chat.html"
    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")          # ...表示必须填写
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 处理是不是流式返回结果
    is_stream = request.is_stream
    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)

    # 每轮新提问都要先把上一轮的节点进度清空。
    # 原因：任务进度是放在内存字典里、以 session_id 为 key 的，同一个会话的第二轮
    # 会复用同一个 key；不清空的话，上一轮"已完成"的节点会一直挂在新消息的进度面板上
    # （表现为第二轮刚开始，所有节点就已经打着勾了）。
    # clear_task 同时会清掉上一轮的 answer/image_urls 等结果，避免上一轮的答案被当成本轮结果。
    clear_task(session_id)

    # 更新任务状态：当前会话id作为key，整体状态处于运行中
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    logger.info(f"开始处理流程... 是否流式: {is_stream}, 其他参数: {user_query}, session_id: {session_id}")

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        logger.info("开始处理结果....")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        # 同步运行
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")

        # 本轮图片由答案节点通过内存任务表带出来
        # （不要再去历史里翻：那边翻到的是"第一条带图的助手消息"，很可能是更早那一轮的图片）
        image_urls = get_task_result(session_id, "image_urls", []) or []

        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "image_urls": image_urls,
            "done_list": []
        }


# 定义查询接口
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    """真正执行 LangGraph 检索图（阻塞式；流式时由后台任务调用）"""
    logger.info(f"开始流程图处理... {session_id} {user_query} {is_stream}")

    init_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream
    }

    try:
        workflow = KBQueryWorkflowV2()
        # 必须遍历这个生成器，图才会真正执行（节点内部会往 SSE 队列推送进度和答案增量）
        for chunk in workflow.run(init_state, stream=is_stream):
            logger.debug(chunk)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.error(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """sse 实时返回结果"""
    logger.info("调用流式 /stream ...")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """清空指定会话的历史记录"""
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """查询当前会话历史记录"""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "image_urls": r.get("image_urls") or [],
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


# 证明服务器启动即可
@app.get("/health")
async def health():
    """检查服务是否正常"""
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
