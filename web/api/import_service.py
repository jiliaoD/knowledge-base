"""
导入流程的 API 接口定义

支持批量上传：一次请求可传多个文件，每个文件独立生成 task_id 并单独跟踪进度。
"""
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from processor.import_processor.config import get_config
from processor.import_processor.main_graph import KBImportWorkflow
from tool.logger import logger
from utils.minio_utils import get_minio_client
from utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    get_task_result,
    set_task_result,
    update_task_status,
)

# 1. 创建应用（标题和描述会在 Swagger 文档中展示）
app = FastAPI(
    title="产品手册智能问答-导入API",
    description="此文档是产品手册智能问答导入流程的API接口说明（支持批量上传）"
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许的源
    allow_credentials=True,     # 允许携带cookie
    allow_methods=["*"],        # 允许的请求方法
    allow_headers=["*"],        # 允许的请求头
)

# 允许导入的文件类型
ALLOWED_EXTENSIONS = {".pdf", ".md"}


def _resolve_upload_root_dir() -> str:
    """
    上传文件的本地存储根目录：
    优先 DATA_BASED_ROOT_DIR，其次 MD_ROOT_DIR（你 .env 里现有配置），最后回落到 ./doc
    """
    return os.getenv("DATA_BASED_ROOT_DIR") or os.getenv("MD_ROOT_DIR") or "./doc"


# 3. 静态页面路由：返回文件导入前端页面
# 访问地址：http://127.0.0.1:8000/import.html
@app.get("/import.html")
async def get_import_page():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "import.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


# 4. 后台任务：LangGraph 全流程执行
# 独立于主请求线程，由 BackgroundTasks 触发，避免阻塞接口响应
def run_graph_task(task_id: str, file_dir: str, import_file_path: str):
    """
    LangGraph 全流程执行后台任务
    核心流程：初始化状态 → 执行图节点 → 更新任务状态 → 异常捕获
    节点进度：由 processor/import_processor/base.py 在每个节点开始/结束时自动记录

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param file_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param import_file_path: 上传文件的本地绝对路径
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")

        # 2. 初始化 LangGraph 状态
        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }

        # 3. 流式执行全流程（消费生成器：图才会真正跑起来）
        workflow = KBImportWorkflow()
        for event in workflow.run(init_state, stream=True):
            logger.debug(f"[{task_id}] 图事件: {event}")

        # 4. 全流程执行完成
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] 导入完成")

    except Exception as e:
        # 5. 捕获全流程异常，标记失败
        update_task_status(task_id, "failed")
        # 记录失败原因，前端/接口可直接查到（便于定位重跑）
        set_task_result(task_id, "error", str(e)[:500])
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)


# 5. 核心接口：文件上传（支持多文件批量）
# 访问地址：http://127.0.0.1:8000/upload （POST，form-data）
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    批量上传核心接口
    1. 接收前端上传的多文件（PDF/MD）
    2. 按「日期/任务ID」分层保存到本地目录，避免多文件重名冲突
    3. 将文件上传至 MinIO 做持久化（失败不影响后续处理）
    4. 为每个文件生成唯一 task_id，启动独立的 LangGraph 后台处理任务
    5. 返回所有 task_id，前端据此轮询各自进度
    """
    if not files:
        raise HTTPException(status_code=400, detail="没有收到任何文件")

    data_based_root_dir = _resolve_upload_root_dir()
    data_dir = os.path.join(data_based_root_dir, datetime.now().strftime("%Y%m%d"))
    task_ids: List[str] = []
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []

    for file in files:
        filename = os.path.basename(file.filename or "")
        ext = os.path.splitext(filename)[1].lower()

        # 类型校验：只接受 PDF / Markdown
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append({"filename": filename, "reason": f"不支持的文件类型 {ext or '(无扩展名)'}"})
            logger.warning(f"[upload] 跳过不支持的文件：{filename}")
            continue

        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        accepted.append({"filename": filename, "task_id": task_id})
        logger.info(f"[{task_id}] 开始处理上传文件：{filename}，类型：{file.content_type}")

        # 标记「文件上传」阶段为运行中
        add_running_task(task_id, "upload_file")

        # 该任务的独立目录：<根目录>/YYYYMMDD/<task_id>
        file_dir = os.path.join(data_dir, task_id)
        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, filename)

        # 保存到本地
        with open(import_file_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地：{import_file_path}")

        # 上传到 MinIO（best-effort，失败不影响导入）
        minio_object_name = f"pdf_files/{datetime.now().strftime('%Y%m%d')}/{filename}"
        try:
            minio_client = get_minio_client()
            bucket_name = get_config().minio_bucket
            minio_client.fput_object(
                bucket_name=bucket_name,
                object_name=minio_object_name,
                file_path=import_file_path,
                content_type=file.content_type
            )
            logger.info(f"[{task_id}] 已上传至MinIO：{bucket_name}/{minio_object_name}")
        except Exception as e:
            logger.warning(f"[{task_id}] 上传MinIO失败，继续本地处理：{str(e)}")

        # 标记「文件上传」阶段完成
        add_done_task(task_id, "upload_file")

        # 交给后台任务执行完整导入流程（多个文件会按顺序依次执行）
        background_tasks.add_task(run_graph_task, task_id, file_dir, import_file_path)
        logger.info(f"[{task_id}] 已加入后台任务，导入流程即将开始")

    logger.info(f"批量上传处理完毕：接收 {len(files)} 个文件，进入导入 {len(task_ids)} 个，跳过 {len(rejected)} 个")
    return {
        "code": 200,
        "message": f"文件上传成功, total: {len(task_ids)}",
        "task_ids": task_ids,
        "files": accepted,
        "rejected": rejected,
    }


# 6. 核心接口：任务状态查询（前端轮询）
# 访问地址：http://127.0.0.1:8000/status/{task_id} （GET）
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每 1.5 秒一次），获取单个文件的实时处理进度。
    数据全部来自内存中的任务字典（task_utils.py），无 IO 开销。
    """
    return {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),          # pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),    # 已完成的节点（中文）
        "running_list": get_running_task_list(task_id),  # 正在运行的节点（中文）
        "error": get_task_result(task_id, "error", "")   # 失败原因（仅失败时非空）
    }


# 7. 存活检查
@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8000)
