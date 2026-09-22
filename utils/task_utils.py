# utils/task_utils.py
"""
内存态任务追踪工具（单进程）

作用：记录一个任务（一次导入 / 一次查询）当前跑到哪个节点了，
      流式模式下顺带把进度推给前端。数据结构都是进程内字典，
      重启即清空——单进程内存态，不引入额外依赖，够用且简单。
"""

from typing import Dict, List

from utils.sse_utils import push_to_session

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

# key: task_id
# value: status 字符串（如 pending/processing/completed/failed）
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: 任务结果（例如 query 的 answer）
_tasks_result: Dict[str, Dict[str, str]] = {}


TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点 ---
    "node_item_name_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _has_task_id(task_id) -> bool:
    """
    是否带了有效的任务ID。
    单节点自测（state 里没有 session_id/task_id）时，所有任务追踪调用都直接跳过，
    避免往字典里塞一个 None 键，也避免报错。
    """
    return bool(task_id)


def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加"正在运行"的节点任务。

    :param task_id: 任务ID
    :param node_name: 节点名称（节点ID）
    :param is_stream: 是否流式（流式时顺带推送一次进度）
    """
    if not _has_task_id(task_id):
        return
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    # 避免重复追加
    if node_name not in running:
        running.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加"已完成"的节点任务。

    注意：添加已完成任务时，会把同名的"正在运行"任务删除。
    """
    if not _has_task_id(task_id):
        return
    _ensure_task(task_id)

    # 1) 从 running 中移除同名节点（可能出现重复，移除所有）
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 2) 追加到 done（保持完成顺序），避免重复
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存储任务结果字段（如 answer / error）。"""
    if not _has_task_id(task_id):
        return
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """获取任务结果字段（如 answer / error）。"""
    if not _has_task_id(task_id):
        return default
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """
    获取当前任务状态。

    :param task_id: 任务ID
    :return: 状态名称；如果未设置过则返回空字符串
    """
    if not _has_task_id(task_id):
        return ""
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """获取已完成节点列表（中文展示）。"""
    if not _has_task_id(task_id):
        return []
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    return [_to_cn(n) for n in done]


def get_running_task_list(task_id: str) -> List[str]:
    """获取正在运行节点列表（中文展示）。"""
    if not _has_task_id(task_id):
        return []
    _ensure_task(task_id)
    running = _tasks_running_list.get(task_id, [])
    return [_to_cn(n) for n in running]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """
    更新任务状态。

    :param task_id: 任务ID
    :param status_name: 状态名称（字符串）
    :param push_queue: 是否顺带推送一次进度
    """
    if not _has_task_id(task_id):
        return
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    """把当前进度（状态 + 已完成 + 进行中）推给前端。"""
    if not _has_task_id(task_id):
        return
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str):
    """清理某个任务的全部内存态数据。"""
    if not _has_task_id(task_id):
        return
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
