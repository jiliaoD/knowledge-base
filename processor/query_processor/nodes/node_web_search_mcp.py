# processor/query_processor/nodes/node_web_search_mcp.py

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task


class NodeWebSearchMcp(NodeBase):
    """
    节点功能：调用百炼 MCP 联网搜索服务，补充知识库以外的实时信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    # 百炼联网搜索的工具名固定；count 为返回条数（官方建议 1-10）
    TOOL_NAME: str = "bailian_web_search"
    TOOL_COUNT: int = 5
    TIMEOUT: int = 15

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑：用改写后的问题联网搜索，输出统一的文档列表
        必要参数：rewritten_query
        更新参数：web_search_docs

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 1、获取查询内容
        query = (state.get("rewritten_query") or "").strip()
        if not query:
            logger.info(f"【{self.name}】没有可用的改写问题，跳过联网搜索")
            return {"web_search_docs": []}

        # 2、调用 MCP 服务（联网搜索属于锦上添花，失败就降级为空结果，不阻断主流程）
        try:
            result = self._run_async(self._mcp_call(query))

            # 3、解析并清洗返回结果
            docs = self._parse_pages(result)
            logger.info(f"【{self.name}】联网搜索完成，有效结果 {len(docs)} 条")

            # 4、更新状态：只返回本节点负责的键，避免并行分支写冲突
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
            return {"web_search_docs": docs}

        except Exception as e:
            logger.error(f"【{self.name}】联网搜索失败，降级为空结果: {e}")
            return {"web_search_docs": []}

    async def _mcp_call(self, query: str):
        """
        连接百炼 MCP 服务并调用联网搜索工具
        :param query: 改写后的用户问题
        :return: MCP 返回的 CallToolResult
        """
        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": self.TIMEOUT,
            },
            cache_tools_list=True,   # 缓存工具列表，避免每次调用都重新拉取
            max_retry_attempts=3,    # 网络抖动时自动重试
        )

        try:
            await search_mcp.connect()
            result = await search_mcp.call_tool(
                tool_name=self.TOOL_NAME,
                arguments={"query": query, "count": self.TOOL_COUNT},
            )
            return result
        finally:
            # 无论成功/失败/异常都要关闭连接，避免资源泄漏
            await search_mcp.cleanup()

    def _parse_pages(self, result) -> list:
        """
        清洗 MCP 返回值，统一成 [{title, url, snippet}] 结构，供后续 RRF/Rerank/答案生成使用
        MCP 原始结构：content[0].text 是一段 JSON 字符串，其中的 pages 才是结果列表
        :param result: MCP 返回的 CallToolResult
        :return: 文档列表，snippet 为空的无效结果会被剔除
        """
        pages = []

        # 1、MCP 返回是 content 数组，逐个文本块尝试解析 JSON，取第一个带 pages 的
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                data = json.loads(text)
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict) and data.get("pages"):
                pages = data["pages"]
                break

        # 2、只保留 title/url/snippet 三个核心字段，并剔除空摘要
        docs = []
        for item in pages:
            snippet = (item.get("snippet") or "").strip()
            if not snippet:
                continue
            docs.append({
                "title": (item.get("title") or "").strip(),
                "url": (item.get("url") or "").strip(),
                "snippet": snippet,
            })

        return docs

    @staticmethod
    def _run_async(coro):
        """
        在同步节点里执行异步协程：
          - 普通场景（graph.invoke）直接用 asyncio.run
          - 如果当前线程已有事件循环（例如以后在 FastAPI 的 async 路由里调用），
            asyncio.run 会直接抛 RuntimeError，这时换一个线程去跑它自己的事件循环
        :param coro: 待执行的协程对象
        :return: 协程的返回值
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))

