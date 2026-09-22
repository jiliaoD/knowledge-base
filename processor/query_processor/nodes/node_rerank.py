# processor/query_processor/nodes/node_rerank.py
from typing import Dict, Any, List

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.reranker_http_utils import rerank_documents
from utils.task_utils import add_done_task

# -----------------------------
# Rerank / TopK 全局常量
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 3 #总数最少条数

# 断崖阈值（绝对，判断高分文档）
# 实测记录（qwen3-rerank，5 个问题的相邻分差样本）：
#   - 同一批"都相关"的文档之间，相邻分差基本在 0.001~0.05
#   - 最大的一处落差只有绝对 0.125 / 相对 0.144（"迅饶网关怎么配置"：0.8720 → 0.7467）
#   - 试过把阈值调成 ABS=0.10 / RATIO=0.12 让它真正触发：那次截掉的是 0.7467 那条文档，
#     结果答案里"开启绿米网关局域网通信协议"的操作细节直接消失（答案自己说"资料中未展开"）；
#     把阈值还原后同一条文档被保留，细节又回来了。
#   - 结论：本知识库的切片短、信息密度高，分数差一点不代表内容没用，断崖检测保持"宁多勿少"，
#     宁可多带几条进提示词，也不要为了省 token 切掉有效内容。想激进省 token 再调这两个值。
RERANK_GAP_ABS: float = 0.5
# 断崖阈值（相对，判断低分文档）
RERANK_GAP_RATIO: float = 0.25


def _to_float(value: Any) -> float | None:
    """
    把文档分数安全地转成 float
    - None、非数字（如字符串乱码）都返回 None，表示"这条无法参与断崖判断"
    - 数字字符串（如 "0.93"）会被正常转换
    :param value: 原始分数值
    :return: float 或 None
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    # 计算相对落差时的数值保护，避免当前分数为 0 时除零
    _SCORE_EPSILON: float = 1e-6

    def _step_1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式"""

        final_docs = []

        # 1. 获取本地 RRF 的文档
        for rrf_doc in state.get('rrf_chunks') or []:

            format_rrf_doc = {
                "content": rrf_doc.get('content'),
                "title": rrf_doc.get('title'),
                "chunk_id": rrf_doc.get('chunk_id'),
                "url": None,
                "source": "local"
            }
            final_docs.append(format_rrf_doc)

        # 2. 获取 web 远程的文档
        for web_doc in state.get('web_search_docs') or []:

            format_web_doc = {
                "content": web_doc.get('snippet'),
                "title": web_doc.get('title'),
                "chunk_id": None,
                "url": web_doc.get('url'),
                "source": "web"
            }
            final_docs.append(format_web_doc)

        return final_docs

    def _step_2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """使用 Reranker 模型对文档进行精排"""

        # 没有任何候选文档时直接返回，避免把空列表发给重排接口
        if not merged_multi_docs:
            logger.info("Merge 后没有候选文档，跳过硬排序")
            return []

        try:
            user_query = state.get('rewritten_query')
            # 获取文档列表的conten字段组成列表
            contents = [doc.get("content") for doc in merged_multi_docs]
            # 调用Rerank模型：交叉编码器（精排阶段）
            # Query 和 Document 联合编码，精度更高
            rerank_scores = rerank_documents(user_query, contents)

            scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
            # 等同如下写法
            # scored_docs = []
            # for doc, score in zip(merged_multi_docs, rerank_scores):
            #     scored_docs.append({
            #         "content": doc.get("content"),
            #         "title": doc.get("title"),
            #         "chunk_id": doc.get("chunk_id"),
            #         "url": doc.get("url"),
            #         "source": doc.get("source"),
            #         "score": float(score),
            #     })

            sorted_score_docs = sorted(
                scored_docs,
                key=lambda x: x["score"],
                reverse=True
            )

            return sorted_score_docs

        except Exception as e:
            logger.error(f"Rerank 重排序失败: {str(e)}")
            # 降级：保留原文档但分数置空，交给下游按"无分数"处理
            return [{**doc, "score": None} for doc in merged_multi_docs]
    def _step_3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        断崖检测截断：相邻得分差距超过阈值时截断。

        前提：ranked_docs 已按 score 降序排列（由上游 _step_2 保证）

        判定规则（两把尺子，满足任一即视为断崖）：
          1. 绝对落差：当前分 - 下一条分 >= RERANK_GAP_ABS
             适用"分数普遍偏高、掉落幅度大"的场景
          2. 相对落差：绝对落差 / |当前分| >= RERANK_GAP_RATIO
             适用"分数普遍偏低、但比例上暴跌"的场景

        检测范围：从第 1 条开始逐对比较（这样"头部断崖"也能被发现）
        截断位置：第一处命中断崖的位置，再做上下限约束
          - 下限：断崖落在前 RERANK_MIN_TOPK 条之内时，至少保留 RERANK_MIN_TOPK 条
          - 上限：最多保留 RERANK_MAX_TOPK 条
          - 一处都没命中则取满上限

        :param ranked_docs: 已降序排列的文档列表，每条需带 score 字段
        :return: 截断后的文档列表（元素仍是原字典对象，不复制、不新增字段）
        """
        if not ranked_docs:
            return []

        # 1、确定保留边界：上限不超过实际条数，下限不超过上限（条数不足时自动收紧）
        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs))
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)

        # 2、默认不截断（取满上限），只有检测到断崖才把截断位置前移
        cutoff_pos = upper_bound
        cliff_found = False

        # 3、从第 1 条开始逐对比较相邻分数，命中第一处断崖即停止
        #    起点用 0：像 [0.95, 0.92, 0.40, ...] 这种"第 2 条之后就是断崖"的情况也能截到
        for idx in range(0, upper_bound - 1):
            current_score = _to_float(ranked_docs[idx].get("score"))
            next_score = _to_float(ranked_docs[idx + 1].get("score"))

            # 分数缺失（如上游重排失败）时无法判断落差，跳过这一对
            if current_score is None or next_score is None:
                continue

            # 已降序时 abs_gap >= 0；用 abs() 保护是为了兼容负数分数
            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + self._SCORE_EPSILON)

            logger.debug(
                f"断崖检测: 第 {idx + 1} 条({current_score:.4f}) → 第 {idx + 2} 条({next_score:.4f}), "
                f"abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}"
            )

            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                # 索引转实际数量：idx=2 表示保留前 3 条
                cutoff_pos = idx + 1
                cliff_found = True
                logger.info(
                    f"断崖检测命中: 第 {idx + 1} 条之后截断, "
                    f"abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}"
                )
                break

        # 4、下限兜底：断崖落在头部时，压到最少保留条数为止，不能再少
        if cliff_found and cutoff_pos < lower_bound:
            cutoff_pos = lower_bound
            logger.info(f"断崖位于前 {RERANK_MIN_TOPK} 条之内, 按最少保留 {cutoff_pos} 条")
        elif not cliff_found:
            logger.info(f"断崖检测: 未发现断崖, 保留 {cutoff_pos} 条")

        return ranked_docs[:cutoff_pos]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        执行重排序
        流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
        :param state: 需包含 rrf_chunks、web_search_docs、rewritten_query
        :return: 更新后的 state，包含 reranked_docs
        """

        # 1. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)

        # 2. Rerank 精排(精排打分)
        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        # 3. 动态 Top_K 截取(断崖检测)
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        # 4. 更新state
        state['reranked_docs'] = cutoff_docs

        # 5. 返回state
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        return state
if __name__ == "__main__":

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(serialize_json(result, indent=4))
