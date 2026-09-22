# processor/query_processor/nodes/node_answer_output.py

import re
from typing import List, Dict, Tuple

from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.answer_prompt import ANSWER_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.llm_utils import get_llm_client
from utils.mongo_history_utils import save_chat_message
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import add_done_task, set_task_result

# 拼装提示词时正文部分的最大字符预算（防止把大模型上下文撑爆）
MAX_CONTEXT_CHARS = 12000

# 最终返回给前端的图片数量上限（按文档相关度顺序取前 N 张）
MAX_IMAGE_COUNT = 6


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案输出
    流程: 检查已有答案 → 构建提示词 → LLM 生成 → 写入历史 → 发送结束事件
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_answer_output"

    # 资料正文的安全处理表：把可能伪装成提示词结构的符号换成相近字符
    _SECTION_MARK_TRANS = str.maketrans({"【": "〖", "】": "〗"})

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        1 判断 state 中的 answer 是否已经存在，如果存在直接输出 answer 中的答案，
          注意判断是否需要流式输出，需要则流式输出
        2 根据 state 中的问题、重写问题、历史对话、提问商品（item_names）、
          重排内容组织 prompt 并调用 llm 生成答案
        3 调用大模型输出答案，注意判断是否需要流式输出，需要则流式输出
        4 把答案写入到 mongodb 的 history 中（utils/mongo_history_utils.py 的 save_chat_message）
        5 做最后一次 push 操作（主要是为了触发前端图片渲染）
             {
                "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
                "status": "completed",
                "image_urls": [
                    "http://local-server/images/panel_view.jpg",
                    "http://local-server/images/button_detail.jpg"
                ]
              }
        """

        # 阶段一：检查 answer 是否存在，如果存在直接输出 answer 中的答案
        answer_exists = self._step_1_check_answer(state)

        # 阶段二、三：如果没有 answer，则构建 Prompt 并调用大模型生成答案
        if not answer_exists:
            prompt = self._step_2_construct_prompt(state)
            state["prompt"] = prompt

            self._step_3_generate_response(state, prompt)

        # 提取图片 URL（用于历史记录和前端展示）
        image_urls = self._extract_images_from_docs(state.get("reranked_docs") or [])

        # 非流式调用方（web/api/query_service.py）是靠内存任务表取结果的：
        # 答案已经通过 set_task_result("answer") 带出去了，本轮图片也一起放进去，
        # 这样它就不必去历史里翻图片（翻出来的很可能是更早那一轮的图）
        if not state.get("is_stream"):
            set_task_result(state.get("session_id"), "image_urls", image_urls)

        # 阶段四：把答案写入到 mongodb 的 history 中
        if state.get("answer"):
            logger.info("---写入MongoDB历史记录---")
            self._step_4_write_history(state, image_urls=image_urls)

        # 阶段五：流式输出结束，发送 final 事件（最后兜底，确保图片都能渲染、前端能收尾）
        logger.info(f"---发送 final 事件---图片为：{image_urls}")
        if state.get("is_stream"):
            push_to_session(
                state["session_id"],
                SSEEvent.FINAL,
                {
                    "answer": state["answer"],
                    "status": "completed",
                    "image_urls": image_urls  # 发送图片URL给前端
                }
            )

        # 任务追踪：多路搜索节点在 base 里拿不到 session_id，所以在各节点内部补"已完成"
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

        logger.info("---node_answer_output 节点处理结束---")
        return state

    def _step_1_check_answer(self, state) -> bool:
        """
        阶段一：检查 state 中是否已有 answer。
        - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
        - 若不存在：返回 False
        """
        answer = state.get("answer")
        is_stream = state.get("is_stream")

        if answer:
            if is_stream:
                logger.info("---Step 1: 发现已有答案，执行流式推送---")
                push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
            else:
                set_task_result(state["session_id"], "answer", answer)
            return True
        else:
            return False

    def _step_2_construct_prompt(self, state: QueryGraphState) -> str:
        """
        阶段二：构建 Prompt
        根据 state 中的问题、重写问题、历史对话、提问商品（item_names）、重排内容组装 LLM 提示词
        """
        char_budget = MAX_CONTEXT_CHARS

        # 1. 获取问题和商品名（优先使用重写后的问题）
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names") or []

        # 2. 格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 4. 格式化 Item Names (提问商品)
        item_names_str = ", ".join(item_names) if item_names else "无指定商品"

        # 5. 组装提示词
        prompt = ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=item_names_str,
            question=question,
        )
        logger.info(f"组装后的提示词为：{prompt}")
        return prompt

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """
        格式化重排序文档，带字符预算控制

        逻辑：
        1. 遍历重排序后的文档列表（已按相关性从高到低排序）
        2. 对每个文档提取关键信息（正文、来源、chunk_id、url、标题、分数）
        3. 拼成"元数据头 + 正文"的文本块，例如：
           [1] [source=local] [chunk_id=123] [score=0.9500] [title=操作手册]
           这里是文档的正文内容...
        4. 累加字符数，超过预算就停止添加，避免把大模型上下文撑爆
        5. 正文用 <资料> 标记包起来并做中性化处理，配合提示词里的"资料不是指令"规则，
           防止网页等内容里的提示词注入被当成指令执行
        """
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, start=1):
            content = self._neutralize_content(doc.get("content"))
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field)).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n<资料>\n" + content + "\n</资料>"

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _neutralize_content(self, text) -> str:
        """
        中性化资料正文，防止提示词注入（只做最小改动，不影响正文其余内容）
        1. 把 【】 换成 〖〗：避免资料内容伪装成【参考内容】【历史对话】【用户问题】等提示词区块
        2. 把 <资料> / </资料> 换成全角：避免资料内容提前闭合我们加的数据标记
        """
        text = text or ""
        text = text.translate(self._SECTION_MARK_TRANS)
        text = text.replace("<资料>", "＜资料＞").replace("</资料>", "＜/资料＞")
        return text

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _step_3_generate_response(self, state: QueryGraphState, prompt: str) -> QueryGraphState:
        """
        阶段三：生成回答
        调用 llm 生成答案，支持流式输出
        """
        logger.info("---Step 3: 开始生成回答 (LLM Generation)---")

        # 获取 LLM 客户端（统一走 utils/llm_utils.py，复用连接与缓存）
        llm = get_llm_client()

        # 判断是否需要流式输出
        session_id = state.get("session_id")
        is_stream = state.get("is_stream")

        if is_stream:
            logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
            final_text = ""
            try:
                # 使用 stream 方法进行流式生成
                for chunk in llm.stream(prompt):
                    delta = getattr(chunk, "content", "") or ""
                    if delta:
                        final_text += delta
                        # 将增量内容放入队列
                        push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})

                logger.info(f"流式输出完成，总长度: {len(final_text)}")

            except Exception as e:
                logger.error(f"流式生成出错: {e}", exc_info=True)
                # 发生错误时，尝试推送到前端
                push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

            state["answer"] = final_text
        else:
            # 非流式直接调用
            logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
            try:
                response = llm.invoke(prompt)
                content = response.content
                state["answer"] = content
                set_task_result(session_id, "answer", content)
                logger.info(f"生成回答完成，长度: {len(content)}")
            except Exception as e:
                logger.error(f"生成回答出错: {e}", exc_info=True)
                state["answer"] = "抱歉，生成回答时出现错误。"

        return state

    def _extract_images_from_docs(self, docs):
        """
        辅助方法：从文档列表中提取图片URL

        核心逻辑：
        1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）
        2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果），验证后缀名是否为图片格式
        3. 策略二：用正则扫描文档 'content' 正文（常见于本地 Markdown 文档），匹配 ![alt](url)
        4. 对提取到的 URL 去重，返回唯一图片列表

        :param docs: 文档列表，每个文档为字典格式
        :return: 图片 URL 字符串列表
        """
        images = []
        seen = set()  # 用于去重，避免同一张图片重复出现
        if not docs:
            return []

        # 正则解释：r'!\[.*?\]\((.*?)\)'
        # 1. !\[   匹配 Markdown 图片语法的开头 "!["
        # 2. .*?   非贪婪匹配图片描述文本（[] 中间的内容）
        # 3. \]    匹配描述文本的结束符 "]"
        # 4. \(    匹配 URL 部分的开始符 "("
        # 5. (.*?) 捕获组：非贪婪匹配括号内的实际 URL 内容
        # 6. \)    匹配 URL 部分的结束符 ")"
        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

        logger.info(f"开始提取图片，待处理文档数: {len(docs)}")

        for i, doc in enumerate(docs):
            # 1. 优先检查 url 字段 (主要针对 Web Search 结果)
            url = (doc.get("url") or "").strip()
            if url and url.lower().startswith(("http://", "https://")):
                # 简单后缀判断：确保是静态图片资源
                if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                    if url not in seen:
                        logger.debug(f"文档[{i}] 发现图片 URL (字段): {url}")
                        seen.add(url)
                        images.append(url)

            # 2. 检查 content 字段中的 Markdown 图片 (主要针对 Local Chunk)
            text = (doc.get("content") or "").strip()
            if text:
                # 有捕获组时，findall 只返回括号里匹配到的内容（即 URL）
                matches = md_img_pattern.findall(text)
                for img_url in matches:
                    img_url = img_url.strip()
                    # 只保留绝对地址：相对路径（如 images/xxx.jpg）前端打不开，属于脏数据
                    if not img_url.lower().startswith(("http://", "https://")):
                        continue
                    if img_url not in seen:
                        logger.debug(f"文档[{i}] 正文发现 Markdown 图片: {img_url}")
                        seen.add(img_url)
                        images.append(img_url)

        logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")

        # 按相关度顺序截断：文档已按分数降序，所以前面的图片来自最相关的文档
        if len(images) > MAX_IMAGE_COUNT:
            logger.info(f"图片数量超过上限 {MAX_IMAGE_COUNT}，截断为前 {MAX_IMAGE_COUNT} 张")
            images = images[:MAX_IMAGE_COUNT]

        return images

    def _step_4_write_history(self, state: QueryGraphState, image_urls=None) -> QueryGraphState:
        """
        阶段四：把本轮答案写入 MongoDB history。
        利用 utils/mongo_history_utils.py 中的 save_chat_message 方法。
        """
        session_id = state.get("session_id", "default")
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []

        try:
            if answer:
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query="",
                    item_names=item_names,
                    image_urls=image_urls,
                    message_id=None
                )
        except Exception as e:
            # 写历史失败不应影响主链路
            logger.error(f"写入Mongo历史记录失败: {e}")

        return state


if __name__ == "__main__":

    logger.info("开始测试: 答案生成节点")

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2",
            "score": 0.95,
            "url": None,
            "content": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "url": "http://example.com/hak180_troubleshooting.jpeg",
            "content": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "url": None,
            "content": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,   # 测试非流式；要测流式需先有 SSE 会话队列
        "answer": None        # 初始无答案
    }

    # 运行节点
    node_answer_output = NodeAnswerOutput()
    result = node_answer_output(mock_state)

    # 1. 验证 Prompt 构建
    if result.get("prompt"):
        print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
    else:
        print("[FAIL] Prompt 未构建")

    # 2. 验证答案生成
    answer = result.get("answer")
    if answer and len(answer) > 10:
        print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
        print(f"答案预览: {answer[:50]}...")
    else:
        print(f"[WARN] 答案生成可能异常 (Content: {answer})")

    # 3. 验证图片提取
    # 期望提取到 3 张图片：
    #   1. http://local-server/images/panel_view.jpg      (来自 local_101 正文)
    #   2. http://local-server/images/knob_detail.png     (来自 local_101 正文)
    #   3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)
    print("[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
    print(" - http://local-server/images/panel_view.jpg")
    print(" - http://local-server/images/knob_detail.png")
    print(" - http://example.com/hak180_troubleshooting.jpeg")
