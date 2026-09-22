import json
import re
from typing import List,Dict

from langchain_core.messages import content, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config.lm_config import lm_config
from config.milves_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from utils.task_utils import add_done_task


# ===== 商品名对齐阈值（想调效果，只改这一块就行）=====
# 实测依据（BGE-M3 混合检索 + 归一化评分，问法都带上型号）：
#   库里确实有的商品，top1 分数大多在 0.80~0.92；
#   库里没有的商品（如"小米15""iPhone17""EpsonL3153"），top1 只有 0.57~0.73；
#   少数库里有的商品 top1 会掉到 0.80 以下，但明显甩开第二名（如"RS-12万用表" 0.7977 vs 0.6107）。
CONFIRM_SCORE = 0.80        # 直接确认：top1 达到这个分数就认为可信
WEAK_CONFIRM_SCORE = 0.72   # 弱匹配区下限：达到它、且领先第二名达到 CONFIRM_GAP，也确认
CONFIRM_GAP = 0.10          # "明显领先第二名"的最小分差
CANDIDATE_SCORE = 0.75      # 进候选区（反问用户）的最低分数，低于它一律判"未找到"
CANDIDATE_LIMIT = 5         # 候选商品名最多给几个
MIN_NAME_OVERLAP = 2        # 名字贴近度：公共子串至少 2 个字才认为"名字确实贴近"


def _normalize_name(name: str) -> str:
    """去掉空格/换行等空白字符，方便两个商品名做比较。"""
    return re.sub(r"\s+", "", name or "")


def _longest_common_substring_len(a: str, b: str) -> int:
    """求两个字符串的最长公共子串长度（滚动数组动态规划）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, ch2 in enumerate(b, start=1):
            if ch == ch2:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _name_affinity(extracted_name: str, candidate_name: str) -> int:
    """
    名字贴近度，用来在多个高分候选里挑"最像用户说的那个"：
      完全相等        → 1000 + 长度（最高优先级）
      互相包含        → 500 + 较短者长度
      其它            → 最长公共子串长度
    """
    e = _normalize_name(extracted_name)
    c = _normalize_name(candidate_name)
    if not e or not c:
        return 0
    if e == c:
        return 1000 + len(e)
    if e in c or c in e:
        return 500 + min(len(e), len(c))
    return _longest_common_substring_len(e, c)


def _pick_by_name_affinity(extracted_name: str, candidates: List[Dict]) -> Dict:
    """多个高分候选里挑一个：名字更贴近的优先，都不贴近才比分数。"""
    scored = [
        (_name_affinity(extracted_name, c.get("item_name", "")), float(c.get("score", 0) or 0), c)
        for c in candidates
    ]
    best_affinity = max(s[0] for s in scored)
    # 只有确实存在一小段共同文字时，才用贴近度压过分数，避免只共用一个字就误判
    if best_affinity >= MIN_NAME_OVERLAP:
        return max(scored, key=lambda s: (s[0], s[1]))[2]
    return max(scored, key=lambda s: s[1])[2]


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def _step_1_validate_param(self, state):

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("参数 original_query 不能为空")

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("参数 session_id 不能为空")

        return session_id, original_query
    def _step_4_extract_info(self, original_query, history):
        try:
            chat_model=ChatOpenAI(
                model=lm_config.item_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                model_kwargs={"response_format": {"type": "json_object"}}
             )

            history_text=""
            for msg in history:
                role=msg.get("role")
                # 历史消息存在 MongoDB 的 text 字段里（答案节点读的也是 text）；
                # 原来读的是 content，取到的永远是 None，导致代词指代消解彻底失效
                # （问"HAK180怎么烫金"能答，接着问"该产品价格"就变成"未找到相关产品"）
                content=msg.get("text") or msg.get("content") or ""
                history_text+=f"{role}: {content}\n"
            user_prompt=ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_text, query=original_query)

            messages=[
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT)
                , HumanMessage(content=user_prompt)
            ]
            response=chat_model.invoke(messages)
            content=response.content

            # 5. 数据清洗：去掉可能的代码围栏（```json 和 ```）
            if content.startswith("```json"):
                # content = content[6:]
                content.replace("```json", "").replace("```", "")

             # 6. JSON数据的解析（将JSON字符串转换为字典）
            result = json.loads(content)
            # 7. 健壮性判断：确保返回结果包含item_names和rewritten_query字段
            if "item_names" not in result:
                logger.warning("大模型返回结果缺少item_names字段")
                result = {"item_names": []}

            if "rewritten_query" not in result:
                logger.warning("大模型返回结果缺少rewritten_query字段")
                result = {"rewritten_query": original_query}

            result["item_names"] = [
                name
                .replace(" ", "")
                .replace("\n", "")
                .replace("\t", "")
                .replace("\r", "")
                for name in result["item_names"]
            ]

            return result
        except Exception as e:
            # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
            logger.error(f"大模型调用异常：{e}")
            # 异常时返回默认结果：空商品名列表+原始查询
            return {"item_names": [], "rewritten_query": None}
    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        """
           把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
           :param item_names: 列表[字符串] - 步骤4中 提取的商品名列表（如["苹果15", "华为P60"]）
           :return: 列表[字典] - 格式：
                [
                    {
                        "extracted_name": "提取的原始商品名",  # 如"苹果15"
                        "matches": [                          # 该商品名的TopN匹配结果，无则空列表
                            {
                                "item_name": "数据库中的商品名",  # Milvus中存储的标准化商品名
                                "score": 0.98                  # 混合搜索的相似度评分（0-1，越高越相似）
                            },
                            ...
                        ]
                    },
                    ...
                ]
        """
        # 1、初始化最终返回结果列表，存储每个商品名的向量化查询结果
        results = []

        # 2、获取Milvus向量数据库的客户端连接对象
        client = get_milvus_client()

        # 3、校验Milvus客户端连接是否成功，失败则记录错误日志并返回空结果
        if not client:
            logger.error("连接 Milvus 失败")
            return results

        # 4、从环境变量中获取Milvus中存储商品名称向量的集合名（表名）
        collection_name = milvus_config.item_name_collection  # kb_item_names

        # 5、对所有商品名称批量生成BGEM3向量（稠密+稀疏），相比逐个生成提升处理效率
        # embeddings格式：{"dense": [向量1, 向量2,...], "sparse": [向量1, 向量2,...]}
        embeddings = generate_embeddings(item_names)

        # 6、遍历每个商品名称，逐个执行向量搜索（保证结果与原始商品名一一对应）
        for i in range(len(item_names)):
            try:
                # 从批量生成的向量结果中，取出当前商品名对应的稠密向量（高维连续值，如[0.12, 0.35,...]）
                dense_vector = embeddings.get("dense")[i]
                # 从批量生成的向量结果中，取出当前商品名对应的稀疏向量（键值对，如{100:0.747, 205:0.664}）
                sparse_vector = embeddings.get("sparse")[i]

                # 构造Milvus混合搜索请求对象，传入稠/稀疏向量，指定返回Top5匹配结果
                # reqs返回格式：[稠密向量搜索请求, 稀疏向量搜索请求]
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )

                # 执行BGEM3混合向量搜索，获取数据库中的匹配结果和评分
                # 默认配置：稠/稀疏向量权重各0.8/0.2，开启评分归一化（将距离值转为0-1相似度评分）
                search_res = hybrid_search(
                    client=client,  # Milvus客户端连接实例
                    collection_name=collection_name,  # 目标向量集合名（存储商品向量的表）
                    reqs=reqs,  # 混合搜索请求对象列表
                    ranker_weights=(0.8, 0.2),  # 稠/稀疏向量评分权重配比（和为1最佳）
                    limit=5,  # 最终返回Top5匹配结果
                    norm_score=True,  # 开启评分归一化，统一评分量级为0-1
                    output_fields=["item_name"]  # 指定返回Milvus中存储的商品名字段（业务字段）
                )

                # 初始化当前商品名的匹配结果列表，存储匹配到的商品名+对应相似度评分
                matches = []
                # 校验搜索结果是否有效（非空且包含数据，适配Milvus批量搜索格式）
                if search_res and len(search_res) > 0:
                    # 遍历当前商品名的Top5匹配结果（search_res[0]为该商品的独立搜索结果集）
                    for hit in search_res[0]:
                        # 提取匹配结果中的商品名和评分，做防KeyError处理（设置默认空字典）
                        # hit格式：{"id": 数据库ID, "distance": 相似度评分, "entity": {"item_name": "标准化商品名"}}
                        matches.append(
                            {
                                "item_name": hit.get("entity", {}).get("item_name"),  # 数据库标准化商品名
                                "score": hit.get("distance"),  # 0-1相似度评分
                            }
                        )

                # 将当前商品名的原始名称+匹配结果，封装后加入最终结果列表
                results.append({
                    "extracted_name": item_names[i],  # step4提取的原始商品名称
                    "matches": matches  # 该商品名的Top5匹配结果（含评分）
                })

            # 捕获单个商品名处理的异常（不中断其他商品名执行），仅记录错误日志
            except Exception as e:
                logger.error(f"查询商品名 '{item_names[i]}' 时出错: {e}")

        # 返回所有商品名的向量化+搜索结果列表
        return results
    def _step_6_align_item_names(self, query_results) -> dict:
        """
        6 根据Milvus搜索评分，逐个对齐step4提取的item_names，生成「确认商品名」和「候选商品名」
        对齐规则（优先级 a > b > b2 > c > d，阈值都定义在本文件顶部）：
                a  只有一个匹配结果评分≥CONFIRM_SCORE(0.80) → 直接确认该商品名
                b  多条评分≥CONFIRM_SCORE(0.80) → 取"名字最贴近提取词"的那条，全都不贴近才取分数最高的
                b2 没有≥0.80的，但top1≥WEAK_CONFIRM_SCORE(0.72)且领先第二名≥CONFIRM_GAP(0.10) → 也确认
                c  以上都不满足 → 把评分≥CANDIDATE_SCORE(0.75)的前5个作为候选，反问用户
                d  没有≥0.75的结果 → 确认和候选都为空，由step7回复"未找到"
        :param query_results: 列表[字典] - step5的返回结果，每个商品名的搜索匹配数据（格式同step5返回值）
        :return: 字典 - 商品名对齐结果，包含确认列表和候选列表，格式：
                 {
                     "confirmed_item_names": ["确认商品名1", "确认商品名2"],  # 去重后的确认商品名，无则空列表
                     "options": ["候选商品名1", "候选商品名2", ...]          # 去重后的候选商品名，无则空列表
                 }
        """
        # 1、初始化确认商品名列表（符合高置信度规则的商品名）
        confirmed_item_names: List[str] = []
        # 2、初始化候选商品名列表（低置信度，需用户确认的商品名）
        options: List[str] = []

        for res in query_results:
            # 提取原始的数据，商品名和匹配结果
            extracted_name = (res.get("extracted_name", "") or  "").strip()
            # 获取匹配的商品名，无就获取空列表
            raw_matches = res.get("matches", []) or []
            # 若无匹配结果，直接跳过当前商品名的对齐
            if not raw_matches:
                continue

            # 先按商品名去重（库里可能存在重复记录，不去重会把"和第一名的分差"算成0），
            # 同一个商品名只保留分数最高的一条，再按分数从高到低排好
            best_score_by_name: Dict[str, float] = {}
            for m in raw_matches:
                name = m.get("item_name")
                if not name:
                    continue
                score = float(m.get("score", 0) or 0)
                if name not in best_score_by_name or score > best_score_by_name[name]:
                    best_score_by_name[name] = score
            matches = [
                {"item_name": name, "score": score}
                for name, score in sorted(best_score_by_name.items(), key=lambda kv: kv[1], reverse=True)
            ]
            if not matches:
                continue

            # top1/top2/分差：判断"是否明显领先第二名"用
            top1 = matches[0]["score"]
            top2 = matches[1]["score"] if len(matches) > 1 else 0.0
            gap = top1 - top2

            # 筛选高置信度匹配结果：评分≥CONFIRM_SCORE
            high = [m for m in matches if m["score"] >= CONFIRM_SCORE]

            # 规则a: 只有一个高置信度结果 → 直接确认该商品名
            if len(high) == 1:
                confirmed_item_names.append(high[0]["item_name"])
                logger.info(
                    f"步骤6：【{extracted_name}】top1={top1:.4f} 达确认阈值 → 确认 {high[0]['item_name']}")
                continue  # 匹配到规则a，跳过后续规则判断

            # 规则b: 多条高置信度结果
            if len(high) > 1:
                # 原实现要求"商品名完全相等"才优先，实际库里名字都带后缀（如"联想至像大象Z1黑白激光
                # 多功能双面一体机"），用户几乎不可能一字不差说出来，所以改成按"名字贴近度"来挑
                picked = _pick_by_name_affinity(extracted_name, high)
                confirmed_item_names.append(picked.get("item_name"))
                logger.info(
                    f"步骤6：【{extracted_name}】有{len(high)}条高分 → 贴近度选中 {picked.get('item_name')}"
                    f"（{picked.get('score'):.4f}）")
                continue  # 匹配到规则b，跳过后续规则判断

            # 规则b2: 没有高分结果，但top1进了弱匹配区且明显领先第二名 → 也确认
            # （专治"RS-12万用表"这类库里确实有、分数却卡在0.8以下的说法）
            if top1 >= WEAK_CONFIRM_SCORE and gap >= CONFIRM_GAP:
                confirmed_item_names.append(matches[0]["item_name"])
                logger.info(
                    f"步骤6：【{extracted_name}】top1={top1:.4f} 领先第二名 {gap:.4f} → 确认"
                    f" {matches[0]['item_name']}")
                continue

            # 规则c: 以上都不满足 → 只有真正沾边（≥CANDIDATE_SCORE）的才进候选，反问用户
            # 注：库里根本没有的商品，top1 通常只有0.6~0.73，会被挡在这里，走规则d
            picked_candidates = [m["item_name"] for m in matches if m["score"] >= CANDIDATE_SCORE]
            options.extend(picked_candidates[:CANDIDATE_LIMIT])
            logger.info(
                f"步骤6：【{extracted_name}】top1={top1:.4f} gap={gap:.4f} 不满足确认条件 → "
                f"候选 {len(picked_candidates[:CANDIDATE_LIMIT])} 个")

            # 规则d: 没有≥CANDIDATE_SCORE的结果 → 不追加任何候选，确认+候选均为空
        # 返回最终对齐结果：确认列表和候选列表均做去重处理（list(set())）
        return {
            "confirmed_item_names": list(dict.fromkeys(confirmed_item_names)),  # 去重（保留原顺序）
            "options": list(dict.fromkeys(options))[:CANDIDATE_LIMIT]  # 去重 + 按分数顺序保留前5个
        }

    def _step_7_check_confirmation(self, state, align_result, history):
        """
        7 检查step6对齐后的商品名状态，分3种分支更新state，并同步更新历史消息的商品名关联
        :param state: 字典 - 原始会话状态，包含session_id/original_query等核心字段
        :param align_result: 字典 - step6的对齐结果
        :param history: 列表[字典] - 近期会话历史
        :return: 字典 - 更新后的会话状态，包含item_names/answer
        """
        # 从对齐结果中提取确认商品名列表，无则空列表
        confirmed = align_result.get("confirmed_item_names", [])
        # 从对齐结果中提取候选商品名列表，无则空列表
        options = align_result.get("options", [])

        # 分支A：有确认的商品名（高置信度，无需用户确认）
        if confirmed:
            # 注意：这里原来会把"历史里所有 item_names 为空的消息"统统回填成本次确认的商品名，
            # 结果会把别的产品的问题也标错（实测问完 HAK180 后，"小米15怎么截屏"那条用户消息
            # 也被标成了 BrotherHAK180烫金机）。指代消解靠的是历史正文（见 _step_4_extract_info），
            # 不依赖这个回填，所以这里不再回填，避免污染历史数据。
            # 如果需要重新启用，把下面两行取消注释即可：
            # ids_to_update = [str(m["_id"]) for m in history if not m.get("item_names") and m.get("_id")]
            # if ids_to_update:
            #     update_message_item_names(ids_to_update, confirmed)

            # 更新会话状态：设置确认商品名、改写后的查询
            state["item_names"] = confirmed
            state["answer"] = ""
            # 返回更新后的状态
            return state

        # 分支B：无确认商品名，但有候选商品名（中置信度，需用户明确）
        if options:
            # 候选商品名拼接为字符串（取前3个，避免过长），格式："商品1、商品2、商品3"
            options_str = "、".join(options[:3])
            # 构造向用户确认的提示语
            answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            # 更新会话状态：设置确认提示语、清空商品名列表
            state["answer"] = answer
            state["item_names"] = []
            return state

        # 分支C：无确认商品名，且无候选商品名（无匹配结果，需用户重新提供）
        state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
        state["item_names"] = []
        return state
    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        """
         8 把本次处理的核心信息（用户问题、助手答案、商品名、改写查询）写入MongoDB的会话历史
         包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
         :param state: 字典 - step6更新后的会话状态，包含answer/item_names等字段
         :param session_id: 字符串 - 会话唯一标识
         :param rewritten_query: 字符串 - step3改写后的完整问题
         :param message_id: 字符串 - 本次用户问题的消息唯一ID
         :return:
         """
        # 这里不再写 assistant 消息：
        # 分支B/C的答案（反问语/未找到）同样会流到下游的 node_answer_output，
        # 由它的 _step_4_write_history 统一写一次。两边都写会让历史里出现两条一模一样的回复。

        # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="user",  # 消息角色：用户
            text=state["original_query"],  # 消息内容：用户原始查询
            rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
            item_names=state.get("item_names", []),  # 补充关联的商品名列表
            message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
        )

        # 返回最终会话状态，供下游节点使用
        return state

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        必要参数：session_id、original_query
        更新参数：history、rewritten_query、item_names、answer

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：校验参数
        session_id, original_query = self._step_1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        # 步骤2：获取历史记录
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        # 更新状态
        state["history"] = history

        # 步骤3：用户初始消息保存
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 步骤4：提取信息
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names")
        rewritten_query = extract_res.get("rewritten_query", original_query)
        # 更新状态
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names

        # 5. & 6. 如果有提取到商品名，进行搜索和对齐
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 7. 检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8. 写入最终历史
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        return state
