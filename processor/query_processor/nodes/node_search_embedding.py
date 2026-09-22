# processor/query_processor/nodes/node_search_embedding.py
from tool.logger import logger
from config.milves_config import milvus_config
from processor.query_processor.base import NodeBase, T
from processor.query_processor.state import QueryGraphState
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.milvus_utils import build_in_expr, create_hybrid_search_requests, hybrid_search, get_milvus_client
from utils.task_utils import add_done_task


class NodeSearchEmbedding(NodeBase):
     """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

     # 覆盖基类的 name 属性，标识节点名称
     name: str = "node_search_embedding"

     def process(self, state: T) -> T:
        """
         节点逻辑
         :param state: 工作流状态对象
         :return: 更新后的状态对象
        """
        try:
             # 1、用户问题和已确认商品名
             query = state.get("rewritten_query")
             item_names = state.get("item_names")

             # 2. 将改写后的问题向量化
             embeddings = generate_embeddings([query])
             dense_vector = embeddings.get("dense")[0]
             sparse_vector = embeddings.get("sparse")[0]

             # 3. 组织查询表达式：匹配已确认商品名
             # 'item_name in ["BrotherHAK-180烫金机","BrotherHAK180烫金机"]'
             expr = build_in_expr("item_name", item_names)
             if expr:
                 logger.info(f"过滤条件: {expr}")
             else:
                 logger.info("未指定商品名过滤，将全库检索")

             # quoted = ", ".join(f"'{v}'" for v in item_names)
             # expr = f"item_name in [{quoted}]"

             # 4. 组织混合向量检索的参数
             reqs = create_hybrid_search_requests(
                 dense_vector=dense_vector,
                 sparse_vector=sparse_vector,
                 expr=expr,
                 limit=10
             )

             # 5. 执行混合向量检索
             client = get_milvus_client()
             collection_name = milvus_config.chunks_collection
             search_res = hybrid_search(
                 client = client,
                 collection_name = collection_name,
                 reqs = reqs,
                 ranker_weights=(0.8,0.2),
                 output_fields=["chunk_id", "content", "item_name"]
             )

             add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
             return {"embedding_chunks":  search_res[0] if search_res else []}

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}

if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(serialize_json(result, indent=4))
