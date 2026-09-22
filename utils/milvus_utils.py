from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from processor.import_processor.config import get_config

_milvus_client = None
config=get_config()

def get_milvus_client():

    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    _milvus_client = MilvusClient(config.milvus_url)
    return _milvus_client
import re   # 文件顶部补一个

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def build_in_expr(field: str, values) -> str | None:
    """
    构造 `字段 in ["值1","值2"]` 形式的 Milvus 过滤表达式。

    为什么需要它：
      - 值里出现引号/反斜杠时，裸拼字符串会让 Milvus 解析失败（code=1100），
        而节点里的 try/except 会把异常吞掉，表现为“什么都检索不到”。
      - 值里出现换行同样会解析失败（实测），所以这里先压掉控制字符。

    处理规则：
      - 兼容 None / 字符串 / 列表；空值、纯空白、重复值自动剔除（保持原顺序）
      - 控制字符（\n \r \t 等）直接删除，空格保留 —— 与导入侧 item_name 的清洗规则一致
      - 每个值经 escape_milvus_string 转义后再用双引号包裹
      - 无有效值时返回 None，调用方按“不加过滤（全库检索）”处理

    :param field: 字段名，如 "item_name"
    :param values: 待匹配的值集合
    :return: 过滤表达式；无有效值时返回 None
    """
    if values is None:
        return None
    if isinstance(values, str):          # 防呆：上游误传单个字符串
        values = [values]

    cleaned = []
    for v in values:
        if v is None:
            continue
        text = _CONTROL_CHARS.sub("", str(v)).strip()
        if text:
            cleaned.append(text)

    cleaned = list(dict.fromkeys(cleaned))   # 保序去重
    if not cleaned:
        return None

    literals = ", ".join(f'"{escape_milvus_string(v)}"' for v in cleaned)
    return f"{field} in [{literals}]"

def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用：
        转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数：
        value: 需要转义的原始字符串
    返回：
        str: 转义后的安全字符串
    """
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value

# utils/milvus_utils.py
# 在这个文件中添加如下方法

def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]
def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算，避免一个得分特别大、另一个特别小导致权重失效。
        # 版本：V2.4
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        from tool.logger import logger
        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None