# processor/import_processor/nodes/node_import_milvus.py
import json
import logging
from typing import Dict, Any, List

from pymilvus import DataType
from processor.import_processor.config import get_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client, escape_milvus_string
config = get_config()


def _truncate_utf8(text, max_bytes: int):
    """
    按 UTF-8 字节数截断字符串。

    注意：Milvus 的 varchar max_length 是按【字节】算的，不是字符数——
    一个中文字符占 3 字节，所以 "100" 实际只够约 33 个汉字，超长会导致整批插入失败。
    这里按字节截断，并避免把一个多字节字符切成两半。
    """
    if not isinstance(text, str):
        return text
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """


    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：Milvus切片数据入库主流程
        执行流程（串行执行，一步一校验，保证数据一致性）：
            1. 输入校验：验证切片有效性、向量字段完整性，提取向量维度
            2. 环境准备：连接Milvus，集合不存在则自动创建Schema+索引
            3. 幂等清理：删除同file_title旧数据，避免重复存储
            4. 批量插入：预处理数据后批量入库，回填Milvus自增chunk_id
            5. 状态更新：将回填了chunk_id的切片更新回全局状态，供下游使用

        异常处理：
            任一步骤失败抛出异常，终止节点执行，保证数据不脏写

        必要参数：chunks
        更新参数：chunks字段回填chunk_id

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据有效性校验
        chunks_json_data, vector_dimension = self._step_1_check_input(state)

        # 步骤2：Milvus客户端连接+集合准备（自动建表）
        client = self._step_2_prepare_collection(vector_dimension)

        # 步骤3：幂等性处理 - 清理同file_title旧数据
        self._step_3_clean_old_data(client, chunks_json_data)

        # 步骤4：批量插入数据+主键chunk_id回填
        updated_chunks = self._step_4_insert_data(client, chunks_json_data)

        # 步骤5：更新全局状态，将回填后的切片回传下游
        state["chunks"] = updated_chunks

        return state

    def _step_1_check_input(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """
        步骤1：输入数据有效性校验
        核心校验项：
            1. chunks非空且为列表类型
            2. 切片包含dense_vector核心字段
            3. 提取向量维度，为集合创建/索引构建提供依据
        参数：
            state: Dict[str, Any] - 流程状态对象，包含上游传入的chunks数据
        返回：
            tuple - (校验通过的切片列表, 稠密向量维度)
        异常：
            任一校验项不通过，抛出ValueError终止入库流程，避免脏数据处理

        """

        # 校验1：chunks非空
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # 校验2：切片包含dense_vector字段
        first_chunk = chunks[0]
        if 'dense_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")

        # 校验3：切片包含 sparse_vector 字段
        if 'sparse_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失sparse_vector字段")

        # 提取向量维度
        vector_dimension = len(first_chunk['dense_vector'])
        return chunks, vector_dimension

    def _step_2_prepare_collection(self, vector_dimension: int):
        """
        步骤2：Milvus客户端连接+集合准备
        核心逻辑：
            1. 获取Milvus单例客户端，验证连接有效性
            2. 集合不存在则自动创建（Schema+索引），存在则直接复用
        参数：
            vector_dimension: int - 稠密向量维度（步骤1提取）
        返回：
            MilvusClient - 已连接、集合准备完成的客户端实例
        异常：
            客户端获取失败/集合名称未配置，抛出异常终止流程
        """

        # 1. 获取milvus客户端对象
        milvus_client = get_milvus_client()
        if not milvus_client:
            self.logger.error("Milvus 连接失败")
            raise MilvusError("Milvus 连接失败")

        # 2. 集合不存在则创建
        collections_name = self.config.chunks_collection
        if not milvus_client.has_collection(collections_name):
            self._create_chunks_collection(collections_name, milvus_client, vector_dimension)

        return milvus_client

    def _create_chunks_collection(self, collections_name, milvus_client, vector_dimension):

        # 1. 创建schem
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 2. 创建列
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
        # 注意：max_length 按字节计（中文 3 字节/字），给足空间避免中文标题超限
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)  # 切片标题
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)  # 父标题
        schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)  # 源文件标题
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)  # 商品名称（幂等性依据）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量

        # 3. 创建索引
        index_params = milvus_client.prepare_index_params()
        # 稠密向量索引：AUTOINDEX自动选最优索引类型+余弦相似度（语义检索常用）
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+内积（IP），适配稀疏向量检索
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        # 创建集合
        milvus_client.create_collection(
            collection_name=collections_name,
            schema=schema,
            index_params=index_params
        )

    def _step_3_clean_old_data(self, client, chunks_json_data):

        """
        幂等清理
        基于每个片段的file_title进行旧数据的清理
        :param client: milvus客户端
        :param chunks_json_data: chunks数据
        :return:
        """

        # 1. 获取查询条件
        file_title = chunks_json_data[0].get("file_title")

        # 2. 执行幂等清理
        self._clear_chunks_by_file_title(client, file_title)

    def _clear_chunks_by_file_title(self, client, file_title):

        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=self.config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")

    def _step_4_insert_data(self, client, chunks_json_data):
        """
        步骤4：批量插入切片数据到Milvus+主键回填
        核心逻辑：
            1. 批量插入数据：提升入库效率，减少Milvus连接次数
            2. 回填chunk_id：将Milvus生成的自增主键回填到切片，供下游业务使用
        参数：
            client - MilvusClient实例
            chunks_json_data: List[Dict[str, Any]] - 待入库的切片列表
        返回：
            List[Dict[str, Any]] - 回填了chunk_id的切片列表
        """
        # 1. 字段裁剪 + 填充 part 字段
        #    Milvus 的 varchar 字段有长度上限（title/parent_title/file_title/item_name 为 100，
        #    content 为 65535），超长会导致整批插入失败，这里做兜底截断
        # 从集合 schema 读取真实的字节上限（Milvus 的 varchar max_length 按字节计，中文 3 字节/字），
        # 读不到时用一套保守的默认值兜底
        field_limits = {}
        try:
            for field in client.describe_collection(self.config.chunks_collection).get("fields", []):
                max_len = (field.get("params") or {}).get("max_length")
                if max_len:
                    # 留 1 字节余量，避免边界上被服务端判超长
                    field_limits[field.get("name")] = max(int(max_len) - 1, 0)
        except Exception as e:
            self.logger.warning(f"读取集合字段长度上限失败，改用默认上限：{e}")
        if not field_limits:
            field_limits = {
                "content": 65535,
                "title": 100,
                "parent_title": 100,
                "file_title": 100,
                "item_name": 100,
            }
        for item in chunks_json_data:
            item["part"] = 0
            for field_name, limit in field_limits.items():
                value = item.get(field_name)
                if isinstance(value, str) and len(value.encode("utf-8")) > limit:
                    self.logger.warning(
                        f"字段 {field_name} 超过 Milvus 上限（{len(value.encode('utf-8'))} 字节 > {limit}），已截断"
                    )
                    item[field_name] = _truncate_utf8(value, limit)

        # 2. 批量插入数据
        result = client.insert(
            collection_name=self.config.chunks_collection,
            data=chunks_json_data
        )

        # 3. 回填chunk_id
        inserted_ids = result.get("ids")
        for idx, item in enumerate(chunks_json_data):
            item["chunk_id"] = inserted_ids[idx]

        return chunks_json_data

if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\hak180产品安全手册\state_vector.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
