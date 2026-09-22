# processor/import_processor/nodes/node_bge_embedding.py
import json
import logging
from typing import Dict, List

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：BGE-M3文本向量化处理
        流程总览：
            1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
            2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            3. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用

        必要参数：chunks
        更新参数：chunks字段新增dense_vector/sparse_vector

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据校验
        chunks = self._step_1_validate_input(state)

        # 步骤2：批量生成双向量，为切片绑定向量字段
        output_data = self._step_2_generate_embeddings(chunks)

        # 步骤3：更新全局状态，将带向量的chunks回传下游
        state['chunks'] = output_data
        return state

    def _step_1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """
        步骤 1：输入数据有效性校验
        核心作用：
            1. 从全局状态提取待向量化的chunks切片列表
            2. 严格校验chunks类型和非空性，无有效数据则终止向量化
        参数：
            state: ImportGraphState - 流程全局状态对象
        返回：
            List[Dict] - 校验通过的文本切片列表
        异常：
            若chunks非列表/为空，抛出ValueError，终止当前向量化流程
        """

        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return chunks

    def _step_2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        步骤 2: 批量生成向量（核心业务逻辑）
        核心逻辑：
            1. 分批处理：避免一次性处理过多数据导致显存溢出（OOM）。
            2. 文本构造：将 item_name 和 content 拼接，增强语义（商品名作为核心特征前置）。
            3. 向量生成：调用模型批量生成 Dense（稠密）和 Sparse（稀疏）向量。
        参数：
            chunks: List[Dict] 待向量化的文本切片列表
        返回：
            List[Dict]: 包含向量字段（dense_vector/sparse_vector）的文本切片列表
        """

        # 定义批处理数量
        batch_size = 3
        # 初始化一个空列表
        output_data = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            # 批量构造文本
            texts = [f"{chunk['item_name']} \n{chunk['content']}" for chunk in batch_chunks]


            # 批量生成向量
            vectors = generate_embeddings(texts)
            dense_vectors = vectors.get("dense")
            sparse_vectors = vectors.get("sparse")

            for j, text in enumerate(texts):
                chunk = batch_chunks[j]
                chunk["dense_vector"] = dense_vectors[j]
                chunk["sparse_vector"] = sparse_vectors[j]
                output_data.append(chunk)

        return output_data



if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\hak180产品安全手册\state.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_bge_embedding = NodeBGEEmbedding()
    result = node_bge_embedding(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))