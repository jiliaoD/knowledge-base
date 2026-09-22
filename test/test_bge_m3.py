# test/bge_m3测试向量获取.py
import logging
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，保证任意 cwd 都能 import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.embedding_utils import get_bge_m3_ef

# embedding_utils 导入时会调用 setup_logging() 把根日志级别设为 INFO，
# 这里调回 DEBUG，让下面的 logger.debug 能正常显示
logging.getLogger().setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)   # 或者 logging.getLogger()


# 加载BGE-M3模型单例
model = get_bge_m3_ef()

# 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
texts = ["测试", "hello"]
embeddings = model.encode_documents(texts)

logger.debug("获取成功！")
logger.debug(embeddings)

# 稠密向量获取
dense_obj = embeddings["dense"]
dense_list = [emb.tolist() for emb in dense_obj]
logger.debug(dense_list)

# 稀疏向量获取：解析为字典格式（适配序列化/存储）
sparse_obj = embeddings["sparse"]
processed_sparse = []
for i in range(len(texts)):

    # 提取第i个文本的稀疏向量索引
    sparse_indices = sparse_obj.indices[
        sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]
    ].tolist()

    # 提取第i个文本的稀疏向量权重
    sparse_data = sparse_obj.data[
        sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]
    ].tolist()

    # 构造{特征索引: 归一化权重}的稀疏向量字典
    # 把两个列表“打包”成一个字典（Milvus 数据库的要求）
    sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
    processed_sparse.append(sparse_dict)
    logger.debug(f"第{i}个文本的稀疏向量: {sparse_dict}")

    # sparse_indices = [0, 5, 23, 156]    # 特征在词汇表中的位置编号
    # sparse_data = [0.8, 0.3, 0.9, 0.2]  # 每个特征的权重
    # zip 后生成：[(0, 0.8), (5, 0.3), (23, 0.9), (156, 0.2)]
    #              ↑   ↑
    #            索引 权重