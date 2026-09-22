# utils/json_format_utils.py

"""
JSON 格式化工具模块

提供统一的 JSON 序列化和格式化功能，确保项目中 JSON 输出的一致性
"""

import json
from typing import Any, Dict
from bson import ObjectId

class CustomJSONEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，支持 MongoDB ObjectId 等特殊类型
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, cls=CustomJSONEncoder)


from bson import ObjectId
import json
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID



class MongoEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，支持MongoDB ObjectId等特殊类型的序列化

    支持的类型：
    - ObjectId: 转换为字符串
    - datetime/date: 转换为ISO格式字符串
    - Decimal: 转换为浮点数
    - UUID: 转换为字符串
    """

    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


def serialize_json(data, ensure_ascii=False, indent=None, **kwargs):
    """
    将数据序列化为JSON字符串，自动处理MongoDB ObjectId等特殊类型

    参数：
        data: 要序列化的数据（字典、列表等）
        ensure_ascii: 是否确保ASCII编码（默认False，支持中文）
        indent: 缩进空格数（默认None，紧凑格式；设为4可美化输出）
        **kwargs: 其他传递给json.dumps的参数

    返回：
        JSON格式的字符串

    示例：
        # 基础用法
        json_str = serialize_json({"_id": ObjectId("..."), "name": "测试"})

        # 美化输出
        json_str = serialize_json(data, indent=4)

        # 在日志中使用
        logger.info(serialize_json(result, indent=2))
    """
    return json.dumps(data, cls=MongoEncoder, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def to_json_file(data, filepath, ensure_ascii=False, indent=4, **kwargs):
    """
    将数据序列化为JSON文件，自动处理MongoDB ObjectId等特殊类型

    参数：
        data: 要序列化的数据
        filepath: 文件保存路径
        ensure_ascii: 是否确保ASCII编码（默认False，支持中文）
        indent: 缩进空格数（默认4，美化输出）
        **kwargs: 其他传递给json.dump的参数

    示例：
        to_json_file(result, "output.json")
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=MongoEncoder, ensure_ascii=ensure_ascii, indent=indent, **kwargs)

