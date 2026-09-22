import logging
from typing import Dict

from processor.import_processor.base import BaseNode, setup_logging


class TestNode(BaseNode):

    name = "node_test"

    def process(self, state: Dict) -> Dict:


        self.logger.error(f"{self.name}正在执行")
        self.logger.debug(f"{self.name}正在debug")
        self.logger.warning(f"{self.name}warning")

        self.log_step("MCP查询","开始执行")



        return {}



if __name__ == "__main__":

    # 激活日志
    setup_logging(logging.INFO)

    test_node = TestNode()
    # 使用 对象() 的方式相当于调用了 对象的__call__()
    test_node({"abc": 456})