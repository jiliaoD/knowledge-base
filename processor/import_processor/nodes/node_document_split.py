# processor/import_processor/nodes/node_document_split.py
import json
import logging
import re
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

        必要参数：md_path(完整流程中非必要，备份测试用的json文件)、md_content、file_title
        更新参数：chunks

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # ===================================== 步骤1：加载并标准化输入数据 =====================================
        # 作用：从状态字典提取MD内容/文件标题，统一换行符消除系统差异
        # 输出：标准化后的md_content、文件标题；
        content, file_title = self._step_1_get_inputs(state)

        # ===================================== 步骤2：按MD标题进行初次切分 ===============================
        # 作用：基于Markdown标题（#/##/###）切分文档为独立章节，自动跳过代码块内的伪标题，保证章节语义完整
        # 输出：初切后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计/日志使用）
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)

        # ===================================== 步骤3：无标题场景兜底处理 ===================================
        # 作用：解决MD文档无任何标题的边界情况，避免后续切分逻辑异常
        # 输出：有标题则返回步骤2的章节列表；无标题则将全文封装为单个「无标题」章节，保证数据格式统一
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # ===================================== 步骤4：Chunk精细化处理（长切短合） ===========================
        # 作用：核心切分逻辑，先将超长章节按「段落→句子」二次切分，再合并同父标题的过短章节，减少碎片化
        # 额外处理：对所有Chunk做parent_title兜底，适配Milvus向量库必填字段要求
        # 输出：长度适中、语义完整、低碎片化的最终Chunk列表（可直接用于向量入库/大模型调用）
        sections = self._step_4_refine_chunks(sections)

        # ===================================== 步骤5：输出文档切分统计信息 ==================================
        # 作用：打印核心统计数据，便于监控切分效果、调试问题（原始行数/最终Chunk数/首个Chunk预览）
        # 输出：无返回值，仅通过logger输出标准化统计日志
        self._step_5_print_stats(lines_count, sections)

        # ===================================== 步骤6：Chunk结果本地JSON备份 + 状态更新 ======================
        # 作用：将最终Chunk列表备份到local_dir目录的chunks.json，便于后续问题排查、数据复用
        # 输出：无返回值
        self._step_6_backup(state, sections)

        # 写入状态字典
        state["chunks"] = sections
        return state

    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        """
        【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
        :param state: 项目状态字典，需包含md_dir（备份目录）
        :param sections: 最终处理后的Chunk列表
        """

        try:
            # 拼接备份文件路径：固定文件名，便于查找
            backup_path = Path(state["md_path"]).parent / "chunks.json"
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看
            with open(backup_path, "w", encoding="utf-8") as f:
                """
                sections是Python 嵌套数据结构（List[Dict[str, str]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
                （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
                json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
                同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
                """
                json.dump(
                    sections,
                    f,
                    # 开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                    # 开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                    ensure_ascii=False,  # 保留中文，不转义为\u编码
                    indent=2  # 格式化缩进，便于阅读
                )
            self.logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
        except Exception as e:
            # 备份失败仅记录日志，不终止主流程
            self.logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """
        【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
        :param lines_count: MD原始文本总行数
        :param sections: 最终处理后的Chunk列表
        """
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
        """
        content = section.get("content")

        # 1. 长度未超限，无需切分，直接返回
        if len(content) <= self.config.max_content_length:
            return [section]

        # 2. 切割文本
        # 章节标题
        title = section.get("title")
        #计算标题的长度
        prefix = f"{title}\n\n" if title else ""

        # 计算正文的长度
        available_len = self.config.max_content_length - len(prefix)

        if available_len <= 0:
            self.logger.warning("章节标题过长，无法切分")
            return [section]


        # 去掉title部分，便于以后判断
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title) :].lstrip()

        # 对正文进行切割
        # langchain扽文本切割器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len, # 切片长度（扣除标题）
            chunk_overlap=0, #重叠部分的长度
            separators=[
                "\n\n",
                "\n",
                "。",
                "？",
                "！",
                "；",
                ".",
                "?",
                "!",
                ";",
                " ",
            ],
        )

        sub_sections = []
        chunks = splitter.split_text(body)
        for idx, chunk in enumerate(chunks, start=1):
            text = chunk.strip()
            if not text:
                continue

            full_text = (prefix + text).strip()
            new_title = f"{title}-{idx}"

            sub_sections.append(
                {
                    "parent_title": title,
                    "title": new_title,
                    "content": full_text,
                    "part": idx,
                    "file_title": section.get("file_title")
                }
            )

        return sub_sections


    def _step_4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
       【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
       执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
       :param sections: 步骤3处理后的章节列表
       :return: 长度适中、低碎片化的最终Chunk列表
       """
        # 阶段1：超长段落切分，所有美容控制在最大字符数内（self.config.max_content_length）
        refined_split = []
        for section in sections:
            sub_sections = self._split_long_section(section)
            refined_split.extend(sub_sections)

        # 阶段2：短段落合并，减少内容碎片化
        final_sections = self._merge_short_sections(refined_split)

        # 阶段3：父标题兜底
        for section in final_sections:
            if not section.get("parent_title"):
                section["parent_title"] = section.get("title") or ""
        return final_sections


    def _merge_short_sections(self, sections: List[Dict[str,str]]) -> List[Dict[str,str]]:

        #边界处理：空列表直接返回
        if not sections:
            return []


        #合并后的结果
        merged_sections = []
        # 保存当前要合并的chunk
        current_chunk = None

        for section in sections:
            #获取第一个section赋值给current_chunk
            if current_chunk is None:
                current_chunk = section
                continue

            #段落是否过短
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == section.get("parent_title")

            if is_current_short and is_same_parent:
                #合并前清理每个content前面重复的parent_title
                parent_title = section.get("parent_title")
                section_content = section.get("content")
                if parent_title and section_content.startswith(parent_title):
                    section_content = section_content[len(parent_title):].lstrip()

                #合并段落
                current_chunk["content"] += "\n\n" + section_content

                #处理part
                if "part" in section:
                    current_chunk["part"] = section["part"]

            else:
                #保存当前段落
                merged_sections.append(current_chunk)
                current_chunk = section

        if current_chunk is not None:
            merged_sections.append(current_chunk)

        return merged_sections


    def _step_3_handle_no_title(self, content: str, sections: List[Dict[str, str]], title_count: int,
                                file_title: str) -> List[Dict[str, str]]:
        """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表
        """
        if title_count == 0:
            # 无标题时调用LLM总结一个标题，失败则回退为"无标题"
            title = self._generate_title(content, file_title)
            return [{
                "title": title,
                "content": content,
                "file_title": file_title
            }]

        return sections


    def _generate_title(self, content: str, file_title: str) -> str:
        """
        调用LLM为无标题文档生成标题
        成功返回清洗后的标题；失败/返回空时回退为"无标题"
        :param content: 标准化后的MD完整内容
        :param file_title: 所属文件标题
        :return: 生成的标题字符串
        """
        try:
            chat_model = ChatOpenAI(
                model=self.config.item_model or self.config.default_model or self.config.vl_model,
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_api_base,
                temperature=self.config.temperature,
            )

            # 标题生成不需要全文，取文档开头一段作为上下文即可
            preview = (content or "")[:self.config.max_content_length]

            prompt = f"""请根据下面的文档内容，为它总结一个简短标题。

要求：
1. 标题不超过20个字符
2. 不要包含空格和换行
3. 只输出标题本身，不要加引号、编号、前缀或多余解释

文档文件名：{file_title}
文档内容（截断预览）：
{preview}"""

            response = chat_model.invoke(prompt)
            title = str(response.content).strip().strip('"').strip("'").strip()
            # 清理标题中的空白字符，并限制长度
            title = re.sub(r"\s+", "", title)[:20]

            if not title:
                raise ValueError("LLM返回了空标题")

            self.logger.info(f"LLM生成标题成功：{title}")
            return title

        except Exception as e:
            self.logger.error(f"LLM生成标题失败，回退为'无标题'：{e}")
            return "无标题"
    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """

        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        title_pattern = r'\s*#{1,6}\s+.+'

        # 2、初始化需要的数据
        lines = content.split("\n")
        sections = []  # 章节列表
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容

        # 3、定义内部函数组装sections列表
        def _flush_section():
            """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                # 每段使用 \n换行区分
                "content": "\n".join(current_lines),
                "file_title": file_title,
            })

        # 4、逐行遍历，识别标题和普通行以及代码块
        fence_char = None  # 当前代码块围栏字符：` 或 ~
        fence_len = 0  # 当前代码块围栏长度（>=3）

        for line in lines:
            striped_line = line.strip()

            # 4.1 正在代码块内：只有「同字符、长度>=开启、闭合后只有空白」的行才闭合
            if fence_char:
                if re.match(rf"^{re.escape(fence_char)}{{{fence_len},}}\s*$", striped_line):
                    fence_char, fence_len = None, 0
                current_lines.append(striped_line)
                continue

            # 4.2 不在代码块内：检测开启围栏 ``` 或 ~~~（可带语言后缀，如 ```python）
            fence_match = re.match(r"^(```+|~~~+)(.*)$", striped_line)
            if fence_match:
                fence_char = fence_match.group(1)[0]
                fence_len = len(fence_match.group(1))
                current_lines.append(striped_line)
                continue

            # 4.3 标题判断（能走到这里，必然不在代码块内，# 天然被保护）
            if re.match(title_pattern, line):
                _flush_section()
                current_title = striped_line
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(striped_line)

        _flush_section()

        return sections, title_count, len(lines)


    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题（无内容则返回None,None）
        """

        # 中间节点也可以只做轻量级的防御性判断，不做完整的磁盘级别的检查 => 前置校验 + 信任状态

        # 1、非空校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        # 2、基础标准化：统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        return md_content, file_title

if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\22857\Desktop\doc\output_md\hak180产品安全手册\hak180产品安全手册_new.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path": md_path,
        "md_content": md_content,
        "file_title": "hak180产品安全手册"
    }

    # 执行核心处理流程
    node_document_split = NodeDocumentSplit()
    result = node_document_split(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))