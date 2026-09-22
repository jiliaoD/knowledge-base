# processor/import_processor/nodes/node_pdf_to_md.py
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests


from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState


# MinerU 相关域名（API / 上传 OSS / 结果 CDN）统一走直连：
# 实测本地代理在长连接和大文件上会重置连接（走代理必失败，直连正常）
MINERU_PROXIES = {"http": None, "https": None}


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        # 1.参数检验返回Path结果
        pdf_path_obj, output_dir_obj = self._step1_validate_paths(state)
        # 2.将PDF上传到mineru并且轮询结果最后得到解压文件路径
        zip_url = self._step2_upload_and_poll(pdf_path_obj)
        self.logger.info(zip_url)
        # 步骤3：下载ZIP包并提取MD文件
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)
        # 4读取md内容
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        # 5更新state
        state.update({"md_content": md_content})
        state.update({"md_path": md_path})
        return state

    def _step1_validate_paths(self, state: ImportGraphState):
        """
               步骤1：校验PDF文件路径和输出目录
               核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
               返回：合法的PDF文件Path对象、输出目录Path对象
               异常：StateFieldError(参数缺失)、FileNotFoundError(文件无效)
        """
        # 1文件非空检验
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", message="PDF文件路径不能为空", expected_type=str)
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", message="输出目录不能为空", expected_type=str)
        # 2转化为Path对象
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 3pdf是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"PDF文件不存在：{pdf_path_obj.name}")
        # 4输出不存在则创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录不存在，正在创建：{file_dir_obj.name}")
            file_dir_obj.mkdir(parents=True, exist_ok=True)
        return pdf_path_obj, file_dir_obj

    def _step2_upload_and_poll(self, pdf_path_obj: Path):
        """
              步骤2：上传PDF至MinerU并轮询解析任务状态
              核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
              参数：pdf_path_obj-已校验的PDF Path对象
              返回：解析结果ZIP包下载链接full_zip_url
              异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
          """
        # 1配置文件检验
        if not self.config.mineru_base_url:
            raise ConfigurationError("MinerU基础URL不能为空, 请检查配置")
        if not self.config.mineru_api_token:
            raise ConfigurationError("MinerU API令牌不能为空, 请检查配置")
        # 2调用MinerU远程接口,获取上传链接
        # 2.1组织数据
        token = self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        # 2.2 发送请求
        # 网络抖动时重试（直连）
        response = None
        for attempt in range(1, 4):
            try:
                response = requests.post(url, headers=header, json=data, timeout=60, proxies=MINERU_PROXIES)
                break
            except Exception as e:
                self.logger.warning(f"获取上传链接失败（第 {attempt}/3 次）：{str(e)[:120]}")
                if attempt == 3:
                    raise PdfConversionError(f"获取上传链接失败（已重试 3 次）：{str(e)[:200]}")
                time.sleep(3)

        # 2.3获取响应结果
        if response.status_code != 200:
            raise PdfConversionError(
                "获取上传链接失败,响应状态:{}, 响应内容:{}".format(response.status_code, response.text))

        result = response.json()
        if result['code'] != 0:
            raise PdfConversionError("获取上传链接失败:返回数据:{}".format(result['msg']))
        # 2.4 获取到上传链接
        # 批量提取任务 id，可用于批量查询解析结果
        batch_id = result["data"]["batch_id"]
        # 文件上传链接
        urls = result["data"]["file_urls"]
        # 3 文件上传（自动重试；大文件优先直连，避免本地代理重置大文件连接）
        upload_ok = False
        last_err = ""
        for attempt in range(1, 4):
            try:
                with open(pdf_path_obj, 'rb') as f:
                    res_upload = requests.put(urls[0], data=f, timeout=180, proxies=MINERU_PROXIES)
                if res_upload.status_code == 200:
                    upload_ok = True
                    self.logger.info(f"{urls[0]}文件上传成功（第 {attempt} 次尝试，直连）")
                    break
                last_err = f"状态码 {res_upload.status_code}"
            except Exception as e:
                last_err = str(e)
            self.logger.warning(
                f"文件上传失败（第 {attempt}/3 次，直连），{3 - attempt} 次重试机会，原因：{last_err}"
            )
            if attempt < 3:
                time.sleep(3)

        if not upload_ok:
            raise PdfConversionError(f"{urls[0]}文件上传失败（已重试 3 次）：{last_err}")

        # 4获取结果
        poll_url= f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"

        # 轮询获取
        start_time = time.time()
        timeout_seconds=600
        poll_interval=3
        self.log_step(
            step_name="轮询开始",
            message=f"轮询间隔: {poll_interval}s, 超时时间: {timeout_seconds}s，batch_id:{batch_id}"
        )
        while True:
            # 已消耗时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"[任务轮询]超时，已消耗时间: {int(elapsed_time)}s，batch_id:{batch_id}")

            # 获取任务结果
            try:
                poll_res = requests.get(poll_url, headers=header, timeout=10, proxies=MINERU_PROXIES)
            except Exception as e:
                self.logger.warning(f"网络请求异常，{poll_interval}s后重试，batch_id:{batch_id}")
                self.logger.warning(f"异常信息：{str(e)}")
                time.sleep(poll_interval)
                continue

            if poll_res.status_code != 200:
                raise PdfConversionError(f'[任务轮询]失败，状态码：{poll_res.status_code}')

            poll_res_json = poll_res.json()
            if poll_res_json["code"] != 0:
                raise PdfConversionError(f'[任务轮询]失败，错误码：{poll_res_json["code"]}')
            extract_results = poll_res_json["data"]["extract_result"]
            extract_result = extract_results[0]
            # 获取任务的状态值
            data_state = extract_result["state"]
            if data_state == "done":
                self.log_step(
                    step_name="任务轮询",
                    message=f"解析完成s, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                full_zip_url = extract_result["full_zip_url"]
                self.log_step(
                    step_name="任务轮询",
                    message=f"获取全量zip地址成功: {full_zip_url}, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                return full_zip_url

            elif data_state == "failed":
                raise PdfConversionError(f'[任务轮询]失败，错误信息: {extract_result["err_msg"]}')

            else:
                self.log_step(
                    step_name="任务轮询",
                    message=f"处理中...... 已耗时: {int(elapsed_time)}s，状态：{data_state}，batch_id:{batch_id}"
                )
                time.sleep(poll_interval)
    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str):
        """
             步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
             核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
             参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
             返回：最终MD文件的字符串格式绝对路径
             异常：RuntimeError(下载失败)
             """
        # 1下载zip
        self.logger.info(f"开始下载ZIP包: {zip_url}")
        # 下载结果 ZIP（直连 + 重试）
        response = None
        for attempt in range(1, 4):
            try:
                response = requests.get(zip_url, timeout=300, proxies=MINERU_PROXIES)
                break
            except Exception as e:
                self.logger.warning(f"下载ZIP包失败（第 {attempt}/3 次）：{str(e)[:120]}")
                if attempt == 3:
                    raise PdfConversionError(f"下载ZIP包失败（已重试 3 次）：{str(e)[:200]}")
                time.sleep(3)

        # 检验响应结果
        if response.status_code != 200:
            raise RuntimeError(f"下载ZIP包失败，状态码: {response.status_code}, 响应内容: {response}")

        # 拼接zip包保存路径并保存
        zip_save_path=output_dir_obj/f"{pdf_stem}_result.zip"
        with open(zip_save_path, 'wb') as f:
            f.write(response.content)
        self.log_step("下载ZIP包完成", f"ZIP包保存路径: {zip_save_path}")

        # 2解压zip
        #解压目录
        extract_target_dir = output_dir_obj / pdf_stem

        # 删除已有目录
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)

        # 创建新目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 解压ZIP
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)

        self.log_step("ZIP下载", "解压完成")

        target_md_file = extract_target_dir / "full.md"
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)

        return str(new_md_path.absolute())



if __name__ == "__main__":
    # 激活日志
    setup_logging()

    init_state = {
        "pdf_path": r"D:\22857\Desktop\doc\hak180产品安全手册.pdf",
        "file_dir": r"D:\22857\Desktop\doc\output_md"
    }
    node_pdf_to_md = NodePDFToMD()
    # 使用 对象() 的方式相当于调用了 对象的__call__()
    result = node_pdf_to_md(init_state)

    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logging.getLogger().info(json_state)
