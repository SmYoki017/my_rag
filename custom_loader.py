import base64
import json
import os
import re

from langchain_community.document_loaders import TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger
from logger import setup_logger
from mineru.cli.common import do_parse, read_fn
from models import get_lc_model_client

os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'
os.environ['MODELSCOPE_CACHE'] = './models'
setup_logger()

class MyCustomLoader(BaseLoader):
    """
    两阶段图片处理 Loader：
      - 检索阶段：VLM 生成图片文字描述，内联到文本中用于语义检索
      - 生成阶段：原始图片路径保存在 metadata["image_paths"]，
                  下游可将原图传给多模态模型生成高质量回答

    TXT/CSV → TextLoader
    PDF/DOCX/MD → MinerU 解析 → _content_list.json 结构化处理
    """
    TXT_EXTENSIONS = {'.txt','.csv'}
    MINERU_OUTPUT_DIR = './chroma/_mineru_cache'

    def __init__(self,file_path:str):
        self.file_path = file_path
        self.file_ext = os.path.splitext(file_path)[-1].lower()
        self._content_list = None
        self._image_dir = None
        self._processed_text = None
        self._image_registry:dict[str,str] = {}

        if self.file_ext in self.TXT_EXTENSIONS:
            self._use_mineru = False
            self.loader = self._create_txt_loader(file_path)
        else:
            self._use_mineru = True
            self.loader = None

    def _create_txt_loader(self,file_path:str) -> BaseLoader:
        try:
            with open(file_path,'r',encoding='utf-8') as f:
                f.read()
            return TextLoader(file_path,encoding='utf-8')
        except UnicodeDecodeError:
            try:
                import chardet
                # rb 二进制读取文件
                with open(file_path,'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result.get('encoding','utf-8')
                logger.info(f"检测到文件编码: {encoding}, 路径: {file_path}")
                return TextLoader(file_path,encoding=encoding)
            except Exception as e:
                logger.info(f"编码检测失败: {e}，使用GBK编码尝试")
                return TextLoader(file_path, encoding='gbk', errors='ignore')

    def _parse_with_mineru(self) -> list[dict]:
        """ MinerU解析 """
        if self._content_list is not None:
            return self._content_list

        # output_dir E:\PycharmProjects\myRag\chroma\_mineru_cache
        output_dir = os.path.abspath(self.MINERU_OUTPUT_DIR)
        # file_stem 2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告
        file_stem = os.path.splitext(os.path.basename(self.file_path))[0]

        if self.file_ext == '.md':
            with open(self.file_path,'r',encoding='utf-8') as f:
                md_text = f.read()
            self._image_dir = os.path.dirname(os.path.abspath(self.file_path))
            self._content_list = [{"type": "text", "text": md_text}]
            return self._content_list
        os.makedirs(output_dir,exist_ok=True)
        json_path = os.path.join(output_dir,file_stem,"auto",f"{file_stem}_content_list.json")
        image_dir = os.path.join(output_dir,file_stem,"auto","images")

        if os.path.exists(json_path):
            logger.info(f"[MinerU] 使用缓存: {json_path}")
            with open(json_path,'r',encoding='utf-8') as f:
                self._content_list = json.load(f)
            self._image_dir = image_dir
            logger.info(f"[MinerU] 解析完毕: {json_path}")
            return self._content_list

        with open(self.file_path,'rb') as f:
            pdf_bytes = f.read()

        logger.info(f"[MinerU] 正在解析:{self.file_path}")
        do_parse(
            output_dir = output_dir,
            pdf_file_names=[file_stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend='pipeline'
        )
        with open(json_path,'r',encoding='utf-8') as f:
            self._content_list = json.load(f)
        self._image_dir = image_dir
        logger.info(f"[MinerU] 解析完毕: {json_path}")
        return self._content_list

        # ------ VLM 图片描述 ------
    def _describe_image(self,image_path:str) -> str:
        if not os.path.exists(image_path):
            logger.warning(f"图片不存在: {image_path}")
            return f"图片 {os.path.basename(image_path)} 文件不存在"
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        ext = os.path.splitext(image_path)[-1].lower().strip('.')
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"

        prompt = "请详细描述这张图片的内容，重点提取关键信息和专业术语，用于知识库检索。"
        client = get_lc_model_client(
            model='gpt-4o-mini',
            api_key=os.getenv('YUNWU_KEY'),
            base_url=os.getenv('YUNWU_BASE_URL')
        )
        message = HumanMessage(content=[
            {"type":"image_url","image_url":{"url":f"data:{mime};base64,{image_data}"}},
            {"type":"text","text":prompt}
        ])
        response = client.invoke([message])
        return response.content

    # ----- HTML 表格转文本 -----

    @staticmethod
    def _html_table_to_text(html: str) -> str:
        if not html:
            return ""
        rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
        text_rows = []
        for row in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            cells = [c.strip() for c in cells]
            text_rows.append(" | ".join(cells))
        return "\n".join(text_rows)

    # ----content_list -> 统一文本(核心方法) -----
    def _content_list_to_text(self,content_list:list[dict]) -> str:
        """
        将 content_list 转为统一文本，同时构建 _image_registry。
        图片描述内联到文本中用于检索，原图路径记录在 registry 中用于生成阶段。
        """
        if self._processed_text is not None:
            return self._processed_text

        image_dir = self._image_dir or os.path.dirname(os.path.abspath(self.file_path))
        parts = []

        for block in content_list:
            block_type = block.get("type","")

            if block_type == 'discarded':
                continue
            elif block_type == 'text':
                text = block.get("text","").strip()
                if not text:
                    continue
                level = block.get("text_level")
                if level and isinstance(level, int) and 1 <= level <= 3:
                    text = "#" * level + " " + text
                parts.append(text)
            elif block_type == 'image':
                img_path = block.get("img_path","")
                abs_img_path = os.path.normpath(os.path.join(image_dir,img_path))
                desc = self._describe_image(abs_img_path)
                captions = block.get("image_caption") or []
                caption = " ".join(captions).strip()
                if caption:
                    desc = f"{caption}：{desc}"
                self._image_registry[desc] = abs_img_path
                parts.append(f"[图片描述: {desc}]")

            elif block_type == 'table':
                table_body = block.get("table_body","")
                table_text = self._html_table_to_text(table_body)
                captions = block.get("table_caption") or []
                caption = " ".join(captions).strip()
                if caption:
                    table_text = f"{caption}\n{table_text}"
                if table_text:
                    parts.append(table_text)

        result = "\n\n".join(parts)

        if re.search(r'!\[.*?\]\(.*?\)', result):
            result = self._replace_inline_images(result,image_dir)

        self._processed_text = result
        return result

    def _replace_inline_images(self, text: str, image_dir: str) -> str:
        def replacer(match):
            img_path = match.group(2)
            abs_path = os.path.normpath(os.path.join(image_dir, img_path))
            desc = self._describe_image(abs_path)
            self._image_registry[desc] = abs_path
            return f"[图片描述: {desc}]"
        return re.sub(r'!\[(.*?)\]\((.*?)\)', replacer, text)

    def _attach_image_paths_to_chunks(self,chunks:list[Document]) -> list[Document]:
        """
        扫描每个chunk中的[图片描述: ...] 标记
        从_image_registry 查找原图路径，写入 metadata["image_paths"]
        """
        for chunk in chunks:
            image_paths = []
            descs = re.findall(r'\[图片描述: (.*?)\]', chunk.page_content)
            for desc in descs:
                img_path = self._image_registry.get(desc)
                if img_path:
                    image_paths.append(img_path)
            if image_paths:
                chunk.metadata["image_paths"] = "|".join(image_paths)

        return chunks

    def get_image_registry(self) -> dict[str, str]:
        """返回图片注册表：{描述文本: 原图绝对路径}"""
        return self._image_registry.copy()

    def _structured_split(self, content_list: list[dict],
                          chunk_size=None, chunk_overlap=None) -> list[Document]:
        if chunk_size is None:
            chunk_size = 500
        if chunk_overlap is None:
            chunk_overlap = int(chunk_size * 0.2)

        full_text = self._content_list_to_text(content_list)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=True
        )
        header_sections = header_splitter.split_text(full_text)
        for section in header_sections:
            section.metadata["source"] = self.file_path

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_documents(header_sections)
        return self._attach_image_paths_to_chunks(chunks)

    def _make_splitter(self, chunk_size=None, chunk_overlap=None):
        if chunk_size is None:
            chunk_size = 500
        if chunk_overlap is None:
            chunk_overlap = int(chunk_size * 0.2)
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def lazy_load(self):
        if self._use_mineru:
            content_list = self._parse_with_mineru()
            full_text = self._content_list_to_text(content_list)
            doc = Document(
                page_content=full_text,
                metadata={"source": self.file_path}
            )
            all_image_paths = list(self._image_registry.values())
            if all_image_paths:
                doc.metadata["image_paths"] = "|".join(all_image_paths)
            yield doc
        else:
            yield from self.loader.lazy_load()

    def load(self):
        return list(self.lazy_load())

    def load_and_split(self, chunk_size=None, chunk_overlap=None):
        if self._use_mineru:
            content_list = self._parse_with_mineru()
            return self._structured_split(content_list, chunk_size, chunk_overlap)
        else:
            documents = self.loader.load()
            splitter = self._make_splitter(chunk_size, chunk_overlap)
            return splitter.split_documents(documents)





if  __name__ == '__main__':
    # file_path = 'myFiles/2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告.pdf'
    # print(os.path.splitext(file_path)[-1].lower())
    print(os.path.abspath('./chroma/_mineru_cache'))
    print(os.path.dirname(os.path.abspath('./chroma/_mineru_cache/2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告/auto/images')))
