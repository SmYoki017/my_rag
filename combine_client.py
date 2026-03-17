import base64
import hashlib
import os
import shutil
import uuid
from typing import Any, Optional, Literal

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever, ParentDocumentRetriever, \
    ContextualCompressionRetriever, RePhraseQueryRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_classic.storage import create_kv_docstore
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.storage import RedisStore

from custom_loader import MyCustomLoader
from models import get_lc_model_client, get_ali_embeddings, get_ali_rerank
from loguru import logger

KNOWLEDGE_DIR = './chroma/knowledge/'
CHROMA_DIR = './chroma'
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
embedding_model = get_ali_embeddings()

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "请对以下文档内容进行简洁摘要，突出核心主题和关键信息，用于知识库语义检索：{doc}"
)
HYPOTHETICAL_QUESTIONS_PROMPT = ChatPromptTemplate.from_template(
    "请根据以下文档内容，生成 2 个用户可能会提问的问题。\n"
    "要求：\n"
    "1. 每个问题单独一行\n"
    "2. 问题应能通过该文档内容直接回答\n"
    "3. 只输出问题本身，不要编号、不要解释\n\n"
    "文档内容：\n{doc}"
)


def get_redis_docstore(namespace: str):
    redis_store = RedisStore(redis_url=REDIS_URL, namespace=namespace)
    return create_kv_docstore(redis_store)


def get_redis_byte_store(namespace: str):
    return RedisStore(redis_url=REDIS_URL, namespace=namespace)


def _encode_image_to_base64(image_path: str) -> Optional[str]:
    """读取图片文件并返回 base64 编码字符串"""
    if not os.path.exists(image_path):
        logger.warning(f"图片不存在，跳过: {image_path}")
        return None
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


class MyKnowledge:
    """知识库管理模块（两阶段图片处理）"""

    __embeddings = embedding_model
    logger.info(f"当前嵌入模型: {__embeddings.model}")

    __retrievers = {}
    __llm = get_lc_model_client(
        model='gpt-4o-mini',
        api_key=os.getenv('YUNWU_KEY'),
        base_url=os.getenv('YUNWU_BASE_URL')
    )

    @staticmethod
    def upload_knowledge(temp_file):
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        file_name = os.path.basename(temp_file)
        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy(temp_file, file_path)

    def load_knowledge(
            self,
            index_type: Literal['normal', 'summary', 'parent_child'] = 'parent_child'
    ):
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        collections = [None]
        logger.info(f"当前知识库列表: {os.listdir(KNOWLEDGE_DIR)}")

        for file in os.listdir(KNOWLEDGE_DIR):
            collections.append(file)
            file_path = os.path.join(KNOWLEDGE_DIR, file)
            logger.info(f"正在加载文件: {file_path}")
            collection_name = get_md5(file)
            retriever_key = f"{collection_name}_{index_type}"

            if retriever_key in self.__retrievers:
                logger.info(f"检索器已存在，跳过：{retriever_key}")
                continue

            loader = MyCustomLoader(file_path)
            if index_type == 'summary':
                retriever = create_summary_index(
                    collection_name=collection_name,
                    loader=loader,
                    embeddings=self.__embeddings,
                    llm=self.__llm
                )
            elif index_type == 'parent_child':
                retriever = create_parent_child_index(
                    collection_name=collection_name,
                    loader=loader,
                    embeddings=self.__embeddings,
                )
            else:
                retriever = create_normal_index(
                    collection_name=collection_name,
                    loader=loader,
                    embeddings=self.__embeddings
                )
            logger.info(f"检索器创建成功: {retriever_key}")

            # parent_child 模式下 bm25 已在 create_parent_child_index 中预建，直接跳过
            if 'bm25' not in retriever:
                bm25_retriever = BM25Retriever.from_documents(retriever['chunks'], k=5)
                retriever['bm25'] = bm25_retriever
            self.__retrievers[retriever_key] = retriever

        return collections

    def get_retriever(
            self,
            file_name: str,
            index_type: Literal['normal', 'summary', 'parent_child'] = 'parent_child'
    ):
        collection_name = get_md5(file_name)
        retriever_key = f"{collection_name}_{index_type}"
        retriever_data = self.__retrievers.get(retriever_key)
        if retriever_data is None:
            raise ValueError(f"未找到检索器: {retriever_key}，请先调用 load_knowledge()")

        vector_retriever = retriever_data['vector']
        bm25_retriever = retriever_data['bm25']

        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

        rerank_retriever = get_ali_rerank(top_n=3)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=rerank_retriever, base_retriever=hybrid_retriever
        )
        logger.debug(f"最终检索器为: {final_retriever}")
        return final_retriever

    def generate_answer(self, query: str, retrieved_docs: list[Document]) -> str:
        """
        两阶段生成：
          1. 从检索结果中收集文本上下文
          2. 检查 metadata["image_paths"]，若有原图则构建多模态消息
          3. 将文本+原图一起发送给多模态 LLM，生成高质量回答
        """
        context_parts = []
        all_image_paths = []

        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[文档片段 {i + 1}]\n{doc.page_content}")
            raw = doc.metadata.get("image_paths", "")
            image_paths = raw.split("|") if isinstance(raw, str) and raw else []
            for p in image_paths:
                if p not in all_image_paths:
                    all_image_paths.append(p)

        context_text = "\n\n".join(context_parts)

        if not all_image_paths:
            prompt = (
                f"请根据以下参考资料回答用户问题。如果资料中没有相关信息，请如实说明。\n\n"
                f"参考资料：\n{context_text}\n\n"
                f"用户问题：{query}"
            )
            response = self.__llm.invoke([HumanMessage(content=prompt)])
            return response.content

        # 有图片 → 构建多模态消息
        logger.info(f"[多模态生成] 检测到 {len(all_image_paths)} 张相关原图，将传递给 VLM")
        content_blocks = [
            {"type": "text", "text": (
                f"请根据以下参考资料和图片回答用户问题。"
                f"图片是从文档中提取的原始图片，请结合图片内容给出准确回答。"
                f"如果资料中没有相关信息，请如实说明。\n\n"
                f"参考资料：\n{context_text}\n\n"
                f"用户问题：{query}"
            )}
        ]

        for img_path in all_image_paths:
            img_b64 = _encode_image_to_base64(img_path)
            if img_b64 is None:
                continue
            ext = os.path.splitext(img_path)[-1].lower().strip('.')
            mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"}
            })

        response = self.__llm.invoke([HumanMessage(content=content_blocks)])
        return response.content

    def retrieve_and_answer(self, query: str, file_name: str,
                            index_type: Literal['normal', 'summary', 'parent_child'] = 'parent_child') -> dict:
        """
        一站式接口：检索 + 多模态生成
        返回 {"answer": str, "source_docs": list[Document], "image_count": int}
        """
        retriever = self.get_retriever(file_name, index_type)
        docs = retriever.invoke(query)
        logger.info(f"[检索] 命中 {len(docs)} 个文档片段")

        image_count = sum(
            len(doc.metadata.get("image_paths", "").split("|")) if doc.metadata.get("image_paths") else 0 for doc in docs
        )
        if image_count > 0:
            logger.info(f"[检索] 其中包含 {image_count} 张原图引用，将使用多模态生成")

        answer = self.generate_answer(query, docs)
        return {
            "answer": answer,
            "source_docs": docs,
            "image_count": image_count
        }


def _get_child_docs_from_chroma(vector_store: Chroma) -> list[Document]:
    """从 Chroma 向量库中提取所有已存储的子块文档（含 doc_id metadata）"""
    results = vector_store._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(results["documents"], results["metadatas"])
        if content
    ]


class BM25ToParentRetriever(BaseRetriever):
    """
    BM25 在子块上做关键词检索，通过 metadata 中的 doc_id 映射回父块，
    保证与 ParentDocumentRetriever 返回的文档粒度一致。
    """
    bm25_retriever: BM25Retriever
    docstore: Any
    id_key: str = "doc_id"

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        child_docs = self.bm25_retriever.invoke(query)
        seen_ids: list[str] = []
        for doc in child_docs:
            pid = doc.metadata.get(self.id_key)
            if pid and pid not in seen_ids:
                seen_ids.append(pid)
        if not seen_ids:
            return []
        results = self.docstore.mget(seen_ids)
        return [doc for doc in results if doc is not None]


#
#  索引创建函数
#

def create_normal_index(
        collection_name: str,
        loader: BaseLoader,
        embeddings: Optional[Embeddings] = None
):
    docs = loader.load_and_split()
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.path.join(CHROMA_DIR, collection_name)
    )
    if db._collection.count() == 0:
        db.add_documents(docs)
        logger.info(f"[普通索引] 已写入 {len(docs)} 个 chunk，collection: {collection_name}")
    else:
        logger.info(f"[普通索引] 使用已有索引，跳过写入，collection: {collection_name}")
    return {
        'vector': db.as_retriever(search_kwargs={"k": 5}),
        'chunks': docs
    }


def create_summary_index(
        collection_name: str,
        loader: MyCustomLoader,
        embeddings: Optional[Embeddings] = None,
        llm=None
):
    docs = loader.load_and_split(chunk_size=1500, chunk_overlap=150)
    logger.info(f"[摘要索引] 加载到 {len(docs)} 个文档块，开始生成摘要...")
    vector_store = Chroma(
        collection_name=f"{collection_name}_summary",
        embedding_function=embeddings,
        persist_directory=os.path.join(CHROMA_DIR, f"{collection_name}_summary")
    )
    byte_store = get_redis_byte_store(namespace=f"{collection_name}_summary")
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vector_store,
        byte_store=byte_store,
        id_key=id_key,
        search_kwargs={"k": 5},
    )
    if vector_store._collection.count() == 0:
        chain = (
            {"doc": lambda x: x.page_content}
            | SUMMARY_PROMPT
            | llm
            | StrOutputParser()
        )
        summaries = chain.batch(docs, {"max_concurrency": 5})
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))
        logger.info(f"[摘要索引] 摘要生成并写入完毕，共 {len(docs)} 条，collection: {collection_name}_summary")
    else:
        logger.info(f"[摘要索引] 使用已有索引，跳过写入，collection: {collection_name}_summary")
    return {
        'vector': retriever,
        'chunks': docs
    }


def create_parent_child_index(
        collection_name: str,
        loader: MyCustomLoader,
        embeddings: Optional[Embeddings] = None,
):
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", " ", ""],
        chunk_size=1000,
        chunk_overlap=100
    )
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", " ", ""],
        chunk_size=200,
        chunk_overlap=20
    )
    vector_store = Chroma(
        collection_name=f"{collection_name}_parent_child",
        embedding_function=embeddings,
        persist_directory=os.path.join(CHROMA_DIR, f"{collection_name}_parent_child")
    )
    docstore = get_redis_docstore(namespace=f"{collection_name}_parent_child")
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 5},
    )

    if vector_store._collection.count() == 0:
        raw_docs = loader.load()
        retriever.add_documents(raw_docs)
        logger.info(f"[父子索引] 原始文档 {len(raw_docs)} 篇已建索引，collection: {collection_name}_parent_child")
    else:
        logger.info(f"[父子索引] 使用已有索引，跳过写入，collection: {collection_name}_parent_child")

    # 从 Chroma 中提取已存储的子块（含 doc_id metadata）
    # BM25 在子块上精确匹配关键词，命中后通过 doc_id 查 Redis 映射回父块，保证粒度一致
    child_docs = _get_child_docs_from_chroma(vector_store)
    bm25_base = BM25Retriever.from_documents(child_docs, k=10)
    bm25_parent_retriever = BM25ToParentRetriever(
        bm25_retriever=bm25_base,
        docstore=docstore,
        id_key="doc_id",
    )
    logger.info(f"[父子索引] BM25 从 {len(child_docs)} 个子块构建，映射回父块")
    return {
        'vector': retriever,
        'chunks': child_docs,
        'bm25': bm25_parent_retriever,
    }


def get_md5(input_string):
    hash_md5 = hashlib.md5()
    hash_md5.update(input_string.encode('utf-8'))
    return hash_md5.hexdigest()


if __name__ == '__main__':
    MyKnowledge.upload_knowledge('myFiles/2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告.pdf')
    kb = MyKnowledge()
    kb.load_knowledge('parent_child')

    result = kb.retrieve_and_answer(
        query='公司资产和负债占比多少',
        file_name='2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告.pdf',
        index_type='parent_child'
    )
    print(result)
    # print(f"回答: {result['answer']}")
    # print(f"引用文档数: {len(result['source_docs'])}")
    # print(f"涉及图片数: {result['image_count']}")