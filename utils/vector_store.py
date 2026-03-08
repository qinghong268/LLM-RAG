import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量数据库
        :param persist_dir: 数据库持久化目录
        :param model_name: 句子嵌入模型名
        """
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(model_name)
        print(f"✅ 加载嵌入模型: {model_name}")

        # 初始化Chroma客户端，设置持久化路径
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False) # 关闭遥测
        )

        # 获取或创建集合（类似于数据库的表）
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"description": "存储RAG项目文档块"}
        )
        print(f"✅ 向量数据库初始化完成，位置: {persist_dir}")

    def add_documents(self, documents: List[Document]):
        """
        将文档列表添加到向量数据库
        """
        print(f"📥 正在将 {len(documents)} 个文档块存入向量数据库...")

        # 准备数据
        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            # 生成唯一ID
            doc_id = f"doc_{i}_{hash(doc.page_content) & 0xffffffff}"
            ids.append(doc_id)
            texts.append(doc.page_content)
            # 将LangChain的metadata转为ChromaDB兼容的格式
            clean_metadata = {k: str(v) for k, v in doc.metadata.items()}
            metadatas.append(clean_metadata)

        # 生成向量嵌入（批量处理，提高效率）
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()

        # 添加到集合
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"🎉 成功入库 {len(documents)} 个文档块！")

    def search(self, query: str, n_results: int = 5):
        """
        在向量数据库中搜索相关文档
        """
        # 将查询文本转为向量
        query_embedding = self.embedding_model.encode([query]).tolist()

        # 执行搜索
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results

# 便捷函数：一键创建向量库
def create_vector_store(documents: List[Document], persist_dir: str = "./chroma_db"):
    """创建并填充向量数据库"""
    vector_store = VectorStore(persist_dir)
    vector_store.add_documents(documents)
    return vector_store