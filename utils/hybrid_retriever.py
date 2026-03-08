from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict, Any
import numpy as np
from .vector_store import VectorStore

class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        """
        初始化混合检索器
        :param vector_store: 向量存储实例
        """
        self.vector_store = vector_store
        self.bm25_index = None
        self.documents = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """从向量数据库构建BM25索引"""
        # 获取所有文档
        all_docs = self.vector_store.collection.get()
        if not all_docs['documents']:
            return

        self.documents = all_docs['documents']

        # 中文需要分词，这里简单按字符分割。如需更准，可集成jieba
        tokenized_docs = [list(doc) for doc in self.documents]  # 按字符分
        # 如需更好效果：import jieba; [list(jieba.cut(doc)) for doc in self.documents]

        self.bm25_index = BM25Okapi(tokenized_docs)
        print(f"✅ BM25索引构建完成，已索引 {len(self.documents)} 个文档")

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        混合检索
        :param query: 查询文本
        :param top_k: 返回结果数量
        :param alpha: 权重系数（0=仅BM25, 1=仅向量）
        :return: 检索结果列表
        """
        if not self.documents:
            return []

        # 1. 向量搜索
        vector_results = self.vector_store.search(query, n_results=top_k*2)

        # 2. BM25搜索
        tokenized_query = list(query)  # 简单按字符分
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # 获取BM25 top结果
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k*2]

        # 3. 结果融合
        all_results = {}

        # 处理向量搜索结果
        if vector_results['documents']:
            for i, (doc, metadata, distance) in enumerate(zip(
                vector_results['documents'][0],
                vector_results['metadatas'][0],
                vector_results['distances'][0]
            )):
                # 从距离转换为相似度分数 (1 - normalized_distance)
                vector_score = 1.0 / (1.0 + distance)  # 简单转换
                doc_id = f"vec_{i}"

                all_results[doc_id] = {
                    'document': doc,
                    'metadata': metadata or {},
                    'vector_score': vector_score,
                    'bm25_score': 0.0,  # 稍后填充
                    'combined_score': 0.0
                }

        # 处理BM25结果并合并
        for idx in bm25_top_indices:
            bm25_score = bm25_scores[idx]
            # 归一化BM25分数
            normalized_bm25 = bm25_score / (bm25_score + 1) if bm25_score > 0 else 0

            doc = self.documents[idx]
            doc_id = f"bm25_{idx}"

            if doc_id in all_results:
                # 如果已存在（来自向量结果），更新BM25分数
                all_results[doc_id]['bm25_score'] = normalized_bm25
            else:
                # 否则添加新条目
                all_results[doc_id] = {
                    'document': doc,
                    'metadata': {},  # BM25不直接提供metadata
                    'vector_score': 0.0,
                    'bm25_score': normalized_bm25,
                    'combined_score': 0.0
                }

        # 计算综合分数
        for result in all_results.values():
            result['combined_score'] = (
                alpha * result['vector_score'] + 
                (1 - alpha) * result['bm25_score']
            )

        # 按综合分数排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]

        return sorted_results

# 便捷函数
def get_hybrid_retriever(persist_dir: str = "./chroma_db"):
    """获取混合检索器实例"""
    vector_store = VectorStore(persist_dir)
    retriever = HybridRetriever(vector_store)
    return retriever