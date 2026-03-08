import sys
import os
from utils.document_loader import batch_load_documents, load_document
from typing import List, Dict, Any
import json
from pathlib import Path
from utils.text_splitter import SmartTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_document_loader():

    """测试文档加载功能"""

    print("LLM+RAG项目-文档加载测试")
    print("=" * 50)
    
    # 测试批量加载
    all_docs = batch_load_documents("./docs")
    
    if not all_docs:
        print("没有加载到任何文档，请检查docs文件夹")
        return
    
    # 显示统计信息
    print("\n文档加载统计:")
    print("-" * 30)
    
    # 按文件类型统计
    stats = {"pdf": 0, "word": 0, "txt": 0, "total_pages": 0}
    
    for doc in all_docs:
        doc_type = doc.metadata.get("type", "unknown")
        if doc_type == "pdf":
            stats["pdf"] += 1
        elif doc_type == "word":
            stats["word"] += 1
        elif doc_type == "txt":
            stats["txt"] += 1
        stats["total_pages"] += 1
    
    print(f"PDF页数: {stats['pdf']}")
    print(f"Word文档数: {stats['word']}")
    print(f"TXT文档数: {stats['txt']}")
    print(f"总文档块数: {stats['total_pages']}")
    
    # 显示前3个文档块的内容预览
    print("\n内容预览 (前3个文档块):")
    print("-" * 30)
    
    for i, doc in enumerate(all_docs[:3]):
        source_name = os.path.basename(doc.metadata.get('source', '未知'))
        page = doc.metadata.get('page', '')
        page_info = f" 第{page}页" if page else ""
        
        print(f"{i+1}. 文件: {source_name}{page_info}")
        print(f"   类型: {doc.metadata.get('type', '未知')}")
        
        # 清理换行符，显示简洁预览
        preview = doc.page_content[:100].replace('\n', ' ').replace('\r', ' ')
        if len(doc.page_content) > 100:
            preview += "..."
        print(f"   内容: {preview}")
        print()
    
    if len(all_docs) > 3:
        print(f"... 还有 {len(all_docs) - 3} 个文档块未显示")
    
    print("\n文档加载测试完成！")
    
    return all_docs

class SemanticChunkManager:
    """语义分块管理器，负责处理分块结果的内存存储"""
    
    def __init__(self):
        self.chunks = []  # 存储所有分块
        self.chunk_index = {}  # 建立索引，便于查找
        self.stats = {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": float('inf'),
            "max_chunk_size": 0,
            "chunks_by_source": {}
        }
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """添加分块到内存存储"""
        start_idx = len(self.chunks)
        
        for chunk_data in chunks:
            # 为每个分块分配唯一ID
            chunk_id = f"chunk_{len(self.chunks):06d}"
            chunk_data["chunk_id"] = chunk_id
            
            # 添加到主列表
            self.chunks.append(chunk_data)
            
            # 建立索引
            source = chunk_data["metadata"].get("source", "unknown")
            if source not in self.chunk_index:
                self.chunk_index[source] = []
            self.chunk_index[source].append(chunk_id)
            
            # 更新统计信息
            chunk_size = len(chunk_data["content"])
            self.stats["total_chunks"] += 1
            self.stats["min_chunk_size"] = min(self.stats["min_chunk_size"], chunk_size)
            self.stats["max_chunk_size"] = max(self.stats["max_chunk_size"], chunk_size)
            
            if source not in self.stats["chunks_by_source"]:
                self.stats["chunks_by_source"][source] = 0
            self.stats["chunks_by_source"][source] += 1
        
        # 更新平均大小
        total_size = sum(len(chunk["content"]) for chunk in self.chunks)
        self.stats["avg_chunk_size"] = total_size / len(self.chunks) if self.chunks else 0
        
        print(f"已添加 {len(chunks)} 个分块，当前总数: {len(self.chunks)}")
        return start_idx, len(self.chunks) - 1
    
    def get_chunks_by_source(self, source: str) -> List[Dict[str, Any]]:
        """根据源文件获取所有分块"""
        chunk_ids = self.chunk_index.get(source, [])
        return [chunk for chunk in self.chunks if chunk["chunk_id"] in chunk_ids]
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """获取所有分块"""
        return self.chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def clear(self):
        """清空所有分块"""
        self.chunks.clear()
        self.chunk_index.clear()
        self.stats = {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": float('inf'),
            "max_chunk_size": 0,
            "chunks_by_source": {}
        }
        print("已清空所有分块")

def semantic_split_documents(documents: List[Any], model_path: str = None, 
                           threshold: float = 0.72) -> List[Dict[str, Any]]:
    """
    对文档列表进行语义分块
    
    参数:
        documents: 文档列表（来自document_loader）
        model_path: 模型路径，如果为None则使用默认模型
        threshold: 语义相似度阈值
        
    返回:
        分块后的文档列表，每个元素是一个字典，包含content和metadata
    """
    if not documents:
        print("警告：没有文档需要分块")
        return []
    
    # 初始化语义分块器
    if model_path and os.path.exists(model_path):
        print(f"使用本地模型: {model_path}")
        splitter = SmartTextSplitter(model_path=model_path, threshold=threshold)
    else:
        print("使用默认模型，将尝试下载...")
        splitter = SmartTextSplitter(
            model_name="all-MiniLM-L6-v2",
            threshold=threshold,
            base_splitter_params={
                "chunk_size": 300,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", "。", "？", "！", "?", "!", ";", "..."]
            }
        )
    
    all_chunks = []
    
    for i, doc in enumerate(documents):
        print(f"正在分块文档 {i+1}/{len(documents)}: {doc.metadata.get('source', '未知')}")
        
        # 进行语义分块
        chunks = splitter.split_text(doc.page_content)
        
        # 为每个块添加元数据
        for j, chunk in enumerate(chunks):
            chunk_data = {
                "content": chunk,
                "metadata": {
                    **doc.metadata,  # 保留原始元数据
                    "chunk_index": j,
                    "total_chunks_in_doc": len(chunks),
                    "original_doc_index": i
                }
            }
            all_chunks.append(chunk_data)
        
        print(f"  → 分割为 {len(chunks)} 个语义块")
    
    return all_chunks

def test_semantic_splitter(documents: List[Any], sample_size: int = 1) -> None:
    """
    测试语义分块器，显示分块效果
    
    参数:
        documents: 文档列表
        sample_size: 测试的文档数量（从第一个开始）
    """
    if not documents:
        print("没有文档可用于测试")
        return
    
    print("\n" + "="*60)
    print("语义分块器测试")
    print("="*60)
    
    # 只测试前几个文档
    test_docs = documents[:sample_size]
    
    # 使用本地模型进行测试
    model_path = "./models/sentence-transformers/all-MiniLM-L6-v2"
    
    if not os.path.exists(model_path):
        print(f"警告：未找到本地模型 {model_path}")
        print("将尝试在线下载模型，这可能需要一些时间...")
        model_path = None
    
    # 进行语义分块
    semantic_chunks = semantic_split_documents(test_docs, model_path, threshold=0.72)
    
    # 使用传统分块作为对比
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    traditional_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "？", "！", "?", "!", ";"]
    )
    
    traditional_chunks = []
    for doc in test_docs:
        chunks = traditional_splitter.split_text(doc.page_content)
        for chunk in chunks:
            traditional_chunks.append(chunk)
    
    # 显示对比结果
    print(f"\n测试文档数: {len(test_docs)}")
    print(f"传统分块数: {len(traditional_chunks)}")
    print(f"语义分块数: {len(semantic_chunks)}")
    
    if traditional_chunks:
        reduction_percent = (len(traditional_chunks) - len(semantic_chunks)) / len(traditional_chunks) * 100
        print(f"语义分块减少块数: {reduction_percent:.1f}%")
    
    # 显示分块示例
    print("\n语义分块示例 (前3个块):")
    print("-" * 50)
    for i, chunk_data in enumerate(semantic_chunks[:3]):
        content_preview = chunk_data["content"][:150] + "..." if len(chunk_data["content"]) > 150 else chunk_data["content"]
        print(f"块 {i+1} (来自: {chunk_data['metadata'].get('source', '未知')}):")
        print(f"大小: {len(chunk_data['content'])} 字符")
        print(f"内容: {content_preview}")
        print()
    
    # 显示块大小统计
    if semantic_chunks:
        sizes = [len(chunk["content"]) for chunk in semantic_chunks]
        print("语义分块大小分布:")
        print(f"  平均大小: {sum(sizes)/len(sizes):.1f} 字符")
        print(f"  最小大小: {min(sizes)} 字符")
        print(f"  最大大小: {max(sizes)} 字符")
        
        # 显示大小分布
        size_ranges = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 500), (501, 1000), (1001, 5000)]
        print("  分布:")
        for min_sz, max_sz in size_ranges:
            count = sum(1 for size in sizes if min_sz <= size <= max_sz)
            if count > 0:
                percentage = count / len(sizes) * 100
                print(f"    {min_sz}-{max_sz} 字符: {count} 块 ({percentage:.1f}%)")
    
    print("\n语义分块测试完成！")
    return semantic_chunks

def save_chunks_to_file(chunks: List[Dict[str, Any]], filepath: str = "semantic_chunks.json"):
    """将分块结果保存到JSON文件"""
    import json
    
    # 简化数据结构以便保存
    chunks_to_save = []
    for chunk in chunks:
        chunks_to_save.append({
            "content": chunk["content"],
            "metadata": chunk["metadata"]
        })
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"分块结果已保存到 {filepath}")
    return filepath

def load_chunks_from_file(filepath: str = "semantic_chunks.json") -> List[Dict[str, Any]]:
    """从JSON文件加载分块结果"""
    import json
    
    if not os.path.exists(filepath):
        print(f"文件 {filepath} 不存在")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    print(f"从 {filepath} 加载了 {len(chunks_data)} 个分块")
    return chunks_data


if __name__ == "__main__":

    # 1. 加载文档
    print("步骤1: 加载文档")
    loaded_documents = test_document_loader()  # 文档已加载到内存，可以继续处理
    
    if not loaded_documents:
        print("没有加载到文档，程序退出")
        exit(1)
    
    # 2. 测试语义分块
    print("\n" + "="*60)
    print("步骤2: 测试语义分块器")
    print("="*60)
    
    # 创建一个分块管理器来存储分块结果
    chunk_manager = SemanticChunkManager()
    
    # 测试分块效果（只测试前2个文档）
    test_chunks = test_semantic_splitter(loaded_documents, sample_size=2)
    
    if test_chunks:
        # 将测试分块添加到管理器
        chunk_manager.add_chunks(test_chunks)
        
        # 显示管理器统计
        stats = chunk_manager.get_stats()
        print("\n分块管理器统计:")
        print(f"  总分块数: {stats['total_chunks']}")
        print(f"  平均大小: {stats['avg_chunk_size']:.1f} 字符")
        print(f"  最小大小: {stats['min_chunk_size']} 字符")
        print(f"  最大大小: {stats['max_chunk_size']} 字符")
        
        # 3. 对全部文档进行分块
        print("\n" + "="*60)
        print("步骤3: 对所有文档进行语义分块")
        print("="*60)
        
        # 清空管理器，准备存储所有分块
        chunk_manager.clear()
        
        # 进行全部分块
        model_path = "./models/sentence-transformers/all-MiniLM-L6-v2"
        if not os.path.exists(model_path):
            print(f"警告：未找到本地模型，将尝试在线下载")
            model_path = None
        
        all_semantic_chunks = semantic_split_documents(
            loaded_documents, 
            model_path=model_path,
            threshold=0.72  # 可以调整这个值来优化分块效果
        )
        
        # 添加到管理器
        start_idx, end_idx = chunk_manager.add_chunks(all_semantic_chunks)
        print(f"分块索引范围: {start_idx} - {end_idx}")
        
        # 4. 保存分块结果（可选）
        print("\n" + "="*60)
        print("步骤4: 保存分块结果")
        print("="*60)
        
        save_option = input("是否将分块结果保存到文件? (y/n): ").strip().lower()
        if save_option == 'y':
            filename = input("输入文件名（默认: semantic_chunks.json）: ").strip()
            if not filename:
                filename = "semantic_chunks.json"
            
            save_path = save_chunks_to_file(all_semantic_chunks, filename)
            print(f"分块结果已保存到: {save_path}")
        
        # 5. 展示分块管理器内容
        print("\n" + "="*60)
        print("步骤5: 分块内容概览")
        print("="*60)
        
        all_chunks = chunk_manager.get_all_chunks()
        print(f"内存中存储的分块总数: {len(all_chunks)}")
        
        # 按源文件统计
        print("\n按源文件分布:")
        for source, count in chunk_manager.stats["chunks_by_source"].items():
            source_name = os.path.basename(source)
            print(f"  {source_name}: {count} 个分块")
        
        # 预览一些分块
        preview_count = min(3, len(all_chunks))
        print(f"\n前 {preview_count} 个分块预览:")
        for i in range(preview_count):
            chunk = all_chunks[i]
            source_name = os.path.basename(chunk["metadata"].get("source", "未知"))
            preview = chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"]
            print(f"  分块 {i+1} ({source_name}): {preview}")
        
        # 6. 后续处理提示
        print("\n" + "="*60)
        print("下一步建议:")
        print("="*60)
        print("1. 分块内容已存储在 chunk_manager 对象中")
        print("2. 可通过 chunk_manager.get_all_chunks() 获取所有分块")
        print("3. 接下来可以:")
        print("   a) 将分块转换为向量（嵌入）")
        print("   b) 存储到向量数据库")
        print("   c) 用于RAG检索")
    
    print("\n语义分块处理完成！分块内容已准备就绪，可用于后续RAG流程。")