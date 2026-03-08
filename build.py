# build.py
"""
LLM+RAG 智能问答系统
使用 LangChain 构建完整的 RAG 流程
"""
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("="*60)
    print("🤖 LLM+RAG 智能问答系统 (LangChain 版本)")
    print("="*60)
    print("实现完整的 RAG 流程:")
    print("1. 📂 DocumentLoader → 2. 🔪 TextSplitter")
    print("3. 🗄️ ChromaDB → 4. 🔍 Hybrid Retriever → 5. 🤖 LLM")
    print("="*60)
    
    # 检查依赖
    try:
        import dashscope
        from langchain.retrievers import BM25Retriever, EnsembleRetriever
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install langchain chromadb sentence-transformers dashscope")
        return
    
    # 检查 docs 文件夹
    docs_folder = Path("./docs")
    if not docs_folder.exists():
        print(f"❌ docs文件夹不存在: {docs_folder}")
        print("请创建docs文件夹并放入文档")
        return
    
    # 显示文档统计
    print(f"\n📁 文档文件夹: {docs_folder}")
    doc_files = list(docs_folder.glob("*"))
    print(f"📄 发现 {len(doc_files)} 个文件/文件夹")
    
    # 询问用户要执行的操作
    print("\n请选择操作:")
    print("1. 构建完整的RAG系统 (加载→分块→存储→检索→问答)")
    print("2. 仅测试检索功能")
    print("3. 仅测试LLM问答")
    print("4. 退出")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == "1":
        run_complete_rag_system()
    elif choice == "2":
        test_retrieval_only()
    elif choice == "3":
        test_llm_only()
    elif choice == "4":
        print("👋 再见!")
    else:
        print("❌ 无效选项")

def run_complete_rag_system():
    """运行完整的RAG系统"""
    print("\n🚀 开始构建完整的RAG系统...")
    print("="*50)
    
    try:
        # 1. 加载文档
        print("\n1. 📂 加载文档...")
        documents = load_documents()
        if not documents:
            print("❌ 没有加载到文档")
            return
        
        print(f"✅ 已加载 {len(documents)} 个文档")
        
        # 2. 文本分块
        print("\n2. 🔪 文本分块...")
        split_docs = split_documents(documents)
        print(f"✅ 生成 {len(split_docs)} 个文本块")
        
        # 3. 创建向量存储
        print("\n3. 🗄️ 创建向量数据库 (ChromaDB)...")
        vectorstore = create_vector_store(split_docs)
        
        # 4. 创建混合检索器
        print("\n4. 🔍 创建混合检索器 (BM25 + 向量)...")
        retriever = create_hybrid_retriever(split_docs, vectorstore)
        
        # 5. 初始化LLM客户端
        print("\n5. 🤖 初始化LLM客户端...")
        from src.llm_api import RAGLLMClient
        llm_client = RAGLLMClient()
        
        print("\n" + "="*50)
        print("🎉 RAG系统构建完成！")
        print("="*50)
        
        # 6. 进入问答循环
        run_qa_loop(retriever, llm_client, vectorstore)
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

def load_documents():
    """加载所有文档"""
    try:
        from utils.document_loader import batch_load_documents
        return batch_load_documents("./docs")
    except Exception as e:
        print(f"❌ 文档加载失败: {e}")
        return []

def split_documents(documents):
    """文本分块"""
    try:
        # 使用你的 SmartTextSplitter
        from utils.text_splitter import SmartTextSplitter
        
        splitter = SmartTextSplitter(
            threshold=0.75,
            base_splitter_params={
                "chunk_size": 500,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", "。", "？", "！", "?", "!", ";", "..."],
                "length_function": len,
                "is_separator_regex": False
            }
        )
        
        all_chunks = []
        for doc in documents:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                # 创建新的Document对象
                from langchain.schema import Document
                new_doc = Document(
                    page_content=chunk,
                    metadata=doc.metadata.copy()
                )
                all_chunks.append(new_doc)
        
        return all_chunks
        
    except Exception as e:
        print(f"❌ 文本分块失败: {e}")
        # 回退到简单分块
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "？", "！", "?", "!", ";"]
        )
        return text_splitter.split_documents(documents)

def create_vector_store(documents):
    """创建ChromaDB向量存储"""
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma
        
        # 使用中文优化的嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 创建Chroma向量存储
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db_langchain",  # 与你的vector_store.py区分
            collection_name="rag_documents"
        )
        
        # 持久化保存
        vectorstore.persist()
        print("✅ 向量数据库已保存到: ./chroma_db_langchain")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ 创建向量存储失败: {e}")
        raise

def create_hybrid_retriever(documents, vectorstore):
    """创建混合检索器 (BM25 + 向量)"""
    try:
        from langchain.retrievers import BM25Retriever, EnsembleRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
        from langchain.retrievers import ContextualCompressionRetriever
        
        # 创建BM25检索器
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5  # 返回5个结果
        
        # 创建向量检索器
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # 创建混合检索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]  # 各占50%权重
        )
        
        print("✅ 混合检索器创建完成 (BM25 + 向量)")
        return ensemble_retriever
        
    except Exception as e:
        print(f"❌ 创建混合检索器失败: {e}")
        # 回退到向量检索器
        return vectorstore.as_retriever(search_kwargs={"k": 5})

def run_qa_loop(retriever, llm_client, vectorstore):
    """运行问答循环"""
    print("\n💬 进入问答模式")
    print("输入 'quit' 或 '退出' 结束")
    print("输入 'show' 查看检索到的文档")
    print("-" * 50)
    
    while True:
        try:
            # 获取用户输入
            question = input("\n请输入问题: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 结束问答")
                break
            
            if question.lower() == 'show':
                show_vectorstore_info(vectorstore)
                continue
            
            if question.lower() == 'help':
                show_help()
                continue
            
            # 回答问题
            start_time = time.time()
            answer = answer_question(retriever, llm_client, question, vectorstore)
            total_time = time.time() - start_time
            
            print(f"\n⏱️  总耗时: {total_time:.2f}秒")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出问答")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def answer_question(retriever, llm_client, question, vectorstore):
    """回答单个问题"""
    print("\n" + "="*50)
    print(f"💬 问题: {question}")
    print("="*50)
    
    # 1. 检索相关文档
    print("🔍 正在检索相关文档...")
    search_start = time.time()
    
    try:
        # 使用混合检索器
        retrieved_docs = retriever.get_relevant_documents(question)
        search_time = time.time() - search_start
        
        if not retrieved_docs:
            print("❌ 没有找到相关文档")
            return "没有找到相关文档信息。"
        
        print(f"✅ 找到 {len(retrieved_docs)} 个相关片段 (检索耗时: {search_time:.2f}s)")
        
        # 显示检索结果
        documents = []
        for i, doc in enumerate(retrieved_docs, 1):
            documents.append(doc.page_content)
            
            # 显示摘要
            print(f"\n📄 【片段 {i}】")
            print(f"   来源: {doc.metadata.get('source', '未知')}")
            
            # 预览内容
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"   预览: {preview}")
        
        # 2. 调用LLM生成答案
        print("\n🤖 正在生成回答...")
        
        # 调用你的RAGLLMClient
        result = llm_client.call_with_context(question, documents)
        
        if result["success"]:
            answer = result["answer"]
            llm_time = result["response_time"]
            input_tokens = result["usage"]["input_tokens"]
            output_tokens = result["usage"]["output_tokens"]
            
            print(f"✅ 生成完成 (耗时: {llm_time:.2f}s)")
            print(f"📊 Token使用: 输入 {input_tokens}, 输出 {output_tokens}")
            
            # 显示答案
            print("\n" + "="*50)
            print("💡 答案:")
            print("-" * 20)
            print(answer)
            print("-" * 20)
            
            return answer
        else:
            error_msg = result.get("error", "未知错误")
            print(f"❌ 生成失败: {error_msg}")
            
            # 显示检索到的文档作为备选
            print("\n📄 检索到的文档:")
            for i, doc in enumerate(documents, 1):
                print(f"\n【片段 {i}】")
                print(doc[:300] + "..." if len(doc) > 300 else doc)
            
            return f"生成答案失败: {error_msg}"
            
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        return f"检索文档时发生错误: {str(e)}"

def show_vectorstore_info(vectorstore):
    """显示向量数据库信息"""
    try:
        # 获取集合信息
        collection = vectorstore._collection
        count = collection.count()
        
        print(f"\n📊 向量数据库信息:")
        print(f"   文档数量: {count}")
        print(f"   存储路径: ./chroma_db_langchain")
        
        # 获取一些示例文档
        if count > 0:
            print(f"\n📄 示例文档:")
            results = collection.get(limit=3)
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                print(f"\n  【示例 {i+1}】")
                print(f"    来源: {metadata.get('source', '未知')}")
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"    内容: {preview}")
                
    except Exception as e:
        print(f"❌ 获取向量数据库信息失败: {e}")

def show_help():
    """显示帮助信息"""
    help_text = """
🤖 LLM+RAG 系统帮助
===================

可用命令:
  help      - 显示此帮助信息
  show      - 查看向量数据库信息
  quit/退出 - 退出问答模式

系统架构:
  • DocumentLoader: 从docs文件夹加载PDF、DOCX、TXT文档
  • TextSplitter: 智能语义分块
  • VectorStore: ChromaDB本地向量数据库
  • Retriever: BM25 + 向量混合检索
  • LLM: 基于Dashscope API的问答模型

技术特点:
  • 支持中文文档处理
  • 语义感知文本分块
  • 混合检索提高准确率
  • 本地向量数据库，数据持久化
  """
    print(help_text)

def test_retrieval_only():
    """仅测试检索功能"""
    print("\n🧪 测试检索功能...")
    print("="*50)
    
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma
        
        # 检查是否已有向量数据库
        if not Path("./chroma_db_langchain").exists():
            print("❌ 未找到向量数据库，请先运行完整流程")
            return
        
        # 加载向量数据库
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        vectorstore = Chroma(
            persist_directory="./chroma_db_langchain",
            embedding_function=embeddings,
            collection_name="rag_documents"
        )
        
        print("✅ 向量数据库加载成功")
        
        # 测试检索
        test_questions = [
            "什么是人工智能？",
            "机器学习有哪些应用？",
            "请介绍一下深度学习"
        ]
        
        for question in test_questions:
            print(f"\n🔍 测试检索: {question}")
            docs = vectorstore.similarity_search(question, k=3)
            
            for i, doc in enumerate(docs):
                print(f"  【结果 {i+1}】相关性: {doc.metadata.get('source', '未知')}")
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"       {preview}")
        
        print("\n🎉 检索测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_llm_only():
    """仅测试LLM问答"""
    print("\n🧪 测试LLM问答...")
    print("="*50)
    
    try:
        from src.llm_api import RAGLLMClient, simple_call_with_context
        
        # 测试1: 简单问答
        print("\n测试1: 简单问答")
        answer = simple_call_with_context(
            "什么是人工智能？",
            ["人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"]
        )
        print(f"✅ 答案: {answer[:200]}...")
        
        # 测试2: 多个上下文
        print("\n测试2: 多个上下文")
        contexts = [
            "机器学习是人工智能的一个重要分支。",
            "深度学习是机器学习的一个子领域。",
            "自然语言处理是人工智能的应用方向之一。"
        ]
        
        answer = simple_call_with_context("请介绍人工智能的相关技术", contexts)
        print(f"✅ 答案: {answer[:200]}...")
        
        print("\n🎉 LLM测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()