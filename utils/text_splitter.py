import os
import torch
from sentence_transformers import SentenceTransformer, util

class SmartTextSplitter:
    """
    智能语义文本分割器
    结合递归字符分割和语义相似度合并
    """
    def __init__(self, model_path=None, model_name=None, threshold=0.75, base_splitter_params=None, device=None):
        """
        初始化语义感知文本分割器
        
        参数:
            model_path: 本地模型路径（优先级最高）
            model_name: 模型名称，如果提供了model_path则忽略此参数
            threshold: 语义相似度阈值，高于此值则合并文本块
            base_splitter_params: 基础递归分割器的参数
            device: 运行设备，如 'cuda' 或 'cpu'
        """
        self.threshold = threshold
        
        # 初始化基础分割器（用于初次分句/分段）
        if base_splitter_params is None:
            base_splitter_params = {
                "chunk_size": 256,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", "。", "？", "！", "；", "?", "!", ";", "..."],
                "length_function": len,
                "is_separator_regex": False
            }
        
        # 注意：这里需要在方法内部导入，避免循环导入问题
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.base_splitter = RecursiveCharacterTextSplitter(**base_splitter_params)
        
        # 设备选择
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 模型加载逻辑
        if model_path is not None and os.path.exists(model_path):
            print(f"[信息] 从本地路径加载模型: {model_path}")
            self.model = SentenceTransformer(model_path, device=device)
        elif model_name is not None:
            print(f"[信息] 加载模型: {model_name}")
            try:
                self.model = SentenceTransformer(model_name, device=device)
            except Exception as e:
                print(f"[错误] 模型加载失败: {e}")
                # 如果在线下载失败，尝试使用默认模型
                print("[信息] 尝试使用默认模型")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        else:
            print("[信息] 使用默认模型: all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        print(f"[信息] 设备: {device}, 阈值: {threshold}")
    
    def split_text(self, text: str):
        """主分割方法"""
        if not text or not text.strip():
            return []
        
        # 第一步：用基础分割器得到初始块
        initial_chunks = self.base_splitter.split_text(text)
        if len(initial_chunks) <= 1:
            return initial_chunks
        
        print(f"[信息] 初始分割为 {len(initial_chunks)} 个块")
        
        # 第二步：计算所有初始块的嵌入向量
        with torch.no_grad():
            embeddings = self.model.encode(
                initial_chunks,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        # 第三步：基于语义相似度合并块
        merged_chunks = []
        current_chunk = initial_chunks[0]
        current_embedding = embeddings[0]
        
        for i in range(1, len(initial_chunks)):
            next_chunk = initial_chunks[i]
            next_embedding = embeddings[i]
            
            # 计算当前块与下一块的语义相似度
            similarity = util.cos_sim(current_embedding, next_embedding).item()
            
            if similarity >= self.threshold:
                # 语义连贯，合并
                current_chunk += " " + next_chunk
                current_embedding = next_embedding  # 近似处理
            else:
                # 语义不连贯，保存当前块，开始新的块
                merged_chunks.append(current_chunk.strip())
                current_chunk = next_chunk
                current_embedding = next_embedding
        
        # 添加最后一个块
        if current_chunk:
            merged_chunks.append(current_chunk.strip())
        
        print(f"[信息] 语义感知合并后为 {len(merged_chunks)} 个块")
        return merged_chunks