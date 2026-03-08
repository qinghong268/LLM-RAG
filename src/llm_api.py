# src/llm_api.py
"""
RAG系统专用LLM API调用模块
适配Dashscope API，优化用于文档问答场景
"""
import dashscope
from http import HTTPStatus
from typing import Optional, Dict, List, Union
import json
import time
from datetime import datetime
import os

class RAGLLMClient:
    """
    RAG专用LLM客户端
    支持多种模型调用，带有重试机制和日志记录
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "deepseek-v3.2",
                 temperature: float = 0.1,
                 max_tokens: int = 2000,
                 system_prompt: str = None):
        """
        初始化LLM客户端
        :param api_key: Dashscope API密钥，默认从环境变量读取
        :param model: 模型名称
        :param temperature: 温度参数，控制随机性
        :param max_tokens: 最大输出token数
        :param system_prompt: 系统提示词
        """
        # 设置API密钥
        if api_key:
            dashscope.api_key = api_key
        elif os.environ.get("DASHSCOPE_API_KEY"):
            dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
        else:
            # 使用您的默认密钥
            dashscope.api_key = "sk-19bef3c06efa46d6a8ed322eed8e0439"
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 设置默认系统提示词（针对RAG场景优化）
        if system_prompt is None:
            self.system_prompt = """你是一个专业、准确的文档问答助手。请根据用户提供的文档片段回答问题。

请遵循以下规则：
1. 只使用提供的文档片段中的信息回答问题
2. 如果文档中没有相关信息，请明确说"根据提供的文档，没有找到相关信息"
3. 回答要简洁、准确，避免冗长
4. 可以引用文档中的具体信息
5. 不要编造文档中没有的信息"""
        else:
            self.system_prompt = system_prompt
        
        # 日志目录
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"✅ LLM客户端已初始化 - 模型: {model}")
    
    def call_with_context(self, 
                         question: str, 
                         context: Union[str, List[str]], 
                         max_retries: int = 3) -> Dict:
        """
        基于上下文调用LLM（RAG专用）
        :param question: 用户问题
        :param context: 检索到的文档上下文（字符串或列表）
        :param max_retries: 最大重试次数
        :return: 包含回答和元数据的字典
        """
        # 构建消息
        messages = self._build_messages(question, context)
        
        # 构建请求参数
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "result_format": "message"
        }
        
        # 记录请求日志
        self._log_request(question, context, request_params)
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                # 调用API
                start_time = time.time()
                response = dashscope.Generation.call(**request_params)
                response_time = time.time() - start_time
                
                # 检查响应
                if response.status_code == HTTPStatus.OK:
                    # 提取回答
                    answer = response.output.choices[0].message.content
                    
                    # 构建结果
                    result = {
                        "success": True,
                        "answer": answer,
                        "question": question,
                        "context": context if isinstance(context, str) else "\n\n".join(context),
                        "model": self.model,
                        "response_time": response_time,
                        "usage": {
                            "input_tokens": response.usage.get("input_tokens", 0),
                            "output_tokens": response.usage.get("output_tokens", 0),
                            "total_tokens": response.usage.get("total_tokens", 0)
                        },
                        "raw_response": response
                    }
                    
                    # 记录响应日志
                    self._log_response(result)
                    
                    return result
                    
                else:
                    error_msg = f"API调用失败 (状态码: {response.status_code}): {response.message}"
                    print(f"❌ 第{attempt+1}次尝试失败: {error_msg}")
                    
                    # 如果不是最后一次尝试，等待后重试
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"⏳ 等待{wait_time}秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise Exception(error_msg)
                        
            except Exception as e:
                print(f"❌ 第{attempt+1}次尝试发生异常: {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    # 返回错误结果
                    return {
                        "success": False,
                        "answer": f"抱歉，生成回答时发生错误: {str(e)}",
                        "question": question,
                        "context": context if isinstance(context, str) else "\n\n".join(context),
                        "error": str(e),
                        "model": self.model
                    }
        
        # 所有重试都失败
        return {
            "success": False,
            "answer": "抱歉，多次尝试后仍无法生成回答。",
            "question": question,
            "error": "Max retries exceeded"
        }
    
    def _build_messages(self, question: str, context: Union[str, List[str]]) -> List[Dict]:
        """
        构建对话消息
        """
        # 格式化上下文
        if isinstance(context, list):
            context_str = "\n\n".join([f"[片段{i+1}] {doc}" for i, doc in enumerate(context)])
        else:
            context_str = context
        
        # 构建用户消息
        user_message = f"""基于以下文档片段，回答用户的问题。

文档片段：
{context_str}

用户问题：{question}

请根据上述文档内容回答问题。如果文档中没有相关信息，请如实说明。"""
        
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_message}
        ]
        
        return messages
    
    def _log_request(self, question: str, context: Union[str, List[str]], params: Dict):
        """记录请求日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "request",
            "question": question,
            "context_length": len(context) if isinstance(context, str) else sum(len(c) for c in context),
            "context_count": 1 if isinstance(context, str) else len(context),
            "model": self.model,
            "temperature": self.temperature,
            "params": {k: v for k, v in params.items() if k != "messages"}  # 不记录完整消息
        }
        
        # 保存到文件
        log_file = os.path.join(self.log_dir, f"llm_requests_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def _log_response(self, result: Dict):
        """记录响应日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "response",
            "question": result["question"],
            "answer": result["answer"],
            "model": result["model"],
            "response_time": result["response_time"],
            "usage": result["usage"],
            "success": result["success"]
        }
        
        # 保存到文件
        log_file = os.path.join(self.log_dir, f"llm_responses_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        # 也打印摘要
        print(f"🤖 LLM调用: {result['response_time']:.2f}s, "
              f"输入:{result['usage']['input_tokens']} tokens, "
              f"输出:{result['usage']['output_tokens']} tokens")

# 便捷函数
def get_llm_client(model: str = "deepseek-v3.2") -> RAGLLMClient:
    """获取LLM客户端实例"""
    return RAGLLMClient(model=model)

def simple_call_with_context(question: str, context: Union[str, List[str]], 
                            model: str = "deepseek-v3.2") -> str:
    """
    简化调用函数
    :return: 模型生成的答案
    """
    client = RAGLLMClient(model=model)
    result = client.call_with_context(question, context)
    return result["answer"] if result["success"] else f"错误: {result.get('error', '未知错误')}"

# 测试函数
def test_rag_llm():
    """测试RAG LLM客户端"""
    print("🧪 测试RAG LLM客户端...")
    
    # 创建客户端
    client = RAGLLMClient()
    
    # 测试问题和上下文
    test_question = "人工智能有哪些主要应用领域？"
    test_context = [
        "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "人工智能在医疗、金融、教育、交通、安防等领域有广泛应用。",
        "机器学习是人工智能的一个重要分支，主要研究计算机如何模拟或实现人类的学习行为。"
    ]
    
    # 调用模型
    result = client.call_with_context(test_question, test_context)
    
    if result["success"]:
        print("✅ 测试成功！")
        print(f"问题: {test_question}")
        print(f"答案: {result['answer']}")
        print(f"响应时间: {result['response_time']:.2f}秒")
        print(f"Token使用: 输入{result['usage']['input_tokens']}, 输出{result['usage']['output_tokens']}")
    else:
        print("❌ 测试失败")
        print(f"错误: {result.get('error')}")

if __name__ == "__main__":
    # 运行测试
    test_rag_llm()