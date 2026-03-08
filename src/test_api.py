# src/test_api.py
"""
兼容旧版本的API调用文件
现在委托给新的RAGLLMClient处理
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_api import simple_call_with_context, RAGLLMClient

# 保持向后兼容的函数名
def call_llm_api(prompt: str, model: str = "deepseek-v3.2") -> str:
    """
    旧版本兼容函数 - 将prompt作为上下文调用LLM
    注意：这个函数将整个prompt作为"上下文"处理
    """
    # 假设prompt中已经包含了问题和上下文
    # 这里我们简单地将整个prompt作为上下文
    client = RAGLLMClient(model=model)
    
    # 创建一个简单的问题
    dummy_question = "请根据提供的文档回答问题"
    
    # 调用
    result = client.call_with_context(dummy_question, prompt)
    
    if result["success"]:
        return result["answer"]
    else:
        return f"调用LLM失败: {result.get('error', '未知错误')}"

def call_qwen_with_prompt(prompt: str = None) -> str:
    """
    旧版本兼容函数 - 调用通义千问
    """
    if prompt is None:
        prompt = "请用一句话介绍人工智能。"
    
    client = RAGLLMClient(model="deepseek-v3.2")
    
    # 构建消息
    dummy_question = "请回答以下问题"
    
    result = client.call_with_context(dummy_question, prompt)
    
    if result["success"]:
        print("通义千问：", result["answer"])
        return result["answer"]
    else:
        print('请求失败:', result.get('error', '未知错误'))
        return ""

# 原始测试代码
if __name__ == '__main__':
    # 测试原始功能
    call_qwen_with_prompt("请用一句话介绍人工智能。")
    
    # 测试新功能
    print("\n" + "="*50)
    print("测试RAG功能:")
    
    context = [
        "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "人工智能在医疗、金融、教育、交通、安防等领域有广泛应用。"
    ]
    
    answer = simple_call_with_context("人工智能有哪些应用？", context)
    print(f"答案: {answer}")