from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
import docx2txt
import os
from win32com import client as wc
import tempfile
import pythoncom
def load_document(file_path: str) -> List[Document]:
    """
    主函数：加载单个文档
    :param file_path: 文档的本地路径
    :return: Document对象列表
    """
    file_ext = os.path.splitext(file_path)[-1].lower()
    
    try:
        if file_ext == ".pdf":
            return _load_pdf(file_path)
        elif file_ext == ".docx":
            return _load_word(file_path)
        elif file_ext == ".txt":
            return _load_txt(file_path)
        else:
            print(f"不支持的文件格式：{file_ext}")
            return []
    except Exception as e:
        print(f"文档加载失败 {file_path}: {str(e)}")
        return []

def _load_pdf(file_path: str) -> List[Document]:
    """加载PDF文档"""
    try:
        docs = []
        reader = PdfReader(file_path)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # 检查是否为空
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path, "page": i+1, "type": "pdf"}
                    )
                )
        
        if docs:
            print(f"PDF加载成功: {os.path.basename(file_path)} ({len(docs)}页)")
        else:
            print(f"PDF无文本内容: {os.path.basename(file_path)}")
            
        return docs
    except Exception as e:
        print(f"PDF加载失败 {file_path}: {str(e)}")
        return []

def convert_doc_to_docx(file_path):
    """
    使用win32com将.doc文件转换为.docx文件,返回转换后.docx文件的临时路径。
    """

    try:
        # 初始化COM组件
        pythoncom.CoInitialize()

        # 创建Word应用程序实例，并设置为后台运行
        word_app = wc.Dispatch("Word.Application")
        word_app.Visible = False
        word_app.DisplayAlerts = False  # 关闭提示框

        # 使用绝对路径打开.doc文档
        abs_path = os.path.abspath(file_path)
        doc = word_app.Documents.Open(abs_path)

        # 创建一个临时文件来保存转换后的.docx
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_docx_path = temp_file.name

        # 关键步骤：另存为.docx格式。FileFormat=16表示.docx格式[3](@ref)
        doc.SaveAs2(temp_docx_path, FileFormat=16)
        doc.Close(SaveChanges=False)
        word_app.Quit()

        # 反初始化COM组件
        pythoncom.CoUninitialize()

        print(f".doc文件已成功转换为.docx临时文件")
        return temp_docx_path

    except Exception as e:
        print(f"使用win32com转换.doc文件失败: {str(e)}")
        try:
            word_app.Quit()
            pythoncom.CoUninitialize()
        except:
            pass
        return None

def _load_word(file_path: str) -> List[Document]:
    """加载Word文档（支持.doc和.docx）"""
    try:
        file_ext = os.path.splitext(file_path)[-1].lower()
        text = ""
        
        if file_ext == ".docx":
            text = docx2txt.process(file_path)
        else:
            print(f"不支持的Word格式: {file_ext}")
            return []
        
        if text.strip():
            print(f"Word加载成功: {os.path.basename(file_path)}")
            return [Document(
                page_content=text, 
                metadata={"source": file_path, "type": "word", "format": file_ext}
            )]
        else:
            print(f"Word文档为空: {os.path.basename(file_path)}")
            return []
            
    except Exception as e:
        print(f"Word加载失败 {file_path}: {str(e)}")
        return []

def _load_txt(file_path: str) -> List[Document]:
    """内部函数：加载TXT文档"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if text.strip():
            print(f"TXT加载成功: {os.path.basename(file_path)}")
            return [Document(
                page_content=text, 
                metadata={"source": file_path, "type": "txt"}
            )]
        else:
            print(f"TXT文档为空: {os.path.basename(file_path)}")
            return []
    except Exception as e:
        print(f"TXT加载失败 {file_path}: {str(e)}")
        return []

def batch_load_documents(docs_folder: str = "./docs") -> List[Document]:
    """
    批量加载docs文件夹下的所有文档
    param docs_folder: docs文件夹路径，默认"./docs"
    :return: 包含所有文档内容的列表
    """
    if not os.path.exists(docs_folder):
        print(f"文件夹不存在: {docs_folder}")
        return []
    
    if not os.path.isdir(docs_folder):
        print(f"不是文件夹: {docs_folder}")
        return []
    
    all_documents = []
    supported_ext = {'.pdf', '.docx', '.txt'}
    
    print(f"开始批量处理文件夹: {docs_folder}")
    print("=" * 50)
    
    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)
        
        # 跳过文件夹，只处理文件
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[-1].lower()
            
            if file_ext in supported_ext:
                print(f"处理文件: {filename}")
                docs = load_document(file_path)
                all_documents.extend(docs)
            else:
                print(f"跳过不支持格式: {filename}")
    
    print("=" * 50)
    print(f"批量处理完成！")
    print(f"总文件数: {len([f for f in os.listdir(docs_folder) if os.path.isfile(os.path.join(docs_folder, f))])}")
    print(f"成功文档块数: {len(all_documents)}")
    
    return all_documents