import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter# 1. 进入项目目录
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import jieba
from langchain_community.retrievers import BM25Retriever
# 如果上面的导入不行，可以尝试这个
from langchain_classic.retrievers import EnsembleRetriever



app = FastAPI()
# 配置环境变量（请替换为你的实际 Key）
os.environ["ZHIPUAI_API_KEY"] = "你的智谱API"
VECTORDB_DIR = "./chroma_db"
EMBEDDING_MODEL = ZhipuAIEmbeddings(model="embedding-2")
# 初始化 GLM-4-Flash 模型 (temperature 设为 0.1 以获得更确定的归纳回答)
LLM_MODEL = ChatZhipuAI(model="glm-4-flash", temperature=0.1)
# ================= Prompt 工程设计 =================
# 采用 ICIO 框架：指令、背景、输入数据、输出指标
SYSTEM_TEMPLATE = """
你是一个专业、严谨的文档问答助手。请严格基于以下检索到的【上下文信息】来回答用户的【问题】。
要求：
1. 归纳总结上下文信息，给出条理清晰的回答。
2. 如果上下文中包含答案，请直接回答，不要附加多余的开场白（如"根据提供的信息"）。
3. 如果上下文中不包含答案，请明确回复："根据当前文档，无法回答该问题。"，严禁自行编造内容。
【上下文信息】：
{context}
"""
HUMAN_TEMPLATE = "{question}"
# 构建 RAG 链的核心组件
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", HUMAN_TEMPLATE),
])


def format_docs(docs):
    """将检索到的文档列表格式化为纯文本字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


# === 新增全局变量：用于存储切好的文本块，供 BM25 使用 ===
# 实际生产环境中，文档块应持久化存储，这里为 Demo 简化为内存存储
global_chunks = []


# === 新增：BM25 中文分词预处理函数 ===
def jieba_preprocess_func(text):
    """由于 BM25 默认按空格分词，对中文支持差，必须用 jieba 强制切分"""
    return list(jieba.cut(text))

# ================= FastAPI 路由 =================
@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>RAG 知识库问答系统</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .upload-box { border: 2px dashed #ccc; padding: 20px; text-align: center; margin-bottom: 30px; }
            .chat-box { border-top: 1px solid #eee; padding-top: 20px; }
            input[type="text"] { width: 80%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            #status { margin-top: 10px; color: green; font-size: 14px; }
            #answer { margin-top: 15px; padding: 15px; background: #f1f1f1; border-radius: 4px; min-height: 50px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📚 RAG 知识库问答系统</h2>
            <div class="upload-box">
                <form id="uploadForm">
                    <input type="file" id="pdfFile" accept="application/pdf" required>
                    <button type="submit">上传并构建向量库</button>
                </form>
                <div id="status"></div>
            </div>
            <div class="chat-box">
                <h3>基于文档提问：</h3>
                <form id="askForm">
                    <input type="text" id="question" placeholder="请输入关于PDF内容的问题..." required>
                    <button type="submit">发送</button>
                </form>
                <div id="answer"><i>系统就绪，请先上传PDF，然后在此处查看回答...</i></div>
            </div>
        </div>
        <script>
            // PDF 上传逻辑
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = "正在处理PDF，请稍候...";
                statusDiv.style.color = "blue";
                const formData = new FormData();
                formData.append('file', document.getElementById('pdfFile').files[0]);
                const response = await fetch('/upload', { method: 'POST', body: formData });
                if (response.ok) {
                    const result = await response.json();
                    statusDiv.textContent = `成功！存入 ${result.chunk_count} 个文本块。现在可以提问了。`;
                    statusDiv.style.color = "green";
                } else {
                    statusDiv.textContent = "处理失败，请查看日志。";
                    statusDiv.style.color = "red";
                }
            });
            // 问答逻辑
            document.getElementById('askForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const answerDiv = document.getElementById('answer');
                const question = document.getElementById('question').value;
                answerDiv.textContent = "正在检索和思考，请稍候...";
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                if (response.ok) {
                    const result = await response.json();
                    answerDiv.textContent = result.answer;
                } else {
                    answerDiv.textContent = "请求出错。";
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global global_chunks
    temp_dir = "./temp_pdf"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    # 关键：将切好的文本块保存到全局变量，供后续 BM25 初始化使用
    global_chunks = chunks
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        persist_directory=VECTORDB_DIR
    )
    os.remove(file_path)
    return {"message": "PDF 处理成功", "chunk_count": len(chunks)}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    # 1. 加载已有的向量数据库
    if not os.path.exists(VECTORDB_DIR):
        return {"answer": "请先上传 PDF 文件构建知识库。"}
    # 1. 初始化稠密检索器 (向量检索)
    vector_db = Chroma(persist_directory=VECTORDB_DIR, embedding_function=EMBEDDING_MODEL)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. 初始化稀疏检索器 (BM25 关键词检索)
    # 传入分词函数 preprocess_func
    bm25_retriever = BM25Retriever.from_documents(global_chunks, preprocess_func=jieba_preprocess_func)
    bm25_retriever.k = 3

    # 3. 构建混合检索器
    # weights=[0.5, 0.5] 表示 BM25 和向量检索的权重各占一半，可根据实际效果调整
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    # 3. 构建 RAG 链
    rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | LLM_MODEL
            | StrOutputParser()
    )
    # 4. 调用链并获取回答 (注意这里使用 req.question)
    try:
        answer = rag_chain.invoke(req.question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"生成回答时出错：{str(e)}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)