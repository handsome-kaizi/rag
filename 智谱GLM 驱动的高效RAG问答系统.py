import os
import shutil
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore


app = FastAPI()
# 配置环境变量（请替换为你的实际 Key）
os.environ["ZHIPUAI_API_KEY"] = "你的智谱api"
VECTORDB_DIR = "./chroma_db"
DOCSTORE_DIR = "./docstore"
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
global_chunks = []
parent_store = None  # 补充这行，预先声明全局变量
 #尝试加载之前保存的 BM25 文档块
BM25_CHUNKS_FILE = "./bm25_chunks.pkl"
if os.path.exists(BM25_CHUNKS_FILE):
    with open(BM25_CHUNKS_FILE, 'rb') as file_obj:
        global_chunks = pickle.load(file_obj)
    print(f"✅ 已加载 {len(global_chunks)} 个 BM25 文档块")
else:
    print("⚠️ 未找到已保存的 BM25 文档块，请先上传 PDF")


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
    global global_chunks, parent_store
    try:
        temp_dir = "./temp_pdf"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # 1. 加载 PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise ValueError("PDF解析内容为空！可能是扫描版图片，PyPDFLoader无法提取文字。")
        # 2. 优先进行简单分块（用于 BM25），确保 chunks 有值
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        global_chunks = chunks
        # 3. 立刻持久化保存 BM25 文档块
        with open(BM25_CHUNKS_FILE, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"💾 已保存 {len(chunks)} 个 BM25 文档块到本地")
        # 4. 再处理父子文档（如果此处报错，不影响 BM25 使用）
        try:
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            vector_db = Chroma(
                collection_name="parent_child_db",
                embedding_function=EMBEDDING_MODEL,
                persist_directory=VECTORDB_DIR
            )
            parent_store = create_kv_docstore(LocalFileStore(DOCSTORE_DIR))
            parent_retriever = ParentDocumentRetriever(
                vectorstore=vector_db,
                docstore=parent_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            parent_retriever.add_documents(documents)
            print("✅ 父子文档检索器构建成功")
        except Exception as inner_e:
            print(f"⚠️ 父子文档构建失败（不影响 BM25 检索）：{str(inner_e)}")
        os.remove(file_path)
        return {
            "message": "PDF 处理成功",
            "chunk_count": len(chunks),
            "parent_docs_count": len(list(parent_store.yield_keys())) if parent_store else 0
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"detail": f"PDF 处理失败：{str(e)}"}, 500

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    global global_chunks, parent_store
    # 1. 加载已有的向量数据库
    if not os.path.exists(VECTORDB_DIR):
        return {"answer": "请先上传 PDF 文件构建知识库。"}

    try:
        # === 方案1：父子文档检索 + BM25 混合检索（推荐）===

        # 2.重新加载向量数据库
        vector_db = Chroma(
            collection_name="parent_child_db",
            embedding_function=EMBEDDING_MODEL,
            persist_directory=VECTORDB_DIR
        )

        # 3. 重建父子分割器
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

        # 4. 重新加载持久化的父文档存储
        parent_store = create_kv_docstore(LocalFileStore(DOCSTORE_DIR))

        # 5. 重建真正的父子检索器
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vector_db,
            docstore=parent_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        # 6. 创建 BM25 检索器（用于关键词匹配）- 添加空值检查与本地加载
        # 在 /ask 路由内部
        if not global_chunks:
            if os.path.exists(BM25_CHUNKS_FILE):
                with open(BM25_CHUNKS_FILE, 'rb') as file_obj:
                    global_chunks = pickle.load(file_obj)
                print(f"✅ 从本地重新加载 {len(global_chunks)} 个 BM25 文档块")
            else:
                print("⚠️ BM25 文档块为空，且无本地缓存，仅使用父子检索器")

        if not global_chunks:
            retrieved_docs = parent_retriever.invoke(req.question)
            # 只使用父子检索器，不使用混合检索器
        else:
            bm25_retriever = BM25Retriever.from_documents(
                global_chunks,
                preprocess_func=jieba_preprocess_func
            )
            bm25_retriever.k = 3

            # 构建混合检索器（父子检索 + BM25）
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, parent_retriever],
                weights=[0.4, 0.6]  # BM25占40%，父子检索占60%
            )
            print(f"✅ 使用混合检索器（BM25: {len(global_chunks)} 个文档块）")

            # 检索文档
            retrieved_docs = ensemble_retriever.invoke(req.question)

        # 7. 格式化文档
        context = format_docs(retrieved_docs)

        # 8. 构建 RAG 链
        rag_chain = (
                {"context": lambda _: context, "question": RunnablePassthrough()}
                | prompt
                | LLM_MODEL
                | StrOutputParser()
        )

        answer = rag_chain.invoke(req.question)
        return {"answer": answer}

    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整错误信息便于调试
        return {"answer": f"生成回答时出错：{str(e)}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)

