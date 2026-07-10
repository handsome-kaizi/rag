import os
import re
import shutil
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import jieba
import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def split_by_articles(documents, chunk_size=800, chunk_overlap=80):
    """
    法律文档专用分块：按'第X条'边界切分，保证每条条文完整不被切断。
    对超出 chunk_size 的单条条文，按句子边界二次切分并保留重叠。
    """
    ARTICLE_PATTERN = r'(?=第[一二三四五六七八九十百零\d]+条)'

    chunks = []
    for doc in documents:
        text = doc.page_content
        # 1. 按"第X条"边界拆分为独立条文
        articles = re.split(ARTICLE_PATTERN, text)
        for article in articles:
            article = article.strip()
            if not article:
                continue

            # 2. 条文在 chunk_size 内 → 直接作为一个chunk
            if len(article) <= chunk_size:
                chunks.append(Document(page_content=article, metadata=doc.metadata))
                continue

            # 3. 超长条文 → 按句子边界二次切分，保留重叠
            sentences = re.split(r'(?<=[。；])', article)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += sent
                else:
                    if current.strip():
                        chunks.append(Document(page_content=current.strip(), metadata=doc.metadata))
                    # 重叠：保留上一chunk末尾部分作为上下文
                    overlap_text = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                    current = overlap_text + sent
            if current.strip():
                chunks.append(Document(page_content=current.strip(), metadata=doc.metadata))

    # 4. 如果没匹配到任何"第X条"（非法律文档），回退到按句子切分
    if not chunks:
        for doc in documents:
            text = doc.page_content
            sentences = re.split(r'(?<=[。！？；.!?;])', text)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += sent
                else:
                    if current.strip():
                        chunks.append(Document(page_content=current.strip(), metadata=doc.metadata))
                    current = sent
            if current.strip():
                chunks.append(Document(page_content=current.strip(), metadata=doc.metadata))

    return chunks


app = FastAPI()
# 配置环境变量（请替换为你的实际 Key）
os.environ["ZHIPUAI_API_KEY"] = "c28067c108d442b9b2f06334d5002d41.veZRK4S1AmB4USdJ"
VECTORDB_DIR = "./chroma_db"
BM25_CHUNKS_FILE = "./bm25_chunks.pkl"
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


def rerank(query, docs, top_k=3):
    """
    Reranker：用 Embedding-2 计算每个 chunk 与 query 的余弦相似度，取 Top-K。
    k=5 粗召回 → 相似度重排 → 取 top-3 精排结果，兼顾召回率和精度。
    """
    if len(docs) <= top_k:
        return docs
    query_emb = np.array(EMBEDDING_MODEL.embed_query(query))
    scored = []
    for doc in docs:
        doc_emb = np.array(EMBEDDING_MODEL.embed_query(doc.page_content))
        sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scored.append((sim, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# === 全局变量：存储切好的文本块，供 BM25 使用 ===
# 每次启动从空开始，必须重新上传 PDF，不再从磁盘加载历史数据
global_chunks = []


# === 新增：BM25 中文分词预处理函数 ===
def jieba_preprocess_func(text):
    """由于 BM25 默认按空格分词，对中文支持差，必须用 jieba 强制切分"""
    return list(jieba.cut(text))


# ================= Query 预处理优化 =================
QUERY_REWRITE_TEMPLATE = """
你是一个专业的查询改写助手。你的任务是将用户输入的原始查询改写为更适合文档检索的版本。

改写原则：
1. 保留原始查询的核心意图和关键信息，绝不偏离语义方向。
2. 补充可能的同义词、相关术语、短语变体，提升命中率。
3. 将口语化、模糊的表述转换为更精确、正式的检索语言。
4. 仅输出改写后的查询文本，不要附加任何解释、说明或前缀。

原始查询：{query}
"""

def compute_cosine_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的余弦相似度，使用智谱 Embedding 模型"""
    emb_a = np.array(EMBEDDING_MODEL.embed_query(text_a))
    emb_b = np.array(EMBEDDING_MODEL.embed_query(text_b))
    dot_product = np.dot(emb_a, emb_b)
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def preprocess_query(original_query: str, similarity_threshold: float = 0.8) -> str:
    """
    Query 预处理主流程：
    1. 调用 LLM 改写原始 Query，生成检索友好的扩展版本
    2. 计算改写后与改写前的余弦相似度
    3. 若相似度 >= threshold，使用改写后的 Query；否则回退到原始 Query
    """
    rewrite_prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_TEMPLATE)
    rewrite_chain = rewrite_prompt | LLM_MODEL | StrOutputParser()
    try:
        rewritten_query = rewrite_chain.invoke({"query": original_query}).strip()
    except Exception:
        return original_query

    if not rewritten_query:
        return original_query

    try:
        similarity = compute_cosine_similarity(original_query, rewritten_query)
    except Exception:
        return original_query

    if similarity >= similarity_threshold:
        return rewritten_query
    else:
        return original_query


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
                    if (result.chunk_count > 0) {
                        statusDiv.textContent = `成功！存入 ${result.chunk_count} 个文本块。现在可以提问了。`;
                        statusDiv.style.color = "green";
                    } else {
                        statusDiv.textContent = `失败：${result.message || '未知原因，请查看终端日志'}`;
                        statusDiv.style.color = "red";
                    }
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
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if not documents or not any(d.page_content.strip() for d in documents):
            return {"message": "PDF 解析内容为空", "chunk_count": 0}

        # 分块：按法律条文"第X条"边界切分，保证条文完整性
        print(f"[upload] PDF共{len(documents)}页, 总字符数: {sum(len(d.page_content) for d in documents)}")
        chunks = split_by_articles(documents, chunk_size=800, chunk_overlap=80)
        print(f"[upload] 分块结果: {len(chunks)}个chunk")
        if not chunks:
            return {"message": "分块失败：未能从PDF中识别到有效文本块，请检查PDF是否为可提取文字的类型", "chunk_count": 0}

        # 新数据确认可用后再清旧数据
        global_chunks = chunks

        # 持久化 BM25 chunks，供评测脚本使用
        with open(BM25_CHUNKS_FILE, 'wb') as f:
            pickle.dump(chunks, f)

        if os.path.exists(VECTORDB_DIR):
            shutil.rmtree(VECTORDB_DIR, ignore_errors=True)

        # 智谱 Embedding 接口单次最多 64 条文本，必须分批入库
        EMBED_BATCH_SIZE = 64
        vector_db = Chroma(embedding_function=EMBEDDING_MODEL, persist_directory=VECTORDB_DIR)
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            vector_db.add_documents(batch)
            print(f"[upload] 已入库 {min(i + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}")
        return {"message": "PDF 处理成功", "chunk_count": len(chunks)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"处理失败：{str(e)}", "chunk_count": 0}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    global global_chunks

    if not global_chunks:
        return {"answer": "请先上传 PDF 文件构建知识库。"}

    try:
        # 0. Query 预处理：改写 + 余弦相似度质量过滤（阈值 0.8）
        processed_query = preprocess_query(req.question, similarity_threshold=0.8)

        # 1. 构建检索器
        vector_db = Chroma(persist_directory=VECTORDB_DIR, embedding_function=EMBEDDING_MODEL)
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        bm25_retriever = BM25Retriever.from_documents(
            global_chunks, preprocess_func=jieba_preprocess_func
        )
        bm25_retriever.k = 5

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )

        # 2. 检索 + Reranker 包装为 Runnable，嵌入 LCEL 管道
        def retrieve_and_rerank(query):
            docs = ensemble_retriever.invoke(query)
            return rerank(query, docs, top_k=3)

        # 3. LangChain LCEL 管道
        rag_chain = (
            {"context": RunnableLambda(retrieve_and_rerank) | format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | LLM_MODEL
            | StrOutputParser()
        )
        answer = rag_chain.invoke(processed_query)
        return {"answer": answer}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"生成回答时出错：{str(e)}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)


