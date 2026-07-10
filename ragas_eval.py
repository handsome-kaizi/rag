"""
RAGAS 风格 RAG 评测脚本
评测四个核心指标：上下文精度、忠实度、回答相关性、上下文召回率
使用智谱 GLM-4-Plus 作为 Judge（生成用 GLM-4-Flash，评测用 GLM-4-Plus，避免同模型自评偏差）
"""

import os
import pickle
import time
import numpy as np
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
import jieba


def safe_invoke(llm, prompt_value, max_retries=5, base_wait=2.0):
    """

    """
    last_err = None
    for attempt in range(max_retries):
        try:
            result = llm.invoke(prompt_value)
            time.sleep(base_wait)  # 控制请求间隔，规避 QPS 限制
            return result
        except Exception as e:
            last_err = e
            # 429 / 限流类错误 → 指数退避重试
            wait = base_wait * (2 ** attempt)
            print(f"  ⚠️ 调用失败({attempt+1}/{max_retries})，{wait:.0f}s后重试：{str(e)[:80]}")
            time.sleep(wait)
    raise last_err

# ================= 配置 =================
# ⚠️ 改成你的真实 key,或者用环境变量(os.getenv("ZHIPUAI_API_KEY"))
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "你的智谱AP")
VECTORDB_DIR = "./chroma_db"
BM25_CHUNKS_FILE = "./bm25_chunks.pkl"

EMBEDDING_MODEL = ZhipuAIEmbeddings(model="embedding-2")
# Judge 使用 glm-4-flashx（Flash 升级版，免费且能力更强），
# 与生成模型 glm-4-flash 区分，避免同模型自评偏差
JUDGE_LLM = ChatZhipuAI(model="glm-4-flashx")

# ================= 加载检索器 =================
def jieba_preprocess_func(text):
    return list(jieba.cut(text))


def load_retrievers():
    """加载向量检索器 + BM25 检索器 → 混合检索器"""
    if not os.path.exists(VECTORDB_DIR):
        raise FileNotFoundError(
            f"向量库目录 '{VECTORDB_DIR}' 不存在！\n"
            "请先用主程序上传 PDF 构建向量库后再运行评测。"
        )
    if not os.path.exists(BM25_CHUNKS_FILE):
        raise FileNotFoundError(
            f"BM25 文档块文件 '{BM25_CHUNKS_FILE}' 不存在！\n"
            "请先用主程序上传 PDF 后再运行评测。"
        )

    vector_db = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=EMBEDDING_MODEL,
        collection_name="langchain"  # 与 Chroma.from_documents 默认名称一致
    )
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    with open(BM25_CHUNKS_FILE, 'rb') as f:
        global_chunks = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(global_chunks, preprocess_func=jieba_preprocess_func)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.6, 0.4]
    )
    return ensemble_retriever


# ================= 指标1：上下文精度（Context Precision）=================
CONTEXT_PRECISION_PROMPT = ChatPromptTemplate.from_template("""\
Determine whether the following document chunk contains information useful for answering the question.
Reply ONLY with YES or NO, no explanation.

Question: {question}
Chunk: {context}

Relevant?""")


def evaluate_context_precision(question, contexts, judge_llm):
    """
    计算上下文精度：检索回来的 chunk 中，有多少比例真正与问题相关。
    加权计算：排序靠前的相关 chunk 得分更高。
    """
    if not contexts:
        return 0.0

    scores = []
    for rank, ctx in enumerate(contexts):
        prompt_text = CONTEXT_PRECISION_PROMPT.format(question=question, context=ctx)
        verdict = safe_invoke(judge_llm, prompt_text).content.strip().upper()
        is_relevant = "YES" in verdict
        # 位置加权：排名越靠前，权重越高 (k=1: weight=1.0, k=3: weight≈0.67)
        weight = 1.0 / (rank + 1)
        scores.append(weight if is_relevant else 0.0)

    if sum(1.0 / (i + 1) for i in range(len(contexts))) == 0:
        return 0.0
    return sum(scores) / sum(1.0 / (i + 1) for i in range(len(contexts)))


# ================= 指标2：忠实度（Faithfulness）===========================
FAITHFULNESS_CLAIMS_PROMPT = ChatPromptTemplate.from_template("""\
Extract all independent, atomic factual claims from the following answer.
Output one claim per line, no numbering or prefixes.

Answer: {answer}

Claims:""")

FAITHFULNESS_VERIFY_PROMPT = ChatPromptTemplate.from_template("""\
Determine whether the following claim can be inferred from the information in the context.
Reply ONLY with YES or NO, no explanation.

Context: {context}

Claim: {claim}

Can this claim be inferred from the context?""")


def evaluate_faithfulness(answer, contexts, judge_llm):
    """
    计算忠实度：回答中的事实陈述有多少比例能从检索结果中验证。
    """
    if not answer or not contexts:
        return 0.0

    # Step 1: 从回答中提取原子事实
    claims_text = safe_invoke(judge_llm, FAITHFULNESS_CLAIMS_PROMPT.format(answer=answer)).content.strip()
    claims = [c.strip() for c in claims_text.split("\n") if c.strip()]
    if not claims:
        return 0.0

    # Step 2: 逐一验证每个事实
    combined_context = "\n\n".join(contexts)
    verified = 0
    for claim in claims:
        verdict = safe_invoke(judge_llm,
            FAITHFULNESS_VERIFY_PROMPT.format(context=combined_context, claim=claim)
        ).content.strip().upper()
        if "YES" in verdict:
            verified += 1

    return verified / len(claims)


# ================= 指标3：回答相关性（Answer Relevancy）===================
ANSWER_RELEVANCY_PROMPT = ChatPromptTemplate.from_template("""\
Generate 3 questions that the following answer would be a suitable response for.
Output one question per line, no numbering.

Answer: {answer}

Generated questions:""")


def evaluate_answer_relevancy(question, answer, judge_llm, embedding_model):
    """
    计算回答相关性：从回答反向生成问题，计算与原始问题的语义相似度。
    """
    if not answer:
        return 0.0

    # Step 1: 从回答反向生成问题
    gen_questions_text = safe_invoke(judge_llm, ANSWER_RELEVANCY_PROMPT.format(answer=answer)).content.strip()
    gen_questions = [q.strip() for q in gen_questions_text.split("\n") if q.strip()]

    if not gen_questions:
        return 0.0

    # Step 2: 计算每个生成问题与原始问题的余弦相似度
    orig_emb = np.array(embedding_model.embed_query(question))
    similarities = []
    for gq in gen_questions:
        gen_emb = np.array(embedding_model.embed_query(gq))
        sim = np.dot(orig_emb, gen_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(gen_emb))
        similarities.append(sim)

    return float(np.mean(similarities))


# ================= 指标4：上下文召回率（Context Recall）===================
CONTEXT_RECALL_EXTRACT_PROMPT = ChatPromptTemplate.from_template("""\
Extract all independent, atomic factual claims from the following reference answer.
Output one claim per line, no numbering or prefixes.

Reference Answer: {reference}

Claims:""")

CONTEXT_RECALL_VERIFY_PROMPT = ChatPromptTemplate.from_template("""\
Determine whether the following claim is supported by ANY of the provided contexts.
Reply ONLY with YES or NO.

Claim: {claim}

Contexts:
{contexts}

Is this claim supported by any context?""")


def evaluate_context_recall(question, contexts, reference, judge_llm):
    """
    计算上下文召回率（标准 RAGAS 实现）：
    1. 将参考答案拆分为原子事实主张
    2. 对每个主张，判断所有检索上下文中是否至少有一条能支撑它
    3. 召回率 = 被支撑的主张数 / 总主张数
    """
    if not contexts or not reference:
        return 0.0

    # Step 1: 从参考答案中提取原子事实
    claims_text = judge_llm.invoke(
        CONTEXT_RECALL_EXTRACT_PROMPT.format(reference=reference)
    ).content.strip()
    claims = [c.strip() for c in claims_text.split("\n") if c.strip()]
    if not claims:
        return 0.0

    # Step 2: 对每个主张，检查是否被任意上下文支撑
    combined_contexts = "\n---\n".join(contexts)
    attributed = 0
    for claim in claims:
        verdict = judge_llm.invoke(
            CONTEXT_RECALL_VERIFY_PROMPT.format(claim=claim, contexts=combined_contexts)
        ).content.strip().upper()
        if "YES" in verdict:
            attributed += 1

    return attributed / len(claims)


def rerank(query, docs, embedding_model, top_k=3):
    """Reranker：Embedding-2 二次打分，取 Top-K"""
    if len(docs) <= top_k:
        return docs
    query_emb = np.array(embedding_model.embed_query(query))
    scored = []
    for doc in docs:
        doc_emb = np.array(embedding_model.embed_query(doc.page_content))
        sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scored.append((sim, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ================= 主评测流程 =================
def run_evaluation(test_cases, show_detail=True):
    """
    test_cases: list of dict, 每个dict包含 question 和 reference（标准答案）
    """
    ensemble_retriever = load_retrievers()
    llm = ChatZhipuAI(model="glm-4-plus", temperature=0.1)

    results = {
        "context_precision": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
    }

    for idx, case in enumerate(test_cases):
        question = case["question"]
        reference = case.get("reference", "")

        # 1. 粗召回(k=5) → Reranker → 精排(top-3)
        retrieved_docs = ensemble_retriever.invoke(question)
        retrieved_docs = rerank(question, retrieved_docs, EMBEDDING_MODEL, top_k=3)
        contexts = [doc.page_content for doc in retrieved_docs]

        # 2. 生成回答
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based on the following context. If the answer cannot be found, say so clearly.\n\nContext: {context}"),
            ("human", "{question}"),
        ])
        context_str = "\n\n".join(contexts)
        answer = llm.invoke(gen_prompt.format(context=context_str, question=question)).content

        # 3. 计算四个指标
        cp = evaluate_context_precision(question, contexts, JUDGE_LLM)
        faith = evaluate_faithfulness(answer, contexts, JUDGE_LLM)
        ar = evaluate_answer_relevancy(question, answer, JUDGE_LLM, EMBEDDING_MODEL)
        cr = evaluate_context_recall(question, contexts, reference, JUDGE_LLM) if reference else None

        results["context_precision"].append(cp)
        results["faithfulness"].append(faith)
        results["answer_relevancy"].append(ar)
        if cr is not None:
            results["context_recall"].append(cr)

        if show_detail:
            print(f"\n{'='*60}")
            print(f"【Q{idx+1}】{question}")
            print(f"{'='*60}")
            print(f"检索到 {len(contexts)} 个片段")
            print(f"回答：{answer[:200]}...")
            print(f"---")
            print(f"  上下文精度 (Context Precision): {cp:.3f}")
            print(f"  忠实度     (Faithfulness):      {faith:.3f}")
            print(f"  回答相关性 (Answer Relevancy):   {ar:.3f}")
            if cr is not None:
                print(f"  上下文召回 (Context Recall):     {cr:.3f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"📊 综合评测结果（共 {len(test_cases)} 个测试用例）")
    print(f"{'='*60}")
    for metric, scores in results.items():
        if scores:
            avg = np.mean(scores)
            print(f"  {metric:20s}: {avg:.3f}  (平均)")

    return results


# ================= 测试用例 =================
if __name__ == "__main__":
    if os.environ.get("ZHIPUAI_API_KEY", "").startswith("你的"):
        print("❌ 请先在脚本第19行设置真实的 ZHIPUAI_API_KEY")
        exit(1)

    # ⚠️ 请根据你上传的 PDF 内容修改以下测试问题
    # reference 字段填写你期望的标准答案，用于计算上下文召回率
    # 如果没有 reference，该指标会跳过

    test_cases = [
        {
            "question": "《中华人民共和国民营经济促进法》是什么时候由哪个机构通过的？",
            "reference": "2025年4月30日由第十四届全国人民代表大会常务委员会第十五次会议通过。"
        },
        {
            "question": "根据该法第三条，民营经济在我国经济中处于什么地位？",
            "reference": "民营经济是社会主义市场经济的重要组成部分，是推进中国式现代化的生力军，是高质量发展的重要基础，是推动我国全面建成社会主义现代化强国、实现中华民族伟大复兴的重要力量。"
        },
        {
            "question": "该法规定了哪些促进民营经济发展的基本原则？",
            "reference": "国家坚持平等对待、公平竞争、同等保护、共同发展的原则。坚持公有制为主体、多种所有制经济共同发展，毫不动摇鼓励、支持、引导非公有制经济发展。"
        },
        {
            "question": "根据该法第二章，市场准入方面有什么规定？",
            "reference": "国家实行全国统一的市场准入负面清单制度。市场准入负面清单以外的领域，包括民营经济组织在内的各类经济组织可以依法平等进入。各级人民政府及其有关部门落实公平竞争审查制度。"
        },
        {
            "question": "该法在投资融资促进方面为民营经济组织提供了哪些金融支持措施？",
            "reference": "包括：支持民营经济组织通过发行股票、债券等方式平等获得直接融资；银行业金融机构接受应收账款、仓单、股权、知识产权等权利质押贷款；金融机构对小型微型民营经济组织实施差异化政策；建立健全信用信息归集共享机制；推动构建民营经济组织融资风险的市场化分担机制。"
        },
        {
            "question": "根据该法第四章，国家如何支持民营经济组织进行科技创新？",
            "reference": "支持民营经济组织参与国家科技攻关项目，支持牵头承担国家重大技术攻关任务，向民营经济组织开放国家重大科研基础设施，支持公共研究开发平台开放共享，鼓励产学研深度融合。加强知识产权保护，实施知识产权侵权惩罚性赔偿制度。"
        },
        {
            "question": "该法对民营经济组织的规范经营提出了哪些具体要求？",
            "reference": "民营经济组织从事生产经营活动应当遵守劳动用工、安全生产、职业卫生、社会保障、生态环境、质量标准、知识产权、网络和数据安全、财政税收、金融等方面的法律法规；不得通过贿赂和欺诈等手段牟取不正当利益；应当完善治理结构和管理制度、强化内部监督；加强财务管理，区分组织财产与经营者个人财产。"
        },
        {
            "question": "根据该法第七章，民营经济组织及其经营者享有哪些权益保护？",
            "reference": "人身权利、财产权利以及经营自主权等合法权益受法律保护。名称权、名誉权、荣誉权和经营者的名誉权、荣誉权、隐私权、个人信息等人格权益受法律保护。禁止利用互联网以侮辱、诽谤等方式恶意侵害人格权益。征收、征用财产应当给予公平合理的补偿。严格区分经济纠纷与经济犯罪，禁止利用行政或刑事手段违法干预经济纠纷。"
        },
        {
            "question": "该法对于国家机关、事业单位和国有企业向民营经济组织支付账款有什么规定？",
            "reference": "国家机关、事业单位、国有企业应当依法或者依合同约定及时向民营经济组织支付账款，不得以人员变更、履行内部付款流程或者在合同未作约定情况下以等待竣工验收批复、决算审计等为由，拒绝或者拖延支付。大型企业向中小民营经济组织采购货物、工程、服务等，应当合理约定付款期限并及时支付账款。"
        },
        {
            "question": "该法第五十一条对民营经济组织的行政处罚有什么原则性规定？",
            "reference": "对民营经济组织及其经营者违法行为的行政处罚应当按照与其他经济组织及其经营者同等原则实施。违法行为依法需要实施行政处罚或者采取其他措施的，应当与违法行为的事实、性质、情节以及社会危害程度相当。具有从轻、减轻或者不予处罚情形的，依照其规定从轻、减轻或者不予处罚。"
        },
    ]

    run_evaluation(test_cases)
