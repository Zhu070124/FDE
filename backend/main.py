"""
CQUPT AI Assistant — FastAPI 服务入口
"""
import sys
from pathlib import Path

# 确保可以导入同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from document_loader import DocumentLoader
from vector_store import VectorStore
from answer_generator import AnswerGenerator, HybridSearcher
from guardrails import Guardrails
import config

# === 初始化 ===
app = FastAPI(
    title="CQUPT AI 学生成长助手",
    description="基于RAG的高校学生成长与心理健康一站式咨询助手",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局组件（延迟初始化）
vector_store: VectorStore = None
answer_gen: AnswerGenerator = None
hybrid_searcher: HybridSearcher = None
guardrails: Guardrails = None


def get_components():
    """延迟初始化全局组件"""
    global vector_store, answer_gen, hybrid_searcher, guardrails
    if vector_store is None:
        vector_store = VectorStore()
    if answer_gen is None:
        answer_gen = AnswerGenerator()

        # 把LLM能力注入到意图路由（轻量调用，只分类）
        def llm_classify(prompt: str, temperature: float, max_tokens: int) -> str:
            return answer_gen._call_doubao_api(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )

        hybrid_searcher = HybridSearcher(vector_store, llm_fn=llm_classify)
    if guardrails is None:
        guardrails = Guardrails()
    return vector_store, answer_gen, hybrid_searcher, guardrails


# === 数据模型 ===
class ChatRequest(BaseModel):
    query: str
    history: list = []  # [{"role": "user/assistant", "content": "..."}]

class ChatResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    sources: list = []
    warning: str | None = None

class IngestRequest(BaseModel):
    collection: str  # policy / exam / psychology
    directory: str = None  # 可选，指定文档目录


# === API 路由 ===

@app.get("/")
async def root():
    """返回前端页面"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"status": "ok", "message": "CQUPT AI Assistant API"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """核心接口：提问 → 意图识别 → 检索 → 生成回答"""
    _, answer_gen, hybrid_searcher, guardrails = get_components()

    # 1. 输入检查
    ok, msg = guardrails.check_query(req.query)
    if not ok:
        if msg == "crisis":
            return ChatResponse(
                answer=guardrails._crisis_response(),
                intent="psychological",
                confidence=1.0,
                warning="高危关键词触发危机干预",
            )
        raise HTTPException(400, msg)

    # 2. 意图路由 + 检索
    result = hybrid_searcher.search(req.query)
    intent = result["intent"]
    strategy = result["strategy"]
    docs = result["retrieved_docs"]

    # 3. 生成回答
    answer = answer_gen.generate(
        query=req.query,
        retrieved_docs=docs,
        intent=intent,
        temperature=strategy["temperature"],
        require_citation=strategy["require_citation"],
        add_disclaimer=strategy.get("add_disclaimer", False),
    )

    # 4. 护栏校验
    ok, warning, answer = guardrails.check_response(
        req.query, answer, intent.value
    )

    # 5. 提取来源
    sources = []
    for doc in docs[:3]:
        sources.append({
            "source": doc.get("metadata", {}).get("source", "未知"),
            "snippet": doc["content"][:100],
            "score": round(doc.get("score", 0), 3),
        })

    return ChatResponse(
        answer=answer,
        intent=intent.value,
        confidence=round(result["confidence"], 3),
        sources=sources,
        warning=warning,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式聊天端点——边生成边返回，大幅降低感知延迟"""
    _, answer_gen, hybrid_searcher, guardrails = get_components()

    # 1. 输入检查
    ok, msg = guardrails.check_query(req.query)
    if not ok:
        if msg == "crisis":
            async def crisis_stream():
                yield guardrails._crisis_response()
            return StreamingResponse(crisis_stream(), media_type="text/plain")

    # 2. 意图路由 + 检索
    result = hybrid_searcher.search(req.query)
    intent = result["intent"]
    strategy = result["strategy"]
    docs = result["retrieved_docs"]

    # 3. 流式生成
    def generate():
        full_response = ""
        for chunk in answer_gen.generate_stream(
            query=req.query,
            retrieved_docs=docs,
            intent=intent,
            temperature=strategy["temperature"],
            require_citation=strategy["require_citation"],
            add_disclaimer=strategy.get("add_disclaimer", False),
            history=req.history,
        ):
            full_response += chunk
            yield chunk

        # 4. 护栏校验（对完整回答做后处理）
        ok, warning, modified = guardrails.check_response(
            req.query, full_response, intent.value
        )
        if warning and ok:
            yield f"\n\n⚠️ {warning}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/api/ingest")
async def ingest_documents(req: IngestRequest):
    """文档导入接口"""
    vs, _, _, _ = get_components()

    doc_dir = Path(req.directory) if req.directory else config.DOCUMENTS_DIR
    if not doc_dir.exists():
        raise HTTPException(404, f"文档目录不存在: {doc_dir}")

    loader = DocumentLoader(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = loader.load_directory(doc_dir)

    if not chunks:
        return {"status": "warning", "message": "未找到可导入的文档", "count": 0}

    vs.add_chunks(chunks, req.collection)

    return {
        "status": "ok",
        "message": f"成功导入 {len(chunks)} 个文档块到 {req.collection}",
        "count": len(chunks),
        "files": list(set(c.metadata["source"] for c in chunks)),
    }


@app.get("/api/stats")
async def get_stats():
    """知识库统计"""
    vs, _, _, _ = get_components()
    return {
        "collections": vs.get_stats(),
        "embedding_model": "BM25 + jieba (纯Python)",
        "doubao_model": config.DOUBAO_MODEL,
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


# === 启动 ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
