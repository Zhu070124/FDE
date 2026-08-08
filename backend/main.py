"""
CQUPT AI Assistant — FastAPI 服务入口
"""
import os
import sys
import time
import uuid
import logging
from pathlib import Path

# 确保可以导入同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from fastapi import UploadFile, Form, Depends

from document_loader import DocumentLoader
from vector_store import VectorStore
from answer_generator import AnswerGenerator, HybridSearcher
from guardrails import Guardrails
from query_decomposer import QueryDecomposer, StepExecutor
from grade_docs import DocumentGrader, HallucinationChecker, AnswerGrader
from reflection_loop import SelfTrainingPipeline
from database import init_db, get_db
from auth import register as auth_register, login as auth_login, get_current_user, require_user
from security import rate_limiter, sanitize_input
from llm_guard import LLMGuard
from feedback import record_feedback, get_feedback_stats, log_query, get_query_analytics
from monitoring import error_tracker, health_check
from content_manager import upload_document, list_documents, delete_document, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from admin import get_user_list, get_dashboard
from nl2sql import get_exam_db
from fanya_adapter import FanyaAdapter
import config

# Setup logging
from logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# === 初始化 ===
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动初始化"""
    try:
        config.validate_config()
    except RuntimeError as e:
        logger.critical("配置错误: %s", e)
        import sys
        sys.exit(1)
    try:
        init_db()
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error("数据库初始化失败: %s", e)
    yield

_school_name = config.get_school_attr("name", "高校")
_school_short = config.get_school_attr("short_name", "")
_school_title = f"{_school_short}AI 学生成长助手" if _school_short else "高校AI学生成长助手"

app = FastAPI(
    title=_school_title,
    description=f"基于RAG的高校学生成长与心理健康一站式咨询助手——当前服务学校：{_school_name}",
    version="0.3.0",
    lifespan=lifespan,
)

# 速率限制中间件
app.middleware("http")(rate_limiter.middleware)

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
query_decomposer: QueryDecomposer = None
step_executor: StepExecutor = None
doc_grader: DocumentGrader = None
hallucination_checker: HallucinationChecker = None
answer_grader: AnswerGrader = None
self_training: SelfTrainingPipeline = None


def get_components():
    """延迟初始化全局组件"""
    global vector_store, answer_gen, hybrid_searcher, guardrails
    global query_decomposer, step_executor, doc_grader, hallucination_checker, answer_grader
    global self_training

    if vector_store is None:
        try:
            vector_store = VectorStore()
            answer_gen = AnswerGenerator()

            def llm_classify(prompt: str, temperature: float, max_tokens: int) -> str:
                return answer_gen._call_doubao_api(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            hybrid_searcher = HybridSearcher(vector_store, llm_fn=llm_classify)
            guardrails = Guardrails()
            guardrails.set_llm_fn(llm_classify)  # 启用LLM语义护栏

            # 多步推理组件
            query_decomposer = QueryDecomposer()
            step_executor = StepExecutor(hybrid_searcher)

            # 检索质量组件
            doc_grader = DocumentGrader()
            hallucination_checker = HallucinationChecker()
            answer_grader = AnswerGrader()

            # 自训练组件
            self_training = SelfTrainingPipeline(vector_store)
        except Exception as e:
            logger.exception("Failed to initialize components: %s", e)
            raise

    return vector_store, answer_gen, hybrid_searcher, guardrails


# === 数据模型 ===
class ChatRequest(BaseModel):
    query: str
    history: list = []        # [{"role": "user/assistant", "content": "..."}]
    scene: str = "policy"     # 前端场景选择
    multi_step: bool = False  # 是否启用多步推理
    self_train: bool = False  # 是否启用自训练反思

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: str  # "like" or "dislike"
    message_id: str = None

class FeedbackWithLogRequest(BaseModel):
    """Feedback that also updates query_log."""
    query: str
    answer: str
    rating: str  # "like" or "dislike"
    message_id: str = None
    query_log_id: int = None  # Optional link to query_log entry

class ChatResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    sources: list = []
    warning: str | None = None
    message_id: str | None = None

class IngestRequest(BaseModel):
    collection: str  # policy / exam / psychology
    directory: str = None  # 可选，指定文档目录


# === API 路由 ===

@app.get("/")
async def root():
    """返回前端页面"""
    try:
        frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
        if frontend_path.exists():
            return FileResponse(frontend_path)
        return {"status": "ok", "message": "CQUPT AI Assistant API"}
    except Exception:
        return {"status": "ok", "message": "CQUPT AI Assistant API"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """核心接口：提问 → 意图识别 → 检索 → 生成回答"""
    start_time = time.time()
    try:
        _, answer_gen, hybrid_searcher, guardrails = get_components()
    except Exception as e:
        logger.exception("Component init failed")
        raise HTTPException(500, f"服务初始化失败: {e}")

    query = sanitize_input(req.query)
    message_id = str(uuid.uuid4())[:8]

    # 1. 输入检查
    try:
        ok, msg = guardrails.check_query(query)
        if not ok:
            if msg == "crisis":
                return ChatResponse(
                    answer=guardrails._crisis_response(),
                    intent="psychological",
                    confidence=1.0,
                    warning="高危关键词触发危机干预",
                    message_id=message_id,
                )
            elif msg in ("high_risk", "semantic_risk"):
                # 高风险或LLM语义检测 → 返回带热线的温和拦截
                hotline = config.get_school_attr("national_hotline", "400-161-9995")
                mental_center = config.get_school_attr("mental_health_center", "学校心理健康中心")
                return ChatResponse(
                    answer=(
                        "我注意到你的话里似乎有一些不太好的信号。"
                        "我只是AI，不能替代真正的关心和帮助。\n\n"
                        "如果你正在经历困难，请记得：\n"
                        f"**{mental_center}** 随时欢迎你去聊聊。\n"
                        f"**全国24小时心理援助热线：{hotline}**\n\n"
                        "有人愿意倾听，你不需要一个人面对这一切。"
                    ),
                    intent="psychological",
                    confidence=0.9,
                    warning="语义护栏检测到潜在风险表达",
                    message_id=message_id,
                )
            raise HTTPException(400, msg)
    except Exception as e:
        logger.exception("Guardrails check_query failed: %s", e)
        # Fail open for non-crisis queries

    # 2. 意图路由 + 检索
    try:
        result = hybrid_searcher.search(query)
        intent = result["intent"]
        strategy = result["strategy"]
        docs = result["retrieved_docs"]
    except Exception as e:
        logger.exception("Search failed: %s", e)
        raise HTTPException(500, f"检索失败: {e}")

    # 3. 生成回答
    try:
        answer = answer_gen.generate(
            query=query,
            retrieved_docs=docs,
            intent=intent,
            temperature=strategy["temperature"],
            require_citation=strategy["require_citation"],
            add_disclaimer=strategy.get("add_disclaimer", False),
        )
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        answer = f"抱歉，我暂时无法生成回答。请稍后重试。（错误: {e}）"

    # 4. 护栏校验
    try:
        ok, warning, answer = guardrails.check_response(
            query, answer, intent.value
        )
    except Exception as e:
        logger.exception("Guardrails check_response failed: %s", e)
        warning = None

    # 5. 提取来源
    sources = []
    try:
        for doc in docs[:3]:
            sources.append({
                "source": doc.get("metadata", {}).get("source", "未知"),
                "snippet": doc["content"][:100],
                "score": round(doc.get("score", 0), 3),
            })
    except Exception:
        pass

    # 6. 记录查询日志
    elapsed_ms = int((time.time() - start_time) * 1000)
    try:
        log_query(
            user_query=query,
            intent=intent.value,
            retrieved_docs=sources,
            response_preview=answer[:200],
            latency_ms=elapsed_ms,
        )
    except Exception as e:
        logger.warning("Failed to log query: %s", e)

    return ChatResponse(
        answer=answer,
        intent=intent.value,
        confidence=round(result["confidence"], 3),
        sources=sources,
        warning=warning,
        message_id=message_id,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式聊天端点——支持多步推理 + 自训练反思"""
    start_time = time.time()
    try:
        _, answer_gen, hybrid_searcher, guardrails = get_components()
    except Exception as e:
        logger.exception("Component init failed for stream")

        async def err_stream():
            yield f"服务初始化失败: {e}"
        return StreamingResponse(err_stream(), media_type="text/plain")

    # 0. 输入清洗 + LLM防护
    query = sanitize_input(req.query)
    try:
        ok, llm_block_reply = LLMGuard.check(query)
        if not ok:
            async def block_stream():
                yield llm_block_reply
            return StreamingResponse(block_stream(), media_type="text/plain")
    except Exception as e:
        logger.warning("LLM guard check failed: %s", e)

    # 1. 输入检查
    try:
        ok, code = guardrails.check_query(query)
        if not ok:
            if code == "crisis":
                async def crisis_stream():
                    yield guardrails._crisis_response()
                return StreamingResponse(crisis_stream(), media_type="text/plain")
            if code in ("high_risk", "semantic_risk"):
                hotline = config.get_school_attr("national_hotline", "400-161-9995")
                mental_center = config.get_school_attr("mental_health_center", "学校心理健康中心")
                async def risk_stream():
                    yield (
                        "我注意到你的话里似乎有一些不太好的信号。"
                        "我只是AI，不能替代真正的关心和帮助。\n\n"
                        "如果你正在经历困难，请记得：\n"
                        f"**{mental_center}** 随时欢迎你去聊聊。\n"
                        f"**全国24小时心理援助热线：{hotline}**\n\n"
                        "有人愿意倾听，你不需要一个人面对这一切。"
                    )
                return StreamingResponse(risk_stream(), media_type="text/plain")
            if code == "too_long":
                async def reject_stream():
                    yield "问题过长，请简洁描述（500字以内）"
                return StreamingResponse(reject_stream(), media_type="text/plain")
    except Exception as e:
        logger.exception("Guardrails check failed in stream: %s", e)

    # 2. 意图路由 + 检索
    result = None
    docs = []
    try:
        if req.multi_step:
            sub_queries = query_decomposer.decompose(req.query)
            if len(sub_queries) > 1:
                step_results = step_executor.execute(sub_queries)
                result = hybrid_searcher.search(req.query)
                docs = result["retrieved_docs"]
                seen = {d["content"][:80] for d in docs}
                for sr in step_results:
                    for d in sr["docs"]:
                        key = d["content"][:80]
                        if key not in seen:
                            seen.add(key)
                            docs.append(d)
            else:
                result = hybrid_searcher.search(req.query)
                docs = result["retrieved_docs"]
        else:
            result = hybrid_searcher.search(req.query)
            docs = result["retrieved_docs"]
    except Exception as e:
        logger.exception("Search failed in stream: %s", e)
        result = {"intent": type("Intent", (), {"value": "policy"}), "strategy": {"temperature": 0.3, "require_citation": True}}

    intent = result.get("intent")
    strategy = result.get("strategy", {"temperature": 0.3, "require_citation": True})

    # 3. 文档质量过滤 + 内部数据过滤
    filtered_docs = docs
    try:
        filtered_docs = [
            d for d in docs
            if d.get("metadata", {}).get("source") != "self-training-reflection"
            and d.get("metadata", {}).get("type") != "feedback"
        ]
        if filtered_docs and len(filtered_docs) > 3:
            try:
                filtered_docs, need_web = doc_grader.grade(req.query, filtered_docs)
            except Exception:
                pass
    except Exception as e:
        logger.warning("Document filtering failed: %s", e)

    # 4. 流式生成
    response_container = [""]

    def generate():
        try:
            for chunk in answer_gen.generate_stream(
                query=query,
                retrieved_docs=filtered_docs,
                intent=intent,
                temperature=strategy["temperature"],
                require_citation=strategy["require_citation"],
                add_disclaimer=strategy.get("add_disclaimer", False),
                history=req.history,
                multi_intents=result.get("multi_intents"),
            ):
                response_container[0] += chunk
                yield chunk
        except Exception as e:
            logger.exception("Stream generation failed: %s", e)
            yield f"\n\n[生成出错: {e}]"

        full_response = response_container[0]

        # 5. 幻觉检测 + 回答质量评估
        if req.self_train and full_response:
            try:
                is_grounded = hallucination_checker.check(filtered_docs, full_response)
                is_useful = answer_grader.grade(req.query, full_response)
                if not is_grounded:
                    yield "\n\n[检测到回答可能不完全基于知识库内容]"
                if not is_useful:
                    yield "\n\n[回答可能未完全解决您的问题]"
            except Exception:
                pass

        # 6. 护栏校验
        try:
            ok, warning, modified = guardrails.check_response(
                req.query, full_response, intent.value if intent else "policy"
            )
            if warning and ok:
                yield f"\n\n[注意: {warning}]"
        except Exception:
            pass

        # 7. 自训练反思（可选）
        if req.self_train and full_response:
            try:
                self_training.process(req.query, full_response, filtered_docs, intent=intent.value if intent else "policy")
            except Exception:
                pass

    # 8. 查询日志 + 自训练反思（后台完成，不阻塞响应）
    def finalize():
        elapsed_ms = int((time.time() - start_time) * 1000)
        full = response_container[0]
        try:
            # Log to conversation_log
            conn = get_db()
            conn.execute(
                "INSERT INTO conversation_log (query, answer_preview, intent, response_time_ms) VALUES (?, ?, ?, ?)",
                (query, full[:200], intent.value if intent else "unknown", elapsed_ms),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        # Log to query_log
        try:
            log_query(
                user_query=query,
                intent=intent.value if intent else "unknown",
                response_preview=full[:200],
                latency_ms=elapsed_ms,
            )
        except Exception:
            pass

        # 自训练反思
        if req.self_train and full:
            try:
                self_training.process(req.query, full, filtered_docs, intent=intent.value if intent else "policy")
            except Exception:
                pass

    return StreamingResponse(generate(), media_type="text/plain", background=finalize())


@app.post("/api/ingest")
async def ingest_documents(req: IngestRequest):
    """文档导入接口——非结构化入向量库，Excel 同时入 SQLite"""
    try:
        vs, _, _, _ = get_components()
    except Exception as e:
        raise HTTPException(500, f"组件初始化失败: {e}")

    doc_dir = Path(req.directory) if req.directory else config.DOCUMENTS_DIR
    if not doc_dir.exists():
        raise HTTPException(404, f"文档目录不存在: {doc_dir}")

    try:
        loader = DocumentLoader(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        chunks = loader.load_directory(doc_dir)
    except Exception as e:
        logger.exception("Document loading failed")
        raise HTTPException(500, f"文档加载失败: {e}")

    if not chunks:
        return {"status": "warning", "message": "未找到可导入的文档", "count": 0}

    try:
        vs.add_chunks(chunks, req.collection)
    except Exception as e:
        logger.exception("Chunk ingestion failed")
        raise HTTPException(500, f"文档导入失败: {e}")

    # NL2SQL：Excel 文件同步导入 SQLite
    nl2sql_result = None
    try:
        exam_db = get_exam_db()
        for file_path in sorted(doc_dir.iterdir()):
            if file_path.suffix.lower() in (".xlsx", ".xls"):
                nl2sql_result = exam_db.ingest_excel(file_path)
                logger.info("NL2SQL ingest: %s", nl2sql_result.get("message", ""))
    except Exception as e:
        logger.warning("NL2SQL ingest skipped: %s", e)

    return {
        "status": "ok",
        "message": f"成功导入 {len(chunks)} 个文档块到 {req.collection}",
        "count": len(chunks),
        "files": list(set(c.metadata["source"] for c in chunks)),
        "nl2sql": nl2sql_result,
    }


@app.get("/api/stats")
async def get_stats():
    """知识库统计——含向量库 + NL2SQL 双轨"""
    try:
        vs, _, _, _ = get_components()
        manifest_status = vs.get_manifest_status()

        # NL2SQL 统计
        nl2sql_stats = None
        try:
            exam_db = get_exam_db()
            nl2sql_stats = exam_db.get_stats()
        except Exception as e:
            logger.warning("NL2SQL stats unavailable: %s", e)

        return {
            "collections": vs.get_stats(),
            "embedding_model": "BM25 + jieba (纯Python)",
            "doubao_model": config.DOUBAO_MODEL,
            "manifest": manifest_status,
            "nl2sql": nl2sql_stats,
        }
    except Exception as e:
        logger.exception("Stats endpoint failed")
        raise HTTPException(500, f"获取统计失败: {e}")


# ===== 认证端点 =====

@app.post("/api/auth/register")
async def api_register(req: RegisterRequest):
    """用户注册"""
    try:
        return auth_register(req.username, req.email, req.password)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Registration failed")
        raise HTTPException(500, f"注册失败: {e}")


@app.post("/api/auth/login")
async def api_login(req: LoginRequest):
    """用户登录"""
    try:
        return auth_login(req.username, req.password)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Login failed")
        raise HTTPException(500, f"登录失败: {e}")


@app.get("/api/auth/me")
async def api_me(user: dict = Depends(require_user)):
    """获取当前用户信息"""
    return user


# ===== 反馈端点 =====

@app.post("/api/feedback")
async def api_feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    """记录用户反馈"""
    try:
        result = record_feedback(
            query=req.query,
            answer=req.answer,
            rating=req.rating,
            user_id=user["user_id"] if user else None,
        )
        # Also update query_log if message_id is provided
        if req.message_id:
            try:
                conn = get_db()
                conn.execute(
                    "UPDATE query_log SET feedback = ? WHERE id = ?",
                    (req.rating, req.message_id),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass
        return result
    except Exception as e:
        logger.exception("Feedback endpoint failed")
        raise HTTPException(500, f"记录反馈失败: {e}")


@app.post("/api/feedback/query-log")
async def api_feedback_with_log(req: FeedbackWithLogRequest, user: dict = Depends(get_current_user)):
    """记录用户反馈并关联到查询日志"""
    try:
        result = record_feedback(
            query=req.query,
            answer=req.answer,
            rating=req.rating,
            user_id=user["user_id"] if user else None,
        )
        # Update query_log with feedback
        if req.query_log_id:
            try:
                conn = get_db()
                conn.execute(
                    "UPDATE query_log SET feedback = ? WHERE id = ?",
                    (req.rating, req.query_log_id),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass
        return result
    except Exception as e:
        logger.exception("Feedback with log endpoint failed")
        raise HTTPException(500, f"记录反馈失败: {e}")


@app.get("/api/feedback/stats")
async def api_feedback_stats():
    """获取反馈统计"""
    try:
        return get_feedback_stats()
    except Exception as e:
        logger.exception("Feedback stats endpoint failed")
        raise HTTPException(500, f"获取反馈统计失败: {e}")


# ===== 分析端点 =====

@app.get("/api/analytics/queries")
async def api_query_analytics(days: int = 7):
    """查询分析：热门查询、延迟分布、意图分布"""
    try:
        return get_query_analytics(days=days)
    except Exception as e:
        logger.exception("Query analytics endpoint failed")
        raise HTTPException(500, f"获取分析数据失败: {e}")


@app.get("/api/analytics/summary")
async def api_analytics_summary():
    """轻量级分析摘要"""
    try:
        analytics = get_query_analytics(days=7)
        feedback = get_feedback_stats(days=7)
        return {
            "queries_7d": analytics.get("total_queries", 0),
            "avg_latency_ms": analytics.get("avg_latency_ms", 0),
            "intent_distribution": analytics.get("intent_distribution", {}),
            "feedback_satisfaction": feedback.get("satisfaction_rate", 0),
        }
    except Exception as e:
        logger.exception("Analytics summary endpoint failed")
        raise HTTPException(500, f"获取分析摘要失败: {e}")


# ===== 管理后台端点 =====

@app.get("/api/admin/dashboard")
async def api_dashboard(user: dict = Depends(require_user)):
    try:
        return get_dashboard()
    except Exception as e:
        logger.exception("Dashboard endpoint failed")
        raise HTTPException(500, f"获取仪表盘失败: {e}")


@app.get("/api/admin/users")
async def api_users(page: int = 1, user: dict = Depends(require_user)):
    try:
        return get_user_list(page=page)
    except Exception as e:
        logger.exception("User list endpoint failed")
        raise HTTPException(500, f"获取用户列表失败: {e}")


# ===== 内容管理端点 =====

@app.post("/api/admin/documents")
async def api_upload_document(
    file: UploadFile,
    collection: str = Form(...),
    user: dict = Depends(require_user),
):
    """上传文档并增量索引"""
    try:
        vs, _, _, _ = get_components()
    except Exception as e:
        raise HTTPException(500, f"组件初始化失败: {e}")

    try:
        result = upload_document(file, collection, user["user_id"], vs)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document upload failed")
        raise HTTPException(500, f"文档上传失败: {e}")


@app.get("/api/admin/documents")
async def api_list_documents(collection: str = None):
    try:
        return list_documents(collection)
    except Exception as e:
        logger.exception("Document list endpoint failed")
        raise HTTPException(500, f"获取文档列表失败: {e}")


@app.delete("/api/admin/documents/{doc_id}")
async def api_delete_document(doc_id: int, user: dict = Depends(require_user)):
    try:
        vs, _, _, _ = get_components()
    except Exception as e:
        raise HTTPException(500, f"组件初始化失败: {e}")

    try:
        return delete_document(doc_id, vs)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document delete failed")
        raise HTTPException(500, f"删除文档失败: {e}")


# ===== 增量文档索引端点 =====

@app.post("/api/admin/documents/incremental")
async def api_incremental_index(
    file: UploadFile,
    collection: str = Form(...),
    user: dict = Depends(require_user),
):
    """增量索引：单个文档上传并加入索引（跳过未变更文件）"""
    try:
        vs, _, _, _ = get_components()
    except Exception as e:
        raise HTTPException(500, f"组件初始化失败: {e}")

    # Save file temporarily
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {suffix}")

    dest_path = config.DOCUMENTS_DIR / file.filename
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        file_size = 0
        with open(dest_path, "wb") as f:
            while chunk := file.file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(400, f"文件过大，最大 {MAX_FILE_SIZE // 1024 // 1024}MB")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("File save failed for incremental ingest")
        raise HTTPException(500, f"文件保存失败: {e}")

    # Incremental add
    try:
        result = vs.add_document(str(dest_path), collection)
    except Exception as e:
        logger.exception("Incremental indexing failed")
        raise HTTPException(500, f"增量索引失败: {e}")

    # NL2SQL：Excel 文件同步入 SQLite
    nl2sql_result = None
    if suffix in (".xlsx", ".xls"):
        try:
            exam_db = get_exam_db()
            nl2sql_result = exam_db.ingest_excel(dest_path)
            logger.info("NL2SQL incremental: %s", nl2sql_result.get("message", ""))
        except Exception as e:
            logger.warning("NL2SQL incremental skipped: %s", e)

    # Record in database
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO documents (filename, collection, uploaded_by, file_size) VALUES (?, ?, ?, ?)",
            (file.filename, collection, user["user_id"], file_size),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Failed to record document in DB: %s", e)

    if nl2sql_result:
        result["nl2sql"] = nl2sql_result

    return result


@app.get("/api/admin/manifest")
async def api_manifest_status(user: dict = Depends(require_user)):
    """查看增量索引 manifest 状态"""
    try:
        vs, _, _, _ = get_components()
        return vs.get_manifest_status()
    except Exception as e:
        logger.exception("Manifest endpoint failed")
        raise HTTPException(500, f"获取manifest失败: {e}")


# ===== 泛雅学习通数据同步（预留接口） =====

@app.post("/api/fanya/sync")
async def api_fanya_sync(
    api_url: str = Form(None),
    api_key: str = Form(None),
    sync_type: str = Form("policy"),
    user: dict = Depends(require_user),
):
    """泛雅学习通平台数据同步——接口已预留，待泛雅 API 对接。

    当前返回接口状态和参数说明。正式对接时：
    1. 调用泛雅开放平台 API 拉取政策文档/考研数据
    2. 自动导入到向量库 + NL2SQL 数据库
    3. 返回同步摘要
    """
    try:
        adapter = FanyaAdapter()
        status = adapter.get_status()
        return {
            "status": "reserved",
            "message": "泛雅学习通数据同步接口已预留，等待泛雅开放平台 API 对接",
            "supported_sync_types": status["sync_types"],
            "required_params": status["required_params"],
            "note": "本接口为面向全国高校规模化部署的长远设计——对接泛雅学习通后，可自动拉取各校真实政策文档与考研数据",
        }
    except Exception as e:
        logger.exception("Fanya sync endpoint failed")
        raise HTTPException(500, f"泛雅接口异常: {e}")


@app.get("/api/fanya/status")
async def api_fanya_status():
    """查看泛雅适配器状态"""
    try:
        adapter = FanyaAdapter()
        return adapter.get_status()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ===== 学校信息（前端多校适配） =====

@app.get("/api/school/info")
async def api_school_info():
    """返回当前激活学校的信息——前端据此动态设置标题/名称。"""
    try:
        return {
            "name": config.get_school_attr("name", ""),
            "short_name": config.get_school_attr("short_name", ""),
            "assistant_name": config.get_school_attr("assistant_name", "小助手"),
            "assistant_greeting": config.get_school_attr("assistant_greeting", "你好！"),
            "subtitle": config.get_school_attr("subtitle", "学生成长一站式咨询"),
            "national_hotline": config.get_school_attr("national_hotline", "400-161-9995"),
        }
    except Exception as e:
        return {"error": str(e)}


# ===== 健康检查 =====

@app.get("/api/health")
async def api_health():
    try:
        return health_check()
    except Exception as e:
        logger.exception("Health check failed")
        return {"api": "error", "error": str(e)}


@app.get("/api/errors/recent")
async def api_recent_errors(minutes: int = 60):
    try:
        return error_tracker.get_recent_errors(minutes)
    except Exception as e:
        logger.exception("Recent errors endpoint failed")
        return []


# === 启动 ===
if __name__ == "__main__":
    import uvicorn
    import signal
    try:
        config.validate_config()
    except RuntimeError as e:
        logger.critical("配置错误: %s", e)
        sys.exit(1)
    logger.info("CQUPT RAG Assistant starting (PID: %d)", os.getpid())
    uvicorn.run(app, host="0.0.0.0", port=8000)
