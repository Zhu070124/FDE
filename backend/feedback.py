"""
反馈闭环：like/dislike评价 + 反馈统计 + 查询日志分析
"""
import json
import logging
from datetime import datetime
from database import get_db

logger = logging.getLogger(__name__)


def record_feedback(
    query: str,
    answer: str,
    rating: str,
    intent: str = None,
    user_id: int = None,
) -> dict:
    """记录用户反馈（like/dislike）"""
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO feedback (user_id, query, answer, rating, intent) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, answer, rating, intent),
        )
        conn.commit()
        conn.close()
        return {"status": "ok", "rating": rating}
    except Exception as e:
        logger.exception("Failed to record feedback: %s", e)
        return {"status": "error", "message": str(e)}


def get_feedback_stats(days: int = 7) -> dict:
    """获取反馈统计（默认最近7天）"""
    try:
        conn = get_db()
        likes = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating='like' AND created_at >= datetime('now', ?)",
            (f"-{days} days",),
        ).fetchone()[0]
        dislikes = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating='dislike' AND created_at >= datetime('now', ?)",
            (f"-{days} days",),
        ).fetchone()[0]
        total = likes + dislikes
        conn.close()
        return {
            "likes": likes,
            "dislikes": dislikes,
            "total": total,
            "satisfaction_rate": round(likes / total * 100, 1) if total > 0 else 0,
            "period_days": days,
        }
    except Exception as e:
        logger.exception("Failed to get feedback stats: %s", e)
        return {"likes": 0, "dislikes": 0, "total": 0, "satisfaction_rate": 0, "period_days": days}


# ===== Query Log =====

def log_query(
    user_query: str,
    intent: str = None,
    retrieved_docs: list = None,
    response_preview: str = None,
    latency_ms: int = None,
    feedback: str = None,
    user_id: int = None,
) -> int:
    """Log a query to the query_log table. Returns the inserted row id."""
    try:
        conn = get_db()
        cursor = conn.execute(
            """INSERT INTO query_log
               (user_query, intent, retrieved_docs, response_preview, latency_ms, feedback, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_query,
                intent,
                json.dumps(retrieved_docs, ensure_ascii=False) if retrieved_docs else None,
                (response_preview[:500] if response_preview else None),
                latency_ms,
                feedback,
                user_id,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id
    except Exception as e:
        logger.exception("Failed to log query: %s", e)
        return -1


def get_query_analytics(days: int = 7) -> dict:
    """Get query log analytics: top queries, avg latency, intent distribution."""
    try:
        conn = get_db()

        # Total queries in period
        total = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE timestamp >= datetime('now', ?)",
            (f"-{days} days",),
        ).fetchone()[0]

        # Avg latency
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) FROM query_log WHERE latency_ms IS NOT NULL AND timestamp >= datetime('now', ?)",
            (f"-{days} days",),
        ).fetchone()[0]
        avg_latency = round(avg_latency, 1) if avg_latency else 0

        # Intent distribution
        intent_rows = conn.execute(
            """SELECT intent, COUNT(*) as cnt FROM query_log
               WHERE intent IS NOT NULL AND timestamp >= datetime('now', ?)
               GROUP BY intent ORDER BY cnt DESC""",
            (f"-{days} days",),
        ).fetchall()
        intent_dist = {row["intent"]: row["cnt"] for row in intent_rows}

        # Top queries (excluding empty)
        top_queries = conn.execute(
            """SELECT user_query, COUNT(*) as cnt FROM query_log
               WHERE timestamp >= datetime('now', ?) AND user_query != ''
               GROUP BY user_query ORDER BY cnt DESC LIMIT 10""",
            (f"-{days} days",),
        ).fetchall()
        top = [{"query": row["user_query"][:80], "count": row["cnt"]} for row in top_queries]

        # Feedback distribution
        feedback_likes = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE feedback='like' AND timestamp >= datetime('now', ?)",
            (f"-{days} days",),
        ).fetchone()[0]
        feedback_dislikes = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE feedback='dislike' AND timestamp >= datetime('now', ?)",
            (f"-{days} days",),
        ).fetchone()[0]

        conn.close()

        return {
            "total_queries": total,
            "avg_latency_ms": avg_latency,
            "intent_distribution": intent_dist,
            "top_queries": top,
            "feedback": {
                "likes": feedback_likes,
                "dislikes": feedback_dislikes,
            },
            "period_days": days,
        }
    except Exception as e:
        logger.exception("Failed to get query analytics: %s", e)
        return {"error": str(e), "total_queries": 0}
