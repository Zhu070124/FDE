"""
管理后台 API：用户管理、数据统计、知识库概览
"""
from fastapi import HTTPException
from database import get_db
from feedback import get_feedback_stats
from monitoring import error_tracker


def get_user_list(page: int = 1, page_size: int = 20) -> dict:
    """用户列表（分页）"""
    conn = get_db()
    offset = (page - 1) * page_size
    users = conn.execute(
        "SELECT id, username, email, role, created_at, last_login FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (page_size, offset),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return {
        "users": [dict(u) for u in users],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


def get_dashboard() -> dict:
    """管理后台仪表盘概览"""
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_conversations = conn.execute("SELECT COUNT(*) FROM conversation_log").fetchone()[0]
    total_documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()

    feedback = get_feedback_stats(days=7)
    recent_errors = len(error_tracker.get_recent_errors(minutes=60))

    return {
        "users": total_users,
        "conversations": total_conversations,
        "documents": total_documents,
        "feedback": feedback,
        "recent_errors_1h": recent_errors,
    }
