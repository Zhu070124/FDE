"""
SQLite 数据库：用户表、反馈表、内容管理表
"""
import sqlite3
import os
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"


def get_db() -> sqlite3.Connection:
    """获取数据库连接（自动创建表）"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """首次运行建表"""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user' CHECK(role IN ('user','admin')),
            created_at TEXT DEFAULT (datetime('now')),
            last_login TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            token TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            expires_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating TEXT CHECK(rating IN ('like','dislike')),
            intent TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            query TEXT NOT NULL,
            answer_preview TEXT,
            intent TEXT,
            response_time_ms INTEGER,
            error TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            collection TEXT NOT NULL CHECK(collection IN ('policy','exam','psychology')),
            uploaded_by INTEGER REFERENCES users(id),
            file_size INTEGER,
            uploaded_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            user_query TEXT NOT NULL,
            intent TEXT,
            retrieved_docs TEXT,
            response_preview TEXT,
            latency_ms INTEGER,
            feedback TEXT,
            user_id INTEGER REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
    import logging
    logging.getLogger(__name__).info("Database initialized")


if __name__ == "__main__":
    init_db()
