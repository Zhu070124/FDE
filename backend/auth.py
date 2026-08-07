"""
JWT认证系统：注册、登录、令牌刷新、中间件
"""
import bcrypt
import jwt
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database import get_db

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is required. Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'")
JWT_EXPIRY_HOURS = 24

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "令牌已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "无效的令牌")


def register(username: str, email: str, password: str) -> dict:
    """注册新用户"""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM users WHERE username=? OR email=?", (username, email)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "用户名或邮箱已存在")

    pw_hash = hash_password(password)
    conn.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
        (username, email, pw_hash),
    )
    conn.commit()
    user = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    conn.close()

    token = create_token(user["id"], username)
    return {"user_id": user["id"], "username": username, "token": token}


def login(username: str, password: str) -> dict:
    """用户登录"""
    conn = get_db()
    user = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username=? OR email=?",
        (username, username),
    ).fetchone()

    if not user or not verify_password(password, user["password_hash"]):
        conn.close()
        raise HTTPException(401, "用户名或密码错误")

    conn.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.utcnow().isoformat(), user["id"]))
    conn.commit()
    conn.close()

    token = create_token(user["id"], user["username"])
    return {"user_id": user["id"], "username": user["username"], "token": token}


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """从请求中提取当前用户（可选——未登录也放行，返回None）"""
    if not credentials:
        return None
    try:
        payload = decode_token(credentials.credentials)
        return {"user_id": payload["user_id"], "username": payload["username"]}
    except HTTPException:
        return None


async def require_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """强制要求登录——未登录返回401"""
    if not credentials:
        raise HTTPException(401, "请先登录")
    return decode_token(credentials.credentials)
