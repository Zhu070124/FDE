"""
安全加固：速率限制、输入清洗、登录锁定
"""
import re
import time
from collections import defaultdict
from fastapi import HTTPException, Request


class RateLimiter:
    """简易内存速率限制器（生产环境换Redis）"""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._store: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        """返回True=放行, False=限流"""
        now = time.time()
        cutoff = now - self.window
        self._store[key] = [t for t in self._store[key] if t > cutoff]

        if len(self._store[key]) >= self.max_requests:
            return False

        self._store[key].append(now)
        return True

    async def middleware(self, request: Request, call_next):
        """FastAPI中间件——按IP限流"""
        ip = request.client.host if request.client else "unknown"
        if not self.check(ip):
            raise HTTPException(429, "请求过于频繁，请稍后再试")
        return await call_next(request)


# 全局实例
rate_limiter = RateLimiter(max_requests=60, window_seconds=60)

# 登录限流（更严格）
login_limiter = RateLimiter(max_requests=10, window_seconds=300)


def sanitize_input(text: str) -> str:
    """输入清洗：移除控制字符，限制长度"""
    # 移除NULL和控制字符（保留换行）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 限制最大长度
    return text[:2000]
