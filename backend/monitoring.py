"""
运维监控：健康检查 + 错误日志 + 飞书通知
"""
import time
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("cqupt.monitoring")

LOG_DIR = Path(__file__).parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ErrorTracker:
    """错误追踪器——记录到文件，批量推送"""

    def __init__(self):
        self.error_log_path = LOG_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.error_count = 0
        self.last_notify_time = 0

    def track(self, error_type: str, detail: str, endpoint: str = ""):
        """记录一条错误"""
        self.error_count += 1
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "detail": detail[:500],
            "endpoint": endpoint,
        }
        with open(self.error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 批量推送间隔（避免频繁通知）
        now = time.time()
        if now - self.last_notify_time > 300 and self.error_count >= 5:
            logger.warning(f"⚠️ 最近5分钟 {self.error_count} 个错误")
            self.last_notify_time = now

    def get_recent_errors(self, minutes: int = 60) -> list:
        """获取最近N分钟的错误"""
        if not self.error_log_path.exists():
            return []
        cutoff = datetime.now().timestamp() - minutes * 60
        errors = []
        with open(self.error_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                    if ts >= cutoff:
                        errors.append(entry)
                except json.JSONDecodeError:
                    pass
        return errors


# 全局实例
error_tracker = ErrorTracker()


def health_check() -> dict:
    """深度健康检查（不只是返回OK）"""
    checks = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat(),
    }

    # 检查豆包API连通性（可选，生产环境启用）
    try:
        import httpx
        r = httpx.get("https://ark.cn-beijing.volces.com/api/v3", timeout=5)
        checks["doubao_api"] = "reachable" if r.status_code < 500 else "degraded"
    except Exception:
        checks["doubao_api"] = "unreachable"

    # 检查数据库
    try:
        from database import get_db
        conn = get_db()
        conn.execute("SELECT 1")
        conn.close()
        checks["database"] = "connected"
    except Exception:
        checks["database"] = "error"

    return checks
