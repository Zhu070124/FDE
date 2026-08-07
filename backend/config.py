"""
配置文件：豆包API、模型路径、知识库设置
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 自动加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# === 项目根目录 ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# === 豆包 API（火山引擎） ===
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_BASE_URL = os.getenv("DOUBAO_BASE_URL")
if not DOUBAO_BASE_URL:
    raise RuntimeError("DOUBAO_BASE_URL is required. Example: https://ark.cn-beijing.volces.com/api/v3")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL")  # 必填，无默认值——fail fast

def validate_config():
    """惰性校验：模块可安全导入，API调用前检查"""
    key = DOUBAO_API_KEY
    if not key or key == "your-api-key-here":
        raise RuntimeError(
            "DOUBAO_API_KEY 环境变量未设置。\n"
            "Linux/macOS: export DOUBAO_API_KEY=你的key\n"
            "Windows: set DOUBAO_API_KEY=你的key"
        )
    if not DOUBAO_MODEL:
        raise RuntimeError(
            "DOUBAO_MODEL 环境变量未设置。\n"
            "请设置火山引擎推理接入点ID，如: set DOUBAO_MODEL=ep-xxxx"
        )

# 仅在此模块被直接 import 且用于非测试场景时才校验
# 调用方在首次API调用前主动调用 validate_config()

# === 豆包 Embedding API（火山引擎） ===
DOUBAO_EMBEDDING_MODEL = os.getenv("DOUBAO_EMBEDDING_MODEL", "doubao-embedding-vision-251215")

# === 检索设置 ===
CHUNK_SIZE = 500       # 中文分块大小（字符）
CHUNK_OVERLAP = 50     # 重叠量
TOP_K_RETRIEVAL = 5    # 检索返回条数

# === 意图分类 ===
INTENT_LABELS = {
    "policy": "政策查询——需要精确匹配文档内容，标注来源",
    "data": "数据查询——需要从表格中提取数值，精确回答",
    "psychological": "心理支持——需要共情回应，标注免责提示",
}

# === 心理回复护栏 ===
PSYCHOLOGICAL_DISCLAIMER = (
    "\n\n---\n"
    "> 💚 我只是AI，能陪你聊天，但不能替代真正的心理咨询。"
    "如果你觉得需要更专业的帮助，学校心理健康中心随时为你开放，"
    "全国心理援助热线 **400-161-9995** 也是24小时免费的。"
)

# === 意图 → 知识库集合映射（vector_store.py / reflection_loop.py 统一引用） ===
INTENT_TO_COLLECTION = {
    "policy": "policy",
    "data": "exam",
    "psychological": "psychology",
}

# === 安全护栏配置（从 YAML 加载） ===
_safety_config: Optional[Dict[str, Any]] = None
SAFETY_CONFIG_PATH = PROJECT_ROOT / "config" / "safety.yaml"


def load_safety_config() -> Dict[str, Any]:
    """加载 safety.yaml 配置，带缓存"""
    global _safety_config
    if _safety_config is not None:
        return _safety_config

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, using built-in safety defaults")
        return {}

    path = SAFETY_CONFIG_PATH
    if not path.exists():
        logger.warning("safety.yaml not found at %s, using built-in defaults", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            _safety_config = yaml.safe_load(f) or {}
        logger.info("Loaded safety config from %s", path)
    except Exception as e:
        logger.error("Failed to load safety.yaml: %s", e)
        _safety_config = {}

    return _safety_config


def get_safety_layer(layer_name: str) -> Optional[Dict[str, Any]]:
    """Get a specific safety layer config, or None if disabled."""
    cfg = load_safety_config()
    layers = cfg.get("layers", {})
    layer = layers.get(layer_name, {})
    if layer.get("enabled", True) is False:
        return None
    return layer


def get_custom_blocklist() -> list:
    """Get custom blocklist words from safety config."""
    cfg = load_safety_config()
    return cfg.get("custom_blocklist", [])
