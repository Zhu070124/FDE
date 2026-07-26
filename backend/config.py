"""
配置文件：豆包API、模型路径、知识库设置
"""
import os
from pathlib import Path

# === 项目根目录 ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# === 豆包 API（火山引擎） ===
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "your-api-key-here")
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL = "ep-m-20260724185722-4b8pb"  # 火山引擎推理接入点

# === 豆包 Embedding API（火山引擎） ===
DOUBAO_EMBEDDING_MODEL = "doubao-embedding-vision-251215"
DOUBAO_EMBEDDING_DIM = 2048  # 向量维度

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
    "> ⚠️ **免责提示**：本回复仅供参考，AI助手无法替代专业心理咨询。"
    "如果你正经历严重的情绪困扰，请及时联系学校心理健康中心或拨打"
    "全国心理援助热线：400-161-9995。"
)

# === 知识库集合名称 ===
COLLECTION_POLICY = "policy_docs"           # 就业政策文档
COLLECTION_EXAM = "exam_data"               # 考研数据（结构化）
COLLECTION_PSYCHOLOGY = "psychology_docs"   # 心理知识文档
