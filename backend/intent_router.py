"""
意图路由器：判断用户问题是 政策查询 / 数据查询 / 心理支持

关键词从 config/safety.yaml 的 intent_router 节点动态加载，
无需改代码即可扩展场景。
"""
import re
import os
from enum import Enum
from typing import List, Tuple, Optional

import yaml


class Intent(Enum):
    POLICY = "policy"                # 政策查询——精确匹配文档
    DATA = "data"                    # 数据查询——结构化查表
    PSYCHOLOGICAL = "psychological"  # 心理支持——共情+免责


# === 从 YAML 动态加载关键词（带硬编码兜底） ===
def _load_intent_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "safety.yaml")
    defaults = {
        "policy_keywords": ["政策", "规定", "奖学金", "助学金"],
        "data_keywords": ["分数线", "报录比", "录取人数"],
        "psychological_keywords": ["压力", "焦虑", "抑郁", "失眠"],
        "psychological_crisis": ["不想活", "活不下去"],
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ir = cfg.get("intent_router", {}) if cfg else {}
        return {
            "policy_keywords": ir.get("policy_keywords", defaults["policy_keywords"]),
            "data_keywords": ir.get("data_keywords", defaults["data_keywords"]),
            "psychological_keywords": ir.get("psychological_keywords", defaults["psychological_keywords"]),
            "psychological_crisis": ir.get("psychological_crisis", defaults["psychological_crisis"]),
        }
    except Exception:
        return defaults

_intent_cfg = _load_intent_config()
POLICY_KEYWORDS = _intent_cfg["policy_keywords"]
DATA_KEYWORDS = _intent_cfg["data_keywords"]
PSYCHOLOGICAL_KEYWORDS = _intent_cfg["psychological_keywords"]
PSYCHOLOGICAL_CRISIS = _intent_cfg["psychological_crisis"]

# === 正则模式 ===
DATA_PATTERNS = [
    re.compile(r"\d{4}年.*(?:分数线|报录比|录取)"),
    re.compile(r"(?:多少|几).*(?:分|人|名)"),
    re.compile(r"(?:历年|往年|去年|今年).*(?:分数|数据|录取)"),
]

POLICY_PATTERNS = [
    re.compile(r"(?:怎么|如何).*(?:办理|申请|报销|登记)"),
    re.compile(r"(?:需要|准备).*(?:材料|证件|证明)"),
]

PSYCHOLOGICAL_PATTERNS = [
    re.compile(r"我.*(?:压力|焦虑|难过|累|失眠|崩溃|不开心)"),
    re.compile(r"(?:帮帮我|救救我)"),
]  # 注：移除了"怎么办"（普通政策/数据问题的合法用词）


class IntentRouter:
    """意图路由器：规则匹配 + LLM辅助判断"""

    # LLM意图分类的prompt（轻量，只分类不生成）
    CLASSIFY_PROMPT = """判断以下学生问题属于哪个类别，只回复一个词（policy/data/psychological）：

- policy: 查询政策、规定、流程、申请条件（如奖学金、助学金、三方协议、考研报名条件）
- data: 查询具体数字、分数线、报录比、排名（如"XX大学去年分数"、"XX专业多少人"）
- psychological: 表达情绪困扰、压力、焦虑、人际关系、寻求安慰（如"压力大"、"失眠"、"和室友吵架"、"想哭"）

注意：提到"考研"但问的是数字/分数线→data；提到"考研"但问的是情绪/压力→psychological。

<query>
{query}
</query>

类别（仅回复 policy / data / psychological 三个词之一）："""

    def __init__(self, use_llm: bool = True, llm_fn=None):
        self.use_llm = use_llm
        self.llm_fn = llm_fn  # 外部注入的LLM调用函数

    def classify(self, query: str) -> Tuple[Intent, float]:
        """
        分类用户意图：规则优先 → 低置信度时LLM兜底
        返回: (Intent, confidence)
        """
        # === 第一步：关键词匹配 ===
        policy_score = self._keyword_score(query, POLICY_KEYWORDS)
        data_score = self._keyword_score(query, DATA_KEYWORDS)
        psych_score = self._keyword_score(query, PSYCHOLOGICAL_KEYWORDS)

        # === 第二步：正则加权 ===
        for pattern in DATA_PATTERNS:
            if pattern.search(query):
                data_score += 0.2
        for pattern in POLICY_PATTERNS:
            if pattern.search(query):
                policy_score += 0.2
        for pattern in PSYCHOLOGICAL_PATTERNS:
            if pattern.search(query):
                psych_score += 0.3

        # === 第三步：心理关键词权重最高（安全优先） ===
        psych_score *= 1.2

        # === 3.5步：政策强信号压制数字弱信号 ===
        # "助学贷款能贷多少" → policy（贷款是政策概念），不是data
        # "奖学金多少钱" → policy，不是data
        policy_strong_signals = ["贷款", "助学金", "奖学金", "补助", "减免", "三方协议", "报到证"]
        if any(s in query for s in policy_strong_signals) and "多少" in query:
            policy_score += 0.4  # 强行拉到政策侧

        # === 第四步：安全检查（高危关键词直接判定） ===
        crisis_keywords = ["不想活", "想死", "结束生命", "自杀", "自残", "活不下去", "结束这一切"]
        if any(kw in query for kw in crisis_keywords):
            return Intent.PSYCHOLOGICAL, 1.0

        # === 决策（平局时心理优先——安全侧倾斜） ===
        scores = {
            Intent.POLICY: policy_score,
            Intent.DATA: data_score,
            Intent.PSYCHOLOGICAL: psych_score,
        }
        # 平局打破：psychological > policy > data
        best_intent = max(scores, key=lambda i: (scores[i], i == Intent.PSYCHOLOGICAL, i == Intent.POLICY))
        best_score = scores[best_intent]
        total = sum(scores.values())

        # 归一化置信度 + 绝对证据门槛
        if total > 0:
            confidence = best_score / total
        else:
            confidence = 0.33
        # 绝对证据弱 → 压低置信度
        if best_score < 0.15:
            confidence = min(confidence, 0.5)

        # === LLM兜底：规则置信度低时用LLM ===
        if self.use_llm and self.llm_fn and (confidence < 0.6 or best_score < 0.15):
            try:
                llm_intent = self._llm_classify(query)
                if llm_intent:
                    return llm_intent, 0.8
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "LLM intent classification failed, using rule-based result", exc_info=True
                )

        # 最低置信度兜底 → 政策（覆盖面最广），非心理
        if best_score < 0.1:
            return Intent.POLICY, 0.3

        return best_intent, min(confidence, 0.99)

    def classify_multi(self, query: str, threshold: float = 0.15) -> List[Tuple[Intent, float]]:
        """
        多标签分类：返回所有置信度高于阈值的意图
        "考研和就业怎么选" → [(DATA, 0.5), (PSYCHOLOGICAL, 0.4)]
        """
        # 复用规则层的计分逻辑（不调LLM，保持轻量）
        policy_score = self._keyword_score(query, POLICY_KEYWORDS)
        data_score = self._keyword_score(query, DATA_KEYWORDS)
        psych_score = self._keyword_score(query, PSYCHOLOGICAL_KEYWORDS)

        for pattern in DATA_PATTERNS:
            if pattern.search(query):
                data_score += 0.2
        for pattern in POLICY_PATTERNS:
            if pattern.search(query):
                policy_score += 0.2
        for pattern in PSYCHOLOGICAL_PATTERNS:
            if pattern.search(query):
                psych_score += 0.3

        psych_score *= 1.2

        # 危机关键词 → 心理分数强制拉高
        if any(kw in query for kw in PSYCHOLOGICAL_CRISIS):
            psych_score = max(psych_score, 1.0)

        if any(s in query for s in ["贷款", "助学金", "奖学金", "补助", "减免", "三方协议", "报到证"]) and "多少" in query:
            policy_score += 0.4

        scores = [
            (Intent.POLICY, policy_score),
            (Intent.DATA, data_score),
            (Intent.PSYCHOLOGICAL, psych_score),
        ]

        results = [(intent, score) for intent, score in scores if score >= threshold]
        if not results:
            return [(Intent.POLICY, 0.3)]

        # 归一化
        total = sum(s for _, s in results)
        return [(i, s / total) for i, s in results]

    def _llm_classify(self, query: str) -> Optional[Intent]:
        """调用LLM做意图分类（轻量版）——严格解析，防止子串误匹配"""
        prompt = self.CLASSIFY_PROMPT.format(query=query)
        result = self.llm_fn(prompt, temperature=0.0, max_tokens=10)
        result = result.strip().lower().replace("。", "").replace(".", "")

        # 严格匹配：只接受精确的三个值
        mapping = {
            "policy": Intent.POLICY,
            "data": Intent.DATA,
            "psychological": Intent.PSYCHOLOGICAL,
        }
        return mapping.get(result)

    def _keyword_score(self, query: str, keywords: list) -> float:
        """关键词匹配得分"""
        hits = sum(1 for kw in keywords if kw in query)
        return hits / max(len(keywords), 1)

    def get_retrieval_strategy(self, intent: Intent) -> dict:
        """根据意图返回检索策略"""
        if intent == Intent.POLICY:
            return {
                "collections": ["policy"],
                "search_mode": "hybrid",     # 向量+关键词
                "require_citation": True,
                "temperature": 0.3,           # 严谨
            }
        elif intent == Intent.DATA:
            return {
                "collections": ["exam", "policy"],
                "search_mode": "structured",  # 优先结构查询
                "require_citation": True,
                "temperature": 0.1,           # 非常精确
            }
        else:  # PSYCHOLOGICAL
            return {
                "collections": ["psychology", "policy"],
                "search_mode": "semantic",    # 语义检索
                "require_citation": False,    # 心理回复不用引用
                "temperature": 0.7,           # 温暖自然
                "add_disclaimer": True,
            }


if __name__ == "__main__":
    router = IntentRouter()

    test_queries = [
        "三方协议怎么签？",
        "重邮计算机去年分数线多少？",
        "我最近压力很大，学不下去了怎么办",
        "国家助学金申请需要什么材料？",
        "重庆大学和西南大学哪个考研好考？",
        "宿舍太吵了睡不好，心态快崩了",
        "跨专业考研有什么条件？",
    ]

    for q in test_queries:
        intent, conf = router.classify(q)
        strategy = router.get_retrieval_strategy(intent)
        print(f"[{intent.value}] (conf={conf:.2f}) {q}")
        print(f"  → collections={strategy['collections']}, t={strategy['temperature']}")
