"""
意图路由器：判断用户问题是 政策查询 / 数据查询 / 心理支持
"""
import re
from enum import Enum
from typing import Tuple, Optional


class Intent(Enum):
    POLICY = "policy"                # 政策查询——精确匹配文档
    DATA = "data"                    # 数据查询——结构化查表
    PSYCHOLOGICAL = "psychological"  # 心理支持——共情+免责


# === 规则层：关键词+正则快速匹配 ===
POLICY_KEYWORDS = [
    "政策", "规定", "流程", "条件", "怎么申请", "需要什么材料",
    "三方协议", "报到证", "档案", "户口", "报到", "派遣",
    "学费", "奖学金", "助学金", "助学贷款", "困难补助",
    "国家规定", "学校规定", "文件", "通知", "要求",
    "毕业", "就业", "创业", "补贴", "税收", "优惠",
    "考研报名", "考研条件", "报考资格", "招生简章",
]

DATA_KEYWORDS = [
    "分数线", "报录比", "录取人数", "多少人", "多少分",
    "排名", "第几", "多少名", "比例", "百分比",
    "去年", "今年", "往年", "历年",
    "哪个学校", "哪个专业", "计算机", "金融", "会计",
]

PSYCHOLOGICAL_KEYWORDS = [
    "压力", "焦虑", "抑郁", "失眠", "难过", "崩溃",
    "不开心", "怎么办", "迷茫", "不知道", "没动力",
    "心烦", "烦躁", "累", "撑不住", "想哭", "孤独",
    "精神", "情绪", "心理", "心态",
    "社恐", "自卑", "内卷", "躺平",
    "人际关系", "室友", "失恋", "分手", "孤独",
]

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
    re.compile(r"(?:怎么办|帮帮我|救救我)"),
]


class IntentRouter:
    """意图路由器：规则匹配 + LLM辅助判断"""

    # LLM意图分类的prompt（轻量，只分类不生成）
    CLASSIFY_PROMPT = """判断以下学生问题属于哪个类别，只回复一个词（policy/data/psychological）：

- policy: 查询政策、规定、流程、申请条件（如奖学金、助学金、三方协议、考研报名条件）
- data: 查询具体数字、分数线、报录比、排名（如"XX大学去年分数"、"XX专业多少人"）
- psychological: 表达情绪困扰、压力、焦虑、人际关系、寻求安慰（如"压力大"、"失眠"、"和室友吵架"、"想哭"）

注意：提到"考研"但问的是数字/分数线→data；提到"考研"但问的是情绪/压力→psychological。

学生问题：{query}
类别："""

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
        data_weak_signals = ["多少", "多少钱", "能贷多少"]
        if any(s in query for s in policy_strong_signals) and any(s in query for s in data_weak_signals):
            policy_score += 0.4  # 强行拉到政策侧

        # === 第四步：安全检查（高危关键词直接判定） ===
        crisis_keywords = ["不想活", "想死", "结束生命", "自杀", "自残", "活不下去", "结束这一切"]
        if any(kw in query for kw in crisis_keywords):
            return Intent.PSYCHOLOGICAL, 1.0

        # === 决策 ===
        scores = {
            Intent.POLICY: policy_score,
            Intent.DATA: data_score,
            Intent.PSYCHOLOGICAL: psych_score,
        }
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        total = sum(scores.values())

        # 归一化置信度
        if total > 0:
            confidence = best_score / total
        else:
            confidence = 0.33

        # === LLM兜底：规则置信度低或分数接近时用LLM ===
        if self.use_llm and self.llm_fn and (confidence < 0.6 or best_score < 0.15):
            try:
                llm_intent = self._llm_classify(query)
                if llm_intent:
                    return llm_intent, 0.8
            except Exception:
                pass  # LLM调用失败，用规则结果

        # 最低置信度兜底
        if best_score < 0.1:
            return Intent.POLICY, 0.3

        return best_intent, min(confidence, 0.99)

    def _llm_classify(self, query: str) -> Optional[Intent]:
        """调用LLM做意图分类（轻量版）"""
        prompt = self.CLASSIFY_PROMPT.format(query=query)
        result = self.llm_fn(prompt, temperature=0.0, max_tokens=10)
        result = result.strip().lower()

        if "psychological" in result or "心理" in result:
            return Intent.PSYCHOLOGICAL
        elif "data" in result or "数据" in result:
            return Intent.DATA
        elif "policy" in result or "政策" in result:
            return Intent.POLICY
        return None

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
