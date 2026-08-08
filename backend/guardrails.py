"""
护栏系统：输出安全校验 + 来源验证 + 心理回复保护
支持从 config/safety.yaml 加载可配置规则
五层防线：显性关键词 → 高风险关键词 → 隐晦正则 → LLM语义 → 医疗拦截
"""
import re
import json
import logging
from typing import Tuple, Optional, Callable

import config

logger = logging.getLogger(__name__)

# LLM 语义护栏——兜底规则漏掉的谐音/拼音/英文/隐喻
SEMANTIC_CHECK_PROMPT = """你是一个安全检测器。判断以下学生输入是否表达了自杀、自残或轻生倾向。

注意识别以下隐蔽表达：
- 谐音：紫砂=自杀、跳篓=跳楼、哥腕=割腕
- 拼音：zisha、tiaolou
- 英文：suicide、kill myself、end my life
- 隐喻：想睡很久不醒来、变成星星、去另一个世界、陪已故亲人、很高的地方看风景
- 睡眠隐喻：永远不醒来、一直睡下去
- 药物试探：吃多少片会死、什么药能让人醒不来

学生输入：
{query}

只输出JSON：
{{"risk": true/false, "reason": "一句话说明（选填）"}}

只输出JSON，不要任何其他内容。"""


class Guardrails:
    """输出护栏：五层校验——规则逐层收窄，LLM兜底"""

    # 一级：立刻干预（主动自伤/自杀计划）
    # Values loaded from YAML; these are fallback defaults
    CRISIS_IMMEDIATE = [
        "自杀", "跳楼", "割腕", "吃安眠药",
        "结束生命", "自残", "不想活了",
        "我要死", "上吊", "同归于尽", "活够了",
    ]

    # 二级：高风险表达
    CRISIS_HIGH_RISK = [
        "想死", "活不下去", "一了百了", "活着没意义",
        "再见这个世界", "我走了（永别）",
        "生不如死", "死了一了百了",
        "另一个世界", "离开这个世界", "去天堂吧",
        "下辈子", "来生", "活着好累",
    ]

    # 二级隐晦模式：需要上下文约束的轻生隐喻（正则匹配）
    CRISIS_HIGH_RISK_PATTERNS = [
        re.compile(r"好想.*(?:另一个世界|去那边|离开这里|消失|不存在)"),
        re.compile(r"(?:如果|要是|觉得|感觉|也许|可能).*没有我.*(?:更好|更轻松|更快乐|无所谓)"),
        re.compile(r"(?:我.*消失了|消失.*我).*(?:记得|怀念|想)"),
        re.compile(r"那边.*(?:风景|世界|生活).*(?:怎么样|会不会|是不是|一定)"),
        re.compile(r"(?:活着|活|过得|真的).*好?累[，。！？\s]*(?:不知道|还能|怎么|什么|多久|坚持)"),
        re.compile(r"(?:不想|不要|不想再).*(?:待|留|活)在.*(?:这里|世上|这个世界)"),
        re.compile(r"(?:我|真的|已经).*(?:撑不住|撑不下去|受不了|熬不住|坚持不住)"),
        re.compile(r"要是.*(?:能|可以).*(?:消失|离开|不再醒来|一觉不醒)"),
        re.compile(r"(?:不知道|不确定).*(?:还能|可以).*(?:坚持|撑|熬|活)多久"),
    ]

    # 禁止输出的内容模式
    BLOCK_PATTERNS = [
        re.compile(r"(建议|推荐|提议)[^。！？\n]{0,30}(服用|吃|购买|试试|用)[^。！？\n]{0,30}"),
        re.compile(r"你(很可能|应该|一定|可能)患有"),
        re.compile(r"(推荐|建议)[^。！？\n]{0,20}(药|药物|药品|用药)"),
    ]

    # 热线常量
    HOTLINE_PREFIX = "400-161"

    def __init__(self):
        self.llm_fn: Optional[Callable] = None  # LLM调用函数 (prompt, temp, max_tokens) -> str
        self.semantic_enabled: bool = False
        self._load_config()

    def set_llm_fn(self, fn: Callable):
        """注入LLM调用函数，启用语义护栏"""
        self.llm_fn = fn
        self.semantic_enabled = True
        logger.info("Semantic guard enabled (LLM-powered)")

    def _load_config(self):
        """从 safety.yaml 加载配置，合并到实例属性"""
        try:
            crisis_layer = config.get_safety_layer("crisis")
            if crisis_layer:
                if crisis_layer.get("immediate_keywords"):
                    self.CRISIS_IMMEDIATE = crisis_layer["immediate_keywords"]
                if crisis_layer.get("high_risk_keywords"):
                    self.CRISIS_HIGH_RISK = crisis_layer["high_risk_keywords"]
                # 隐晦模式：用正则不用裸关键词
                pattern_strs = crisis_layer.get("high_risk_patterns", [])
                if pattern_strs:
                    self.CRISIS_HIGH_RISK_PATTERNS = [re.compile(p) for p in pattern_strs]
                self.crisis_sensitivity = crisis_layer.get("sensitivity", "medium")
                self.crisis_enabled = True
                logger.info("Crisis layer: enabled, sensitivity=%s, subtle_patterns=%d",
                            self.crisis_sensitivity, len(self.CRISIS_HIGH_RISK_PATTERNS))
            else:
                self.crisis_enabled = False
                self.crisis_sensitivity = "medium"
                logger.info("Crisis layer: disabled")

            # Medical layer
            medical_layer = config.get_safety_layer("medical")
            if medical_layer:
                self.medical_enabled = True
                self.medical_sensitivity = medical_layer.get("sensitivity", "high")
                # Rebuild block patterns from config
                pattern_strs = medical_layer.get("block_patterns", [])
                if pattern_strs:
                    self.BLOCK_PATTERNS = [re.compile(p) for p in pattern_strs]
                # Custom blocklist words
                extra_words = medical_layer.get("blocklist_words", [])
                if extra_words:
                    for word in extra_words:
                        self.BLOCK_PATTERNS.append(re.compile(re.escape(word)))
                logger.info("Medical layer: enabled, sensitivity=%s", self.medical_sensitivity)
            else:
                self.medical_enabled = False
                logger.info("Medical layer: disabled")

            # Disclaimer layer
            self.disclaimer_enabled = config.get_safety_layer("disclaimer") is not None

            # Citation layer
            citation_layer = config.get_safety_layer("citation")
            self.citation_enabled = citation_layer is not None
            self.warn_missing_citation = (
                citation_layer.get("warn_missing_citation", True)
                if citation_layer else True
            )

            # Custom blocklist
            self.custom_blocklist = config.get_custom_blocklist()

        except Exception as e:
            logger.error("Failed to load safety config: %s, using defaults", e)
            self.crisis_enabled = True
            self.medical_enabled = True
            self.disclaimer_enabled = True
            self.citation_enabled = True
            self.warn_missing_citation = True
            self.custom_blocklist = []

    def check_response(
        self, query: str, response: str, intent: str
    ) -> Tuple[bool, Optional[str], str]:
        """
        校验回复
        返回: (通过?, 警告信息, 修改后的回复)
        """
        try:
            modified = response

            # === 1. 危机分级检测 ===
            if self.crisis_enabled:
                level, detail = self._assess_crisis(query)
                if level == "immediate":
                    return False, detail, self._crisis_response()
                elif level == "high_risk":
                    if "热线" not in modified and "400-161" not in modified:
                        modified += (
                            "\n\n---\n"
                            "> 如果你正在经历困难时刻，请记住："
                            "全国24小时心理援助热线 **400-161-9995**，"
                            "有人愿意倾听。"
                        )

            # === 2. 心理回复：确保有边界说明 ===
            if self.disclaimer_enabled and intent == "psychological":
                if "不能替代" not in modified and self.HOTLINE_PREFIX not in modified:
                    try:
                        disclaimer_layer = config.get_safety_layer("disclaimer")
                        if disclaimer_layer and disclaimer_layer.get("auto_append", True):
                            disclaimer_text = disclaimer_layer.get("text", config.PSYCHOLOGICAL_DISCLAIMER)
                            modified += "\n\n---\n" + disclaimer_text
                        else:
                            modified += config.PSYCHOLOGICAL_DISCLAIMER
                    except Exception:
                        modified += config.PSYCHOLOGICAL_DISCLAIMER

            # === 3. 禁止内容检测 ===
            if self.medical_enabled:
                for pattern in self.BLOCK_PATTERNS:
                    if pattern.search(response):
                        medical_sens = getattr(self, "medical_sensitivity", "high")
                        if medical_sens == "high":
                            return False, "回复包含不适当内容（医疗建议/诊断），已拦截", self._fallback(intent)
                        elif medical_sens == "medium":
                            logger.warning("Medical pattern matched (medium sensitivity, not blocking): %s", pattern.pattern)

            # Custom blocklist
            if self.custom_blocklist:
                for word in self.custom_blocklist:
                    if word in response:
                        return False, f"回复包含被屏蔽内容，已拦截", self._fallback(intent)

            # === 4. 政策/数据：检查来源标注 ===
            if self.citation_enabled and self.warn_missing_citation:
                if intent in ("policy", "data"):
                    if "来源" not in response and "【参考" not in response:
                        logger.warning("policy/data 回复缺少来源标注: %s", response[:100])

            return True, None, modified

        except Exception as e:
            logger.exception("Guardrails check_response failed: %s", e)
            return True, None, response  # Fail open — don't block on guardrail error

    def _semantic_check(self, query: str) -> Tuple[bool, str]:
        """LLM语义护栏——兜底规则漏掉的谐音/拼音/英文/隐喻/文化引用。

        只在规则层返回 level='none' 时触发，cost-controlled。
        返回: (is_risk, reason)
        """
        if not self.llm_fn:
            return False, ""

        prompt = SEMANTIC_CHECK_PROMPT.format(query=query[:500])
        try:
            raw = self.llm_fn(prompt, temperature=0.0, max_tokens=64)
            raw = raw.strip()
            # 提取JSON
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
            is_risk = result.get("risk", False)
            reason = result.get("reason", "")
            return is_risk, reason
        except json.JSONDecodeError:
            # LLM没输出JSON → 按内容判断
            lowered = raw.lower()
            if any(w in lowered for w in ["true", "risk", "yes", "是", "危险"]):
                return True, "semantic fallback match"
            return False, ""
        except Exception as e:
            logger.warning("Semantic check error: %s", e)
            return False, ""

    def _assess_crisis(self, query: str) -> Tuple[str, str]:
        """
        三级危机评估（归一化输入防止空格/标点绕过）
        返回: (level, detail)

        检测优先级：
        1. immediate 关键词 → BLOCK
        2. high_risk 关键词 → WARN + hotline
        3. high_risk 隐晦正则（诗意化/隐喻化轻生表达）→ WARN + hotline
        """
        try:
            normalized = re.sub(r"[\s\W_]+", "", query)

            # === 一级：紧急关键词 ===
            for kw in self.CRISIS_IMMEDIATE:
                if kw in normalized:
                    return "immediate", f"检测到紧急高危表达: {kw}"

            # === 二级：高风险关键词 ===
            if self.crisis_sensitivity in ("medium", "high"):
                for kw in self.CRISIS_HIGH_RISK:
                    if kw in normalized:
                        return "high_risk", f"检测到高风险表达: {kw}"

                # === 二级增强：隐晦轻生模式（正则匹配） ===
                for pattern in self.CRISIS_HIGH_RISK_PATTERNS:
                    if pattern.search(query):  # 注意：此处用原始query而非normalized，保留标点语义
                        return "high_risk", "检测到隐晦轻生表达（诗意化/隐喻化），触发高风险警告"

            return "none", ""
        except Exception as e:
            logger.exception("Crisis assessment failed: %s", e)
            return "none", ""
        except Exception as e:
            logger.exception("Crisis assessment failed: %s", e)
            return "none", ""

    def _crisis_response(self) -> str:
        """一级危机干预回复——使用当前学校配置"""
        hotline = config.get_school_attr("national_hotline", "400-161-9995")
        mental_center = config.get_school_attr("mental_health_center", "学校心理健康中心")
        mental_contact = config.get_school_attr("mental_health_contact", "校内咨询")
        # 也可从 safety.yaml 覆盖热线
        try:
            crisis_layer = config.get_safety_layer("crisis")
            if crisis_layer:
                hotline = crisis_layer.get("hotline", hotline)
        except Exception:
            pass
        return (
            "我注意到你可能正在经历非常困难的时刻。"
            "我不是专业的心理咨询师，不能提供真正的危机干预。\n\n"
            "**请立即联系以下资源，有受过训练的人能帮你：**\n\n"
            f"**全国24小时心理危机干预热线**：{hotline}\n"
            f"**希望24热线**：{hotline}\n"
            f"**{mental_center}**：{mental_contact}\n\n"
            "**你不需要一个人面对这一切。请现在就打电话。**"
        )

    def _fallback(self, intent: str) -> str:
        """非危机场景的兜底回复——使用当前学校配置"""
        hotline = config.get_school_attr("national_hotline", "400-161-9995")
        mental_center = config.get_school_attr("mental_health_center", "学校心理健康中心")
        if intent == "psychological":
            return (
                "抱歉，这个问题超出了我能提供建议的范围。\n\n"
                f"如果你正经历情绪困扰，建议联系{mental_center}，"
                f"或拨打全国心理援助热线：{hotline}。"
            )
        return "抱歉，我暂时无法回答这个问题。请咨询学校相关部门获取准确信息。"

    def check_query(self, query: str) -> Tuple[bool, str]:
        """输入侧检测——规则优先 → LLM语义兜底。返回 (ok, machine_code)"""
        try:
            if self.crisis_enabled:
                level, detail = self._assess_crisis(query)
                if level == "immediate":
                    return False, "crisis"
                if level == "high_risk":
                    return False, "high_risk"  # 高风险也拦截，返回热线警告

            # === 规则层全部放行 → LLM语义兜底 ===
            if self.semantic_enabled and self.llm_fn:
                try:
                    is_risk, reason = self._semantic_check(query)
                    if is_risk:
                        logger.info("Semantic guard caught: %s", reason)
                        return False, "semantic_risk"
                except Exception as e:
                    logger.warning("Semantic check failed (fail open): %s", e)

            if len(query) > 500:
                return False, "too_long"
            return True, ""
        except Exception as e:
            logger.exception("check_query failed: %s", e)
            return True, ""  # Fail open


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from logger_config import get_logger
    get_logger(__name__)

    g = Guardrails()

    tests = [
        ("我想自杀，活不下去了", "心理回复……"),
        ("考研压力好大，失眠怎么办", "你可以试试深呼吸……"),
        ("三方协议怎么签", "三方协议签订流程：1.……来源：《就业指南》"),
    ]

    for query, resp in tests:
        ok, warn, modified = g.check_response(query, resp, "psychological")
        print(f"OK={ok} | {warn or '通过'} | {modified[:60]}...")
