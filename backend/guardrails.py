"""
护栏系统：输出安全校验 + 来源验证 + 心理回复保护
支持从 config/safety.yaml 加载可配置规则
"""
import re
import logging
from typing import Tuple, Optional

import config

logger = logging.getLogger(__name__)


class Guardrails:
    """输出护栏：多层校验——危机分三级，不搞一刀切"""

    # 一级：立刻干预（主动自伤/自杀计划）
    # Values loaded from YAML; these are fallback defaults
    CRISIS_IMMEDIATE = [
        "我要自杀", "我要跳楼", "割腕自杀", "吃安眠药自杀",
        "结束生命", "自残", "不想活了",
    ]

    # 二级：高风险表达
    CRISIS_HIGH_RISK = [
        "想死", "活不下去", "一了百了", "活着没意义",
        "再见这个世界", "我走了（永别）",
    ]

    # 禁止输出的内容模式
    BLOCK_PATTERNS = [
        re.compile(r"建议[^。！？\n]{0,20}服用[^。！？\n]{0,10}药"),
        re.compile(r"你(很可能|应该|一定|可能)患有"),
    ]

    # 热线常量
    HOTLINE_PREFIX = "400-161"

    def __init__(self):
        self._load_config()

    def _load_config(self):
        """从 safety.yaml 加载配置，合并到实例属性"""
        try:
            crisis_layer = config.get_safety_layer("crisis")
            if crisis_layer:
                if crisis_layer.get("immediate_keywords"):
                    self.CRISIS_IMMEDIATE = crisis_layer["immediate_keywords"]
                if crisis_layer.get("high_risk_keywords"):
                    self.CRISIS_HIGH_RISK = crisis_layer["high_risk_keywords"]
                self.crisis_sensitivity = crisis_layer.get("sensitivity", "medium")
                self.crisis_enabled = True
                logger.info("Crisis layer: enabled, sensitivity=%s", self.crisis_sensitivity)
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

    def _assess_crisis(self, query: str) -> Tuple[str, str]:
        """
        三级危机评估（归一化输入防止空格/标点绕过）
        返回: (level, detail)
        """
        try:
            normalized = re.sub(r"[\s\W_]+", "", query)
            for kw in self.CRISIS_IMMEDIATE:
                if kw in normalized:
                    return "immediate", f"检测到紧急高危表达: {kw}"

            if self.crisis_sensitivity in ("medium", "high"):
                for kw in self.CRISIS_HIGH_RISK:
                    if kw in normalized:
                        return "high_risk", f"检测到高风险表达: {kw}"
            return "none", ""
        except Exception as e:
            logger.exception("Crisis assessment failed: %s", e)
            return "none", ""

    def _crisis_response(self) -> str:
        """一级危机干预回复"""
        hotline = "400-161-9995"
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
            "**重庆邮电大学心理健康中心**：校内咨询（请查看学校官网获取地址和预约方式）\n\n"
            "**你不需要一个人面对这一切。请现在就打电话。**"
        )

    def _fallback(self, intent: str) -> str:
        """非危机场景的兜底回复"""
        if intent == "psychological":
            hotline = "400-161-9995"
            return (
                "抱歉，这个问题超出了我能提供建议的范围。\n\n"
                "如果你正经历情绪困扰，建议联系学校心理健康中心，"
                f"或拨打全国心理援助热线：{hotline}。"
            )
        return "抱歉，我暂时无法回答这个问题。请咨询学校相关部门获取准确信息。"

    def check_query(self, query: str) -> Tuple[bool, str]:
        """输入侧检测——危机优先于长度。返回 (ok, machine_code)"""
        try:
            if self.crisis_enabled:
                level, _ = self._assess_crisis(query)
                if level == "immediate":
                    return False, "crisis"
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
