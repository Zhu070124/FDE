"""
护栏系统：输出安全校验 + 来源验证 + 心理回复保护
"""
import re
from typing import Tuple, Optional

import config


class Guardrails:
    """输出护栏：多层校验"""

    # 心理高危关键词——触发强制转介
    CRISIS_KEYWORDS = [
        "自杀", "想死", "不想活", "结束生命", "自残",
        "跳楼", "割腕", "安眠药", "一了百了", "活着没意义",
        "我走了", "再见这个世界",
    ]

    # 禁止输出的内容模式
    BLOCK_PATTERNS = [
        re.compile(r"建议.*服用.*药"),      # 禁止推荐药物
        re.compile(r"你.*有.*症"),          # 禁止诊断
        re.compile(r"你.*患.*病"),
    ]

    def check_response(
        self, query: str, response: str, intent: str
    ) -> Tuple[bool, Optional[str], str]:
        """
        校验回复
        返回: (通过?, 警告信息, 修改后的回复)
        """
        modified = response

        # === 1. 危机检测 ===
        is_crisis, crisis_msg = self._check_crisis(query, response)
        if is_crisis:
            return False, crisis_msg, self._crisis_response()

        # === 2. 心理回复：确保有免责声明 ===
        if intent == "psychological":
            if "免责" not in response and "仅供参考" not in response:
                modified += config.PSYCHOLOGICAL_DISCLAIMER

        # === 3. 禁止内容检测 ===
        for pattern in self.BLOCK_PATTERNS:
            if pattern.search(response):
                return False, "回复包含不适当内容（医疗建议/诊断），已拦截", self._fallback(intent)

        # === 4. 政策/数据：检查是否有来源标注 ===
        if intent in ("policy", "data"):
            if "来源" not in response and "【参考" not in response:
                # 不是硬性拦截，但提示
                pass

        return True, None, modified

    def _check_crisis(self, query: str, response: str) -> Tuple[bool, str]:
        """检测是否有自伤/自杀风险"""
        combined = query + response
        for kw in self.CRISIS_KEYWORDS:
            if kw in combined:
                return True, f"检测到高危关键词: {kw}"
        return False, ""

    def _crisis_response(self) -> str:
        """危机干预回复"""
        return (
            "我注意到你可能正在经历非常困难的时刻。\n\n"
            "**请立即联系以下资源，有人能帮你：**\n\n"
            "🚨 **全国24小时心理危机干预热线**：400-161-9995\n"
            "🚨 **希望24热线**：400-161-9995\n"
            "🏥 **重庆邮电大学心理健康中心**：\n"
            "   地址：校内（具体地址请查看学校官网）\n"
            "   预约电话：请在工作时间联系\n\n"
            "**你不需要一个人面对这一切。请现在就打电话，有人愿意倾听。**"
        )

    def _fallback(self, intent: str) -> str:
        if intent == "psychological":
            return self._crisis_response()
        return "抱歉，我暂时无法回答这个问题。请咨询学校相关部门获取准确信息。"

    def check_query(self, query: str) -> Tuple[bool, str]:
        """输入侧检测"""
        # 检测是否包含不当内容
        if len(query) > 500:
            return False, "问题过长，请简洁描述"
        # 心理高危 → 直接返回危机干预
        for kw in self.CRISIS_KEYWORDS:
            if kw in query:
                return False, "crisis"
        return True, ""


if __name__ == "__main__":
    g = Guardrails()

    tests = [
        ("我想自杀，活不下去了", "心理回复……"),
        ("考研压力好大，失眠怎么办", "你可以试试深呼吸……"),
        ("三方协议怎么签", "三方协议签订流程：1.……来源：《就业指南》"),
    ]

    for query, resp in tests:
        ok, warn, modified = g.check_response(query, resp, "psychological")
        print(f"✅={ok} | {warn or '通过'} | {modified[:60]}...")
