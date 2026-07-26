"""
回答生成器：调用豆包API + 检索上下文生成回答
支持流式输出 + httpx连接池
"""
import json
import re
from typing import List, Dict, Optional, Generator
import httpx

import config
from intent_router import Intent, IntentRouter


class AnswerGenerator:
    """豆包 API 回答生成器（火山引擎 Ark）"""

    def __init__(self):
        self.api_key = config.DOUBAO_API_KEY
        self.base_url = config.DOUBAO_BASE_URL
        self.model = config.DOUBAO_MODEL

    def generate(
        self,
        query: str,
        retrieved_docs: List[dict],
        intent: Intent,
        temperature: float = 0.3,
        require_citation: bool = True,
        add_disclaimer: bool = False,
        history: List[dict] = None,
    ) -> str:
        """主生成入口"""
        system_prompt = self._build_system_prompt(
            intent, require_citation, add_disclaimer
        )
        user_prompt = self._build_user_prompt(query, retrieved_docs, intent)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-12:])
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._call_doubao_api(messages, temperature)
            # 后处理：心理回复加免责
            if add_disclaimer:
                response += config.PSYCHOLOGICAL_DISCLAIMER
            return response
        except Exception as e:
            return self._fallback_response(intent, str(e))

    def _call_doubao_api(self, messages: List[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """调用火山引擎豆包API（非流式，同步返回）"""
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def generate_stream(
        self,
        query: str,
        retrieved_docs: List[dict],
        intent: Intent,
        temperature: float = 0.3,
        require_citation: bool = True,
        add_disclaimer: bool = False,
        history: List[dict] = None,
    ) -> Generator[str, None, None]:
        """流式生成——边生成边返回，支持对话历史"""
        system_prompt = self._build_system_prompt(intent, require_citation, add_disclaimer)
        user_prompt = self._build_user_prompt(query, retrieved_docs, intent)

        # 构建消息列表：system + history(最近6轮) + user
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            # 只保留最近6轮（12条消息），token友好
            messages.extend(history[-12:])
        messages.append({"role": "user", "content": user_prompt})

        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
            "stream": True,
        }

        disclaimer_appended = False

        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    url,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"\n\n[生成出错: {e}]"

        # 心理回复追加免责
        if add_disclaimer and not disclaimer_appended:
            yield config.PSYCHOLOGICAL_DISCLAIMER

    # ===== Prompt 构建 =====

    def _build_system_prompt(
        self, intent: Intent, require_citation: bool, add_disclaimer: bool
    ) -> str:
        """根据意图构建不同的 system prompt"""

        base = '你是一个高校学生成长咨询助手，名叫"小邮"。你的回答应当专业、准确、温暖。'

        if intent == Intent.POLICY:
            return base + (
                "\n## 当前场景：就业/政策咨询\n"
                "规则：\n"
                "1. 回答必须基于提供的参考文档，不得编造\n"
                "2. 每条关键信息必须标注来源编号，如【来源：参考1】\n"
                "3. 如果文档中没有相关信息，诚实告知'我目前没有找到相关政策信息'\n"
                "4. 复杂流程用 步骤1→步骤2→步骤3 的方式呈现\n"
                "5. 回答末尾必须列出【参考来源】段落"
            )
        elif intent == Intent.DATA:
            return base + (
                "\n## 当前场景：考研数据查询\n"
                "规则：\n"
                "1. 回答必须基于提供的表格数据，精确引用数字\n"
                "2. 每个数据点必须标注来源编号，如【来源：参考1】\n"
                "3. 如果数据不完整，明确说明'根据已有数据'\n"
                "4. 用表格或列表呈现对比数据\n"
                "5. 回答末尾必须列出【参考来源】段落\n"
                "6. 不要对数据做主观解读，只呈现事实"
            )
        else:  # PSYCHOLOGICAL
            return base + (
                "\n## 当前场景：心理健康支持\n"
                "规则：\n"
                "1. 以温暖、共情的语气回应，让同学感到被理解\n"
                "2. 不诊断、不乱给建议，提供科学的心理自助方法\n"
                "3. 推荐学校的心理健康资源（心理咨询中心、24小时热线等）\n"
                "4. 如果对方表现出严重情绪困扰，温和建议寻求专业帮助\n"
                "5. 回答结尾必须包含免责提示"
            )

    def _build_user_prompt(
        self, query: str, retrieved_docs: List[dict], intent: Intent
    ) -> str:
        """构建包含检索上下文的 user prompt"""

        parts = [f"学生提问：{query}"]

        if retrieved_docs:
            parts.append("\n--- 以下是从知识库中检索到的相关信息 ---")
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.get("metadata", {}).get("source", "未知来源")
                score = doc.get("score", 0)
                parts.append(
                    f"\n【参考{i}】来源: {source} | 相关度: {score:.2f}\n"
                    f"{doc['content']}"
                )
            parts.append("\n--- 请基于以上参考信息回答学生的问题 ---")
        else:
            parts.append("\n（知识库中暂无相关信息，请据此诚实回答）")

        return "\n".join(parts)

    def _fallback_response(self, intent: Intent, error: str) -> str:
        """API调用失败时的兜底回复"""
        print(f"⚠️ 豆包API调用失败: {error}")
        if intent == Intent.PSYCHOLOGICAL:
            return (
                "我在倾听，也很想帮你。以下是你可以尝试的自助方法：\n\n"
                "1. 梳理当前的压力来源，把大问题拆成小步骤逐一解决\n"
                "2. 尝试腹式呼吸放松：吸气4秒→屏息4秒→呼气6秒，重复5次\n"
                "3. 每天30分钟有氧运动（慢跑、散步），帮助身体释放压力\n"
                "4. 找信任的朋友倾诉，不要一个人憋着\n"
                "5. 如果持续两周以上情绪低落，请预约学校心理咨询中心\n\n"
                "重庆邮电大学心理健康中心随时为你开放。"
                "你不需要一个人扛着，有人愿意帮你。"
                + config.PSYCHOLOGICAL_DISCLAIMER
            )
        return (
            "抱歉，我暂时无法访问知识库。请稍后重试，"
            "或在工作时间直接联系学校相关部门获取帮助。"
        )


class EntityExtractor:
    """从自然语言中提取结构化查询字段（学校、专业、年份等）"""

    # 已知学校关键词
    SCHOOLS = [
        "重庆邮电大学", "重庆大学", "西南大学", "重庆交通大学",
        "重庆理工大学", "重庆师范大学", "重庆工商大学", "四川大学",
        "电子科技大学", "北京大学", "清华大学",
    ]

    # 已知专业关键词
    MAJORS = [
        "计算机科学与技术", "计算机技术", "软件工程", "信息与通信工程",
        "控制科学与工程", "教育学", "电子信息", "人工智能",
        "金融", "会计", "工商管理", "法学", "医学",
    ]

    # 年份模式
    YEAR_PATTERNS = [
        (r"(\d{4})年", 1),       # 2024年
        (r"去年", "去年"),       # 动态计算
        (r"今年", "今年"),
        (r"前年", "前年"),
    ]

    @classmethod
    def extract(cls, query: str) -> dict:
        """从查询中提取实体"""
        entities = {}

        # 提取学校
        for school in cls.SCHOOLS:
            if school in query:
                entities["学校"] = school
                break
        else:
            # 简称匹配
            if "重邮" in query:
                entities["学校"] = "重庆邮电大学"
            elif "重大" in query and "重邮" not in query:
                entities["学校"] = "重庆大学"
            elif "西大" in query:
                entities["学校"] = "西南大学"

        # 提取专业
        for major in cls.MAJORS:
            if major in query:
                entities["专业"] = major
                break
        else:
            if "计算机" in query:
                entities["专业"] = "计算机科学与技术"
            elif "通信" in query:
                entities["专业"] = "信息与通信工程"
            elif "软件" in query:
                entities["专业"] = "软件工程"

        # 提取年份 → 映射到数据中的列
        import re
        for pattern, group in cls.YEAR_PATTERNS:
            m = re.search(pattern, query)
            if m:
                if isinstance(group, int):
                    entities["年份"] = m.group(group)
                break

        return entities


class HybridSearcher:
    """混合检索：结构化查询 + BM25关键词 + 语义检索"""

    def __init__(self, vector_store, llm_fn=None):
        self.vs = vector_store
        self.router = IntentRouter(use_llm=llm_fn is not None, llm_fn=llm_fn)

    def search(self, query: str) -> dict:
        """执行混合检索，根据意图选择不同检索策略"""
        intent, confidence = self.router.classify(query)
        strategy = self.router.get_retrieval_strategy(intent)

        collections = strategy["collections"]
        mode = strategy["search_mode"]

        if mode == "structured":
            # === 数据查询：优先结构化精确匹配 ===
            docs = self._structured_search(query, collections)
        elif mode == "hybrid":
            # === 政策查询：BM25全文本搜索 ===
            docs = self.vs.search(query, collections, top_k=config.TOP_K_RETRIEVAL)
        else:
            # === 心理支持：BM25语义搜索 ===
            docs = self.vs.search(query, collections, top_k=config.TOP_K_RETRIEVAL)

        return {
            "intent": intent,
            "confidence": confidence,
            "strategy": strategy,
            "retrieved_docs": docs,
        }

    def _structured_search(self, query: str, collections: list) -> list:
        """结构化精确查询：从NL中提取实体 → 精确匹配Excel字段 → BM25补充"""
        entities = EntityExtractor.extract(query)
        exact_docs = []
        bm25_docs = []

        # 1. 精确字段匹配（如果提取到了实体）
        if entities:
            # 多轮尝试：全匹配 → 部分匹配
            for attempt in [entities, {k: v for k, v in entities.items() if k != "年份"}]:
                if not attempt:
                    continue
                exact_docs = self.vs.search_structured(attempt, "exam")
                if exact_docs:
                    break

        # 2. BM25补充（特别是年份信息）
        bm25_docs = self.vs.search(query, collections, top_k=3)

        # 3. 精确匹配优先，BM25补充（去重）
        seen = set()
        merged = []
        for doc in exact_docs + bm25_docs:
            key = doc["content"][:80]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                if len(merged) >= 5:
                    break

        return merged if merged else bm25_docs


if __name__ == "__main__":
    gen = AnswerGenerator()
    # 测试用 — 不实际调用API
    prompt = gen._build_system_prompt(Intent.PSYCHOLOGICAL, False, True)
    print(prompt)
