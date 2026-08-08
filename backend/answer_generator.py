"""
回答生成器：调用豆包API + 检索上下文生成回答
支持流式输出 + httpx连接池
"""
import json
import re
import logging
from typing import List, Tuple, Generator
import httpx

import config

logger = logging.getLogger(__name__)
from intent_router import Intent, IntentRouter


class AnswerGenerator:
    """豆包 API 回答生成器（火山引擎 Ark）"""

    def __init__(self):
        self.api_key = config.DOUBAO_API_KEY
        self.base_url = config.DOUBAO_BASE_URL
        self.model = config.DOUBAO_MODEL
        # 共享连接池——避免每次请求新建TCP+TLS连接
        self._client = httpx.Client(timeout=60.0)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

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
            logger.exception("Doubao API call failed, falling back: %s", e)
            return self._fallback_response(intent, str(e))

    def _call_doubao_api(self, messages: List[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """调用火山引擎豆包API（非流式）——复用共享连接池"""
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            resp = self._client.post(
                url, json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error("Doubao API HTTP error: %s, body: %s", e, e.response.text[:200] if e.response else "")
            raise
        except Exception as e:
            logger.exception("Doubao API call failed: %s", e)
            raise

    def generate_stream(
        self,
        query: str,
        retrieved_docs: List[dict],
        intent: Intent,
        temperature: float = 0.3,
        require_citation: bool = True,
        add_disclaimer: bool = False,
        history: List[dict] = None,
        multi_intents: List[Tuple] = None,
    ) -> Generator[str, None, None]:
        """流式生成——支持多标签融合"""
        system_prompt = self._build_system_prompt(intent, require_citation, add_disclaimer, multi_intents)
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

        stream_failed = False

        try:
            with self._client.stream(  # 复用共享连接池
                "POST", url, json=body,
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
                        except json.JSONDecodeError:
                            continue
                        # 跳过无choices的chunk（usage/error事件等）
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
        except Exception as e:
            stream_failed = True
            logger.error("Doubao streaming failed: %s", e)
            yield "\n\n[生成出错，请稍后重试]"

        # 心理回复追加免责（仅在生成成功时）
        if add_disclaimer and not stream_failed:
            yield config.PSYCHOLOGICAL_DISCLAIMER

    # ===== Prompt 构建 =====

    def _build_system_prompt(
        self, intent: Intent, require_citation: bool, add_disclaimer: bool,
        multi_intents: List[Tuple] = None,
    ) -> str:
        """根据意图构建 system prompt，支持多标签融合"""

        base = (
            '你是一个高校学生成长咨询助手，名叫"小邮"。'
            '你面对的是和你一样的大学生——他们可能正在为毕业焦虑、'
            '为考研熬夜、为各种流程手续感到迷茫。'
            '你的回答应当专业、准确，同时像一个学长/学姐一样有温度。'
        )

        # 多标签融合：叠加多个场景的温度规则
        if multi_intents and len(multi_intents) > 1:
            scenes = []
            rules = []
            has_psych = any(i == Intent.PSYCHOLOGICAL for i, _ in multi_intents)
            has_policy = any(i == Intent.POLICY for i, _ in multi_intents)
            has_data = any(i == Intent.DATA for i, _ in multi_intents)

            if has_policy:
                scenes.append("就业/政策咨询")
                rules.append("政策信息要精确标注来源")
            if has_data:
                scenes.append("考研数据查询")
                rules.append("数据要精确，用表格呈现")
            if has_psych:
                scenes.append("心理健康支持")
                rules.append("先看见情绪再给建议，结尾用自己的话说明边界")

            return base + (
                f"\n## 当前场景：{' + '.join(scenes)}（跨场景）\n"
                "语气：这些需求往往是交织在一起的——对方可能在为流程焦虑、"
                "为数据紧张、内心也需要被安抚。你既要给出准确的信息，"
                "也要看见数字和流程背后的那个人。\n"
                "规则：\n" +
                "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules)) + "\n" +
                f"{len(rules)+1}. 回答末尾列出【参考来源】段落"
            )

        # 单标签：走原有路径
        if intent == Intent.POLICY:
            return base + (
                "\n## 当前场景：就业/政策咨询\n"
                "语气：清晰有条理，但别冷冰冰地甩条款。开头可以用一两句话承接学生的情绪"
                "（如'这个流程确实有点绕，别急，一步步来'），再进入正文。\n"
                "规则：\n"
                "1. 回答必须基于提供的参考文档，不得编造\n"
                "2. 每条关键信息必须标注来源编号，如【来源：参考1】\n"
                "3. 如果文档中没有相关信息，诚实告知'我目前没有找到相关政策信息'"
                "——但不要让这句话成为终结，可以补充建议（如'建议咨询辅导员或学校相关部门'）\n"
                "4. 复杂流程用 步骤1→步骤2→步骤3 的方式呈现\n"
                "5. 回答末尾必须列出【参考来源】段落"
            )
        elif intent == Intent.DATA:
            return base + (
                "\n## 当前场景：考研数据查询\n"
                "语气：数据要精确，但数字背后的焦虑你也要看见。呈现数据前可以先肯定对方"
                "（如'你在认真对比学校，这是个很重要的决定'），"
                "然后客观呈现数据，不替对方做判断。\n"
                "规则：\n"
                "1. 回答必须基于提供的表格数据，精确引用数字\n"
                "2. 每个数据点必须标注来源编号，如【来源：参考1】\n"
                "3. 如果数据不完整，明确说明'根据已有数据'\n"
                "4. 用表格或列表呈现对比数据\n"
                "5. 回答末尾必须列出【参考来源】段落\n"
                "6. 不替用户做'哪个更好'的决定，只呈现事实供对方自己判断"
            )
        else:  # PSYCHOLOGICAL
            return base + (
                "\n## 当前场景：心理健康支持\n"
                "语气：你是深夜可以倾诉的朋友。先看见对方的痛苦（'我能感觉到你积攒了很多'），"
                "再给建议。永远不要让对方觉得'你只是想应付我'。\n"
                "规则：\n"
                "1. 以温暖、共情的语气回应，让同学感到被理解\n"
                "2. 不诊断、不乱给建议，提供科学的心理自助方法\n"
                "3. 如果对方表现出严重情绪困扰，温和建议寻求专业帮助\n"
                "4. 回答结尾用你自己的话温和地说明你的边界，"
                "参考语气：'我只是AI，能陪你聊天但不能替代心理咨询，"
                "如果你需要更专业的帮助，学校心理中心随时为你开放'\n"
                "5. 不要使用'免责提示'这四个字——太生硬了，用自己的话说"
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
        hotline = config.get_school_attr("national_hotline", "400-161-9995")
        mental_center = config.get_school_attr("mental_health_center", "学校心理健康中心")
        if intent == Intent.PSYCHOLOGICAL:
            return (
                "我在倾听，也很想帮你。以下是你可以尝试的自助方法：\n\n"
                "1. 梳理当前的压力来源，把大问题拆成小步骤逐一解决\n"
                "2. 尝试腹式呼吸放松：吸气4秒→屏息4秒→呼气6秒，重复5次\n"
                "3. 每天30分钟有氧运动（慢跑、散步），帮助身体释放压力\n"
                "4. 找信任的朋友倾诉，不要一个人憋着\n"
                "5. 如果持续两周以上情绪低落，请预约学校心理咨询中心\n\n"
                f"{mental_center}随时为你开放。"
                "你不需要一个人扛着，有人愿意帮你。"
                + config.PSYCHOLOGICAL_DISCLAIMER
            )
        return (
            "抱歉，我暂时无法访问知识库。请稍后重试，"
            "或在工作时间直接联系学校相关部门获取帮助。"
        )


class EntityExtractor:
    """从自然语言中提取结构化查询字段（学校、专业、年份等）
    学校/专业列表从 config/schools.yaml 的 entity_registry 动态加载。
    """

    # 年份模式（不需要配置化）
    YEAR_PATTERNS = [
        (r"(\d{4})年", None),     # 绝对年份
        (r"去年", -1),            # 相对年份-1
        (r"今年", 0),             # 当前年份
        (r"前年", -2),            # 相对年份-2
    ]

    @classmethod
    def _get_registry(cls) -> dict:
        """延迟加载实体注册表"""
        return config.get_entity_registry()

    @classmethod
    def extract(cls, query: str) -> dict:
        """从查询中提取实体"""
        from datetime import datetime
        registry = cls._get_registry()
        entities = {}

        # 提取学校（从配置加载）
        schools = registry.get("schools", [])
        aliases = registry.get("school_aliases", {})

        for school in schools:
            if school in query:
                entities["学校"] = school
                break
        else:
            # 简称匹配
            for alias, full_name in aliases.items():
                if alias in query:
                    entities["学校"] = full_name
                    break

        # 提取专业（从配置加载）
        majors = registry.get("majors", [])
        for major in majors:
            if major in query:
                entities["专业"] = major
                break
        else:
            # 简称匹配
            if "计算机" in query and "计算机" not in str(entities.get("专业", "")):
                entities["专业"] = "计算机科学与技术"
            elif "通信" in query:
                entities["专业"] = "信息与通信工程"
            elif "软件" in query:
                entities["专业"] = "软件工程"

        # 提取年份（保持不变）
        current_year = datetime.now().year
        for pattern, offset in cls.YEAR_PATTERNS:
            m = re.search(pattern, query)
            if m:
                if offset is None:
                    entities["年份"] = m.group(1)
                else:
                    entities["年份"] = str(current_year + offset)
                break

        return entities


class HybridSearcher:
    """混合检索：NL2SQL + 结构化查询 + BM25关键词 + 语义检索"""

    def __init__(self, vector_store, llm_fn=None):
        self.vs = vector_store
        self.router = IntentRouter(use_llm=llm_fn is not None, llm_fn=llm_fn)
        self.llm_fn = llm_fn
        # NL2SQL 延迟初始化
        self._exam_db = None

    @property
    def exam_db(self):
        """延迟加载 ExamDatabase（避免 import 时的 config 校验失败）。"""
        if self._exam_db is None:
            try:
                from nl2sql import get_exam_db
                self._exam_db = get_exam_db()
            except Exception:
                pass
        return self._exam_db

    def search(self, query: str, multi_label: bool = True) -> dict:
        """执行混合检索，支持多标签跨场景"""
        import logging
        logger = logging.getLogger(__name__)

        # 主意图：走完整 classify（含LLM兜底）
        primary_intent, confidence = self.router.classify(query)

        # 多标签补充：仅用于跨场景检索，不影响主意图
        if multi_label:
            extra_labels = self.router.classify_multi(query)
        else:
            extra_labels = [(primary_intent, confidence)]

        # 合并多意图的检索结果
        all_docs = []
        seen = set()

        for intent, weight in extra_labels:
            strategy = self.router.get_retrieval_strategy(intent)
            collections = strategy["collections"]
            mode = strategy["search_mode"]

            if mode == "structured":
                docs = self._structured_search(query, collections)
            else:
                docs = self.vs.search(query, collections, top_k=config.TOP_K_RETRIEVAL)

            for doc in docs:
                key = doc["content"][:80]
                if key not in seen:
                    seen.add(key)
                    doc["score"] = doc.get("score", 0) * weight
                    all_docs.append(doc)

        all_docs.sort(key=lambda x: x["score"], reverse=True)

        return {
            "intent": primary_intent,
            "confidence": confidence,
            "multi_intents": extra_labels if len(extra_labels) > 1 else None,
            "strategy": self.router.get_retrieval_strategy(primary_intent),
            "retrieved_docs": all_docs[:config.TOP_K_RETRIEVAL * 2],
        }

    def _structured_search(self, query: str, collections: list) -> list:
        """结构化精确查询：NL2SQL → 实体精确匹配 → BM25 三级降级"""
        import logging
        logger = logging.getLogger(__name__)

        # === 第1级：NL2SQL 精确查询（主路径） ===
        if self.exam_db and self.llm_fn:
            try:
                tables = self.exam_db.get_tables()
                if tables:
                    nl2sql_results = self.exam_db.query(query, llm_fn=self.llm_fn)
                    if nl2sql_results:
                        logger.info("NL2SQL: %d results for '%s'", len(nl2sql_results), query[:40])
                        return nl2sql_results[:config.TOP_K_RETRIEVAL]
            except Exception as e:
                logger.warning("NL2SQL failed, falling back to entity matching: %s", e)

        # === 第2级：实体提取 + JSON 精确匹配（兜底） ===
        entities = EntityExtractor.extract(query)
        exact_docs = []

        if entities:
            for attempt in [entities, {k: v for k, v in entities.items() if k != "年份"}]:
                if not attempt:
                    continue
                exact_docs = self.vs.search_structured(attempt, "exam")
                if exact_docs:
                    break

        # === 第3级：BM25 补充 ===
        bm25_docs = self.vs.search(query, collections, top_k=3)

        # 合并去重
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
