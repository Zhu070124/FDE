"""
自训练反射回环 — 回答 → 自我反思 → 修正 → 记录学习
灵感来源: langgraph-course reflexion-agent
"""
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import httpx

import config

logger = logging.getLogger(__name__)


@dataclass
class ReflectionRecord:
    """一次反思记录"""
    timestamp: str
    question: str
    original_answer: str
    critique: str         # 自我批判：缺了什么
    improved_answer: str  # 修正后回答
    is_useful: bool       # 这次反思是否有帮助


class ReflectionEngine:
    """自训练引擎：回答 → 自我反思 → 修正 → 记录到知识库"""

    REFLECT_PROMPT = """你是回答质量审查员。对以下回答进行严格自我批判：

用户问题：{question}
原始回答：{answer}
参考文档：{documents}

请评估：
1. 回答是否完整？遗漏了什么关键信息？
2. 回答是否准确？有没有编造文档中没有的内容？
3. 如果有改进空间，给出修正后的回答。

输出JSON：
{{
    "is_complete": true/false,
    "critique": "批判意见（一句话）",
    "improved_answer": "修正后的回答（如果不需要修正，留空）"
}}

只输出JSON，不要其他内容。"""

    def __init__(self, persist_dir: Path = None):
        self.api_key = config.DOUBAO_API_KEY
        self.base_url = config.DOUBAO_BASE_URL
        self.model = config.DOUBAO_MODEL

        self.persist_dir = persist_dir or config.DATA_DIR / "reflections"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[ReflectionRecord] = []

    def reflect(
        self,
        question: str,
        answer: str,
        documents: List[dict],
    ) -> ReflectionRecord:
        """对一次回答进行反思，如果需要修正则生成改进版"""
        docs_text = "\n".join(d["content"][:300] for d in documents[:3])

        critique = ""
        improved = ""
        is_useful = False

        try:
            result = self._call_reflection(question, answer, docs_text)

            if isinstance(result, dict):
                critique = result.get("critique", "")
                is_useful = not result.get("is_complete", True)
                if is_useful:
                    improved = result.get("improved_answer", "")

        except Exception as e:
            critique = f"反思调用失败: {e}"

        record = ReflectionRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            question=question,
            original_answer=answer,
            critique=critique,
            improved_answer=improved if improved else answer,
            is_useful=is_useful,
        )

        self.records.append(record)
        return record

    def save(self):
        """持久化反思记录到磁盘"""
        filepath = self.persist_dir / f"reflections_{time.strftime('%Y%m%d')}.json"
        data = [
            {
                "timestamp": r.timestamp,
                "question": r.question,
                "original_answer": r.original_answer[:500],
                "critique": r.critique,
                "is_useful": r.is_useful,
            }
            for r in self.records
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_feedback_loop_data(self) -> List[dict]:
        """提取可用于改进检索策略的反馈数据"""
        # 找那些反思发现不完整的case
        improvement_cases = []
        for r in self.records:
            if r.is_useful and r.critique:
                improvement_cases.append({
                    "question": r.question,
                    "gap": r.critique,  # 缺了什么
                    "improved": r.improved_answer[:300],
                })
        return improvement_cases

    def _call_reflection(self, question: str, answer: str, docs: str) -> dict:
        """调用LLM做反思"""
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.REFLECT_PROMPT.format(
                    question=question, answer=answer[:1500], documents=docs
                )}
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                url, json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())


class SelfTrainingPipeline:
    """自训练管道：回答 → 反思 → 反馈数据 → 知识库增量更新"""

    def __init__(self, vector_store):
        self.engine = ReflectionEngine()
        self.vs = vector_store

    def process(
        self,
        question: str,
        answer: str,
        documents: List[dict],
        intent: str = "policy",
    ) -> ReflectionRecord:
        """处理一次问答：反思 → 记录 → 反馈数据写入知识库"""
        record = self.engine.reflect(question, answer, documents)

        # 如果反思发现不完整 → 把改进版回答写入知识库
        if record.is_useful and record.improved_answer:
            self._inject_feedback(record, intent)

        # 每次处理完自动持久化
        self.engine.save()

        return record

    def _inject_feedback(self, record: ReflectionRecord, intent: str):
        """把改进版回答注入对应知识库集合"""
        from document_loader import DocumentChunk

        # 按意图路由到正确的集合（引用config统一映射）
        collection_name = config.INTENT_TO_COLLECTION.get(intent, "policy")

        chunk = DocumentChunk(
            content=f"Q: {record.question}\nA: {record.improved_answer}",
            metadata={
                "source": "self-training-reflection",
                "type": "feedback",
                "critique": record.critique,
            }
        )
        try:
            self.vs.add_chunks([chunk], collection_name)
            logger.info("Self-training: injected into '%s' for '%s...'", collection_name, record.question[:40])
        except Exception as e:
            logger.warning("Self-training injection failed: %s", e)


if __name__ == "__main__":
    engine = ReflectionEngine()
    record = engine.reflect(
        "重邮计算机分数线多少？",
        "重邮计算机2024年复试线是310分。",
        [{"content": "2024年重邮计算机科学与技术复试线310分，报录比8:1，招生120人"}]
    )
    print(f"Critique: {record.critique}")
    print(f"Useful: {record.is_useful}")
