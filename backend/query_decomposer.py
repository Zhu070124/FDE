"""
多步推理分解器 — 把复杂问题拆成子问题链
灵感来源: langgraph-course agentic-rag Adaptive Router
"""
from typing import List, Dict
from dataclasses import dataclass
import json
import httpx

import config


@dataclass
class SubQuery:
    """子查询"""
    question: str
    intent: str          # policy / data / psychological
    depends_on: int = None  # 依赖前面哪个子查询的结果


class QueryDecomposer:
    """判断是否需要多步推理 + 分解问题"""

    DECOMPOSE_PROMPT = """你是查询分解器。判断以下学生问题是否需要多步推理。

需要多步推理的场景：
- 对比类："A和B哪个更好" → 需先查A、再查B、最后对比
- 条件类："满足XX条件的有哪些" → 需先查条件、再匹配
- 组合类："A的分数线和报录比" → 需查分数线、再查报录比

不需要多步推理的场景：
- 简单查询："XX分数线多少"、"助学金怎么申请"
- 情感支持："压力大怎么办"

学生问题：{query}

输出JSON格式：
{{
    "needs_decomposition": true/false,
    "sub_queries": [
        {{"question": "子问题1", "intent": "data", "depends_on": null}},
        {{"question": "子问题2", "intent": "data", "depends_on": 0}}
    ]
}}

只输出JSON，不要其他内容。"""

    def __init__(self):
        self.api_key = config.DOUBAO_API_KEY
        self.base_url = config.DOUBAO_BASE_URL
        self.model = config.DOUBAO_MODEL

    def analyze(self, query: str) -> dict:
        """分析是否需要分解，返回子查询列表"""
        try:
            result = self._call_llm(query)
            return json.loads(result)
        except Exception:
            # 分解失败 → 单步执行
            return {"needs_decomposition": False, "sub_queries": []}

    def decompose(self, query: str) -> List[SubQuery]:
        """分解复杂问题为子查询链"""
        analysis = self.analyze(query)

        if not analysis.get("needs_decomposition"):
            # 不需要分解 → 就是原问题
            return [SubQuery(question=query, intent="data")]

        subs = []
        for sq in analysis.get("sub_queries", []):
            subs.append(SubQuery(
                question=sq["question"],
                intent=sq.get("intent", "data"),
                depends_on=sq.get("depends_on"),
            ))

        return subs if subs else [SubQuery(question=query, intent="data")]

    def _call_llm(self, query: str) -> str:
        """调用LLM做分解判断"""
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.DECOMPOSE_PROMPT.format(query=query)}
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    url, json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # 提取JSON（LLM可能在前后加markdown）
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                return content.strip()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Query decomposer LLM call failed: %s", e)
            raise


class StepExecutor:
    """逐个执行子查询，收集结果"""

    def __init__(self, searcher):
        self.searcher = searcher  # HybridSearcher

    def execute(self, sub_queries: List[SubQuery]) -> List[dict]:
        """串行执行子查询链（处理依赖关系）"""
        results = []

        for i, sq in enumerate(sub_queries):
            # 如果有依赖，把前面结果附加到问题中
            question = sq.question
            if sq.depends_on is not None and sq.depends_on < len(results):
                prev = results[sq.depends_on]
                docs_preview = " ".join(
                    d["content"][:100] for d in prev.get("docs", [])[:2]
                )
                question = f"{question}\n（参考前面的结果：{docs_preview}）"

            # 检索
            search_result = self.searcher.search(question)
            results.append({
                "sub_query": sq,
                "question": question,
                "docs": search_result["retrieved_docs"],
                "intent": search_result["intent"].value,
            })

        return results

    def combine(self, results: List[dict]) -> str:
        """将多步结果合并为综合上下文"""
        parts = []
        for i, r in enumerate(results):
            parts.append(f"\n=== 子问题{i+1}: {r['sub_query'].question} ===")
            for doc in r["docs"][:3]:
                parts.append(f"- {doc['content'][:200]}")
        return "\n".join(parts)


if __name__ == "__main__":
    decomposer = QueryDecomposer()

    tests = [
        "重邮和重大计算机哪个更好考？",
        "国家助学金怎么申请？",
        "重邮计算机的分数线和报录比分别是多少？",
    ]

    for q in tests:
        result = decomposer.analyze(q)
        print(f"\n{q}")
        print(f"  needs_decomposition: {result.get('needs_decomposition')}")
        for sq in result.get("sub_queries", []):
            print(f"  → {sq['question']} [{sq.get('intent')}]")
