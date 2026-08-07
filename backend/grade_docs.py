"""
检索质量打分 + 幻觉检测 + 回答质量评估
灵感来源: langgraph-course agentic-rag 三级打分链
"""
from typing import List, Dict, Tuple
import json
import logging
import httpx

import config

logger = logging.getLogger(__name__)

# 共享连接池（三个Grader复用）
_shared_client = None

def _get_client() -> httpx.Client:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.Client(timeout=15.0)
    return _shared_client


class DocumentGrader:
    """检索文档相关性打分 —— 过滤低质量检索结果"""

    PROMPT = """判断以下检索到的文档是否与用户问题相关。

用户问题：{question}
检索文档：{document}

只回答"yes"或"no"。yes=文档包含与问题相关的信息，no=无关。"""

    def __init__(self):
        self.api_key = config.DOUBAO_API_KEY
        self.base_url = config.DOUBAO_BASE_URL
        self.model = config.DOUBAO_MODEL

    def grade(self, question: str, documents: List[dict]) -> Tuple[List[dict], bool]:
        """
        逐条打分 → 过滤低相关文档
        返回: (高相关文档列表, 是否需要联网搜索)
        """
        relevant = []
        need_web_search = False
        consecutive_failures = 0
        max_failures = 3  # 连续失败上限——超过则不过滤，全保留

        for doc in documents:
            try:
                result = self._call_grader(question, doc["content"][:500])
                if result.strip().lower() == "yes":
                    relevant.append(doc)
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                # 连续失败超过阈值 → 停止过滤，全量保留
                if consecutive_failures >= max_failures:
                    return documents, False
                # 偶发失败 → 保守保留
                relevant.append(doc)

        if not relevant:
            need_web_search = True

        return relevant, need_web_search

    def _call_grader(self, question: str, document: str) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.PROMPT.format(
                    question=question, document=document
                )}
            ],
            "temperature": 0.0,
            "max_tokens": 5,
        }

        try:
            client = _get_client()  # shared pooled client
            resp = client.post(
                url, json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("Document grader API call failed: %s", e)
            raise


class HallucinationChecker:
    """幻觉检测 —— 验证生成回答是否基于检索文档"""

    PROMPT = """判断以下回答是否基于提供的参考文档。

参考文档：{documents}
生成回答：{generation}

只回答"yes"或"no"。yes=回答内容可以在参考文档中找到依据，no=回答编造了文档中没有的信息。"""

    def check(self, documents: List[dict], generation: str) -> bool:
        """返回: True=回答有依据, False=存在幻觉"""
        docs_text = "\n".join(d["content"][:300] for d in documents[:3])

        try:
            result = self._call_checker(docs_text, generation)
            return result.strip().lower() == "yes"
        except Exception:
            return True  # 检查失败 → 保守通过

    def _call_checker(self, docs_text: str, generation: str) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.PROMPT.format(
                    documents=docs_text, generation=generation[:1000]
                )}
            ],
            "temperature": 0.0,
            "max_tokens": 5,
        }

        try:
            client = _get_client()  # shared pooled client
            resp = client.post(
                url, json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("Hallucination checker API call failed: %s", e)
            raise


class AnswerGrader:
    """回答质量评估 —— 回答是否真正解决了用户的问题"""

    PROMPT = """判断以下回答是否真正解决了用户的问题。

用户问题：{question}
生成回答：{generation}

只回答"yes"或"no"。yes=回答解决了问题，no=回答没有真正回答用户想问的。"""

    def grade(self, question: str, generation: str) -> bool:
        """返回: True=回答有用, False=答非所问"""
        try:
            result = self._call_grader(question, generation)
            return result.strip().lower() == "yes"
        except Exception:
            return True

    def _call_grader(self, question: str, generation: str) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.PROMPT.format(
                    question=question, generation=generation[:1000]
                )}
            ],
            "temperature": 0.0,
            "max_tokens": 5,
        }

        try:
            client = _get_client()  # shared pooled client
            resp = client.post(
                url, json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("Answer grader API call failed: %s", e)
            raise


if __name__ == "__main__":
    # 本地测试
    g = DocumentGrader()
    docs, need_web = g.grade("计算机分数线", [
        {"content": "2024年重邮计算机复试线310分"},
        {"content": "北京烤鸭的做法是..."},
    ])
    print(f"Relevant: {len(docs)}, Need web: {need_web}")
