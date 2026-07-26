"""
检索存储：豆包Embedding API（主）+ BM25关键词（兜底）
"""
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import jieba
import httpx
from rank_bm25 import BM25Okapi

from document_loader import DocumentChunk
import config


class DoubaoEmbedder:
    """豆包 Embedding API 封装（多模态模型，逐条调用）"""

    def __init__(self):
        self.api_key = config.DOUBAO_API_KEY
        self.base_url = config.DOUBAO_BASE_URL
        self.model = config.DOUBAO_EMBEDDING_MODEL
        self._dim = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """文本 → 向量（逐条调 multimodal 端点）"""
        url = f"{self.base_url}/embeddings/multimodal"
        all_embeddings = []

        with httpx.Client(timeout=60.0) as client:
            for i, text in enumerate(texts):
                body = {
                    "model": self.model,
                    "input": [{"type": "text", "text": text}],
                }

                for attempt in range(3):
                    try:
                        resp = client.post(
                            url, json=body,
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {self.api_key}",
                            },
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        # multimodal返回 data.embedding（dict非list）
                        emb = data["data"]["embedding"]
                        all_embeddings.append(emb)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise RuntimeError(f"Embedding failed for chunk {i}: {e}")
                        time.sleep(1.0)

                # 进度提示
                if (i + 1) % 10 == 0:
                    print(f"    Embedded {i + 1}/{len(texts)}...")

        return all_embeddings

    @property
    def dim(self) -> int:
        """获取向量维度（懒加载）"""
        if self._dim is None:
            try:
                emb = self.embed(["test"])
                self._dim = len(emb[0])
            except Exception:
                self._dim = 2048
        return self._dim


class VectorStore:
    """
    混合检索存储：
    - 豆包Embedding：语义向量检索
    - BM25：关键词检索兜底
    - 数据自动持久化到JSON
    """

    def __init__(self, persist_dir: Path = None):
        self.persist_dir = persist_dir or config.DATA_DIR / "store"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collections: Dict[str, List[dict]] = {}        # 文档存储
        self.bm25_indexes: Dict[str, tuple] = {}             # BM25索引
        self.embeddings: Dict[str, List[List[float]]] = {}   # 向量存储
        self._init_collections()

        # 尝试加载豆包Embedder
        self.embedder = None
        self.embedding_enabled = False
        try:
            self.embedder = DoubaoEmbedder()
            self.embedding_enabled = True
            print(f"  Embedding: doubao-embedding-vision ({self.embedder.dim}d)")
        except Exception as e:
            print(f"  Embedding disabled (API error: {e}), using BM25 only")

        self._load_from_disk()

    def _init_collections(self):
        for name in ["policy", "exam", "psychology"]:
            if name not in self.collections:
                self.collections[name] = []

    # ===== Tokenization (BM25用) =====
    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^一-鿿\w\s]', ' ', text)
        words = list(jieba.cut(text))
        return [w.strip() for w in words if w.strip() and len(w.strip()) > 1]

    # ===== Add =====
    def add_chunks(self, chunks: List[DocumentChunk], collection_name: str):
        if not chunks:
            return

        texts = [c.content for c in chunks]
        collection = self.collections.get(collection_name, [])

        # 1. 存储文档
        for c in chunks:
            collection.append({
                "content": c.content,
                "metadata": c.metadata,
                "tokens": self._tokenize(c.content),
            })
        self.collections[collection_name] = collection

        # 2. 重建BM25索引
        tokenized = [d["tokens"] for d in collection]
        if tokenized:
            self.bm25_indexes[collection_name] = (BM25Okapi(tokenized), collection)

        # 3. 豆包向量化（新增/追加）
        if self.embedding_enabled and self.embedder:
            try:
                new_embeddings = self.embedder.embed(texts)
                if collection_name not in self.embeddings:
                    self.embeddings[collection_name] = []
                self.embeddings[collection_name].extend(new_embeddings)
            except Exception as e:
                print(f"  ⚠️ Embedding failed for {collection_name}: {e}")

        print(f"  Added {len(chunks)} chunks to '{collection_name}' (total: {len(collection)})")
        self._save_to_disk()

    # ===== Search =====
    def search(
        self,
        query: str,
        collection_names: List[str],
        top_k: int = None
    ) -> List[dict]:
        """混合检索：向量相似度 + BM25关键词，加权融合"""
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL

        vector_results = {}
        bm25_results = {}

        # === 向量检索 ===
        if self.embedding_enabled and self.embedder:
            try:
                query_vec = self.embedder.embed([query])[0]
                for name in collection_names:
                    if name not in self.embeddings or name not in self.collections:
                        continue
                    collection = self.collections[name]
                    for i, doc_vec in enumerate(self.embeddings[name]):
                        sim = self._cosine_similarity(query_vec, doc_vec)
                        if sim > 0.3:  # 阈值过滤
                            key = f"{name}_{i}"
                            vector_results[key] = {
                                "content": collection[i]["content"],
                                "metadata": collection[i]["metadata"],
                                "score": float(sim),
                                "collection": name,
                            }
            except Exception as e:
                print(f"  ⚠️ Vector search failed: {e}")

        # === BM25检索 ===
        query_tokens = self._tokenize(query)
        for name in collection_names:
            if name not in self.bm25_indexes:
                continue
            bm25, collection = self.bm25_indexes[name]
            scores = bm25.get_scores(query_tokens)
            if max(scores) > 0:
                for i, score in enumerate(scores):
                    if score > 0:
                        key = f"bm25_{name}_{i}"
                        bm25_results[key] = {
                            "content": collection[i]["content"],
                            "metadata": collection[i]["metadata"],
                            "score": float(score),
                            "collection": name,
                        }

        # === 融合：向量优先，BM25补充 ===
        merged = {}

        # 向量结果（归一化后优先）
        for key, doc in vector_results.items():
            merged[key] = doc

        # BM25结果补充（去重）
        for key, doc in bm25_results.items():
            content_key = doc["content"][:80]
            if not any(d["content"][:80] == content_key for d in merged.values()):
                # BM25分数归一化
                doc["score"] = doc["score"] * 0.7  # BM25权重稍低
                merged[key] = doc

        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        # 归一化
        if results and results[0]["score"] > 0:
            max_score = results[0]["score"]
            for r in results:
                r["score"] = min(r["score"] / max_score, 1.0)

        return results[:top_k]

    def search_structured(
        self, filters: dict, collection_name: str = "exam"
    ) -> List[dict]:
        """结构化数据精确查询"""
        if collection_name not in self.collections:
            return []

        collection = self.collections[collection_name]
        results = []

        for doc in collection:
            matched = False

            # 方式1：record JSON精确匹配
            record_str = doc["metadata"].get("record")
            if record_str:
                try:
                    record = json.loads(record_str)
                    if all(str(record.get(k, "")).strip() == str(v).strip() for k, v in filters.items()):
                        matched = True
                except json.JSONDecodeError:
                    pass

            # 方式2：content文本模糊匹配
            if not matched:
                content = doc.get("content", "")
                if all(str(v) in content for v in filters.values()):
                    matched = True

            if matched:
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": 1.0,
                    "collection": collection_name,
                })

        # 降级：部分匹配
        if not results and filters:
            for doc in collection:
                content = doc.get("content", "")
                if any(str(v) in content for v in filters.values()):
                    results.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": 0.8,
                        "collection": collection_name,
                    })

        return results

    # ===== 向量工具 =====
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    # ===== Persistence =====
    def _save_to_disk(self):
        for name, collection in self.collections.items():
            filepath = self.persist_dir / f"{name}.json"
            data = [{"content": d["content"], "metadata": d["metadata"]} for d in collection]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        # 向量单独存储（较大）
        if self.embeddings:
            emb_path = self.persist_dir / "embeddings.json"
            with open(emb_path, "w", encoding="utf-8") as f:
                json.dump(self.embeddings, f, ensure_ascii=False)

    def _load_from_disk(self):
        for name in ["policy", "exam", "psychology"]:
            filepath = self.persist_dir / f"{name}.json"
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for doc in data:
                    self.collections[name].append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "tokens": self._tokenize(doc["content"]),
                    })
                if self.collections[name]:
                    tokenized = [d["tokens"] for d in self.collections[name]]
                    self.bm25_indexes[name] = (BM25Okapi(tokenized), self.collections[name])
                print(f"  Loaded {len(data)} docs from {filepath.name}")

        # 加载向量（如果有）
        emb_path = self.persist_dir / "embeddings.json"
        if emb_path.exists() and self.embedding_enabled:
            with open(emb_path, "r", encoding="utf-8") as f:
                self.embeddings = json.load(f)
            print(f"  Loaded embeddings from disk")

    def get_stats(self) -> dict:
        return {name: len(coll) for name, coll in self.collections.items()}


if __name__ == "__main__":
    vs = VectorStore()
    print(f"Stats: {vs.get_stats()}")
    print(f"Embedding enabled: {vs.embedding_enabled}")
