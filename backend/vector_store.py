"""
检索存储：豆包Embedding API（主）+ BM25关键词（兜底）
支持增量索引 + manifest追踪
"""
import json
import time
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import jieba
import httpx
from rank_bm25 import BM25Okapi

from document_loader import DocumentChunk
import config

logger = logging.getLogger(__name__)


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
                        emb = data["data"]["embedding"]
                        all_embeddings.append(emb)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise RuntimeError(f"Embedding failed for chunk {i}: {e}")
                        time.sleep(1.0)

                # 进度提示
                if (i + 1) % 10 == 0:
                    logger.info("Embedded %d/%d...", i + 1, len(texts))

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
    - 增量索引 + manifest追踪
    """

    def __init__(self, persist_dir: Path = None):
        self.persist_dir = persist_dir or config.DATA_DIR / "store"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collections: Dict[str, List[dict]] = {}        # 文档存储
        self.bm25_indexes: Dict[str, tuple] = {}             # BM25索引
        self.embeddings: Dict[str, List[List[float]]] = {}   # 向量存储
        self._init_collections()

        # Manifest for incremental indexing
        self.manifest_path = self.persist_dir / "manifest.json"
        self.manifest: Dict[str, dict] = {}
        self._load_manifest()

        # 尝试加载豆包Embedder
        self.embedder = None
        self.embedding_enabled = False
        try:
            self.embedder = DoubaoEmbedder()
            self.embedding_enabled = True
            logger.info("Embedding: doubao-embedding-vision (%dd)", self.embedder.dim)
        except Exception as e:
            logger.warning("Embedding disabled (API error: %s), using BM25 only", e)

        self._load_from_disk()

    def _init_collections(self):
        for name in ["policy", "exam", "psychology"]:
            if name not in self.collections:
                self.collections[name] = []

    # ===== Manifest (incremental tracking) =====
    def _load_manifest(self):
        """Load indexed-file manifest from disk."""
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                logger.info("Loaded manifest: %d tracked files", len(self.manifest))
            else:
                self.manifest = {}
        except Exception as e:
            logger.error("Failed to load manifest: %s", e)
            self.manifest = {}

    def _save_manifest(self):
        """Persist manifest to disk."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save manifest: %s", e)

    @staticmethod
    def _file_hash(file_path: Path) -> str:
        """Compute MD5 hash of a file for change detection."""
        try:
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _content_hash(content: str) -> str:
        """Compute MD5 hash of content string."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    # ===== Add Document (incremental) =====
    def add_document(
        self,
        file_path: str,
        collection_name: str,
        force: bool = False,
    ) -> dict:
        """
        Incrementally index a single document file.
        Skips if unchanged (tracked by file hash in manifest).

        Args:
            file_path: Path to the document file.
            collection_name: Target collection (policy/exam/psychology).
            force: If True, re-index even if unchanged.

        Returns:
            dict with status, chunks_added, and skipped reason if any.
        """
        path = Path(file_path)
        if not path.exists():
            msg = f"File not found: {file_path}"
            logger.error(msg)
            return {"status": "error", "message": msg}

        file_key = str(path.absolute())
        file_hash = self._file_hash(path)

        # Check manifest — skip if unchanged
        if not force and file_key in self.manifest:
            cached = self.manifest[file_key]
            if cached.get("hash") == file_hash:
                logger.info("Skipping unchanged file: %s (hash: %s)", path.name, file_hash[:8])
                return {
                    "status": "skipped",
                    "message": f"File unchanged: {path.name}",
                    "hash": file_hash[:8],
                }

        # Load and chunk
        try:
            from document_loader import DocumentLoader
            loader = DocumentLoader(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )
            chunks = loader.load_file(path)
        except Exception as e:
            logger.exception("Failed to load file %s", path.name)
            return {"status": "error", "message": f"Failed to load: {e}"}

        if not chunks:
            return {"status": "warning", "message": f"No chunks extracted from {path.name}", "chunks_added": 0}

        # Add chunks to collection
        try:
            self.add_chunks(chunks, collection_name)
        except Exception as e:
            logger.exception("Failed to add chunks to collection %s", collection_name)
            return {"status": "error", "message": f"Failed to index: {e}"}

        # Update manifest
        self.manifest[file_key] = {
            "hash": file_hash,
            "filename": path.name,
            "collection": collection_name,
            "chunks": len(chunks),
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_manifest()

        logger.info("Indexed %s → %d chunks in '%s'", path.name, len(chunks), collection_name)
        return {
            "status": "ok",
            "filename": path.name,
            "chunks_added": len(chunks),
            "collection": collection_name,
        }

    def get_manifest_status(self) -> dict:
        """Return manifest summary for admin dashboard."""
        files_by_collection = defaultdict(list)
        for file_key, info in self.manifest.items():
            files_by_collection[info.get("collection", "unknown")].append({
                "filename": info.get("filename", Path(file_key).name),
                "chunks": info.get("chunks", 0),
                "hash": info.get("hash", "")[:8],
                "indexed_at": info.get("indexed_at", ""),
            })
        return {
            "total_files": len(self.manifest),
            "by_collection": dict(files_by_collection),
        }

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
            try:
                self.bm25_indexes[collection_name] = (BM25Okapi(tokenized), collection)
            except Exception as e:
                logger.error("BM25 index rebuild failed for %s: %s", collection_name, e)

        # 3. 豆包向量化（新增/追加）
        if self.embedding_enabled and self.embedder:
            try:
                new_embeddings = self.embedder.embed(texts)
                if collection_name not in self.embeddings:
                    self.embeddings[collection_name] = []
                self.embeddings[collection_name].extend(new_embeddings)
            except Exception as e:
                logger.warning("Embedding failed for %s: %s", collection_name, e)

        logger.info("Added %d chunks to '%s' (total: %d)", len(chunks), collection_name, len(collection))
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
                logger.warning("Vector search failed: %s", e)

        # === BM25检索 ===
        query_tokens = self._tokenize(query)
        for name in collection_names:
            if name not in self.bm25_indexes:
                continue
            try:
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
            except Exception as e:
                logger.error("BM25 search failed for %s: %s", name, e)

        # === 融合：向量优先，BM25补充 ===
        merged = {}

        for key, doc in vector_results.items():
            merged[key] = doc

        for key, doc in bm25_results.items():
            content_key = doc["content"][:80]
            if not any(d["content"][:80] == content_key for d in merged.values()):
                doc["score"] = doc["score"] * 0.7  # BM25权重稍低
                merged[key] = doc

        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        # 过滤内部元数据
        results = [
            r for r in results
            if r.get("metadata", {}).get("source") != "self-training-reflection"
            and r.get("metadata", {}).get("type") != "feedback"
        ]

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

            record_str = doc["metadata"].get("record")
            if record_str:
                try:
                    record = json.loads(record_str)
                    if all(str(record.get(k, "")).strip() == str(v).strip() for k, v in filters.items()):
                        matched = True
                except json.JSONDecodeError:
                    pass

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
        try:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
        except Exception:
            return 0.0

    # ===== Persistence =====
    def _save_to_disk(self):
        for name, collection in self.collections.items():
            try:
                filepath = self.persist_dir / f"{name}.json"
                data = [{"content": d["content"], "metadata": d["metadata"]} for d in collection]
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error("Failed to save collection %s: %s", name, e)

        if self.embeddings:
            try:
                emb_path = self.persist_dir / "embeddings.json"
                with open(emb_path, "w", encoding="utf-8") as f:
                    json.dump(self.embeddings, f, ensure_ascii=False)
            except Exception as e:
                logger.error("Failed to save embeddings: %s", e)

    def _load_from_disk(self):
        for name in ["policy", "exam", "psychology"]:
            filepath = self.persist_dir / f"{name}.json"
            if filepath.exists():
                try:
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
                    logger.info("Loaded %d docs from %s", len(data), filepath.name)
                except Exception as e:
                    logger.error("Failed to load collection %s: %s", name, e)

        # 加载向量
        emb_path = self.persist_dir / "embeddings.json"
        if emb_path.exists() and self.embedding_enabled:
            try:
                with open(emb_path, "r", encoding="utf-8") as f:
                    self.embeddings = json.load(f)
                logger.info("Loaded embeddings from disk")
            except Exception as e:
                logger.error("Failed to load embeddings: %s", e)

    def get_stats(self) -> dict:
        return {name: len(coll) for name, coll in self.collections.items()}


if __name__ == "__main__":
    from logger_config import get_logger
    get_logger(__name__)

    vs = VectorStore()
    print(f"Stats: {vs.get_stats()}")
    print(f"Embedding enabled: {vs.embedding_enabled}")
    print(f"Manifest files: {len(vs.manifest)}")
