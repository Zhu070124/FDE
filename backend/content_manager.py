"""
内容管理：文档上传 + 自动重新向量化
"""
import shutil
import logging
from pathlib import Path
from datetime import datetime

from fastapi import UploadFile, HTTPException

from database import get_db
from document_loader import DocumentLoader
import config

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def upload_document(
    file: UploadFile,
    collection: str,
    uploaded_by: int = None,
    vector_store=None,
) -> dict:
    """上传文档 → 保存 → 重新向量化"""
    # 1. 校验
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {suffix}，支持: {', '.join(ALLOWED_EXTENSIONS)}")

    # 2. 保存到磁盘
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = config.DOCUMENTS_DIR / file.filename

    file_size = 0
    with open(dest_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):  # 1MB chunks
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                dest_path.unlink(missing_ok=True)
                raise HTTPException(400, f"文件过大，最大 {MAX_FILE_SIZE // 1024 // 1024}MB")
            f.write(chunk)

    # 3. 加载+向量化
    chunks_added = 0
    if vector_store:
        try:
            loader = DocumentLoader(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            chunks = loader.load_file(dest_path)
            if chunks:
                vector_store.add_chunks(chunks, collection)
                chunks_added = len(chunks)
            logger.info("Uploaded %s → %d chunks to '%s'", file.filename, chunks_added, collection)
        except Exception as e:
            logger.exception("Vectorization failed for %s", file.filename)
            # Don't fail the upload — file is already saved
            pass

    # 4. 记录到数据库
    conn = get_db()
    conn.execute(
        "INSERT INTO documents (filename, collection, uploaded_by, file_size) VALUES (?, ?, ?, ?)",
        (file.filename, collection, uploaded_by, file_size),
    )
    conn.commit()
    conn.close()

    return {
        "status": "ok",
        "filename": file.filename,
        "collection": collection,
        "file_size_kb": round(file_size / 1024, 1),
        "chunks_added": chunks_added,
    }


def list_documents(collection: str = None) -> list:
    """列出已上传的文档"""
    conn = get_db()
    if collection:
        rows = conn.execute(
            "SELECT id, filename, collection, uploaded_at, file_size FROM documents WHERE collection=? ORDER BY uploaded_at DESC",
            (collection,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, filename, collection, uploaded_at, file_size FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document(doc_id: int, vector_store=None) -> dict:
    """删除文档（从数据库+磁盘+向量库）"""
    conn = get_db()
    doc = conn.execute("SELECT id, filename FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(404, "文档不存在")

    # 删除磁盘文件
    filepath = config.DOCUMENTS_DIR / doc["filename"]
    if filepath.exists():
        filepath.unlink()

    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

    # TODO: 从向量库移除该文档的向量（需要向量库支持按source删除）

    return {"status": "deleted", "filename": doc["filename"]}
