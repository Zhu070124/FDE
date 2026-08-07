"""测试完整流水线：文档加载→向量化→检索（含断言）"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from document_loader import DocumentLoader
from vector_store import VectorStore
import config


def classify_chunks(chunks):
    """基于文件名+内容特征分类。优先级：扩展名 > 文件名关键词 > 内容关键词"""
    policy_chunks, exam_chunks, psych_chunks = [], [], []
    unmatched = []

    for c in chunks:
        src = c.metadata.get("source", "").lower()
        content = c.content  # 全文，不截断

        # 1. 结构化数据优先：电子表格按扩展名归入exam
        if src.endswith(".xlsx") or src.endswith(".xls"):
            exam_chunks.append(c)
        # 2. 心理文档：文件名含心理关键词
        elif "心理" in src:
            psych_chunks.append(c)
        # 3. 政策文档：文件名关键词
        elif any(k in src for k in ["资助", "就业", "政策", "奖学金", "助学"]):
            policy_chunks.append(c)
        # 4. 考研数据：文件名关键词
        elif any(k in src for k in ["考研", "分数线", "报录比"]):
            exam_chunks.append(c)
        # 5. 内容二次判断（仅当前面规则都没匹配时）
        elif any(kw in content for kw in ["心理", "焦虑", "失眠", "抑郁"]):
            psych_chunks.append(c)
        elif any(kw in content for kw in ["奖学金", "助学金", "贷款", "三方协议"]):
            policy_chunks.append(c)
        elif any(kw in content for kw in ["复试线", "报录比", "招生人数"]):
            exam_chunks.append(c)
        else:
            unmatched.append(c)

    if unmatched:
        print(f"  Warning: {len(unmatched)} chunks unmatched, routing to policy:")
        for c in unmatched:
            print(f"     - {c.metadata.get('source', '?')}: {c.content[:60]}...")
        policy_chunks.extend(unmatched)

    return policy_chunks, exam_chunks, psych_chunks


def main():
    """主流程——仅当 __name__ == '__main__' 时执行"""
    # 1. Load documents
    print("=" * 50)
    print("Step 1: Loading documents...")
    loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
    all_chunks = loader.load_directory(config.DOCUMENTS_DIR)

    if not all_chunks:
        print("ERROR: No documents found in data/documents/. Aborting.")
        sys.exit(1)
    print(f"  Total chunks: {len(all_chunks)}")

    # 2. Categorize
    policy_chunks, exam_chunks, psych_chunks = classify_chunks(all_chunks)
    print(f"  Policy: {len(policy_chunks)} | Exam: {len(exam_chunks)} | Psychology: {len(psych_chunks)}")

    # 3. Isolated vector store (does not corrupt persisted data)
    print()
    print("Step 2: Initializing isolated vector store...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="cqupt_test_"))
    vs = VectorStore(persist_dir=tmp_dir)

    print()
    print("Step 3: Adding chunks to collections...")
    if policy_chunks:
        vs.add_chunks(policy_chunks, "policy")
    if exam_chunks:
        vs.add_chunks(exam_chunks, "exam")
    if psych_chunks:
        vs.add_chunks(psych_chunks, "psychology")

    # 4. Test search with explicit validation (no bare assert)
    print()
    print("Step 4: Testing search...")
    tests = [
        ("奖学金申请条件", "policy"),
        ("计算机专业分数线", "exam"),
        ("考试焦虑失眠", "psychology"),
    ]
    failed = 0

    for q, expected_coll in tests:
        results = vs.search(q, ["policy", "exam", "psychology"], top_k=2)

        if not results:
            failed += 1
            print(f"  FAIL [{q}]: returned no results (expected: {expected_coll})")
            continue

        top_coll = results[0]["collection"]
        match = "PASS" if top_coll == expected_coll else "FAIL"
        if top_coll != expected_coll:
            failed += 1
        print(f"  {match} [{q}] → [{top_coll}] (expected: {expected_coll})")
        for r in results:
            print(f"       score={r['score']:.3f} | {r['content'][:80]}...")

    print()
    print("=" * 50)
    if failed > 0:
        print(f"FAIL: {failed}/{len(tests)} queries failed")
        sys.exit(1)
    else:
        print("PASS: all assertions green")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
