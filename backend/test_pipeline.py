"""测试完整流水线：文档加载→向量化→检索"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from document_loader import DocumentLoader
from vector_store import VectorStore
import config

# 1. Load documents
print("=" * 50)
print("Step 1: Loading documents...")
loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
all_chunks = loader.load_directory(config.DOCUMENTS_DIR)
print(f"Total chunks: {len(all_chunks)}")

# 2. Categorize
policy_chunks = []
exam_chunks = []
psych_chunks = []

for c in all_chunks:
    src = c.metadata.get("source", "").lower()
    if any(k in src for k in ["资助", "就业", "政策"]):
        policy_chunks.append(c)
    elif any(k in src for k in ["考研", "分数线", "xlsx"]):
        exam_chunks.append(c)
    elif "心理" in src:
        psych_chunks.append(c)
    else:
        policy_chunks.append(c)

print(f"  Policy: {len(policy_chunks)} chunks")
print(f"  Exam: {len(exam_chunks)} chunks")
print(f"  Psychology: {len(psych_chunks)} chunks")

# 3. Vector store + embedding
print()
print("Step 2: Initializing vector store + BGE model...")
print("(First run downloads BGE-large-zh-v1.5, ~1.3GB)")
vs = VectorStore()

print()
print("Step 3: Adding chunks to collections...")
if policy_chunks:
    vs.add_chunks(policy_chunks, "policy")
if exam_chunks:
    vs.add_chunks(exam_chunks, "exam")
if psych_chunks:
    vs.add_chunks(psych_chunks, "psychology")

# 4. Test search
print()
print("Step 4: Testing search...")
test_queries = [
    ("奖学金申请条件", "policy"),
    ("计算机专业分数线", "exam"),
    ("考试焦虑失眠", "psychology"),
]
for q, expected_coll in test_queries:
    results = vs.search(q, ["policy", "exam", "psychology"], top_k=2)
    print(f"  Query: {q}")
    for r in results:
        print(f"    [{r['collection']}] score={r['score']:.3f} | {r['content'][:80]}...")

print()
print("=" * 50)
print("Pipeline test complete!")
