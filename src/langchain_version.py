"""
langchain_version.py - the same retrieval step, built with LangChain, compared with ours.

It uses exactly the same chunks (data/chunks/chunks.jsonl), the same 34 questions
(eval/questions.csv), the same relevance rule and the same metrics as evaluate.py, so any
difference in the numbers comes from the retrievers themselves:

    from scratch (retrieval.py)          LangChain
    BM25Retriever (stemming, stopwords)  langchain_community BM25Retriever (rank_bm25, whitespace split)
    DenseRetriever (our numpy search)    InMemoryVectorStore + HuggingFaceEmbeddings (same MiniLM model)
    HybridRetriever (our RRF)            EnsembleRetriever (LangChain's own rank fusion)

Install (once, in your .venv):   pip install -r requirements-langchain.txt
Run from the project folder:     python src/langchain_version.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from langchain_community.retrievers import BM25Retriever as LCBM25
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    try:                                   # LangChain 1.x keeps the older retrievers here
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:                    # older LangChain versions
        from langchain.retrievers import EnsembleRetriever
except ImportError as e:
    sys.exit(f"A LangChain package is missing ({e}).\nRun: pip install -r requirements-langchain.txt")

from chunk_experiment import evaluate as run_metrics  # noqa: E402  (same metric code as the other experiments)
from evaluate import is_relevant, load_questions  # noqa: E402
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, chunk_text, load_chunks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "eval" / "langchain_comparison.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LangChainAdapter:
    """Gives a LangChain retriever the same .search(query, k) interface as our own retrievers."""

    def __init__(self, lc_retriever):
        self.lc = lc_retriever

    def search(self, query, k=5):
        docs = self.lc.invoke(query)[:k]
        return [{"chunk_id": d.metadata["chunk_id"], "doc_id": d.metadata["doc_id"],
                 "text": d.metadata["text"]} for d in docs]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    k = args.k

    chunks = load_chunks()
    questions = []
    for q in load_questions():
        q["total_relevant"] = sum(is_relevant(c, q) for c in chunks)
        if q["total_relevant"] > 0:
            questions.append(q)

    # ---- our own retrievers
    ours_bm25 = BM25Retriever(chunks)
    ours_dense = DenseRetriever(chunks)
    ours_hybrid = HybridRetriever(chunks, [ours_bm25, ours_dense])

    # ---- LangChain retrievers over the same chunks (title + text, as in our own indexing)
    documents = [
        Document(page_content=chunk_text(c),
                 metadata={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["text"]})
        for c in chunks
    ]
    lc_bm25 = LCBM25.from_documents(documents, k=k)
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    store = InMemoryVectorStore.from_documents(documents, embeddings)
    lc_dense = store.as_retriever(search_kwargs={"k": k})
    lc_hybrid = EnsembleRetriever(retrievers=[lc_bm25, lc_dense])  # default: equal weights

    contenders = [
        ("from scratch", "bm25", ours_bm25), ("LangChain", "bm25", LangChainAdapter(lc_bm25)),
        ("from scratch", "dense", ours_dense), ("LangChain", "dense", LangChainAdapter(lc_dense)),
        ("from scratch", "hybrid", ours_hybrid), ("LangChain", "hybrid", LangChainAdapter(lc_hybrid)),
    ]

    rows = []
    print(f"{len(questions)} questions, {len(chunks)} chunks (same chunks for every row), K={k}\n")
    print(f"{'method':<8}{'build':<14}{f'hit@{k}':>8}{f'prec@{k}':>10}{f'recall@{k}':>11}{'mrr':>8}{'ms/query':>10}")
    for build, method, retriever in contenders:
        retriever.search("warm up", k)
        start = time.perf_counter()
        hit, prec, rec, mrr = run_metrics(retriever, questions, k)
        ms = (time.perf_counter() - start) * 1000 / len(questions)
        print(f"{method:<8}{build:<14}{hit:>8.3f}{prec:>10.3f}{rec:>11.3f}{mrr:>8.3f}{ms:>10.1f}")
        rows.append({"method": method, "build": build, f"hit@{k}": round(hit, 3),
                     f"precision@{k}": round(prec, 3), f"recall@{k}": round(rec, 3),
                     "mrr": round(mrr, 3), "ms_per_query": round(ms, 1)})

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
