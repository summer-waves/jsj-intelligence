"""
llamaindex_version.py - the same retrieval step, built with LlamaIndex, compared with ours.

Same chunks (data/chunks/chunks.jsonl), same 34 questions (eval/questions.csv), same relevance
rule and same metrics as evaluate.py and langchain_version.py, so any difference in the numbers
comes from the retrievers themselves:

    from scratch (retrieval.py)          LlamaIndex
    BM25Retriever (stemming, stopwords)  llama_index BM25Retriever (bm25s library)
    DenseRetriever (our numpy search)    VectorStoreIndex + HuggingFaceEmbedding (same MiniLM model)
    HybridRetriever (our RRF)            QueryFusionRetriever, mode "reciprocal_rerank"

Install (once, in your .venv):   pip install -r requirements-llamaindex.txt
Run from the project folder:     python src/llamaindex_version.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever as LIBM25
except ImportError as e:
    sys.exit(f"A LlamaIndex package is missing ({e}).\nRun: pip install -r requirements-llamaindex.txt")

from chunk_experiment import evaluate as run_metrics  # noqa: E402  (same metric code as the other experiments)
from evaluate import is_relevant, load_questions  # noqa: E402
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, chunk_text, load_chunks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "eval" / "llamaindex_comparison.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LlamaIndexAdapter:
    """Gives a LlamaIndex retriever the same .search(query, k) interface as our own retrievers.
    LlamaIndex returns nodes; we look the chunk up by its id (the node id IS our chunk_id)."""

    def __init__(self, li_retriever, chunks_by_id):
        self.li = li_retriever
        self.chunks_by_id = chunks_by_id

    def search(self, query, k=5):
        found = self.li.retrieve(query)[:k]
        out = []
        for hit in found:
            c = self.chunks_by_id[hit.node.node_id]
            out.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["text"]})
        return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    k = args.k

    chunks = load_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    questions = []
    for q in load_questions():
        q["total_relevant"] = sum(is_relevant(c, q) for c in chunks)
        if q["total_relevant"] > 0:
            questions.append(q)

    # ---- our own retrievers
    ours_bm25 = BM25Retriever(chunks)
    ours_dense = DenseRetriever(chunks)
    ours_hybrid = HybridRetriever(chunks, [ours_bm25, ours_dense])

    # ---- LlamaIndex retrievers over the same chunks (title + text, as in our own indexing)
    embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)   # normalizes vectors by default
    Settings.embed_model = embed_model
    Settings.llm = None   # we only retrieve; this stops LlamaIndex from looking for an OpenAI key

    nodes = [TextNode(text=chunk_text(c), id_=c["chunk_id"]) for c in chunks]
    li_bm25 = LIBM25.from_defaults(nodes=nodes, similarity_top_k=k)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    li_dense = index.as_retriever(similarity_top_k=k)
    li_hybrid = QueryFusionRetriever(
        [li_bm25, li_dense], similarity_top_k=k,
        num_queries=1,                 # 1 = just the original question (no LLM query rewriting)
        mode="reciprocal_rerank",      # reciprocal rank fusion, like ours
        use_async=False,
    )

    contenders = [
        ("from scratch", "bm25", ours_bm25), ("LlamaIndex", "bm25", LlamaIndexAdapter(li_bm25, chunks_by_id)),
        ("from scratch", "dense", ours_dense), ("LlamaIndex", "dense", LlamaIndexAdapter(li_dense, chunks_by_id)),
        ("from scratch", "hybrid", ours_hybrid), ("LlamaIndex", "hybrid", LlamaIndexAdapter(li_hybrid, chunks_by_id)),
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
