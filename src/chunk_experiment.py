import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chunk as chunk_module  # noqa: E402
import retrieval  # noqa: E402
from evaluate import is_relevant, load_questions  # noqa: E402
from retrieval import BM25Retriever, HybridRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "eval" / "chunk_experiment.csv"
EXPERIMENT_DIR = ROOT / "data" / "chunks" / "experiment"

# (MAX_WORDS, OVERLAP_WORDS). The overlap is about 20% of the chunk size.
SETTINGS = [(80, 15), (120, 25), (180, 35), (250, 50), (400, 80)]


def build_chunks(max_words, overlap_words):
    """Same logic as chunk.py's main(), but returns the chunks instead of writing a file."""
    chunk_module.MAX_WORDS = max_words        # pack_units() reads these two module settings
    chunk_module.OVERLAP_WORDS = overlap_words
    log = chunk_module.load_source_log()
    chunks = []
    for path in sorted(chunk_module.CLEAN_DIR.glob("jsj_*.txt")):
        doc_id = path.stem
        units = chunk_module.split_units(path.read_text(encoding="utf-8"))
        for i, chunk_units in enumerate(chunk_module.pack_units(units), start=1):
            text = " ".join(chunk_units)
            chunks.append({
                "chunk_id": f"{doc_id}_c{i:02d}", "doc_id": doc_id, "title": log[doc_id]["title"],
                "site": log[doc_id]["site"], "url": log[doc_id]["url"],
                "chunk_index": i, "word_count": len(text.split()), "text": text,
            })
    return chunks


def evaluate(retriever, questions, k):
    hits = precisions = recalls = rr = 0.0
    for q in questions:
        results = retriever.search(q["question"], k)
        flags = [is_relevant(r, q) for r in results]
        n_rel = sum(flags)
        first = next((i for i, f in enumerate(flags, start=1) if f), None)
        hits += 1.0 if n_rel else 0.0
        precisions += n_rel / k
        recalls += n_rel / q["total_relevant"]
        rr += 1.0 / first if first else 0.0
    n = len(questions)
    return hits / n, precisions / n, recalls / n, rr / n


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    k = args.k

    # Keep the experiment's embeddings away from the real ones used by the app.
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    have_dense = True
    try:
        from retrieval import DenseRetriever  # noqa: F401
        import sentence_transformers  # noqa: F401
    except (ImportError, SystemExit):
        have_dense = False
        print("sentence-transformers not available: running BM25 only.\n")

    all_questions = load_questions()
    rows = []
    for max_words, overlap in SETTINGS:
        chunks = build_chunks(max_words, overlap)
        questions = []
        for q in all_questions:
            q = dict(q)
            q["total_relevant"] = sum(is_relevant(c, q) for c in chunks)
            if q["total_relevant"] > 0:   # a question whose key phrase was split by a chunk boundary is skipped
                questions.append(q)
        skipped = len(all_questions) - len(questions)
        avg_words = sum(c["word_count"] for c in chunks) / len(chunks)
        print(f"MAX_WORDS={max_words}, OVERLAP={overlap}: {len(chunks)} chunks, "
              f"avg {avg_words:.0f} words, {len(questions)} questions scored ({skipped} skipped)")

        # Separate embedding files for this setting
        tag = f"{max_words}_{overlap}"
        retrieval.EMBEDDINGS_PATH = EXPERIMENT_DIR / f"embeddings_{tag}.npy"
        retrieval.EMBEDDINGS_META_PATH = EXPERIMENT_DIR / f"embeddings_{tag}_meta.json"

        bm25 = BM25Retriever(chunks)
        retrievers = {"bm25": bm25}
        if have_dense:
            dense = retrieval.DenseRetriever(chunks)
            retrievers["dense"] = dense
            retrievers["hybrid"] = HybridRetriever(chunks, [bm25, dense])

        for name, retriever in retrievers.items():
            hit, prec, rec, mrr = evaluate(retriever, questions, k)
            rows.append({
                "max_words": max_words, "overlap_words": overlap, "chunks": len(chunks),
                "avg_words": round(avg_words), "method": name, "questions": len(questions),
                f"hit@{k}": round(hit, 3), f"precision@{k}": round(prec, 3),
                f"recall@{k}": round(rec, 3), "mrr": round(mrr, 3),
            })

    methods = sorted({r["method"] for r in rows}, key=["bm25", "dense", "hybrid"].index)
    for metric in (f"hit@{k}", "mrr", f"recall@{k}", f"precision@{k}"):
        print(f"\n{metric}")
        print(f"{'MAX/OVERLAP':<14}{'chunks':>8}" + "".join(f"{m:>10}" for m in methods))
        for max_words, overlap in SETTINGS:
            sel = {r["method"]: r for r in rows if r["max_words"] == max_words}
            first = next(iter(sel.values()))
            print(f"{max_words}/{overlap:<10}{first['chunks']:>8}"
                  + "".join(f"{sel[m][metric]:>10.3f}" for m in methods))

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {RESULTS_PATH.relative_to(ROOT)}")
    print("Note: precision and recall change with the number of chunks, so compare hit and MRR first.")


if __name__ == "__main__":
    main()
