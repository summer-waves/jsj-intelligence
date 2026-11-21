"""
evaluate.py - Step 4 of the JSJ Intelligence pipeline.

Measures how good each retrieval method is, using the questions in eval/questions.csv.

For every question the script asks each method for its top K chunks, then checks each
returned chunk against the answer key. A chunk counts as RELEVANT when:
    1. its doc_id is listed in the question's relevant_docs, AND
    2. its text contains every word/phrase listed in must_contain (lowercase match).
Because relevance is decided by text rather than by chunk number, the answer key keeps
working if you change the chunk size in chunk.py and re-run.

Metrics (all averaged over the questions):
    Hit@K        share of questions with at least one relevant chunk in the top K
    Precision@K  share of the K returned chunks that are relevant
    Recall@K     relevant chunks found in the top K / all relevant chunks that exist
    MRR          mean of 1 / (rank of the first relevant chunk); 0 if none in the top K
    Latency      average milliseconds per question

Run it from the project folder:
    python src/evaluate.py
    python src/evaluate.py --k 3
    python src/evaluate.py --methods bm25 hybrid
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import BM25Retriever, HybridRetriever, load_chunks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.csv"
RESULTS_PATH = ROOT / "eval" / "results.csv"


def split_list(cell):
    return [x.strip().lower() for x in cell.split(";") if x.strip()]


def load_questions(path=None):
    with open(path or QUESTIONS_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["relevant_docs"] = set(split_list(r["relevant_docs"]))
        r["must_contain"] = split_list(r["must_contain"])
    return rows


def is_relevant(chunk, question):
    """The answer-key rule described at the top of this file."""
    if chunk["doc_id"].lower() not in question["relevant_docs"]:
        return False
    text = chunk["text"].lower()
    return all(term in text for term in question["must_contain"])


def build_retrievers(chunks, wanted):
    """Create the requested retrievers. 'dense' and 'hybrid' need sentence-transformers."""
    retrievers = {}
    bm25 = BM25Retriever(chunks)
    if "bm25" in wanted:
        retrievers["bm25"] = bm25
    if "dense" in wanted or "hybrid" in wanted:
        try:
            from retrieval import DenseRetriever
            dense = DenseRetriever(chunks)
        except SystemExit as e:  # retrieval.py exits when sentence-transformers is missing
            print(f"Skipping dense and hybrid: {e}")
            dense = None
        if dense is not None:
            if "dense" in wanted:
                retrievers["dense"] = dense
            if "hybrid" in wanted:
                retrievers["hybrid"] = HybridRetriever(chunks, [bm25, dense])
    return retrievers


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate the retrieval methods.")
    parser.add_argument("--k", type=int, default=5, help="how many chunks each method returns")
    parser.add_argument("--methods", nargs="+", default=["bm25", "dense", "hybrid"],
                        choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--questions", default=None,
                        help="questions CSV to use instead of eval/questions.csv")
    args = parser.parse_args()
    k = args.k

    questions_path = None
    results_path = RESULTS_PATH
    if args.questions:
        questions_path = Path(args.questions)
        if not questions_path.is_absolute():
            questions_path = ROOT / questions_path
        results_path = ROOT / "eval" / f"results_{questions_path.stem}.csv"

    chunks = load_chunks()
    questions = load_questions(questions_path)

    # Check the answer key first: a question with no relevant chunk can never be answered,
    # so it would unfairly count against every method.
    bad = []
    for q in questions:
        q["total_relevant"] = sum(is_relevant(c, q) for c in chunks)
        if q["total_relevant"] == 0:
            bad.append(q)
    if bad:
        print("LABELING PROBLEM - these questions have no relevant chunk at all:")
        for q in bad:
            print(f"  {q['question_id']}: {q['question']}  "
                  f"(docs={sorted(q['relevant_docs'])}, must_contain={q['must_contain']})")
        print("Fix relevant_docs / must_contain in eval/questions.csv. They are skipped below.\n")
        questions = [q for q in questions if q["total_relevant"] > 0]

    retrievers = build_retrievers(chunks, args.methods)
    print(f"{len(questions)} questions, {len(chunks)} chunks, K={k}\n")

    summary = {}
    rows = []
    by_type = defaultdict(lambda: defaultdict(list))  # method -> type -> list of hit flags

    for name, retriever in retrievers.items():
        retriever.search("warm up", k)  # first call is slow (model loading); keep it out of timing
        hits, precisions, recalls, rrs, latencies = [], [], [], [], []
        failed = []
        for q in questions:
            start = time.perf_counter()
            results = retriever.search(q["question"], k)
            latencies.append((time.perf_counter() - start) * 1000)

            flags = [is_relevant(r, q) for r in results]
            n_rel = sum(flags)
            first = next((i for i, f in enumerate(flags, start=1) if f), None)

            hit = 1.0 if n_rel else 0.0
            hits.append(hit)
            precisions.append(n_rel / k)
            recalls.append(n_rel / q["total_relevant"])
            rrs.append(1.0 / first if first else 0.0)
            by_type[name][q["type"]].append(hit)
            if not n_rel:
                failed.append(q)

            rows.append({
                "method": name, "question_id": q["question_id"], "type": q["type"],
                "question": q["question"], "hit": int(hit), "first_relevant_rank": first or "",
                "relevant_in_top_k": n_rel, "total_relevant": q["total_relevant"],
                "top_chunks": " ".join(r["chunk_id"] for r in results),
            })

        n = len(questions)
        summary[name] = {
            f"Hit@{k}": sum(hits) / n, f"Precision@{k}": sum(precisions) / n,
            f"Recall@{k}": sum(recalls) / n, "MRR": sum(rrs) / n,
            "Latency ms": sum(latencies) / n,
        }
        summary[name]["_failed"] = failed

    # ---- main table
    cols = [f"Hit@{k}", f"Precision@{k}", f"Recall@{k}", "MRR", "Latency ms"]
    print(f"{'method':<10}" + "".join(f"{c:>15}" for c in cols))
    for name, s in summary.items():
        print(f"{name:<10}" + "".join(
            f"{s[c]:>15.1f}" if c == "Latency ms" else f"{s[c]:>15.3f}" for c in cols))

    # ---- Hit@K split by question type
    types = sorted({q["type"] for q in questions})
    print(f"\nHit@{k} by question type")
    print(f"{'type':<16}{'n':>4}" + "".join(f"{m:>10}" for m in summary))
    for t in types:
        n_t = sum(q["type"] == t for q in questions)
        print(f"{t:<16}{n_t:>4}" + "".join(
            f"{sum(by_type[m][t]) / len(by_type[m][t]):>10.2f}" for m in summary))

    # ---- failed questions
    for name, s in summary.items():
        print(f"\n{name}: {len(s['_failed'])} question(s) with no relevant chunk in the top {k}")
        for q in s["_failed"]:
            print(f"  {q['question_id']} [{q['type']}] {q['question']}")

    # ---- save per-question results
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-question results saved to {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()