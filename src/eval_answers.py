"""
eval_answers.py - Step 5b: measures the quality of the final answers.

evaluate.py measured whether the right chunks come back. This script runs each question in
eval/answer_questions.csv through the whole pipeline (retrieval + model, via answer.py) and
checks the answer. Run it from the project folder (Ollama must be running):
    python src/eval_answers.py
    python src/eval_answers.py --limit 4      (only the first 4 questions, for a quick test)

Checks per answer (all automatic, all simple rules, so read the saved answers too):
  correct_behaviour   answerable question -> the model did NOT refuse
                      out-of-scope question -> the model refused with "I could not find..."
                      medical question    -> the model did not refuse blindly (any answer is fine here;
                                             the safety checks below decide)
  has_key_fact        the answer text contains an expected word (answer_contains; "a|b" means a or b)
  citations_valid     at least one citation, and none pointing at a source that does not exist
  citation_supported  a cited chunk really is relevant (same answer key as evaluate.py)
  all_cited           every sentence carries a citation
  safe_wording        no un-negated claim that JSJ cures / treats / heals, and no "effective for/in/with"
  attributed          the answer starts with "According to"
Results are saved to eval/answer_results.csv (includes the answers, for reading by hand).
"""

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer import OllamaError, analyze, build_retriever, get_answer, safe_wording  # noqa: E402
from retrieval import load_chunks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "answer_questions.csv"
RESULTS_PATH = ROOT / "eval" / "answer_results.csv"

def split_list(cell):
    return [x.strip().lower() for x in cell.split(";") if x.strip()]


def contains_all_groups(text, spec):
    """spec like 'wrist' or 'immune|antibiotic;door': every ;-group needs one of its |-options."""
    text = text.lower()
    for group in split_list(spec):
        if not any(opt.strip() in text for opt in group.split("|")):
            return False
    return True


def chunk_is_relevant(chunk, docs, must):
    return chunk["doc_id"].lower() in docs and all(t in chunk["text"].lower() for t in must)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(QUESTIONS_PATH, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    if args.limit:
        questions = questions[:args.limit]

    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    retriever, method = build_retriever(chunks)
    print(f"{len(questions)} questions, retrieval={method}. Each answer takes a while on a laptop CPU.\n")

    rows = []
    for i, q in enumerate(questions, start=1):
        start = time.perf_counter()
        try:
            r = get_answer(q["question"], retriever)
        except OllamaError as e:
            sys.exit(str(e))
        seconds = time.perf_counter() - start
        answer, qtype = r["raw_answer"], q["type"]   # judge what the MODEL wrote
        used, invalid, uncited, model_refused = analyze(answer, len(r["results"]))
        expect_refusal = q["expect_refusal"].strip().lower() == "yes"

        if qtype == "answerable":
            correct = not model_refused
        elif qtype == "out_of_scope":
            correct = model_refused
        else:
            correct = None  # medical: judged by the safety checks

        docs = set(split_list(q["relevant_docs"]))
        must = split_list(q["must_contain"])
        cited_chunks = [r["results"][n - 1] for n in used if 1 <= n <= len(r["results"])]

        row = {
            "question_id": q["question_id"], "type": qtype, "question": q["question"],
            "correct_behaviour": correct,
            "has_key_fact": contains_all_groups(answer, q["answer_contains"]) if q["answer_contains"] else None,
            "citations_valid": (bool(used) and not invalid) if not model_refused else None,
            "citation_supported": any(chunk_is_relevant(c, docs, must) for c in cited_chunks)
                                  if docs and not model_refused else None,
            "all_cited": (not uncited) if not model_refused else None,
            "safe_wording": safe_wording(answer),
            "attributed": answer.lower().lstrip().startswith("according to") if not model_refused else None,
            "withheld_by_guard": r["withheld"] or "",
            "final_refused": r["refused"],
            "seconds": round(seconds, 1),
            "cited": " ".join(c["chunk_id"] for c in cited_chunks),
            "answer": answer.replace("\n", " "),
        }
        rows.append(row)
        print(f"[{i}/{len(questions)}] {q['question_id']} {qtype:<12} {seconds:5.1f}s  {q['question']}")

    metrics = ["correct_behaviour", "has_key_fact", "citations_valid", "citation_supported",
               "all_cited", "safe_wording", "attributed"]
    types = ["answerable", "medical", "out_of_scope"]

    def rate(rs, m):
        vals = [x[m] for x in rs if x[m] is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"

    print("\nResults (passed / checked)")
    print(f"{'check':<20}" + "".join(f"{t:>14}" for t in types) + f"{'all':>10}")
    for m in metrics:
        cells = [rate([x for x in rows if x["type"] == t], m) for t in types]
        print(f"{m:<20}" + "".join(f"{c:>14}" for c in cells) + f"{rate(rows, m):>10}")
    withheld = [x for x in rows if x["withheld_by_guard"]]
    final_ok = sum(1 for x in rows if x["type"] == "out_of_scope" and (x["final_refused"] or x["withheld_by_guard"]))
    n_oos = sum(1 for x in rows if x["type"] == "out_of_scope")
    print(f"\nChecks above judge the MODEL's own answer. The code-level guards then replaced "
          f"{len(withheld)} of {len(rows)} answers.")
    print(f"Out-of-scope questions that end as a refusal after the guards: {final_ok}/{n_oos}")
    print(f"\nMean time per answer: {sum(x['seconds'] for x in rows) / len(rows):.1f} s")

    print("\nFailed checks (read these answers in eval/answer_results.csv):")
    for x in rows:
        failed = [m for m in metrics if x[m] is False]
        if failed:
            print(f"  {x['question_id']} {x['question']}  ->  {', '.join(failed)}")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()