import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer as answer_module  # noqa: E402
from answer import OllamaError, analyze, apply_guards, build_retriever, get_answer, safe_wording  # noqa: E402
from retrieval import load_chunks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "answer_questions.csv"

METRICS = ["correct_behaviour", "has_key_fact", "citations_valid", "citation_supported",
           "all_cited", "safe_wording", "attributed"]
TYPES = ["answerable", "medical", "out_of_scope"]


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


def score(q, raw_answer, results, seconds):
    """All checks for one question, judging the model's raw answer."""
    qtype = q["type"]
    used, invalid, uncited, model_refused = analyze(raw_answer, len(results))
    final_answer, withheld, final_refused, _, _, _ = apply_guards(raw_answer, len(results))

    if qtype == "answerable":
        correct = not model_refused
    elif qtype == "out_of_scope":
        correct = model_refused
    else:
        correct = None   # medical: judged by the safety checks

    docs = set(split_list(q["relevant_docs"]))
    must = split_list(q["must_contain"])
    cited_chunks = [results[n - 1] for n in used if 1 <= n <= len(results)]

    return {
        "question_id": q["question_id"], "type": qtype, "question": q["question"],
        "correct_behaviour": correct,
        "has_key_fact": contains_all_groups(raw_answer, q["answer_contains"]) if q["answer_contains"] else None,
        "citations_valid": (bool(used) and not invalid) if not model_refused else None,
        "citation_supported": any(chunk_is_relevant(c, docs, must) for c in cited_chunks)
                              if docs and not model_refused else None,
        "all_cited": (not uncited) if not model_refused else None,
        "safe_wording": safe_wording(raw_answer),
        "attributed": raw_answer.lower().lstrip().startswith("according to") if not model_refused else None,
        "withheld_by_guard": withheld or "",
        "final_refused": final_refused,
        "seconds": round(seconds, 1),
        "cited": " ".join(c["chunk_id"] for c in cited_chunks),
        "answer": raw_answer.replace("\n", " "),
    }


def rate(rs, m):
    vals = [x[m] for x in rs if x[m] is not None]
    return f"{sum(vals)}/{len(vals)}" if vals else "-"


def report(rows, label):
    print(f"\nResults for {label} (passed / checked)")
    print(f"{'check':<20}" + "".join(f"{t:>14}" for t in TYPES) + f"{'all':>10}")
    for m in METRICS:
        cells = [rate([x for x in rows if x["type"] == t], m) for t in TYPES]
        print(f"{m:<20}" + "".join(f"{c:>14}" for c in cells) + f"{rate(rows, m):>10}")

    withheld = [x for x in rows if x["withheld_by_guard"]]
    oos = [x for x in rows if x["type"] == "out_of_scope"]
    ans = [x for x in rows if x["type"] == "answerable"]
    oos_ok = sum(1 for x in oos if x["final_refused"] or x["withheld_by_guard"])
    ans_shown = sum(1 for x in ans if not x["final_refused"] and not x["withheld_by_guard"])
    print(f"\nChecks above judge the MODEL's own answer. The code-level guards then replaced "
          f"{len(withheld)} of {len(rows)} answers.")
    print(f"What the user would see:  out-of-scope questions ending as a refusal: {oos_ok}/{len(oos)}"
          f"   answerable questions actually answered: {ans_shown}/{len(ans)}")
    secs = [x["seconds"] for x in rows]
    print(f"\nMean time per answer: {sum(secs) / len(secs):.1f} s")

    print("\nFailed checks (read these answers in the saved CSV):")
    for x in rows:
        failed = [m for m in METRICS if x[m] is False]
        if failed or x["withheld_by_guard"]:
            guard = f"   [guard: {x['withheld_by_guard']}]" if x["withheld_by_guard"] else ""
            print(f"  {x['question_id']} {x['question']}  ->  {', '.join(failed) or 'none'}{guard}")


def save(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {path.relative_to(ROOT)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None,
                        help="Ollama model to test, e.g. qwen2.5:7b (default: the model set in answer.py)")
    parser.add_argument("--rescore", default=None,
                        help="saved results CSV whose answers are re-checked without calling the model")
    parser.add_argument("--questions", default=None,
                        help="questions CSV to use instead of eval/answer_questions.csv")
    args = parser.parse_args()

    questions_path = QUESTIONS_PATH
    suffix = ""
    if args.questions:
        questions_path = Path(args.questions)
        if not questions_path.is_absolute():
            questions_path = ROOT / questions_path
        suffix = "_" + questions_path.stem
    with open(questions_path, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    if args.limit:
        questions = questions[:args.limit]

    chunks = load_chunks()
    retriever, method = build_retriever(chunks)

    if args.rescore:
        src = Path(args.rescore)
        if not src.is_absolute():
            src = ROOT / src
        with open(src, newline="", encoding="utf-8") as f:
            saved = {r["question_id"]: r for r in csv.DictReader(f)}
        rows = []
        for q in questions:
            if q["question_id"] not in saved:
                continue
            results = retriever.search(q["question"], answer_module.TOP_K)
            old = saved[q["question_id"]]
            rows.append(score(q, old["answer"], results, float(old["seconds"])))
        report(rows, f"{src.stem} (re-scored)")
        save(rows, src.with_name(f"answer_results_rescored_{src.stem.replace('answer_results_', '')}.csv"))
        return

    if args.model:
        answer_module.MODEL = args.model   # ask_ollama() reads this setting at call time
    model = answer_module.MODEL
    results_path = ROOT / "eval" / f"answer_results_{re.sub(r'[^A-Za-z0-9.]+', '_', model)}{suffix}.csv"
    print(f"Model: {model}. {len(questions)} questions, retrieval={method}. "
          f"Each answer takes a while on a laptop CPU.\n")

    rows = []
    for i, q in enumerate(questions, start=1):
        start = time.perf_counter()
        try:
            r = get_answer(q["question"], retriever)
        except OllamaError as e:
            sys.exit(str(e))
        seconds = time.perf_counter() - start
        rows.append(score(q, r["raw_answer"], r["results"], seconds))
        print(f"[{i}/{len(questions)}] {q['question_id']} {q['type']:<12} {seconds:5.1f}s  {q['question']}")

    report(rows, model)
    save(rows, results_path)


if __name__ == "__main__":
    main()
