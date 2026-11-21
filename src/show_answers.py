"""
show_answers.py - print saved answers from an answer_results_*.csv so you can read them.

Usage (from the project folder):
    python src/show_answers.py eval/answer_results_llama3.2_independent_answer_questions.csv i02 i06 i07
Give no question ids to print every answer.
"""

import csv
import sys
import textwrap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 2:
    sys.exit("Usage: python src/show_answers.py <results csv> [question ids...]")

path, wanted = sys.argv[1], set(sys.argv[2:])
with open(path, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if wanted and r["question_id"] not in wanted:
            continue
        failed = [m for m in ("correct_behaviour", "has_key_fact", "citations_valid", "citation_supported",
                              "all_cited", "safe_wording", "attributed") if r[m] == "False"]
        print(f"=== {r['question_id']} [{r['type']}] {r['question']}")
        print(f"cited chunks: {r['cited'] or '-'}   failed checks: {', '.join(failed) or 'none'}"
              + (f"   guard: {r['withheld_by_guard']}" if r["withheld_by_guard"] else ""))
        print(textwrap.fill("ANSWER: " + r["answer"], width=110) + "\n")