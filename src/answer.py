"""
answer.py - Step 5 of the JSJ Intelligence pipeline.

Answers a question about Jin Shin Jyutsu using ONLY the document library:
    1. finds the best chunks (hybrid search from retrieval.py)
    2. gives those chunks to a local Ollama model, with strict rules
    3. prints the answer, then a list of the sources that were actually retrieved
    4. adds a fixed safety note that does not depend on what the model says

Run it from the project folder (Ollama must be running, and llama3.2 downloaded):
    python src/answer.py "Which finger do I hold for fear?"
    python src/answer.py                      (no question = ask questions in a loop)
    python src/answer.py "..." --show-context (also prints the chunks the model saw)
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import BM25Retriever, HybridRetriever, load_chunks  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
TOP_K = 5

SYSTEM_RULES = """You are JSJ Intelligence, an assistant that answers questions about Jin Shin Jyutsu (JSJ) using ONLY the numbered SOURCES provided in the user's message.

Rules you must follow:
1. Use only information found in the SOURCES. Do not use outside knowledge. Do not guess.
2. After every sentence that uses a source, add its number in square brackets, for example [2]. Use only the numbers of the SOURCES given.
3. Begin the answer with "According to the JSJ teachings in the library," or a similar phrase. Describe what JSJ teaches or claims. Never state it as established medical fact.
4. Never say JSJ cures, treats, heals, or is effective for any disease, and never tell the person to stop or avoid medical care.
   If a SOURCE mentions a condition, it is one author's or practitioner's personal report. Write it as "one source reports that..." and never as a general fact.
   If the question asks whether JSJ can cure or treat a condition, say the library does not show that JSJ cures or treats it, then say what the sources report, with their numbers.
5. If the SOURCES do not contain the answer, reply exactly: "I could not find this in the JSJ library." Do not say anything else.
6. Keep the answer short: at most 5 sentences."""

SAFETY_NOTE = (
    "Note: This describes what Jin Shin Jyutsu teachings say. It is not medical advice, and "
    "the sources are not scientific proof that it works. If you have symptoms, pain, or a "
    "health condition, please see a doctor or other qualified health professional."
)

EMERGENCY_NOTE = (
    "IMPORTANT: Your question may describe an emergency. Please call your local emergency "
    "number now (911 in the US). If you are thinking about harming yourself, call or text 988 "
    "(US Suicide & Crisis Lifeline). Do not rely on this tool in an emergency."
)

HEALTH_NOTE = (
    "IMPORTANT: The JSJ library does not establish that Jin Shin Jyutsu cures or treats any "
    "medical condition. Please keep following the treatment your doctor recommends, and do "
    "not stop or change medication because of anything below."
)

HEALTH_WORDS = [
    "cure", "treat", "heal", "diabet", "cancer", "chemo", "medication", "medicine",
    "disease", "diagnos", "tumor", "infection", "blood pressure", "depression", "pregnan",
]

EMERGENCY_WORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe", "trouble breathing",
    "stroke", "overdose", "suicid", "kill myself", "end my life", "severe bleeding",
    "unconscious", "poison",
]


class OllamaError(Exception):
    """Ollama is not running or not reachable."""


def build_retriever(chunks):
    """Hybrid search if sentence-transformers is installed, otherwise keyword search only."""
    bm25 = BM25Retriever(chunks)
    try:
        from retrieval import DenseRetriever
        return HybridRetriever(chunks, [bm25, DenseRetriever(chunks)]), "hybrid"
    except SystemExit:
        print("(sentence-transformers not available: using BM25 only)")
        return bm25, "bm25"


def build_prompt(question, results):
    sources = []
    for n, r in enumerate(results, start=1):
        sources.append(f"[{n}] ({r['title']})\n{r['text']}")
    return "SOURCES:\n\n" + "\n\n".join(sources) + f"\n\nQUESTION: {question}"


def ask_ollama(prompt):
    body = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise OllamaError(f"Could not reach Ollama at {OLLAMA_URL} ({e.reason}). "
                          f"Open the Ollama app (or run: ollama serve) and try again.")


def check_citations(answer, n_sources):
    """Return the citation numbers the model used, and any that do not exist."""
    used = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    invalid = [x for x in used if x < 1 or x > n_sources]
    return used, invalid


REFUSAL_TEXT = "I could not find this in the JSJ library."
UNSAFE_TEXT = (
    "I can't give a reliable summary for this question. The JSJ library does not show that "
    "Jin Shin Jyutsu cures or treats any medical condition. The passages that were found are "
    "listed under Sources so you can read them yourself."
)

# A claim that JSJ cures / treats / heals / "is effective for..." is unsafe unless negated nearby.
CLAIM = re.compile(r"\b(cure[sd]?|heal[s]?|treat[s]?|effective (?:for|in|with|at))\b", re.I)
NEGATION = re.compile(r"(not|n't|no|never|cannot|nor|neither|without)\b", re.I)


def safe_wording(answer):
    """False if the answer makes an un-negated cure/treat/heal/effective claim."""
    for m in CLAIM.finditer(answer):
        before = answer[max(0, m.start() - 45):m.start()]
        if not NEGATION.search(before):
            return False
    return True


def analyze(answer, n_sources):
    """Citation facts about an answer text: numbers used, invalid numbers, uncited sentences, refused?"""
    used, invalid = check_citations(answer, n_sources)
    refused = "could not find this in the jsj library" in answer.lower()
    uncited = []
    if not refused and used:
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", answer) if x.strip()]
        uncited = [x for x in sentences if not re.search(r"\[\d+\]", x)]
    return used, invalid, uncited, refused


def get_answer(question, retriever):
    """Retrieve, ask the model, and run every check. Returns a dict; prints nothing.
    Used by both the command line (below) and the Streamlit page (app.py)."""
    results = retriever.search(question, TOP_K)
    raw_answer = ask_ollama(build_prompt(question, results))
    lowered = question.lower()

    used, invalid, uncited, refused = analyze(raw_answer, len(results))
    answer, withheld = raw_answer, None

    # Guards that do not depend on the model behaving:
    if not refused and (not used or invalid):
        answer, withheld, refused = REFUSAL_TEXT, "no valid citations", True
    elif not refused and not safe_wording(raw_answer):
        answer, withheld = UNSAFE_TEXT, "claims JSJ cures or treats a condition"
    if withheld:
        used, invalid, uncited = [], [], []

    return {
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,   # what the model actually wrote
        "withheld": withheld,       # None, or the reason the answer was replaced
        "results": results,
        "used": used,
        "invalid": invalid,
        "uncited": uncited,
        "refused": refused,
        "no_citations": False,
        "emergency": any(w in lowered for w in EMERGENCY_WORDS),
        "health": any(w in lowered for w in HEALTH_WORDS),
    }


def answer_question(question, retriever, show_context=False):
    """Command-line version: get the answer and print it."""
    print("\nThinking (a local model on a laptop CPU can take 10-60 seconds)...\n")
    try:
        r = get_answer(question, retriever)
    except OllamaError as e:
        sys.exit(str(e))

    if show_context:
        for n, c in enumerate(r["results"], start=1):
            print(f"--- context [{n}] {c['chunk_id']} ---\n{c['text']}\n")
    if r["emergency"]:
        print(EMERGENCY_NOTE + "\n")
    if r["health"]:
        print(HEALTH_NOTE + "\n")

    if r["withheld"]:
        print(f"(The model's answer was replaced by a safety check: {r['withheld']}.)\n")
    print(r["answer"] + "\n")

    if r["invalid"]:
        print(f"WARNING: the answer cites source number(s) {r['invalid']} that do not exist. "
              f"Do not trust those citations.\n")
    if r["uncited"]:
        print(f"WARNING: {len(r['uncited'])} sentence(s) have no citation and cannot be verified:")
        for x in r["uncited"]:
            print(f"  - {x}")
        print()
    if r["no_citations"]:
        print("WARNING: the answer contains no citations, so it cannot be verified.\n")

    if not r["refused"]:
        print("Sources:")
        for n, c in enumerate(r["results"], start=1):
            mark = "  <- cited" if n in r["used"] else ""
            print(f"  [{n}] {c['title']} ({c['chunk_id']}){mark}")
            print(f"      {c['url']}")
        print()
    print(SAFETY_NOTE + "\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Ask the JSJ library a question.")
    parser.add_argument("question", nargs="?", help="your question, in quotes")
    parser.add_argument("--show-context", action="store_true",
                        help="print the chunks given to the model")
    args = parser.parse_args()

    chunks = load_chunks()
    retriever, method = build_retriever(chunks)
    print(f"Retrieval: {method}   Model: {MODEL}")

    if args.question:
        answer_question(args.question, retriever, args.show_context)
        return
    while True:
        q = input("Question (press Enter alone to quit): ").strip()
        if not q:
            break
        answer_question(q, retriever, args.show_context)


if __name__ == "__main__":
    main()