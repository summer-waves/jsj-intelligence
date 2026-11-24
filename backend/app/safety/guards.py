"""
guards.py - the safety checks that run in code, independent of the model.

The model is asked to follow rules, but a small model can break them. These checks look at the finished
answer and replace it when needed: an answer without a valid citation becomes a refusal, an answer that
claims a cure or treatment is withheld, and citations that point to non-existent sources are flagged.
Same logic as the original src/answer.py; here it is separate, so it can be tested on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A citation is a square bracket holding one or more source numbers: [2]   [1, 2, 5]   [1][2]
CITATION = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
CITATION_AFTER_STOP = re.compile(r"([.!?])\s*((?:\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]\s*)+)")
NEGATION = re.compile(r"(not|n't|no|never|cannot|nor|neither|without)\b", re.I)

REFUSAL_TEXT = "I could not find this in the JSJ library."
REFUSAL_MARKER = "could not find this in the jsj library"
UNSAFE_TEXT = (
    "I can't give a reliable summary for this question. The JSJ library does not show that "
    "Jin Shin Jyutsu cures or treats any medical condition. The passages that were found are "
    "listed under Sources so you can read them yourself."
)
WITHHELD_NO_CITATIONS = "no valid citations"
WITHHELD_CLAIM = "claims JSJ cures or treats a condition"

HEALTH_WORDS = [
    "cure", "treat", "heal", "diabet", "cancer", "chemo", "medication", "medicine",
    "disease", "diagnos", "tumor", "infection", "blood pressure", "depression", "pregnan",
    "palpitation", "panic", "anxiety", "chronic", "medical", "symptom", "pain",
]

EMERGENCY_WORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe", "trouble breathing",
    "stroke", "overdose", "suicid", "kill myself", "end my life", "severe bleeding",
    "unconscious", "poison",
]


@dataclass
class GuardResult:
    answer: str
    withheld: str | None
    refused: bool
    used: list = field(default_factory=list)
    invalid: list = field(default_factory=list)
    uncited: list = field(default_factory=list)


def question_flags(question):
    """(emergency, health): does the question mention emergency or medical wording?"""
    lowered = question.lower()
    return (any(w in lowered for w in EMERGENCY_WORDS), any(w in lowered for w in HEALTH_WORDS))


def check_citations(answer, n_sources):
    """Return the citation numbers the model used, and any that do not exist."""
    numbers = set()
    for group in CITATION.findall(answer):
        numbers.update(int(x) for x in re.findall(r"\d+", group))
    used = sorted(numbers)
    invalid = [x for x in used if x < 1 or x > n_sources]
    return used, invalid


def safe_wording(answer, claim_pattern):
    """False if the answer makes an un-negated cure/treat/heal claim."""
    for m in claim_pattern.finditer(answer):
        before = answer[max(0, m.start() - 45):m.start()]
        if not NEGATION.search(before):
            return False
    return True


def analyze(answer, n_sources):
    """Citation facts about an answer text: numbers used, invalid numbers, uncited sentences, refused?"""
    used, invalid = check_citations(answer, n_sources)
    refused = REFUSAL_MARKER in answer.lower()
    uncited = []
    if not refused and used:
        # Some models write the citation AFTER the full stop ("... for fear. [1][2]"). Move it in front of the
        # full stop, so the citation stays with its own sentence instead of the next one.
        tidy = CITATION_AFTER_STOP.sub(lambda m: " " + m.group(2).strip() + m.group(1) + " ", answer)
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", tidy) if x.strip()]
        uncited = [x for x in sentences if not CITATION.search(x)]
    return used, invalid, uncited, refused


def apply_guards(raw_answer, n_sources, claim_pattern):
    """Run the code-level guards on the model's answer and return a GuardResult."""
    used, invalid, uncited, refused = analyze(raw_answer, n_sources)
    answer, withheld = raw_answer, None
    if not refused and (not used or invalid):
        answer, withheld, refused = REFUSAL_TEXT, WITHHELD_NO_CITATIONS, True
    elif not refused and not safe_wording(raw_answer, claim_pattern):
        answer, withheld = UNSAFE_TEXT, WITHHELD_CLAIM
    if withheld:
        used, invalid, uncited = [], [], []
    return GuardResult(answer=answer, withheld=withheld, refused=refused,
                       used=used, invalid=invalid, uncited=uncited)
