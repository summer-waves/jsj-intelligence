"""
profiles.py - the wording that depends on which library is loaded.

"private": your Jin Shin Jyutsu library (wording identical to the original project).
"demo":    the five public NIH (NCCIH) pages, used for the public website.

A profile bundles the instructions given to the model, the fixed notes shown to the reader, and the
cure/treat pattern used by the safety guard. Keeping them together means a new library only needs a new profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMERGENCY_NOTE = (
    "IMPORTANT: Your question may describe an emergency. Please call your local emergency "
    "number now (911 in the US). If you are thinking about harming yourself, call or text 988 "
    "(US Suicide & Crisis Lifeline). Do not rely on this tool in an emergency."
)

# The model is always told to refuse with this exact sentence; display() may reword it for the reader.
REFUSAL_PHRASE = "I could not find this in the JSJ library."


@dataclass(frozen=True)
class Profile:
    name: str
    system_rules: str
    safety_note: str
    health_note: str
    claim_pattern: re.Pattern
    replacements: tuple = ()

    def display(self, text):
        """Reword fixed messages for the reader (used by the demo profile only)."""
        for old, new in self.replacements:
            text = text.replace(old, new)
        return text


PRIVATE = Profile(
    name="private",
    system_rules="""You are JSJ Intelligence, an assistant that answers questions about Jin Shin Jyutsu (JSJ) using ONLY the numbered SOURCES provided in the user's message.

Rules you must follow:
1. Use only information found in the SOURCES. Do not use outside knowledge. Do not guess.
2. After every sentence that uses a source, add its number in square brackets, for example [2]. Use only the numbers of the SOURCES given.
3. Begin the answer with "According to the JSJ teachings in the library," or a similar phrase. Describe what JSJ teaches or claims. Never state it as established medical fact.
4. Never say JSJ cures, treats, heals, or is effective for any disease, and never tell the person to stop or avoid medical care.
   If a SOURCE mentions a condition, it is one author's or practitioner's personal report. Write it as "one source reports that..." and never as a general fact.
   If the question asks whether JSJ can cure or treat a condition, say the library does not show that JSJ cures or treats it, then say what the sources report, with their numbers.
5. If the SOURCES do not contain the answer, reply exactly: "I could not find this in the JSJ library." Do not say anything else.
6. Keep the answer short: at most 5 sentences.""",
    safety_note=(
        "Note: This describes what Jin Shin Jyutsu teachings say. It is not medical advice, and "
        "the sources are not scientific proof that it works. If you have symptoms, pain, or a "
        "health condition, please see a doctor or other qualified health professional."
    ),
    health_note=(
        "IMPORTANT: The JSJ library does not establish that Jin Shin Jyutsu cures or treats any "
        "medical condition. Please keep following the treatment your doctor recommends, and do "
        "not stop or change medication because of anything below."
    ),
    # A claim that JSJ cures / treats / heals / "is effective for..." is unsafe unless negated nearby.
    claim_pattern=re.compile(r"\b(cure[sd]?|heal[s]?|treat[s]?|effective (?:for|in|with|at))\b", re.I),
)

DEMO = Profile(
    name="demo",
    system_rules="""You are a demo assistant that answers questions about relaxation techniques, stress, anxiety, yoga, massage and other mind-body practices using ONLY the numbered SOURCES provided in the user's message. The SOURCES are public pages from the U.S. National Center for Complementary and Integrative Health (NCCIH).

Rules you must follow:
1. Use only information found in the SOURCES. Do not use outside knowledge. Do not guess.
2. After every sentence that uses a source, add its number in square brackets, for example [2]. Use only the numbers of the SOURCES given.
3. Begin the answer with "According to the NCCIH pages in the library," or a similar phrase. Report what the sources say, including how strong the evidence is. Never state a practice as a proven medical treatment.
4. Never say a practice cures, treats, or heals a disease, and never tell the person to stop or avoid medical care. If the sources say the evidence is limited or low quality, say so.
5. If the SOURCES do not contain the answer, reply exactly: "I could not find this in the JSJ library." Do not say anything else.
6. Keep the answer short: at most 5 sentences.""",
    safety_note=(
        "Note: This summarizes public NIH (NCCIH) pages. It is not medical advice. If you have "
        "symptoms, pain, or a health condition, please see a doctor or other qualified health professional."
    ),
    health_note=(
        "IMPORTANT: This demo only summarizes what the library pages say. It does not tell you what to "
        "do about a medical condition. Please keep following the treatment your doctor recommends."
    ),
    # demo: hedged NIH wording like "effective for" is allowed
    claim_pattern=re.compile(r"\b(cure[sd]?|heal[s]?|treat[s]?)\b", re.I),
    replacements=(
        ("I could not find this in the JSJ library.", "I could not find this in the demo library."),
        ("claims JSJ cures or treats a condition", "claims a practice cures or treats a condition"),
        ("The JSJ library does not show that Jin Shin Jyutsu cures or treats any medical condition.",
         "The library does not show that any practice cures or treats a medical condition."),
    ),
)


def get_profile(library_mode):
    return DEMO if library_mode == "demo" else PRIVATE
