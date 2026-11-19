"""
ingest.py - Step 1 of the JSJ Intelligence pipeline.

What this script does:
  1. Reads data/source_log.csv to learn which documents exist (and which to include).
  2. Opens every PDF in data/raw/ and extracts its text.
  3. Removes website clutter (menus, page headers/footers, link lists, promo banners).
  4. Saves one clean text file per document to data/clean/ (named jsj_001.txt, ...).
  5. Prints a summary so you can spot problems.

Run it from the project folder with:   python src/ingest.py
"""

import csv
import logging
import re
from pathlib import Path

import pdfplumber

# pdfplumber prints harmless font warnings; hide them so the summary stays readable.
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# 1. Paths (all relative to the project folder, so the script works anywhere)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
SOURCE_LOG = PROJECT_ROOT / "data" / "source_log.csv"

# ----------------------------------------------------------------------------
# 2. Which PDF file belongs to which doc_id.
#    Each key is a piece of the PDF's file name (lowercase). The first part of
#    the match must be unique: exactly one PDF may match each key.
# ----------------------------------------------------------------------------
FILENAME_TO_DOC_ID = {
    "how to soothe your nervous system": "jsj_001",
    "unlock calm and clarity": "jsj_002",
    "self help for anxiety": "jsj_003",
    "what is jin shin jyutsu_ - flows": "jsj_004",
    "what is jin shin jyutsu_ part 1": "jsj_005",
    "frequently asked questions": "jsj_006",
    "finger method": "jsj_007",
    "physio-philosophy": "jsj_008",
    "universal art to boost energy": "jsj_009",
    "renew myself": "jsj_010",
    "keep the body fit": "jsj_011",
    "colds and flu": "jsj_012",
    "people sometimes ask me": "jsj_013",
    "the mindful word": "jsj_014",
    "ancient healing art for today": "jsj_015",
    "the art of jin shin jyutsu.pdf": "jsj_016",
}

# ----------------------------------------------------------------------------
# 3. Special rules for the few PDFs that are laid out differently.
#    crop_right: ignore everything to the right of this x position on the page
#                (MarinHealth has a sidebar that mixes into the article text).
#    start_after: throw away every line up to and including the first line that
#                 starts with this text (removes the website menu at the top).
#    end_at: stop at the first line that starts with this text.
# ----------------------------------------------------------------------------
DOC_RULES = {
    "jsj_015": {
        "crop_right": 405,
        "start_after": "Author:",
        "end_at": "Previous Post",
    },
}

# ----------------------------------------------------------------------------
# 4. Generic cleaning rules for the Flows for Life articles and similar pages.
# ----------------------------------------------------------------------------
# Everything from the first line that starts with one of these is removed.
END_MARKERS = [
    "want to practice",
    "want to learn more",
    "you may also like",
    "related articles",
    "join the practice",
    "join our jin shin jyutsu community",
    "see more about",
    "see also",
    "part 2",
    "expand your knowledge",
    "join our",
]

# "11/18/26, 6:46 PM Page title..." - the browser's header on each printed page
BROWSER_HEADER = re.compile(r"^\d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s*[AP]M\b")
# "https://www.site.com/page/ 2/12" - the browser's footer (URL + page counter)
BROWSER_FOOTER = re.compile(r"^https?://\S+(\s+\d+/\d+)?$")
# "2/12" on its own line
PAGE_COUNTER = re.compile(r"^\d+/\d+$")
# Menu links printed in capitals, e.g. "ANXIETY (HTTPS://WWW.FLOWSFORLIFE.COM/...)"
MENU_LINK = re.compile(r"\(HTTPS?://")
# A web address inside a sentence, with or without brackets around it
INLINE_URL = re.compile(r"\(?https?://[^\s)]*\)?")
# A leftover piece of a web address that wrapped onto its own line
URL_FRAGMENT = re.compile(r"^[\w\-./:()?=&%]+$")
# "BY ASTRID KAUFFMANN" / "BY FLOWSFORLIFE" byline lines in capitals
BYLINE_CAPS = re.compile(r"^BY [A-Z0-9 ]+$")
# The end of a web address that wrapped onto the next line, e.g. "fingers/),"
URL_TAIL = re.compile(r"\S*/\)")
# A wrapped web address piece made only of lowercase words joined by hyphens
HYPHEN_FRAGMENT = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]*){3,}$")
# Icon-font symbols (menu buttons etc.) that show up as invisible characters
ICON_CHARS = re.compile(r"[\ue000-\uf8ff]")
# Author-name lines from the page footer
AUTHOR_LINES = {"flowsforlife", "astrid kauffmann"}


def extract_text(pdf_path, crop_right=None):
    """Read all pages of a PDF and return the text as one string."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if crop_right:
                # keep only the left part of the page (x from 0 to crop_right)
                page = page.crop((0, 0, crop_right, page.height))
            pages.append(page.extract_text(y_tolerance=6) or "")
    return "\n".join(pages)


def clean_text(raw_text, rules):
    """Remove website clutter from extracted text and return the article text."""
    lines = [line.strip() for line in raw_text.split("\n")]

    # Special rule: drop everything up to the line where the article really starts.
    start_after = rules.get("start_after")
    if start_after:
        for i, line in enumerate(lines):
            if line.startswith(start_after):
                lines = lines[i + 1:]
                break

    # Stop at the first "end of article" marker (promo banners, related articles...).
    end_markers = [m.lower() for m in END_MARKERS]
    if rules.get("end_at"):
        end_markers.append(rules["end_at"].lower())
    for i, line in enumerate(lines):
        if any(line.lower().startswith(m) for m in end_markers):
            lines = lines[:i]
            break

    kept = []
    for line in lines:
        line = ICON_CHARS.sub("", line).strip()
        if not line:
            continue
        if BROWSER_HEADER.match(line) or BROWSER_FOOTER.match(line):
            continue
        if PAGE_COUNTER.match(line) or MENU_LINK.search(line):
            continue
        if BYLINE_CAPS.match(line):
            continue
        if len(line) <= 3:  # single characters from social-media links, etc.
            continue
        if URL_FRAGMENT.match(line) and "/" in line:
            continue
        if HYPHEN_FRAGMENT.match(line) or line.lower() in AUTHOR_LINES:
            continue
        line = INLINE_URL.sub("", line)
        line = URL_TAIL.sub("", line)
        line = line.lstrip(",.;: ").rstrip("| ").strip()
        # skip a line that repeats the line before it (e.g. a title printed twice)
        if line and not (kept and kept[-1] == line):
            kept.append(line)

    return "\n".join(kept) + "\n"


def load_source_log():
    """Return {doc_id: row} from data/source_log.csv."""
    with open(SOURCE_LOG, newline="", encoding="utf-8") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f)}


def find_doc_id(pdf_name):
    """Return the doc_id for a PDF file name, or None if there is no match."""
    name = pdf_name.lower()
    matches = [doc_id for key, doc_id in FILENAME_TO_DOC_ID.items() if key in name]
    return matches[0] if len(matches) == 1 else None


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    log = load_source_log()
    results = []

    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        doc_id = find_doc_id(pdf_path.name)
        if doc_id is None:
            print(f"WARNING: no unique doc_id for '{pdf_path.name}' - skipped")
            continue
        if log[doc_id]["include_v1"].strip().lower() != "yes":
            print(f"skipped {doc_id} (include_v1 = no): {log[doc_id]['title']}")
            continue

        rules = DOC_RULES.get(doc_id, {})
        raw_text = extract_text(pdf_path, rules.get("crop_right"))
        clean = clean_text(raw_text, rules)
        (CLEAN_DIR / f"{doc_id}.txt").write_text(clean, encoding="utf-8")
        results.append((doc_id, len(raw_text.split()), len(clean.split()), log[doc_id]["title"]))

    print()
    print(f"{'doc_id':8} {'raw words':>10} {'clean words':>12}  title")
    for doc_id, raw_n, clean_n, title in sorted(results):
        flag = "   <-- check this one" if clean_n < 150 else ""
        print(f"{doc_id:8} {raw_n:>10} {clean_n:>12}  {title[:55]}{flag}")
    print(f"\nDone: {len(results)} documents saved to {CLEAN_DIR}")


if __name__ == "__main__":
    main()
