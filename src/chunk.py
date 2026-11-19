"""
chunk.py - Step 2 of the JSJ Intelligence pipeline.

What this script does:
  1. Reads every clean text file in data/clean/ (made by ingest.py).
  2. Splits each document into sentences, keeping headings as their own pieces.
  3. Packs the sentences into chunks of about MAX_WORDS words. Neighbouring chunks
     share about OVERLAP_WORDS words, so an idea that sits on a boundary is still
     whole in at least one chunk.
  4. Saves all chunks to data/chunks/chunks.jsonl (one JSON record per line),
     together with the document's title, site and URL (needed later for citations).

Run it from the project folder with:   python src/chunk.py
"""

import csv
import json
import re
from pathlib import Path

# ----------------------------------------------------------------------------
# Settings you can change later to test how chunk size affects search quality.
# ----------------------------------------------------------------------------
MAX_WORDS = 180      # a chunk is closed once adding the next sentence would pass this
OVERLAP_WORDS = 35   # how much of the end of a chunk is repeated at the start of the next

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
SOURCE_LOG = PROJECT_ROOT / "data" / "source_log.csv"

# A sentence ends at . ! ? (optionally followed by a closing quote), then a space,
# and the next sentence starts with a capital letter, quote or bracket.
SENTENCE_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][”\"’']))\s+(?=[A-Z“‘\"'(])")


def is_heading(line):
    """A short line with no sentence punctuation at the end is treated as a heading."""
    return len(line.split()) <= 8 and line[-1] not in ".!?,;:”\"’)"


def split_units(text):
    """Turn a document into a list of 'units': sentences and headings."""
    units = []
    paragraph = []  # consecutive normal lines that belong to the same block of text

    def flush():
        if paragraph:
            joined = " ".join(paragraph)
            units.extend(SENTENCE_SPLIT.split(joined))
            paragraph.clear()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if is_heading(line):
            flush()
            units.append(line)
        else:
            paragraph.append(line)
    flush()
    return [u.strip() for u in units if u.strip()]


def word_count(unit):
    return len(unit.split())


def pack_units(units):
    """Group units into chunks of up to MAX_WORDS words, with overlap."""
    chunks = []
    current = []
    current_words = 0

    for unit in units:
        n = word_count(unit)
        if current and current_words + n > MAX_WORDS:
            # A heading must not be the last thing in a chunk: move it to the next chunk.
            carried = []
            while len(current) > 1 and is_heading(current[-1]):
                carried.insert(0, current.pop())
            chunks.append(current)

            # Start the next chunk with the last few sentences of this one (the overlap).
            overlap, overlap_words = [], 0
            for prev in reversed(current):
                if overlap_words + word_count(prev) > OVERLAP_WORDS:
                    break
                overlap.insert(0, prev)
                overlap_words += word_count(prev)
            current = overlap + carried
            current_words = sum(word_count(u) for u in current)

        current.append(unit)
        current_words += n

    if current:
        chunks.append(current)
    return chunks


def load_source_log():
    with open(SOURCE_LOG, newline="", encoding="utf-8") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f)}


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    log = load_source_log()
    records = []

    for path in sorted(CLEAN_DIR.glob("jsj_*.txt")):
        doc_id = path.stem
        meta = log[doc_id]
        units = split_units(path.read_text(encoding="utf-8"))
        for i, chunk_units in enumerate(pack_units(units), start=1):
            text = " ".join(chunk_units)
            records.append({
                "chunk_id": f"{doc_id}_c{i:02d}",
                "doc_id": doc_id,
                "title": meta["title"],
                "site": meta["site"],
                "url": meta["url"],
                "chunk_index": i,
                "word_count": len(text.split()),
                "text": text,
            })

    out_path = CHUNKS_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    print(f"{'doc_id':8} {'chunks':>6}  {'words per chunk (min-max)':>26}")
    for doc_id in sorted({r["doc_id"] for r in records}):
        counts = [r["word_count"] for r in records if r["doc_id"] == doc_id]
        print(f"{doc_id:8} {len(counts):>6}  {min(counts):>12} - {max(counts):<12}")
    all_counts = [r["word_count"] for r in records]
    print(f"\nTotal: {len(records)} chunks, average {sum(all_counts) / len(all_counts):.0f} words "
          f"(settings: MAX_WORDS={MAX_WORDS}, OVERLAP_WORDS={OVERLAP_WORDS})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
