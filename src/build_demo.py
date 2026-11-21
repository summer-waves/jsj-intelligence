import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk import pack_units, split_units  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"


def main():
    with open(DEMO_DIR / "sources.csv", newline="", encoding="utf-8") as f:
        sources = list(csv.DictReader(f))

    records = []
    for meta in sources:
        path = DEMO_DIR / "raw" / f"{meta['doc_id']}.txt"
        if not path.exists():
            print(f"skipped {meta['doc_id']}: {path.name} not found")
            continue
        units = split_units(path.read_text(encoding="utf-8"))
        for i, chunk_units in enumerate(pack_units(units), start=1):
            text = " ".join(chunk_units)
            records.append({
                "chunk_id": f"{meta['doc_id']}_c{i:02d}",
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "site": meta["site"],
                "url": meta["url"],
                "chunk_index": i,
                "word_count": len(text.split()),
                "text": text,
            })

    out_path = DEMO_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for doc_id in sorted({r["doc_id"] for r in records}):
        print(doc_id, sum(1 for r in records if r["doc_id"] == doc_id), "chunks")
    print(f"Total: {len(records)} chunks. Saved to {out_path}")


if __name__ == "__main__":
    main()
