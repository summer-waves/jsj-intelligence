import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = CHUNKS_DIR / "embeddings.npy"
EMBEDDINGS_META_PATH = CHUNKS_DIR / "embeddings_meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    """Read data/chunks/chunks.jsonl into a list of dictionaries."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_text(chunk):
    """The text we search: the document title plus the chunk itself.
    Adding the title tells the search which article a passage comes from."""
    return f"{chunk['title']}. {chunk['text']}"


class Retriever:
    """Base class: subclasses only have to provide scores(query)."""

    name = "base"
    min_score = None  # results scoring at or below this are dropped

    def __init__(self, chunks):
        self.chunks = chunks

    def scores(self, query):
        raise NotImplementedError

    def search(self, query, k=5):
        """Return the k best chunks for a query, best first."""
        scores = self.scores(query)
        results = []
        for rank, i in enumerate(np.argsort(-scores)[:k], start=1):
            if self.min_score is not None and scores[i] <= self.min_score:
                break
            c = self.chunks[i]
            results.append({
                "rank": rank,
                "score": float(scores[i]),
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                "url": c["url"],
                "text": c["text"],
            })
        return results


STOPWORDS = set(
    "a an and are as at be but by for from has have how i in is it its of on or "
    "that the their this to was were what when where which who why will with you your "
    "do does can could should would about into than then there these those not no".split()
)


def stem(word):
    """Very simple stemmer so 'holding', 'holds' and 'hold' count as the same word."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text):
    """Lowercase, split into words, drop common filler words, apply the stemmer."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


class BM25Retriever(Retriever):
    name = "bm25"
    min_score = 0.0  # a chunk sharing no words with the question is not a result

    def __init__(self, chunks, k1=1.5, b=0.75):
        super().__init__(chunks)
        self.k1 = k1  # how quickly repeating a word stops adding to the score
        self.b = b    # how strongly long chunks are penalised

        self.term_counts = [Counter(tokenize(chunk_text(c))) for c in chunks]
        self.lengths = np.array([sum(tc.values()) for tc in self.term_counts])
        self.avg_length = self.lengths.mean()

        # idf: rare words get a high weight, words found in many chunks a low one
        n = len(chunks)
        doc_freq = Counter()
        for tc in self.term_counts:
            doc_freq.update(tc.keys())
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def scores(self, query):
        query_terms = tokenize(query)
        scores = np.zeros(len(self.chunks))
        for i, tc in enumerate(self.term_counts):
            length_factor = 1 - self.b + self.b * self.lengths[i] / self.avg_length
            for term in query_terms:
                f = tc.get(term, 0)
                if f:
                    scores[i] += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * length_factor)
        return scores


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, chunks, model_name=MODEL_NAME):
        super().__init__(chunks)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit("sentence-transformers is not installed. Run: pip install sentence-transformers")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self):
        """Embed every chunk once and save the result, so later runs start instantly."""
        expected = {"model": self.model_name, "chunk_ids": [c["chunk_id"] for c in self.chunks]}
        if EMBEDDINGS_PATH.exists() and EMBEDDINGS_META_PATH.exists():
            saved = json.loads(EMBEDDINGS_META_PATH.read_text(encoding="utf-8"))
            if saved == expected:
                return np.load(EMBEDDINGS_PATH)

        print(f"Embedding {len(self.chunks)} chunks (first run only, or after the chunks changed)...")
        embeddings = self.model.encode(
            [chunk_text(c) for c in self.chunks],
            batch_size=32,
            normalize_embeddings=True,  # unit-length vectors: dot product = cosine similarity
            show_progress_bar=True,
        )
        np.save(EMBEDDINGS_PATH, embeddings)
        EMBEDDINGS_META_PATH.write_text(json.dumps(expected), encoding="utf-8")
        return embeddings

    def scores(self, query):
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        return self.embeddings @ query_vector  # cosine similarity with every chunk


class HybridRetriever(Retriever):
    """Each method ranks every chunk. A chunk earns 1 / (rrf_k + rank) from each method
    that ranks it, and the earned amounts are added up. A chunk that is high in both
    lists wins; a chunk that is first in only one list still scores well."""

    name = "hybrid"

    def __init__(self, chunks, retrievers, rrf_k=60):
        super().__init__(chunks)
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def scores(self, query):
        fused = np.zeros(len(self.chunks))
        for r in self.retrievers:
            s = r.scores(query)
            for rank, i in enumerate(np.argsort(-s), start=1):
                if r.min_score is not None and s[i] <= r.min_score:
                    break  # e.g. BM25 chunks that share no words with the question
                fused[i] += 1.0 / (self.rrf_k + rank)
        return fused


def print_results(label, results):
    print(f"\n=== {label} ===")
    if not results:
        print("(no matching chunks)")
    for r in results:
        snippet = " ".join(r["text"].split())[:230]
        print(f"{r['rank']}. [{r['score']:.3f}] {r['chunk_id']}  {r['title']}")
        print(f"   {snippet}...")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Search the JSJ chunks.")
    parser.add_argument("query", help="your question, in quotes")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid", "both"], default="both")
    parser.add_argument("--k", type=int, default=5, help="how many results to show")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Question: {args.query}")
    bm25 = BM25Retriever(chunks) if args.method != "dense" else None
    dense = DenseRetriever(chunks) if args.method != "bm25" else None
    if args.method in ("bm25", "both"):
        print_results("Keyword search (BM25)", bm25.search(args.query, args.k))
    if args.method in ("dense", "both"):
        print_results("Meaning search (embeddings)", dense.search(args.query, args.k))
    if args.method in ("hybrid", "both"):
        hybrid = HybridRetriever(chunks, [BM25Retriever(chunks), dense or DenseRetriever(chunks)])
        print_results("Hybrid (BM25 + embeddings, RRF)", hybrid.search(args.query, args.k))


if __name__ == "__main__":
    main()
