"""
retrieval.py - finds the chunks that best match a question.

This is the same search code as src/retrieval.py in the original project (BM25 written from scratch,
embedding search, and a hybrid of both with reciprocal rank fusion). The scoring is unchanged, so the
measured results in the README still describe it. Only the file paths now come from config.py, and a
missing sentence-transformers package raises a normal error that the backend can catch.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseUnavailable(Exception):
    """sentence-transformers is not installed, so meaning search cannot be used."""


def load_chunks(path):
    """Read a chunks.jsonl file (one JSON record per line) into a list of dictionaries."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_text(chunk):
    """The text we search: the document title plus the chunk itself."""
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


# ----------------------------------------------------------------------------
# Keyword search: BM25
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Meaning search: embeddings
# ----------------------------------------------------------------------------
class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, chunks, embeddings_path, embeddings_meta_path, model_name=MODEL_NAME):
        super().__init__(chunks)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise DenseUnavailable("sentence-transformers is not installed") from e
        self.model_name = model_name
        self.embeddings_path = embeddings_path
        self.embeddings_meta_path = embeddings_meta_path
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self):
        """Embed every chunk once and save the result, so later runs start instantly."""
        expected = {"model": self.model_name, "chunk_ids": [c["chunk_id"] for c in self.chunks]}
        if self.embeddings_path.exists() and self.embeddings_meta_path.exists():
            saved = json.loads(self.embeddings_meta_path.read_text(encoding="utf-8"))
            if saved == expected:
                return np.load(self.embeddings_path)

        embeddings = self.model.encode(
            [chunk_text(c) for c in self.chunks],
            batch_size=32,
            normalize_embeddings=True,  # unit-length vectors: dot product = cosine similarity
            show_progress_bar=False,
        )
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, embeddings)
        self.embeddings_meta_path.write_text(json.dumps(expected), encoding="utf-8")
        return embeddings

    def scores(self, query):
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        return self.embeddings @ query_vector


# ----------------------------------------------------------------------------
# Hybrid search: BM25 + embeddings combined with reciprocal rank fusion (RRF)
# ----------------------------------------------------------------------------
class HybridRetriever(Retriever):
    """Each method ranks every chunk. A chunk earns 1 / (rrf_k + rank) from each method that ranks it,
    and the earned amounts are added up."""

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


def build_retriever(chunks, settings):
    """Hybrid search if allowed and sentence-transformers is installed, otherwise keyword search only.
    Returns (retriever, method_name)."""
    bm25 = BM25Retriever(chunks)
    if settings.retrieval == "bm25":
        return bm25, "bm25"
    try:
        dense = DenseRetriever(chunks, settings.embeddings_path, settings.embeddings_meta_path)
    except DenseUnavailable:
        return bm25, "bm25"
    return HybridRetriever(chunks, [bm25, dense]), "hybrid"
