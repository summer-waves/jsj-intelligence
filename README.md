# JSJ Intelligence

A Retrieval-Augmented Generation (RAG) assistant that answers questions about **Jin Shin Jyutsu (JSJ)**, a Japanese self-help and energy-work practice, using only a curated library of documents. Every answer cites its sources, and the retrieval step is measured with a labeled evaluation set.

The goal of the project is to build each part of a RAG pipeline from scratch (ingestion, chunking, keyword search, embeddings, hybrid search, evaluation, grounded generation) so that every design choice can be tested and explained.

> **Not medical advice.** The assistant reports what the JSJ teachings in the library say. It does not claim that JSJ treats or cures any condition, and the library is not scientific evidence. Anyone with symptoms should see a doctor.

## How it works

```
PDF articles --> ingest.py --> clean text --> chunk.py --> chunks.jsonl
                                                              |
question --> retrieval.py (BM25 + embeddings, fused) --> top 5 chunks
                                                              |
                          answer.py --> local LLM (Ollama) --> cited answer + safety notes
```

| Step | File | What it does |
|---|---|---|
| 1 | `src/ingest.py` | Reads the PDFs with pdfplumber and removes web-page clutter (menus, URLs, footers, sidebars). Writes one clean text file per document. |
| 2 | `src/chunk.py` | Splits each document into sentence-based chunks of up to 180 words with 35 words of overlap. Headings stay with the text that follows them. 80 chunks in total. |
| 3 | `src/retrieval.py` | Three retrievers: **BM25** (keyword search written from scratch), **dense** (sentence-transformers `all-MiniLM-L6-v2` embeddings), and **hybrid** (reciprocal rank fusion of the two). |
| 4 | `src/evaluate.py` | Scores each retriever on `eval/questions.csv`: Hit@K, Precision@K, Recall@K, MRR, latency. |
| 5 | `src/answer.py` | Sends the top chunks to a local Ollama model (`llama3.2`) with strict rules, checks the citations, and adds fixed safety notes. |

## Retrieval results

34 labeled questions, top 5 chunks returned.

| Method | Hit@5 | Precision@5 | Recall@5 | MRR | Latency |
|---|---|---|---|---|---|
| BM25 | 0.882 | 0.406 | 0.546 | 0.815 | 0.7 ms |
| Dense (embeddings) | 0.912 | 0.388 | 0.557 | 0.770 | 17.8 ms |
| **Hybrid** | **0.941** | **0.435** | **0.617** | 0.809 | 17.7 ms |

Hit@5 by question type:

| Type | n | BM25 | Dense | Hybrid |
|---|---|---|---|---|
| conceptual | 7 | 1.00 | 1.00 | 1.00 |
| exact term (e.g. "SEL 17") | 5 | 0.80 | 0.40 | 0.80 |
| factual | 9 | 1.00 | 1.00 | 1.00 |
| multi-document | 2 | 1.00 | 1.00 | 1.00 |
| safety | 3 | 0.33 | 1.00 | 1.00 |
| vocabulary mismatch | 8 | 0.88 | 1.00 | 0.88 |

**Findings**
- Keyword search is stronger on exact identifiers. Embeddings missed the chunk for "Where is Safety Energy Lock 17 located?" and for "Safety Energy Lock 1", while BM25 found them.
- Embeddings are stronger when the question uses different words from the document (for example, safety questions about pregnancy or evidence).
- Hybrid search combined both strengths and had the best Hit, Precision and Recall, so it is the default retriever.
- One question ("Which Safety Energy Locks are held in the flow for colds?") failed on all three methods: the answer sits in procedural step-by-step chunks that share little vocabulary with the question.

**How relevance is judged.** A returned chunk counts as relevant if its document is listed for the question and its text contains the required keyword(s). Labels therefore survive changes to the chunk size.

**Limits of this evaluation.** 34 questions is small, so one question is worth about 3 percentage points; the differences between methods of one or two questions should not be over-read. The questions were written after reading the documents, so their wording leans toward the source text. Precision@5 is low for every method because most questions have only one or two relevant chunks.

## Answer generation and safety

`answer.py` constrains the model to the retrieved sources and does not rely on the model alone for safety:

- The prompt requires answers built only from the numbered sources, a citation after each sentence, the phrase "According to the JSJ teachings...", and the exact reply "I could not find this in the JSJ library." when the sources do not contain the answer.
- Code-level checks (independent of the model): citations that point to non-existent sources are flagged; sentences with no citation are flagged; a fixed "not medical advice, see a doctor" note is always printed; a fixed notice is printed for questions about curing, treating, or medication; an emergency notice (911 / 988) is printed for emergency wording.

**Tests**
- "Who won the 2022 World Cup?" returned "I could not find this in the JSJ library." with no sources, as intended.
- "Can Jin Shin Jyutsu cure my diabetes?" showed the fixed health notice, cited the one source that mentions diabetes (a practitioner's personal report), and flagged an uncited sentence. The wording was still stronger than that source supports. This is a known limit of the small 3B-parameter local model, and it is the reason the safety notices and citation checks are implemented in code.

## Run it

Requirements: Python 3.11, [Ollama](https://ollama.com) with `llama3.2` pulled (`ollama pull llama3.2`).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/ingest.py       # PDFs in data/raw  -> data/clean
python src/chunk.py        # data/clean        -> data/chunks/chunks.jsonl
python src/evaluate.py     # retrieval metrics
python src/answer.py "Which finger do I hold for fear?"
```

## Data and copyright

The source PDFs are web articles that are mostly "All rights reserved" and are used here privately. **They are not included in this repository** (`data/raw`, `data/clean` and `data/chunks` are git-ignored). `data/source_log.csv` records each document's title, author, URL and reuse status. To reproduce the project, collect your own documents and list them in the source log and in `FILENAME_TO_DOC_ID` in `src/ingest.py`.

## Roadmap

- [x] Ingestion, chunking, BM25 / dense / hybrid retrieval, evaluation, grounded answers with citations
- [ ] Streamlit chat interface
- [ ] Answer-quality evaluation (faithfulness, citation accuracy) and a larger model comparison
- [ ] Chunk-size experiment; more evaluation questions written by people who have not read the documents
- [ ] Re-implementation with LangChain or LlamaIndex for comparison; Docker