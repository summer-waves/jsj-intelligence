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
| 5b | `src/eval_answers.py` | Runs 17 questions through the whole pipeline and checks the answers: correct fact, valid and supported citations, correct refusals, safe wording. |
| 6 | `app.py` | Streamlit chat page: cited answers, short source excerpts with links, safety notes and warnings. |

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
- At the chosen chunk size (180 words) hybrid search combined both strengths and had the best Hit, Precision and Recall. Across other chunk sizes no method wins consistently (see the chunk-size experiment below), so hybrid is used as the default because it is the most consistent, not because it always wins.
- One question ("Which Safety Energy Locks are held in the flow for colds?") failed on all three methods: the answer sits in procedural step-by-step chunks that share little vocabulary with the question.

**How relevance is judged.** A returned chunk counts as relevant if its document is listed for the question and its text contains the required keyword(s). Labels therefore survive changes to the chunk size.

**Limits of this evaluation.** 34 questions is small, so one question is worth about 3 percentage points; the differences between methods of one or two questions should not be over-read. The questions were written after reading the documents, so their wording leans toward the source text. Precision@5 is low for every method because most questions have only one or two relevant chunks.

## Chunk-size experiment

`src/chunk_experiment.py` re-cuts the documents at five sizes and re-runs the same 34 retrieval questions (K = 5). The saved results are in `eval/chunk_experiment.csv`.

| Chunk size (words, overlap) | Chunks | BM25 Hit / MRR | Dense Hit / MRR | Hybrid Hit / MRR |
|---|---|---|---|---|
| 80 (15) | 177 | 0.824 / 0.738 | 0.882 / 0.809 | 0.853 / 0.795 |
| 120 (25) | 123 | 0.912 / 0.810 | 0.853 / 0.782 | 0.882 / 0.833 |
| **180 (35), used** | 80 | 0.882 / 0.815 | 0.912 / 0.770 | 0.941 / 0.809 |
| 250 (50) | 58 | 0.971 / 0.896 | 1.000 / 0.824 | 0.971 / 0.882 |
| 400 (80) | 39 | 0.941 / 0.892 | 0.971 / 0.664 | 0.941 / 0.811 |

**Findings**
- All three methods did best at roughly 220-word chunks (the 250 setting), and worst at the smallest size.
- Dense retrieval degrades on the largest chunks (MRR 0.664 at 400 words). The embedding model reads only about the first 256 tokens of each chunk, so later text in a large chunk is not represented.
- Averaged over the five sizes: dense had the best Hit (0.924), BM25 the best MRR (0.830), and hybrid was close to the top on both (0.918, 0.826).
- Hit@K favors large chunks, because the top 5 cover a larger share of the library when there are fewer chunks. Precision and recall also change with the number of chunks.
- With 34 questions, one question is 0.029, so most differences between methods at one size are one or two questions.

**Decision.** The project keeps 180 words. The 250-word setting scores higher on retrieval, but the hybrid gain is about one question, and larger chunks make prompts longer (answers already take about 42 s on a laptop CPU), make citations point to bigger blocks of text, and would require re-running the answer evaluation.

## LangChain comparison

To check my from-scratch retrieval against a standard framework, `src/langchain_version.py`
rebuilds the same three retrievers with LangChain (`BM25Retriever`, `InMemoryVectorStore` with
the same MiniLM model, and `EnsembleRetriever`) and scores them on the same chunks and the
same 34 questions (K=5).

| Method | Build | Hit@5 | Prec@5 | Recall@5 | MRR |
|---|---|---|---|---|---|
| bm25 | from scratch | 0.882 | 0.406 | 0.546 | 0.815 |
| bm25 | LangChain | 0.853 | 0.312 | 0.459 | 0.704 |
| dense | from scratch | 0.912 | 0.388 | 0.557 | 0.770 |
| dense | LangChain | 0.912 | 0.388 | 0.557 | 0.770 |
| hybrid | from scratch | 0.941 | 0.435 | 0.617 | 0.809 |
| hybrid | LangChain | 0.971 | 0.382 | 0.578 | 0.767 |

- Dense results are identical, as expected: same model, same chunks.
- LangChain's BM25 scores lower. A likely reason is that it splits on whitespace without
  stemming or stopword removal, while my version does both (not tested separately).
- Hybrid is mixed: LangChain's Hit@5 is one question higher (0.971 vs 0.941), but my version
  ranks relevant chunks higher (better Precision, Recall and MRR).
- Retrieval speed is similar; embedding the query dominates.

Run it with `pip install -r requirements-langchain.txt` and `python src/langchain_version.py`.

## Answer generation and safety

`answer.py` constrains the model to the retrieved sources and does not rely on the model alone for safety:

- The prompt requires answers built only from the numbered sources, a citation after each sentence, the phrase "According to the JSJ teachings...", and the exact reply "I could not find this in the JSJ library." when the sources do not contain the answer.
- Code-level checks (independent of the model): citations that point to non-existent sources are flagged; sentences with no citation are flagged; a fixed "not medical advice, see a doctor" note is always printed; a fixed notice is printed for questions about curing, treating, or medication; an emergency notice (911 / 988) is printed for emergency wording.
- Code-level guards that replace the model's answer: an answer with no valid citation becomes "I could not find this in the JSJ library."; an answer that claims JSJ cures, treats, heals, or "is effective for" something is withheld, and only the retrieved passages are shown. The page tells the user when this happened.
- Small Sources excerpts (40 words) with a link to the original article, so the page does not reproduce the articles.

### Answer-quality results

17 questions (`eval/answer_questions.csv`): 10 answerable, 3 medical, 4 out of scope. Checks are simple automatic rules. Local model `llama3.2` (3B) on a laptop CPU, about 42 seconds per answer.

| Check (model's own answer) | Answerable | Medical | Out of scope |
|---|---|---|---|
| Answers or refuses correctly | 10/10 | - | 2/4 |
| Contains the expected fact | 10/10 | - | - |
| Citations valid | 10/10 | 3/3 | 1/2 |
| A cited chunk supports the answer | 9/10 | - | - |
| Every sentence cited | 8/10 | 0/3 | 2/2 |
| No cure/treat claim | 10/10 | 2/3 | 4/4 |
| Starts with "According to" | 10/10 | 3/3 | 1/2 |

With the code-level guards, 2 of 17 answers were replaced (a sourdough answer with no citation, and a diabetes answer that overstated a source), and 3 of 4 out-of-scope questions ended as a refusal.

**Findings**
- Prompt rules alone were not reliable for a 3B model: it answered a cooking question from its own knowledge and, on the diabetes question, phrased one practitioner's personal report as an effect. Moving these rules into code fixed both cases.
- The "every sentence cited" check is strict. On medical questions the uncited sentence is probably a "see a doctor" sentence, which does not need a citation.
- One out-of-scope question (acupuncture needle placement) is borderline, because the library mentions acupuncture only as a comparison.
- An early version of the citation checker read only single citations such as `[2]`, and wrongly failed answers that cite `[1, 2]`. This was found by reading the saved answers, fixed, and all results were re-scored (the numbers above are the corrected ones). Automatic checks on model output need to be read by hand at least once.
- Results vary slightly between runs, and 17 questions is a small sample.

### Model comparison

The same 17 questions, same retrieval, two local models (CPU only, 11.8 GB RAM laptop). Checks judge the model's own answer; the guards are then applied.

| | `llama3.2` (3B) | `qwen2.5:7b` |
|---|---|---|
| Out-of-scope refused by the model itself | 2/4 | **4/4** |
| Out-of-scope ending as a refusal (after guards) | 3/4 | **4/4** |
| Answerable questions answered (after guards) | 10/10 | 10/10 |
| Expected fact present | **10/10** | 9/10 |
| Citations valid (answerable) | 10/10 | 10/10 |
| A cited chunk supports the answer | 9/10 | 9/10 |
| Every sentence cited (answerable) | **8/10** | 6/10 |
| No cure/treat claim (medical) | **2/3** | 1/3 |
| Answers replaced by the guards | 2 of 17 | 2 of 17 |
| Mean seconds per answer | **42** | 58 |

**Findings**
- The larger model refused every out-of-scope question on its own, while the 3B model needed the guard for one of them. The larger model was also slower, cited fewer of its sentences, and made more cure/treat-style claims on the medical questions.
- With the guards in place both models behave acceptably, so the guards matter more than model size here. The only end-to-end difference is one borderline out-of-scope question (acupuncture).
- 17 questions and one run per model is a small sample and the models are not fully deterministic, so these are indications, not rankings.

**Decision.** The project keeps `llama3.2` as the default because it is about 27% faster and equal or better on most checks. To try another model: `$env:JSJ_MODEL = "qwen2.5:7b"` (PowerShell) before `streamlit run app.py`, or `python src/eval_answers.py --model qwen2.5:7b`.

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
python src/eval_answers.py # answer-quality checks (about 10 minutes)
python src/eval_answers.py --rescore eval/answer_results.csv   # re-check saved answers in seconds
streamlit run app.py       # chat page at http://localhost:8501
```

## Data and copyright

The source PDFs are web articles that are mostly "All rights reserved" and are used here privately. **They are not included in this repository** (`data/raw`, `data/clean` and `data/chunks` are git-ignored). `data/source_log.csv` records each document's title, author, URL and reuse status. To reproduce the project, collect your own documents and list them in the source log and in `FILENAME_TO_DOC_ID` in `src/ingest.py`.

## Run with Docker

The Streamlit chat page can run in a container, so nobody has to install Python or the libraries.

- The image contains only the code. The source articles and chunk files (`data/`) are not
  copied into it (see the copyright note); `docker-compose.yml` mounts your local `data/`
  folder at run time. Run the pipeline first (`ingest.py`, `chunk.py`) so `data/chunks/` exists.
- Ollama is not in the container. It runs on the host computer, and the container reaches it
  at `host.docker.internal:11434` (set by the `OLLAMA_URL` environment variable).
- The page is published on port 8502 (`http://localhost:8502`).

```
docker compose up --build     # first run: builds the image (several minutes)
docker compose up             # later runs
docker compose down           # stop and remove the container
```

Requires Docker Desktop and a running Ollama with `llama3.2` pulled.


## Roadmap

- [x] Ingestion, chunking, BM25 / dense / hybrid retrieval, evaluation, grounded answers with citations
- [x] Streamlit chat interface
- [x] Answer-quality evaluation and code-level safety guards
- [x] Larger model comparison
- [x] Chunk-size experiment
- [ ] More evaluation questions written by people who have not read the documents
- [x] Re-implementation with LangChain for comparison (see "LangChain comparison" above)
- [x] Docker
- [ ] LlamaIndex comparision