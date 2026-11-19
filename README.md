# JSJ Intelligence

A Retrieval-Augmented Generation (RAG) assistant that answers questions about Jin Shin Jyutsu (JSJ) using a curated document library, with source citations and a measured evaluation of retrieval quality.

## Status

Version 1 is in progress.

- [x] Collect documents and write the source log
- [ ] Ingestion: read PDFs, remove website clutter, save clean text
- [ ] Chunking
- [ ] Embeddings and semantic search
- [ ] Evaluation (Precision@K, Recall@K, MRR)
- [ ] Answer generation with citations
- [ ] Streamlit interface

## Project structure

```
data/
  raw/            original source files (not uploaded to GitHub)
  clean/          cleaned text files (not uploaded to GitHub)
  source_log.csv  one row per document: title, byline, URL, reuse status
src/              Python code
eval/             test questions and evaluation results
```

## About the data

The documents are articles and pages from public websites. Most are copyrighted, so the text is not included in this repository. `data/source_log.csv` lists every source with its URL.

## Disclaimer

This is an educational project about what Jin Shin Jyutsu teaches. Its answers describe JSJ teachings and are not medical advice or established medical evidence.
