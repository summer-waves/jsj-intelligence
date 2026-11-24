from app.rag.retrieval import BM25Retriever, load_chunks, tokenize


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The holding of hands") == ["hold", "hand"]


def test_bm25_finds_relevant_document(settings):
    r = BM25Retriever(load_chunks(settings.chunks_path))
    top = r.search("What does the NIH say about yoga safety?", 5)
    assert top and top[0]["rank"] == 1
    assert any(x["doc_id"] == "demo_003" for x in top)


def test_unrelated_query_returns_nothing(settings):
    r = BM25Retriever(load_chunks(settings.chunks_path))
    assert r.search("zzzqqq xylophone", 5) == []
