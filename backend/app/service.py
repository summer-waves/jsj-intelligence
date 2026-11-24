"""
service.py - the whole question-to-answer flow in one place.

    question -> retrieve chunks -> build prompt -> ask the model -> run the safety guards -> AskResponse

It prints nothing and knows nothing about web servers, so it can be tested directly and reused by the API.
"""

from __future__ import annotations

import time

from .rag.retrieval import build_retriever, load_chunks
from .safety import guards
from .safety.profiles import EMERGENCY_NOTE, get_profile
from .schemas import AskResponse, Notices, Source, Timings


def excerpt(text, n):
    """The first n words of a passage. The API never returns a whole chunk."""
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def build_prompt(question, results):
    sources = []
    for n, r in enumerate(results, start=1):
        sources.append(f"[{n}] ({r['title']})\n{r['text']}")
    return "SOURCES:\n\n" + "\n\n".join(sources) + f"\n\nQUESTION: {question}"


class AssistantService:
    def __init__(self, settings, retriever, retrieval_method, llm, profile, chunk_count):
        self.settings = settings
        self.retriever = retriever
        self.retrieval_method = retrieval_method
        self.llm = llm
        self.profile = profile
        self.chunk_count = chunk_count

    def answer(self, question) -> AskResponse:
        t0 = time.perf_counter()
        results = self.retriever.search(question, self.settings.top_k)
        t1 = time.perf_counter()
        raw_answer = self.llm.generate(self.profile.system_rules, build_prompt(question, results))
        t2 = time.perf_counter()

        g = guards.apply_guards(raw_answer, len(results), self.profile.claim_pattern)
        emergency, health = guards.question_flags(question)

        # Like the original page: when the model refused, no sources are shown.
        sources = []
        if not g.refused:
            for n, r in enumerate(results, start=1):
                sources.append(Source(
                    number=n, chunk_id=r["chunk_id"], doc_id=r["doc_id"], title=r["title"], url=r["url"],
                    excerpt=excerpt(r["text"], self.settings.excerpt_words),
                    score=r["score"], cited=n in g.used,
                ))

        return AskResponse(
            question=question,
            answer=self.profile.display(g.answer),
            refused=g.refused,
            withheld_reason=self.profile.display(g.withheld) if g.withheld else None,
            citations_used=g.used,
            invalid_citations=g.invalid,
            uncited_sentences=g.uncited,
            sources=sources,
            notices=Notices(
                safety=self.profile.safety_note,
                health=self.profile.health_note if health else None,
                emergency=EMERGENCY_NOTE if emergency else None,
            ),
            model=self.llm.name,
            retrieval=self.retrieval_method,
            library_mode=self.settings.library_mode,
            timings=Timings(
                retrieval_ms=round((t1 - t0) * 1000),
                generation_ms=round((t2 - t1) * 1000),
                total_ms=round((t2 - t0) * 1000),
            ),
        )


def build_service(settings, llm=None) -> AssistantService:
    """Load the library, build the search, pick the model. Runs once when the server starts."""
    from .llm.factory import create_llm

    chunks = load_chunks(settings.chunks_path)
    retriever, method = build_retriever(chunks, settings)
    return AssistantService(
        settings=settings, retriever=retriever, retrieval_method=method,
        llm=llm or create_llm(settings), profile=get_profile(settings.library_mode),
        chunk_count=len(chunks),
    )
