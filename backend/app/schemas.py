"""
schemas.py - the exact shape of what goes into and comes out of the API.

Pydantic checks these shapes automatically: a request with a missing or too long question is rejected
before any of our code runs, and a response that does not match is an error in the code, not a surprise
for the website.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field, StringConstraints


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]


class Source(BaseModel):
    number: int = Field(description="The number used in the answer, for example 2 for [2]")
    chunk_id: str
    doc_id: str
    title: str
    url: str
    excerpt: str = Field(description="A short excerpt only; the full passage stays on the server")
    score: float
    cited: bool = Field(description="True if the answer cites this source")


class Notices(BaseModel):
    safety: str
    health: Optional[str] = None      # set when the question touches a medical topic
    emergency: Optional[str] = None   # set when the question looks like an emergency


class Timings(BaseModel):
    retrieval_ms: int
    generation_ms: int
    total_ms: int


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool = Field(description="True when the library does not contain the answer")
    withheld_reason: Optional[str] = Field(default=None, description="Set when a safety check replaced the model's answer")
    citations_used: list[int]
    invalid_citations: list[int]
    uncited_sentences: list[str]
    sources: list[Source]
    notices: Notices
    model: str
    retrieval: str
    library_mode: str
    timings: Timings


class HealthResponse(BaseModel):
    status: str
    library_mode: str
    chunks: int
    retrieval: str
    model: str
