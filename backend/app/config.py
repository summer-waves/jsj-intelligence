"""
config.py - every setting of the backend, read from environment variables.

Nothing secret is stored in the code. The API key is only read from the environment
(on your computer: $env:ANTHROPIC_API_KEY = "..." ; in the cloud: Secret Manager).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# backend/app/config.py -> parents[0]=app, [1]=backend, [2]=the repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_CHUNKS_PATH = Path(os.environ.get("JSJ_CHUNKS_PATH", REPO_ROOT / "data" / "chunks" / "chunks.jsonl"))
DEMO_CHUNKS_PATH = REPO_ROOT / "demo" / "chunks.jsonl"  # public-domain NIH pages, committed to Git


@dataclass(frozen=True)
class Settings:
    library_mode: str        # "private" (your JSJ library) or "demo" (public NIH pages)
    chunks_path: Path
    embeddings_path: Path
    embeddings_meta_path: Path
    retrieval: str           # "hybrid" or "bm25"
    llm_backend: str         # "auto", "anthropic" or "ollama"
    anthropic_model: str
    ollama_model: str
    ollama_url: str
    top_k: int
    excerpt_words: int
    cors_origins: tuple
    anthropic_key: str | None = field(default=None, repr=False)  # repr=False: never printed in logs

    @classmethod
    def from_env(cls) -> "Settings":
        forced = os.environ.get("JSJ_LIBRARY_MODE", "").strip().lower()
        if forced in ("private", "demo"):
            mode = forced
        else:
            mode = "private" if PRIVATE_CHUNKS_PATH.exists() else "demo"

        chunks_dir = PRIVATE_CHUNKS_PATH.parent
        if mode == "private":
            chunks_path = PRIVATE_CHUNKS_PATH
            embeddings = chunks_dir / "embeddings.npy"                # the file your current project already made
            embeddings_meta = chunks_dir / "embeddings_meta.json"
        else:
            chunks_path = DEMO_CHUNKS_PATH
            embeddings = chunks_dir / "embeddings_demo.npy"           # separate files, so the private ones are never overwritten
            embeddings_meta = chunks_dir / "embeddings_demo_meta.json"

        origins = os.environ.get("JSJ_CORS_ORIGINS", "http://localhost:3000")
        return cls(
            library_mode=mode,
            chunks_path=chunks_path,
            embeddings_path=embeddings,
            embeddings_meta_path=embeddings_meta,
            retrieval=os.environ.get("JSJ_RETRIEVAL", "hybrid").strip().lower(),
            llm_backend=os.environ.get("JSJ_LLM", "auto").strip().lower(),
            anthropic_model=os.environ.get("JSJ_HOSTED_MODEL", "claude-haiku-5-5"),
            ollama_model=os.environ.get("JSJ_MODEL", "llama3.2"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat"),
            top_k=int(os.environ.get("JSJ_TOP_K", "5")),
            excerpt_words=int(os.environ.get("JSJ_EXCERPT_WORDS", "40")),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            anthropic_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        )
