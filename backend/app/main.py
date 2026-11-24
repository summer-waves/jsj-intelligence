"""
main.py - the web API.

    GET  /health   is the server up, which library and model are loaded
    POST /ask      {"question": "..."} -> a cited, safety-checked answer
    GET  /docs     automatic interactive documentation (try /ask right in the browser)

Start it from the backend folder:   uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import Settings
from .llm.base import LLMError
from .schemas import AskRequest, AskResponse, HealthResponse
from .service import AssistantService, build_service

log = logging.getLogger("jsj")


def create_app(service: AssistantService | None = None) -> FastAPI:
    """Build the app. Tests pass a ready-made service; normally the service is built at startup."""
    settings = Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_service(settings)
        yield

    app = FastAPI(title="JSJ Intelligence API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),  # the website's address(es); set JSJ_CORS_ORIGINS to change
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        svc = request.app.state.service
        return HealthResponse(
            status="ok", library_mode=svc.settings.library_mode, chunks=svc.chunk_count,
            retrieval=svc.retrieval_method, model=svc.llm.name,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        svc = request.app.state.service
        try:
            result = svc.answer(body.question)
        except LLMError as e:
            log.warning("model call failed: %s", e)
            raise HTTPException(status_code=503, detail=str(e))
        log.info("answered refused=%s withheld=%s total_ms=%d", result.refused,
                 result.withheld_reason, result.timings.total_ms)  # the question text itself is not logged
        return result

    return app


app = create_app()
