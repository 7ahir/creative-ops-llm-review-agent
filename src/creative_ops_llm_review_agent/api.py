from __future__ import annotations

from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .config import Settings
from .models import CreativeBrief, ReviewOnlyRequest
from .observability import load_recent_traces, summarize_traces
from .pipeline import CreativeOpsPipeline


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    app = FastAPI(title="Creative Ops LLM Review Agent", version="0.1.0")
    active_settings = settings or Settings.load()
    pipeline = CreativeOpsPipeline(active_settings)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "metrics": summarize_traces(active_settings),
                "recent_traces": load_recent_traces(active_settings, limit=5),
            },
        )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/api/generate")
    def generate(
        brief: CreativeBrief,
        strategy: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> dict:
        try:
            response = pipeline.generate_and_review(brief, strategy=strategy, provider_name=provider)
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return response.model_dump(mode="json")

    @app.get("/api/providers")
    def providers() -> dict:
        return {
            "default_provider": active_settings.provider_name,
            "providers": ["heuristic", "openai", "ollama"],
            "default_model": active_settings.openai_model,
            "mcp_endpoint": "http://%s:%s%s"
            % (active_settings.mcp_host, active_settings.mcp_port, active_settings.mcp_path),
            "ollama_base_url": active_settings.ollama_base_url,
            "ollama_model": active_settings.ollama_model,
            "ollama_latency_budget_ms": active_settings.ollama_latency_budget_ms,
            "ollama_fallback_provider": active_settings.ollama_fallback_provider,
            "ollama_preemptive_routing_enabled": active_settings.ollama_preemptive_routing_enabled,
            "ollama_preemptive_min_samples": active_settings.ollama_preemptive_min_samples,
            "ollama_preemptive_percentile": active_settings.ollama_preemptive_percentile,
        }

    @app.post("/api/review")
    def review(request: ReviewOnlyRequest) -> dict:
        return {"reviews": [item.model_dump(mode="json") for item in pipeline.review_existing(request)]}

    @app.get("/api/traces")
    def traces(limit: int = 10) -> dict:
        return {"traces": load_recent_traces(active_settings, limit=limit)}

    @app.get("/api/traces/{trace_id}")
    def trace_detail(trace_id: str) -> dict:
        path = active_settings.traces_dir / ("%s.json" % trace_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Trace not found")
        return {"trace": path.read_text()}

    @app.get("/api/metrics/summary")
    def metrics_summary() -> dict:
        return summarize_traces(active_settings)

    return app


app = create_app()


def serve() -> None:
    uvicorn.run("creative_ops_llm_review_agent.api:app", host="127.0.0.1", port=8000, reload=False)
