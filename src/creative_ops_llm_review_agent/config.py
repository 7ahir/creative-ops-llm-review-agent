from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    project_root: Path
    data_dir: Path
    runs_dir: Path
    traces_dir: Path
    evals_dir: Path
    logs_path: Path
    spans_path: Path
    provider_name: str
    openai_model: str
    openai_input_cost_per_1k: float
    openai_output_cost_per_1k: float
    ollama_base_url: str
    ollama_model: str
    ollama_api_key: str
    ollama_think: bool
    mcp_host: str
    mcp_port: int
    mcp_path: str
    default_strategy: str = "tool_aware"
    log_level: str = "INFO"
    ollama_latency_budget_ms: int = 0
    ollama_fallback_provider: str = "heuristic"
    ollama_fallback_on_error: bool = True
    ollama_preemptive_routing_enabled: bool = True
    ollama_preemptive_min_samples: int = 2
    ollama_preemptive_percentile: int = 95

    @classmethod
    def load(cls, project_root: Optional[Path] = None) -> "Settings":
        root = project_root or Path(__file__).resolve().parents[2]
        data_dir = root / "data"
        runs_dir = root / "runs"
        traces_dir = runs_dir / "traces"
        evals_dir = runs_dir / "evals"
        logs_path = runs_dir / "app.jsonl"
        spans_path = runs_dir / "spans.jsonl"

        traces_dir.mkdir(parents=True, exist_ok=True)
        evals_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            project_root=root,
            data_dir=data_dir,
            runs_dir=runs_dir,
            traces_dir=traces_dir,
            evals_dir=evals_dir,
            logs_path=logs_path,
            spans_path=spans_path,
            provider_name=os.getenv("CREATIVE_OPS_PROVIDER", "heuristic"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            openai_input_cost_per_1k=float(os.getenv("OPENAI_INPUT_COST_PER_1K", "0")),
            openai_output_cost_per_1k=float(os.getenv("OPENAI_OUTPUT_COST_PER_1K", "0")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3"),
            ollama_api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            ollama_think=_env_bool("OLLAMA_THINK", False),
            mcp_host=os.getenv("CREATIVE_OPS_MCP_HOST", "127.0.0.1"),
            mcp_port=int(os.getenv("CREATIVE_OPS_MCP_PORT", "8002")),
            mcp_path=os.getenv("CREATIVE_OPS_MCP_PATH", "/mcp"),
            default_strategy=os.getenv("CREATIVE_OPS_DEFAULT_STRATEGY", "tool_aware"),
            log_level=os.getenv("CREATIVE_OPS_LOG_LEVEL", "INFO"),
            ollama_latency_budget_ms=int(os.getenv("OLLAMA_LATENCY_BUDGET_MS", "0")),
            ollama_fallback_provider=os.getenv("OLLAMA_FALLBACK_PROVIDER", "heuristic"),
            ollama_fallback_on_error=_env_bool("OLLAMA_FALLBACK_ON_ERROR", True),
            ollama_preemptive_routing_enabled=_env_bool("OLLAMA_PREEMPTIVE_ROUTING_ENABLED", True),
            ollama_preemptive_min_samples=int(os.getenv("OLLAMA_PREEMPTIVE_MIN_SAMPLES", "2")),
            ollama_preemptive_percentile=int(os.getenv("OLLAMA_PREEMPTIVE_PERCENTILE", "95")),
        )
