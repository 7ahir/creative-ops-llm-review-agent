import json
from pathlib import Path

from creative_ops_review_agent.config import Settings
from creative_ops_review_agent.observability import summarize_traces


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_settings(tmp_path: Path) -> Settings:
    runs_dir = tmp_path / "runs"
    traces_dir = runs_dir / "traces"
    evals_dir = runs_dir / "evals"
    traces_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=PROJECT_ROOT / "data",
        runs_dir=runs_dir,
        traces_dir=traces_dir,
        evals_dir=evals_dir,
        logs_path=runs_dir / "app.jsonl",
        spans_path=runs_dir / "spans.jsonl",
        provider_name="heuristic",
        openai_model="gpt-5-mini",
        openai_input_cost_per_1k=0.0,
        openai_output_cost_per_1k=0.0,
        ollama_base_url="http://localhost:11434/v1/",
        ollama_model="qwen3",
        ollama_api_key="ollama",
        ollama_think=False,
        mcp_host="127.0.0.1",
        mcp_port=8002,
        mcp_path="/mcp",
    )


def write_trace(path: Path, trace_id: str, provider: str, latency_ms: float, fallback=None) -> None:
    payload = {
        "trace_id": trace_id,
        "provider": provider,
        "strategy": "tool_aware",
        "best_variant_id": "variant-a",
        "metrics": {
            "total_latency_ms": latency_ms,
            "average_variant_score": 0.95,
            "passed_variant_count": 2,
            "total_variant_count": 3,
            "estimated_cost_usd": 0.01,
        },
        "fallback": fallback,
        "reviews": [
            {
                "variant_id": "variant-a",
                "passed": True,
                "score": {
                    "overall": 1.0,
                    "length_compliance": 1.0,
                    "policy_compliance": 1.0,
                    "brand_alignment": 1.0,
                    "message_clarity": 1.0,
                },
                "issues": [],
            },
            {
                "variant_id": "variant-b",
                "passed": False,
                "score": {
                    "overall": 0.8,
                    "length_compliance": 0.7,
                    "policy_compliance": 1.0,
                    "brand_alignment": 0.8,
                    "message_clarity": 0.7,
                },
                "issues": [{"severity": "medium", "category": "headline_length", "message": "Too long"}],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_traces_reports_fallbacks_p95_and_provider_mix(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    write_trace(settings.traces_dir / "2026040301.json", "2026040301", "heuristic-provider", 100.0)
    write_trace(
        settings.traces_dir / "2026040302.json",
        "2026040302",
        "heuristic-provider",
        900.0,
        fallback={
            "requested_provider": "ollama",
            "fallback_provider": "heuristic",
            "final_provider": "heuristic-provider",
            "reason": "latency_budget_exceeded",
            "latency_budget_ms": 600.0,
            "observed_latency_ms": 900.0,
            "error": None,
        },
    )
    write_trace(settings.traces_dir / "2026040303.json", "2026040303", "openai-responses", 400.0)

    summary = summarize_traces(settings)

    assert summary["request_count"] == 3
    assert summary["average_latency_ms"] == 466.67
    assert summary["p95_latency_ms"] == 900.0
    assert summary["fallback_count"] == 1
    assert summary["fallback_rate"] == 0.333
    assert summary["fallback_reasons"] == {"latency_budget_exceeded": 1}
    assert summary["provider_mix"] == {"heuristic-provider": 2, "openai-responses": 1}
    assert summary["top_issue_categories"] == {"headline_length": 3}
