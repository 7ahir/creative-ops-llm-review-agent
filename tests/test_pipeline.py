import json
from pathlib import Path
import time

from fastapi.testclient import TestClient

from creative_ops_llm_review_agent.api import create_app
from creative_ops_llm_review_agent.config import Settings
from creative_ops_llm_review_agent.models import CreativeBrief
from creative_ops_llm_review_agent.pipeline import CreativeOpsPipeline
from creative_ops_llm_review_agent.providers.heuristic import HeuristicCreativeProvider


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


def sample_brief() -> dict:
    return {
        "brand_key": "acme",
        "campaign_name": "Test Campaign",
        "product_name": "Acme Creative Studio",
        "audience": "performance marketers",
        "objective": "Speed up review",
        "key_message": "Review banner copy against brand and placement constraints before launch.",
        "tone": "confident",
        "offer_details": "Built for busy teams. Terms apply.",
        "required_terms": ["Terms apply"],
        "forbidden_terms": ["guaranteed", "instant"],
        "call_to_action": "Book a demo",
        "channel": "display",
        "placement": "display_300x250"
    }


def write_trace(path: Path, provider_key: str, provider: str, strategy: str, latency_ms: float) -> None:
    payload = {
        "trace_id": path.stem,
        "provider": provider,
        "strategy": strategy,
        "metrics": {
            "total_latency_ms": latency_ms,
            "average_variant_score": 1.0,
            "passed_variant_count": 3,
            "total_variant_count": 3,
            "estimated_cost_usd": 0.0,
        },
        "reviews": [],
        "events": [
            {
                "message": "provider_completed",
                "payload": {
                    "provider": provider,
                    "provider_key": provider_key,
                    "attempt": "primary",
                    "observed_latency_ms": latency_ms,
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_trace(settings: Settings, trace_path: str) -> dict:
    return json.loads((settings.runs_dir.parent / trace_path).read_text())


def test_tool_aware_pipeline_outscores_baseline(tmp_path: Path) -> None:
    pipeline = CreativeOpsPipeline(settings=build_settings(tmp_path))
    brief = CreativeBrief.model_validate(sample_brief())
    baseline = pipeline.generate_and_review(brief, strategy="baseline")
    improved = pipeline.generate_and_review(brief, strategy="tool_aware")

    assert improved.metrics.average_variant_score > baseline.metrics.average_variant_score
    assert improved.metrics.passed_variant_count >= baseline.metrics.passed_variant_count


def test_api_generate_returns_trace_and_metrics(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path))
    client = TestClient(app)

    response = client.post("/api/generate?strategy=tool_aware", json=sample_brief())
    payload = response.json()

    assert response.status_code == 200
    assert payload["trace_id"]
    assert payload["metrics"]["average_variant_score"] > 0
    assert len(payload["variants"]) == 3


def test_api_providers_endpoint_exposes_mcp_info(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path))
    client = TestClient(app)

    response = client.get("/api/providers")
    payload = response.json()

    assert response.status_code == 200
    assert "heuristic" in payload["providers"]
    assert payload["mcp_endpoint"].endswith("/mcp")
    assert payload["ollama_fallback_provider"] == "heuristic"
    assert payload["ollama_preemptive_routing_enabled"] is True


def test_api_metrics_summary_exposes_latency_and_fallback_fields(tmp_path: Path) -> None:
    app = create_app(build_settings(tmp_path))
    client = TestClient(app)

    generate_response = client.post("/api/generate?strategy=tool_aware", json=sample_brief())
    assert generate_response.status_code == 200

    response = client.get("/api/metrics/summary")
    payload = response.json()

    assert response.status_code == 200
    assert "p95_latency_ms" in payload
    assert "fallback_count" in payload
    assert "fallback_rate" in payload
    assert "provider_mix" in payload


class SlowOllamaProvider:
    name = "ollama-chat"

    def __init__(self, delay_seconds: float = 0.01) -> None:
        self.delay_seconds = delay_seconds
        self.delegate = HeuristicCreativeProvider()

    def generate(self, brief, brand_rules, channel_spec, policy_rules, strategy):
        time.sleep(self.delay_seconds)
        output = self.delegate.generate(brief, brand_rules, channel_spec, policy_rules, strategy)
        return output.model_copy(update={"provider_name": self.name, "prompt_version": "ollama-test"})


class FailingOllamaProvider:
    name = "ollama-chat"

    def generate(self, brief, brand_rules, channel_spec, policy_rules, strategy):
        raise RuntimeError("invalid JSON from local model")


class UnexpectedOllamaProvider:
    name = "ollama-chat"

    def generate(self, brief, brand_rules, channel_spec, policy_rules, strategy):
        raise AssertionError("Ollama should not be called when preemptive routing is active")


def test_pipeline_falls_back_when_ollama_exceeds_latency_budget(tmp_path: Path, monkeypatch) -> None:
    settings = build_settings(tmp_path)
    settings.provider_name = "ollama"
    settings.ollama_latency_budget_ms = 1
    pipeline = CreativeOpsPipeline(settings=settings)

    def fake_get_provider(name: str, active_settings: Settings, tool_runtime):
        if name == "ollama":
            return SlowOllamaProvider()
        if name == "heuristic":
            return HeuristicCreativeProvider()
        raise AssertionError("Unexpected provider request: %s" % name)

    monkeypatch.setattr("creative_ops_llm_review_agent.pipeline.get_provider", fake_get_provider)

    response = pipeline.generate_and_review(CreativeBrief.model_validate(sample_brief()), strategy="tool_aware")

    assert response.provider == "heuristic-provider"
    assert response.fallback is not None
    assert response.fallback.reason == "latency_budget_exceeded"
    assert response.fallback.requested_provider == "ollama"
    assert response.fallback.fallback_provider == "heuristic"
    assert response.fallback.observed_latency_ms > response.fallback.latency_budget_ms

    trace_payload = read_trace(settings, response.trace_path)
    events = [event["message"] for event in trace_payload["events"]]
    assert "provider_fallback_triggered" in events
    assert "provider_fallback_completed" in events


def test_pipeline_falls_back_when_ollama_errors(tmp_path: Path, monkeypatch) -> None:
    settings = build_settings(tmp_path)
    settings.provider_name = "ollama"
    settings.ollama_latency_budget_ms = 0
    pipeline = CreativeOpsPipeline(settings=settings)

    def fake_get_provider(name: str, active_settings: Settings, tool_runtime):
        if name == "ollama":
            return FailingOllamaProvider()
        if name == "heuristic":
            return HeuristicCreativeProvider()
        raise AssertionError("Unexpected provider request: %s" % name)

    monkeypatch.setattr("creative_ops_llm_review_agent.pipeline.get_provider", fake_get_provider)

    response = pipeline.generate_and_review(CreativeBrief.model_validate(sample_brief()), strategy="tool_aware")

    assert response.provider == "heuristic-provider"
    assert response.fallback is not None
    assert response.fallback.reason == "provider_error"
    assert "invalid JSON" in response.fallback.error
    assert not response.trace_path.startswith("/")


def test_pipeline_routes_preemptively_when_recent_ollama_latency_predicts_miss(
    tmp_path: Path, monkeypatch
) -> None:
    settings = build_settings(tmp_path)
    settings.provider_name = "ollama"
    settings.ollama_latency_budget_ms = 1000
    pipeline = CreativeOpsPipeline(settings=settings)

    write_trace(settings.traces_dir / "recent-a.json", "ollama", "ollama-chat", "tool_aware", 120000.0)
    write_trace(settings.traces_dir / "recent-b.json", "ollama", "ollama-chat", "tool_aware", 118000.0)

    def fake_get_provider(name: str, active_settings: Settings, tool_runtime):
        if name == "ollama":
            return UnexpectedOllamaProvider()
        if name == "heuristic":
            return HeuristicCreativeProvider()
        raise AssertionError("Unexpected provider request: %s" % name)

    monkeypatch.setattr("creative_ops_llm_review_agent.pipeline.get_provider", fake_get_provider)

    response = pipeline.generate_and_review(CreativeBrief.model_validate(sample_brief()), strategy="tool_aware")

    assert response.provider == "heuristic-provider"
    assert response.fallback is not None
    assert response.fallback.reason == "predicted_latency_exceeded"
    assert response.fallback.predicted_latency_ms >= 118000.0
    assert response.fallback.observed_latency_ms is None

    trace_payload = read_trace(settings, response.trace_path)
    fallback_event = next(
        event for event in trace_payload["events"] if event["message"] == "provider_fallback_triggered"
    )
    assert fallback_event["payload"]["predicted_latency_ms"] >= 118000.0
