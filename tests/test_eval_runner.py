from datetime import datetime, timezone
from pathlib import Path

from creative_ops_llm_review_agent.config import Settings
from creative_ops_llm_review_agent.eval_runner import run_benchmark_matrix, run_evaluation
from creative_ops_llm_review_agent.models import (
    CreativeResponse,
    GeneratedVariant,
    RequestMetrics,
    ReviewIssue,
    StageRecord,
    VariantReview,
    VariantScore,
    FallbackDecision,
)


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


def test_eval_report_shows_positive_delta(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    report = run_evaluation(settings, PROJECT_ROOT / "data" / "golden_set.json")

    assert report.cases_run == 3
    assert report.score_delta > 0
    assert report.pass_rate_delta >= 0


def test_benchmark_matrix_captures_provider_mix_and_fallbacks(tmp_path: Path, monkeypatch) -> None:
    settings = build_settings(tmp_path)
    dataset_path = PROJECT_ROOT / "data" / "golden_set.json"

    class FakePipeline:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def generate_and_review(self, brief, strategy: str, provider_name: str) -> CreativeResponse:
            fallback = None
            provider = "heuristic-provider"
            latency_ms = 20.0
            if provider_name == "ollama" and self.settings.ollama_latency_budget_ms == 0:
                provider = "ollama-chat"
                latency_ms = 120.0
            elif provider_name == "ollama":
                fallback = FallbackDecision(
                    requested_provider="ollama",
                    fallback_provider="heuristic",
                    final_provider="heuristic-provider",
                    reason="latency_budget_exceeded",
                    latency_budget_ms=float(self.settings.ollama_latency_budget_ms),
                    observed_latency_ms=120.0,
                )
                latency_ms = 121.0

            variant = GeneratedVariant(
                variant_id="variant-a",
                headline="Teams first",
                description="Built for busy teams. Terms apply.",
                cta=brief.call_to_action,
                rationale="Test variant",
            )
            review = VariantReview(
                variant_id="variant-a",
                passed=True,
                score=VariantScore(
                    overall=1.0,
                    length_compliance=1.0,
                    policy_compliance=1.0,
                    brand_alignment=1.0,
                    message_clarity=1.0,
                ),
                issues=[],
            )
            return CreativeResponse(
                trace_id="trace-%s-%s" % (provider_name, brief.campaign_name.replace(" ", "-")),
                created_at=datetime.now(timezone.utc),
                provider=provider,
                strategy=strategy,
                brief=brief,
                variants=[variant],
                reviews=[review],
                best_variant_id="variant-a",
                prompt_version="test",
                metrics=RequestMetrics(
                    total_latency_ms=latency_ms,
                    average_variant_score=1.0,
                    passed_variant_count=1,
                    total_variant_count=1,
                    estimated_cost_usd=0.0,
                ),
                stages=[StageRecord(name="generate_variants", duration_ms=latency_ms, metadata={})],
                tools_consulted=[],
                fallback=fallback,
                trace_path=str(tmp_path / ("%s.json" % brief.campaign_name.replace(" ", "-"))),
            )

    monkeypatch.setattr("creative_ops_llm_review_agent.eval_runner.CreativeOpsPipeline", FakePipeline)

    report = run_benchmark_matrix(
        settings=settings,
        dataset_path=dataset_path,
        ollama_model="qwen3:4b",
        fallback_budget_ms=1,
        limit=2,
    )

    assert report.cases_run == 2
    assert len(report.scenarios) == 3
    assert report.scenarios[0].provider_mix == {"heuristic-provider": 2}
    assert report.scenarios[1].provider_mix == {"ollama-chat": 2}
    assert report.scenarios[2].fallback_rate == 1.0
    assert report.scenarios[2].fallback_count == 2
