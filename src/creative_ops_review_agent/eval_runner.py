from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional

from .config import Settings
from .models import (
    BenchmarkMatrixReport,
    BenchmarkScenarioSummary,
    EvalCase,
    EvalReport,
    EvalStrategySummary,
)
from .pipeline import CreativeOpsPipeline


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_evaluation(settings: Settings, dataset_path: Path) -> EvalReport:
    cases = [EvalCase.model_validate(item) for item in json.loads(dataset_path.read_text())]
    pipeline = CreativeOpsPipeline(settings=settings)
    strategy_summaries: List[EvalStrategySummary] = []

    for strategy in ("baseline", "tool_aware"):
        score_total = 0.0
        pass_total = 0
        latency_total = 0.0
        cost_total = 0.0
        issue_counts: Counter = Counter()

        for case in cases:
            response = pipeline.generate_and_review(case.brief, strategy=strategy)
            best_review = next(review for review in response.reviews if review.variant_id == response.best_variant_id)
            score_total += best_review.score.overall
            pass_total += 1 if best_review.passed else 0
            latency_total += response.metrics.total_latency_ms
            cost_total += response.metrics.estimated_cost_usd
            for issue in best_review.issues:
                issue_counts[issue.category] += 1

        strategy_summaries.append(
            EvalStrategySummary(
                strategy=strategy,
                average_score=round(score_total / len(cases), 3),
                pass_rate=round(pass_total / len(cases), 3),
                average_latency_ms=round(latency_total / len(cases), 2),
                average_cost_usd=round(cost_total / len(cases), 4),
                issue_counts=dict(issue_counts),
            )
        )

    baseline, improved = strategy_summaries
    report = EvalReport(
        generated_at=datetime.now(timezone.utc),
        dataset_path=_relative_path(dataset_path, settings.project_root),
        cases_run=len(cases),
        summaries=strategy_summaries,
        score_delta=round(improved.average_score - baseline.average_score, 3),
        pass_rate_delta=round(improved.pass_rate - baseline.pass_rate, 3),
    )
    output_path = settings.evals_dir / ("eval-%s.json" % report.generated_at.strftime("%Y%m%d%H%M%S"))
    output_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    return report


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    provider: str
    strategy: str
    ollama_model: str
    ollama_latency_budget_ms: int
    ollama_fallback_provider: str
    ollama_fallback_on_error: bool


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(ordered[index], 2)


def _load_cases(dataset_path: Path, limit: Optional[int] = None) -> List[EvalCase]:
    cases = [EvalCase.model_validate(item) for item in json.loads(dataset_path.read_text())]
    return cases[:limit] if limit else cases


def _benchmark_settings(
    settings: Settings,
    benchmark_root: Path,
    scenario: BenchmarkScenario,
) -> Settings:
    traces_dir = benchmark_root / "traces"
    evals_dir = benchmark_root / "evals"
    traces_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        settings,
        runs_dir=benchmark_root,
        traces_dir=traces_dir,
        evals_dir=evals_dir,
        logs_path=benchmark_root / "app.jsonl",
        spans_path=benchmark_root / "spans.jsonl",
        provider_name=scenario.provider,
        ollama_model=scenario.ollama_model,
        ollama_latency_budget_ms=scenario.ollama_latency_budget_ms,
        ollama_fallback_provider=scenario.ollama_fallback_provider,
        ollama_fallback_on_error=scenario.ollama_fallback_on_error,
    )


def _benchmark_scenarios(settings: Settings, ollama_model: str, fallback_budget_ms: int) -> List[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            name="heuristic_tool_aware",
            provider="heuristic",
            strategy="tool_aware",
            ollama_model=ollama_model,
            ollama_latency_budget_ms=0,
            ollama_fallback_provider=settings.ollama_fallback_provider,
            ollama_fallback_on_error=False,
        ),
        BenchmarkScenario(
            name="ollama_tool_aware_direct",
            provider="ollama",
            strategy="tool_aware",
            ollama_model=ollama_model,
            ollama_latency_budget_ms=0,
            ollama_fallback_provider=settings.ollama_fallback_provider,
            ollama_fallback_on_error=False,
        ),
        BenchmarkScenario(
            name="ollama_tool_aware_fallback",
            provider="ollama",
            strategy="tool_aware",
            ollama_model=ollama_model,
            ollama_latency_budget_ms=fallback_budget_ms,
            ollama_fallback_provider=settings.ollama_fallback_provider,
            ollama_fallback_on_error=True,
        ),
    ]


def run_benchmark_matrix(
    settings: Settings,
    dataset_path: Path,
    ollama_model: str,
    fallback_budget_ms: int = 1,
    limit: Optional[int] = None,
) -> BenchmarkMatrixReport:
    cases = _load_cases(dataset_path, limit=limit)
    generated_at = datetime.now(timezone.utc)
    benchmark_id = generated_at.strftime("%Y%m%d%H%M%S")
    benchmark_root = settings.runs_dir / "benchmarks" / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)

    scenarios: List[BenchmarkScenarioSummary] = []
    for scenario in _benchmark_scenarios(settings, ollama_model, fallback_budget_ms):
        # Each scenario gets its own trace sink so the summary can compare routing decisions cleanly.
        scenario_settings = _benchmark_settings(settings, benchmark_root, scenario)
        pipeline = CreativeOpsPipeline(settings=scenario_settings)
        latencies: List[float] = []
        score_total = 0.0
        pass_total = 0
        cost_total = 0.0
        fallback_count = 0
        provider_mix: Counter = Counter()
        issue_counts: Counter = Counter()
        trace_paths: List[str] = []

        for case in cases:
            response = pipeline.generate_and_review(case.brief, strategy=scenario.strategy, provider_name=scenario.provider)
            best_review = next(review for review in response.reviews if review.variant_id == response.best_variant_id)
            latencies.append(response.metrics.total_latency_ms)
            score_total += best_review.score.overall
            pass_total += 1 if best_review.passed else 0
            cost_total += response.metrics.estimated_cost_usd
            fallback_count += 1 if response.fallback else 0
            provider_mix[response.provider] += 1
            trace_paths.append(response.trace_path or "")
            for issue in best_review.issues:
                issue_counts[issue.category] += 1

        scenarios.append(
            BenchmarkScenarioSummary(
                scenario=scenario.name,
                provider=scenario.provider,
                strategy=scenario.strategy,
                model=scenario.ollama_model if scenario.provider == "ollama" else "deterministic",
                average_score=round(score_total / len(cases), 3),
                pass_rate=round(pass_total / len(cases), 3),
                average_latency_ms=round(sum(latencies) / len(latencies), 2),
                p95_latency_ms=_percentile(latencies, 95),
                average_cost_usd=round(cost_total / len(cases), 4),
                fallback_count=fallback_count,
                fallback_rate=round(fallback_count / len(cases), 3),
                provider_mix=dict(sorted(provider_mix.items(), key=lambda item: (-item[1], item[0]))),
                issue_counts=dict(sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))),
                trace_paths=[
                    _relative_path(Path(path), settings.project_root) for path in trace_paths if path
                ],
            )
        )

    report = BenchmarkMatrixReport(
        generated_at=generated_at,
        dataset_path=_relative_path(dataset_path, settings.project_root),
        cases_run=len(cases),
        benchmark_root=_relative_path(benchmark_root, settings.project_root),
        fallback_budget_ms=fallback_budget_ms,
        scenarios=scenarios,
    )
    output_path = settings.evals_dir / ("benchmark-%s.json" % generated_at.strftime("%Y%m%d%H%M%S"))
    output_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline evaluation for the Creative Ops Review Agent.")
    parser.add_argument(
        "--dataset",
        default="data/golden_set.json",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the local benchmark matrix across heuristic, direct Ollama, and fallback-routed Ollama.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Override the Ollama model for matrix runs.",
    )
    parser.add_argument(
        "--fallback-budget-ms",
        type=int,
        default=1,
        help="Latency budget used for the fallback-routed Ollama scenario.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of evaluation cases.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _run_from_args(args)


def _run_from_args(args: argparse.Namespace) -> None:
    settings = Settings.load()
    dataset_path = settings.project_root / args.dataset
    if args.matrix:
        report = run_benchmark_matrix(
            settings=settings,
            dataset_path=dataset_path,
            ollama_model=args.ollama_model or settings.ollama_model,
            fallback_budget_ms=args.fallback_budget_ms,
            limit=args.limit,
        )
    else:
        report = run_evaluation(settings, dataset_path)
    print(json.dumps(report.model_dump(mode="json"), indent=2))


def benchmark_main() -> None:
    parser = _build_parser()
    cli_args = sys.argv[1:]
    if "--matrix" not in cli_args:
        cli_args = ["--matrix", *cli_args]
    args = parser.parse_args(cli_args)
    _run_from_args(args)
