from __future__ import annotations

import json
import logging
import math
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from .config import Settings
from .models import StageRecord


_LOCK = threading.Lock()
_CONFIGURED = False


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "payload"):
            payload["payload"] = _serialize(record.payload)
        return json.dumps(payload)


class JsonFileSpanExporter(SpanExporter):
    def __init__(self, path: Path) -> None:
        self.path = path

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        with _LOCK:
            with self.path.open("a", encoding="utf-8") as handle:
                for span in spans:
                    payload = {
                        "trace_id": format(span.context.trace_id, "032x"),
                        "span_id": format(span.context.span_id, "016x"),
                        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
                        "name": span.name,
                        "start_time_unix_nano": span.start_time,
                        "end_time_unix_nano": span.end_time,
                        "attributes": _serialize(dict(span.attributes)),
                    }
                    handle.write(json.dumps(payload) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def configure_observability(settings: Settings) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger = logging.getLogger("creative_ops")
    logger.setLevel(settings.log_level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(settings.logs_path)
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonFileSpanExporter(settings.spans_path)))
    trace.set_tracer_provider(provider)

    _CONFIGURED = True


def get_logger() -> logging.Logger:
    return logging.getLogger("creative_ops")


class TraceRecorder:
    def __init__(self, settings: Settings, trace_id: str, strategy: str) -> None:
        self.settings = settings
        self.trace_id = trace_id
        self.strategy = strategy
        self.started_at = time.perf_counter()
        self.stages: List[StageRecord] = []
        self.events: List[Dict[str, Any]] = []
        self.tracer = trace.get_tracer("creative_ops_llm_review_agent")

    @contextmanager
    def stage(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        stage_meta = metadata or {}
        stage_start = time.perf_counter()
        with self.tracer.start_as_current_span(name) as span:
            span.set_attribute("creative_ops.trace_id", self.trace_id)
            for key, value in stage_meta.items():
                span.set_attribute(key, str(value))
            try:
                yield
            finally:
                duration = round((time.perf_counter() - stage_start) * 1000, 2)
                self.stages.append(StageRecord(name=name, duration_ms=duration, metadata=stage_meta))

    def log_event(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "payload": _serialize(payload or {}),
        }
        self.events.append(entry)
        get_logger().info(message, extra={"payload": {"trace_id": self.trace_id, **entry["payload"]}})

    @property
    def total_latency_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 2)

    def persist(self, payload: Dict[str, Any]) -> str:
        target = self.settings.traces_dir / ("%s.json" % self.trace_id)
        with target.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(_serialize(payload), indent=2))
        # Return a repo-relative path so API responses and screenshots do not leak local home directories.
        try:
            return str(target.relative_to(self.settings.runs_dir.parent))
        except ValueError:
            pass
        try:
            return str(target.relative_to(self.settings.project_root))
        except ValueError:
            return str(target)


def load_recent_traces(settings: Settings, limit: int = 10) -> List[Dict[str, Any]]:
    traces = sorted(settings.traces_dir.glob("*.json"), reverse=True)[:limit]
    payloads: List[Dict[str, Any]] = []
    for path in traces:
        payloads.append(json.loads(path.read_text()))
    return payloads


def _primary_provider_key(trace_payload: Dict[str, Any]) -> str:
    fallback = trace_payload.get("fallback") or {}
    if fallback.get("requested_provider"):
        return str(fallback["requested_provider"])

    # Fall back to the primary provider event so routing summaries stay stable even when
    # the final provider was a heuristic fallback.
    for event in trace_payload.get("events", []):
        if event.get("message") != "provider_completed":
            continue
        payload = event.get("payload", {})
        if payload.get("attempt") != "primary":
            continue
        provider_key = payload.get("provider_key")
        if provider_key:
            return str(provider_key)
    return str(trace_payload.get("provider", "unknown"))


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(ordered[index], 2)


def estimate_provider_latency_ms(
    settings: Settings,
    provider_name: str,
    strategy: str,
    percentile: int = 95,
    min_samples: int = 2,
    limit: int = 20,
) -> Optional[float]:
    traces = sorted(settings.traces_dir.glob("*.json"), reverse=True)
    latencies: List[float] = []
    for path in traces:
        payload = json.loads(path.read_text())
        if payload.get("strategy") != strategy:
            continue
        if _primary_provider_key(payload) != provider_name:
            continue
        metrics = payload.get("metrics", {})
        total_latency_ms = metrics.get("total_latency_ms")
        if total_latency_ms is None:
            continue
        latencies.append(float(total_latency_ms))
        if len(latencies) >= limit:
            break

    if len(latencies) < max(min_samples, 1):
        return None
    # Use recent persisted traces as a cheap predictor for the next local-model call.
    return _percentile(list(reversed(latencies)), percentile)


def summarize_traces(settings: Settings) -> Dict[str, Any]:
    traces = list(settings.traces_dir.glob("*.json"))
    if not traces:
        return {
            "request_count": 0,
            "average_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "average_cost_usd": 0.0,
            "pass_rate": 0.0,
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "fallback_reasons": {},
            "provider_mix": {},
            "top_issue_categories": {},
        }

    latencies: List[float] = []
    total_latency = 0.0
    total_cost = 0.0
    total_variants = 0
    total_passed = 0
    issue_counts: Dict[str, int] = {}
    fallback_count = 0
    fallback_reasons: Dict[str, int] = {}
    provider_mix: Dict[str, int] = {}

    for path in traces:
        payload = json.loads(path.read_text())
        metrics = payload["metrics"]
        latency_ms = metrics["total_latency_ms"]
        latencies.append(latency_ms)
        total_latency += metrics["total_latency_ms"]
        total_cost += metrics["estimated_cost_usd"]
        total_variants += metrics["total_variant_count"]
        total_passed += metrics["passed_variant_count"]
        provider_name = payload.get("provider", "unknown")
        provider_mix[provider_name] = provider_mix.get(provider_name, 0) + 1
        fallback = payload.get("fallback")
        if fallback:
            fallback_count += 1
            reason = fallback.get("reason", "unknown")
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        for review in payload["reviews"]:
            for issue in review["issues"]:
                issue_counts[issue["category"]] = issue_counts.get(issue["category"], 0) + 1

    return {
        "request_count": len(traces),
        "average_latency_ms": round(total_latency / len(traces), 2),
        "p95_latency_ms": _percentile(latencies, 95),
        "average_cost_usd": round(total_cost / len(traces), 4),
        "pass_rate": round(total_passed / max(total_variants, 1), 3),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(traces), 3),
        "fallback_reasons": dict(sorted(fallback_reasons.items(), key=lambda item: (-item[1], item[0]))),
        "provider_mix": dict(sorted(provider_mix.items(), key=lambda item: (-item[1], item[0]))),
        "top_issue_categories": dict(sorted(issue_counts.items(), key=lambda item: item[1], reverse=True)[:5]),
    }
