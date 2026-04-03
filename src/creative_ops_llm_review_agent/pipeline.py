from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import time
from typing import List, Optional, Tuple

from .config import Settings
from .knowledge import KnowledgeStore
from .models import (
    BrandRuleSet,
    ChannelSpec,
    CreativeBrief,
    CreativeResponse,
    FallbackDecision,
    GeneratedVariant,
    PolicyRuleSet,
    ProviderOutput,
    RequestMetrics,
    ReviewOnlyRequest,
    VariantReview,
)
from .observability import TraceRecorder, configure_observability, estimate_provider_latency_ms
from .providers.factory import get_provider
from .repair import repair_variant_set
from .scoring import review_variant
from .tool_runtime import ConstraintToolRuntime


class ProviderAttemptError(RuntimeError):
    def __init__(self, provider_name: str, latency_ms: float, original: Exception) -> None:
        super().__init__(str(original))
        self.provider_name = provider_name
        self.latency_ms = latency_ms
        self.original = original


class CreativeOpsPipeline:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.load()
        configure_observability(self.settings)
        self.knowledge = KnowledgeStore(self.settings)
        self.tool_runtime = ConstraintToolRuntime(self.knowledge)

    def generate_and_review(
        self,
        brief: CreativeBrief,
        strategy: Optional[str] = None,
        provider_name: Optional[str] = None,
    ) -> CreativeResponse:
        selected_strategy = strategy or self.settings.default_strategy
        selected_provider_name = provider_name or self.settings.provider_name
        recorder = TraceRecorder(self.settings, trace_id=self._new_trace_id(), strategy=selected_strategy)
        recorder.log_event(
            "request_started",
            {
                "strategy": selected_strategy,
                "provider": selected_provider_name,
                "campaign_name": brief.campaign_name,
            },
        )

        with recorder.stage("load_context", {"brand_key": brief.brand_key, "placement": brief.placement}):
            brand_rules = self.knowledge.brand_rules(brief.brand_key)
            channel_spec = self.knowledge.channel_spec(brief.placement)
            policy_rules = self.knowledge.policy_rules()

        provider_output, fallback = self._generate_with_optional_fallback(
            brief=brief,
            brand_rules=brand_rules,
            channel_spec=channel_spec,
            policy_rules=policy_rules,
            strategy=selected_strategy,
            provider_name=selected_provider_name,
            recorder=recorder,
        )
        variants = provider_output.variants

        with recorder.stage("review_variants", {"variant_count": len(variants)}):
            reviews = [
                review_variant(
                    brief=brief,
                    variant=variant,
                    brand_rules=brand_rules,
                    channel_spec=channel_spec,
                    policy_rules=policy_rules,
                )
                for variant in variants
            ]

        if selected_strategy == "tool_aware":
            failing_variant_count = len([review for review in reviews if not review.passed])
            if failing_variant_count:
                with recorder.stage("repair_variants", {"failing_variant_count": failing_variant_count}):
                    variants, reviews, repaired_count = repair_variant_set(
                        brief=brief,
                        variants=variants,
                        reviews=reviews,
                        brand_rules=brand_rules,
                        channel_spec=channel_spec,
                        policy_rules=policy_rules,
                    )
                    recorder.log_event(
                        "repair_completed",
                        {
                            "failing_variant_count": failing_variant_count,
                            "repaired_variant_count": repaired_count,
                        },
                    )

        with recorder.stage("select_best_variant", {"variant_count": len(variants)}):
            best_review = sorted(
                reviews,
                key=lambda item: (item.passed, item.score.overall, -len(item.issues)),
                reverse=True,
            )[0]

        response = CreativeResponse(
            trace_id=recorder.trace_id,
            created_at=datetime.now(timezone.utc),
            provider=provider_output.provider_name,
            strategy=selected_strategy,
            brief=brief,
            variants=variants,
            reviews=reviews,
            best_variant_id=best_review.variant_id,
            prompt_version=provider_output.prompt_version,
            metrics=RequestMetrics(
                total_latency_ms=recorder.total_latency_ms,
                average_variant_score=round(sum(review.score.overall for review in reviews) / len(reviews), 3),
                passed_variant_count=len([review for review in reviews if review.passed]),
                total_variant_count=len(reviews),
                estimated_cost_usd=provider_output.usage.estimated_cost_usd,
            ),
            stages=recorder.stages,
            tools_consulted=provider_output.tools_consulted,
            fallback=fallback,
        )

        with recorder.stage("persist_trace", {"trace_id": recorder.trace_id}):
            trace_path = recorder.persist(
                {
                    **response.model_dump(mode="json"),
                    "provider_usage": provider_output.usage.model_dump(),
                    "prompt_preview": provider_output.prompt_preview,
                    "provider_raw_output_text": provider_output.raw_output_text,
                    "provider_tool_events": provider_output.tool_events,
                    "events": recorder.events,
                }
            )
            response.trace_path = trace_path

        recorder.log_event(
            "request_completed",
            {
                "best_variant_id": response.best_variant_id,
                "average_variant_score": response.metrics.average_variant_score,
                "passed_variant_count": response.metrics.passed_variant_count,
                "provider": response.provider,
                "fallback_reason": fallback.reason if fallback else None,
            },
        )
        return response

    def review_existing(self, request: ReviewOnlyRequest) -> List[VariantReview]:
        brand_rules = self.knowledge.brand_rules(request.brief.brand_key)
        channel_spec = self.knowledge.channel_spec(request.brief.placement)
        policy_rules = self.knowledge.policy_rules()
        return [
            review_variant(request.brief, variant, brand_rules, channel_spec, policy_rules)
            for variant in request.variants
        ]

    def failure_taxonomy(self, response: CreativeResponse) -> Counter:
        counter: Counter = Counter()
        for review in response.reviews:
            for issue in review.issues:
                counter[issue.category] += 1
        return counter

    @staticmethod
    def _new_trace_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    def _generate_with_optional_fallback(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
        provider_name: str,
        recorder: TraceRecorder,
    ) -> Tuple[ProviderOutput, Optional[FallbackDecision]]:
        preemptive_prediction_ms = self._predicted_latency_ms(provider_name, strategy)
        if preemptive_prediction_ms is not None:
            # Skip the local model entirely when recent traces already show it will miss budget.
            return self._run_fallback_attempt(
                brief=brief,
                brand_rules=brand_rules,
                channel_spec=channel_spec,
                policy_rules=policy_rules,
                strategy=strategy,
                provider_name=provider_name,
                recorder=recorder,
                reason="predicted_latency_exceeded",
                latency_budget_ms=self._latency_budget_ms(provider_name),
                predicted_latency_ms=preemptive_prediction_ms,
            )

        provider = get_provider(provider_name, self.settings, self.tool_runtime)
        try:
            primary_output, primary_latency_ms = self._run_provider_attempt(
                provider=provider,
                provider_name=provider_name,
                brief=brief,
                brand_rules=brand_rules,
                channel_spec=channel_spec,
                policy_rules=policy_rules,
                strategy=strategy,
                recorder=recorder,
                stage_name="generate_variants",
                attempt="primary",
            )
        except ProviderAttemptError as exc:
            if not self._should_fallback_on_error(provider_name):
                raise RuntimeError(str(exc.original)) from exc.original
            fallback = self._run_fallback_attempt(
                brief=brief,
                brand_rules=brand_rules,
                channel_spec=channel_spec,
                policy_rules=policy_rules,
                strategy=strategy,
                provider_name=provider_name,
                recorder=recorder,
                reason="provider_error",
                observed_latency_ms=exc.latency_ms,
                error=str(exc.original),
            )
            return fallback

        latency_budget_ms = self._latency_budget_ms(provider_name)
        if latency_budget_ms and primary_latency_ms > latency_budget_ms and self._fallback_provider_name(provider_name):
            return self._run_fallback_attempt(
                brief=brief,
                brand_rules=brand_rules,
                channel_spec=channel_spec,
                policy_rules=policy_rules,
                strategy=strategy,
                provider_name=provider_name,
                recorder=recorder,
                reason="latency_budget_exceeded",
                observed_latency_ms=primary_latency_ms,
                latency_budget_ms=latency_budget_ms,
            )

        return primary_output, None

    def _run_fallback_attempt(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
        provider_name: str,
        recorder: TraceRecorder,
        reason: str,
        observed_latency_ms: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
        predicted_latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> Tuple[ProviderOutput, FallbackDecision]:
        fallback_provider_name = self._fallback_provider_name(provider_name)
        if not fallback_provider_name:
            raise RuntimeError("No fallback provider configured for %s." % provider_name)

        recorder.log_event(
            "provider_fallback_triggered",
            {
                "requested_provider": provider_name,
                "reason": reason,
                "latency_budget_ms": latency_budget_ms,
                "observed_latency_ms": observed_latency_ms,
                "predicted_latency_ms": predicted_latency_ms,
                "fallback_provider": fallback_provider_name,
                "error": error,
            },
        )

        fallback_provider = get_provider(fallback_provider_name, self.settings, self.tool_runtime)
        fallback_output, _ = self._run_provider_attempt(
            provider=fallback_provider,
            provider_name=fallback_provider_name,
            brief=brief,
            brand_rules=brand_rules,
            channel_spec=channel_spec,
            policy_rules=policy_rules,
            strategy=strategy,
            recorder=recorder,
            stage_name="generate_variants_fallback",
            attempt="fallback",
        )
        fallback_decision = FallbackDecision(
            requested_provider=provider_name,
            fallback_provider=fallback_provider_name,
            final_provider=fallback_output.provider_name,
            reason=reason,
            latency_budget_ms=latency_budget_ms,
            observed_latency_ms=observed_latency_ms,
            predicted_latency_ms=predicted_latency_ms,
            error=error,
        )
        recorder.log_event("provider_fallback_completed", fallback_decision.model_dump(mode="json"))
        return fallback_output, fallback_decision

    def _run_provider_attempt(
        self,
        provider,
        provider_name: str,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
        recorder: TraceRecorder,
        stage_name: str,
        attempt: str,
    ) -> Tuple[ProviderOutput, float]:
        with recorder.stage(stage_name, {"strategy": strategy, "provider": provider_name, "attempt": attempt}):
            started_at = time.perf_counter()
            try:
                provider_output = provider.generate(
                    brief=brief,
                    brand_rules=brand_rules,
                    channel_spec=channel_spec,
                    policy_rules=policy_rules,
                    strategy=strategy,
                )
            except ValueError:
                raise
            except Exception as exc:
                latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                recorder.log_event(
                    "provider_failed",
                    {
                        "provider": provider_name,
                        "attempt": attempt,
                        "observed_latency_ms": latency_ms,
                        "error": str(exc),
                    },
                )
                raise ProviderAttemptError(provider_name=provider_name, latency_ms=latency_ms, original=exc) from exc

            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            recorder.log_event(
                "provider_completed",
                {
                    "provider": provider_output.provider_name,
                    "provider_key": provider_name,
                    "attempt": attempt,
                    "observed_latency_ms": latency_ms,
                    "prompt_version": provider_output.prompt_version,
                    "tool_events": provider_output.tool_events,
                },
            )
            return provider_output, latency_ms

    def _latency_budget_ms(self, provider_name: str) -> int:
        if provider_name == "ollama":
            return max(self.settings.ollama_latency_budget_ms, 0)
        return 0

    def _fallback_provider_name(self, provider_name: str) -> Optional[str]:
        if provider_name != "ollama":
            return None
        fallback_provider_name = self.settings.ollama_fallback_provider.strip()
        if not fallback_provider_name or fallback_provider_name == provider_name:
            return None
        return fallback_provider_name

    def _should_fallback_on_error(self, provider_name: str) -> bool:
        return provider_name == "ollama" and self.settings.ollama_fallback_on_error and bool(
            self._fallback_provider_name(provider_name)
        )

    def _predicted_latency_ms(self, provider_name: str, strategy: str) -> Optional[float]:
        if provider_name != "ollama":
            return None
        if not self.settings.ollama_preemptive_routing_enabled:
            return None
        latency_budget_ms = self._latency_budget_ms(provider_name)
        if latency_budget_ms <= 0:
            return None
        if not self._fallback_provider_name(provider_name):
            return None

        predicted_latency_ms = estimate_provider_latency_ms(
            settings=self.settings,
            provider_name=provider_name,
            strategy=strategy,
            percentile=self.settings.ollama_preemptive_percentile,
            min_samples=self.settings.ollama_preemptive_min_samples,
        )
        # This is intentionally conservative: if the predicted percentile still fits the budget,
        # we let Ollama run and rely on the regular fallback path if the call still misbehaves.
        if predicted_latency_ms is None or predicted_latency_ms <= latency_budget_ms:
            return None
        return predicted_latency_ms
