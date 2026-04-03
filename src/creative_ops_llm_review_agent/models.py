from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CreativeBrief(BaseModel):
    brief_id: Optional[str] = None
    brand_key: str = "acme"
    campaign_name: str
    product_name: str
    audience: str
    objective: str
    key_message: str
    tone: Literal["playful", "confident", "premium", "direct", "helpful"] = "confident"
    offer_details: Optional[str] = None
    required_terms: List[str] = Field(default_factory=list)
    forbidden_terms: List[str] = Field(default_factory=list)
    call_to_action: str = "Shop now"
    channel: Literal["display", "social", "native"] = "display"
    placement: str = "display_300x250"
    landing_page: Optional[str] = None
    locale: str = "en-US"


class ChannelSpec(BaseModel):
    placement: str
    channel: str
    max_headline_chars: int
    max_description_chars: int
    max_cta_chars: int
    requires_cta: bool = True
    notes: List[str] = Field(default_factory=list)


class BrandRuleSet(BaseModel):
    brand_name: str
    approved_tone_descriptors: List[str] = Field(default_factory=list)
    avoid_terms: List[str] = Field(default_factory=list)
    required_phrases: List[str] = Field(default_factory=list)
    style_notes: List[str] = Field(default_factory=list)


class PolicyRuleSet(BaseModel):
    disallowed_claims: List[str] = Field(default_factory=list)
    required_disclaimer_terms: List[str] = Field(default_factory=list)
    sensitive_terms: List[str] = Field(default_factory=list)


class GeneratedVariant(BaseModel):
    variant_id: str
    headline: str
    description: str
    cta: str
    rationale: str


class UsageMetrics(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


class ProviderOutput(BaseModel):
    provider_name: str
    prompt_version: str
    strategy: str
    variants: List[GeneratedVariant]
    tools_consulted: List[str] = Field(default_factory=list)
    usage: UsageMetrics
    prompt_preview: str
    raw_output_text: Optional[str] = None
    tool_events: List[Dict[str, Any]] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    severity: Literal["low", "medium", "high"]
    category: str
    message: str


class VariantScore(BaseModel):
    overall: float
    length_compliance: float
    policy_compliance: float
    brand_alignment: float
    message_clarity: float


class VariantReview(BaseModel):
    variant_id: str
    passed: bool
    score: VariantScore
    issues: List[ReviewIssue] = Field(default_factory=list)


class StageRecord(BaseModel):
    name: str
    duration_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RequestMetrics(BaseModel):
    total_latency_ms: float
    average_variant_score: float
    passed_variant_count: int
    total_variant_count: int
    estimated_cost_usd: float


class FallbackDecision(BaseModel):
    requested_provider: str
    fallback_provider: str
    final_provider: str
    reason: Literal["latency_budget_exceeded", "provider_error", "predicted_latency_exceeded"]
    latency_budget_ms: Optional[float] = None
    observed_latency_ms: Optional[float] = None
    predicted_latency_ms: Optional[float] = None
    error: Optional[str] = None


class CreativeResponse(BaseModel):
    trace_id: str
    created_at: datetime
    provider: str
    strategy: str
    brief: CreativeBrief
    variants: List[GeneratedVariant]
    reviews: List[VariantReview]
    best_variant_id: str
    prompt_version: str
    metrics: RequestMetrics
    stages: List[StageRecord]
    tools_consulted: List[str] = Field(default_factory=list)
    fallback: Optional[FallbackDecision] = None
    trace_path: Optional[str] = None


class ReviewOnlyRequest(BaseModel):
    brief: CreativeBrief
    variants: List[GeneratedVariant]


class EvalCase(BaseModel):
    case_id: str
    brief: CreativeBrief
    notes: str = ""


class EvalStrategySummary(BaseModel):
    strategy: str
    average_score: float
    pass_rate: float
    average_latency_ms: float
    average_cost_usd: float
    issue_counts: Dict[str, int] = Field(default_factory=dict)


class EvalReport(BaseModel):
    generated_at: datetime
    dataset_path: str
    cases_run: int
    summaries: List[EvalStrategySummary]
    score_delta: float
    pass_rate_delta: float


class BenchmarkScenarioSummary(BaseModel):
    scenario: str
    provider: str
    strategy: str
    model: str
    average_score: float
    pass_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    average_cost_usd: float
    fallback_count: int
    fallback_rate: float
    provider_mix: Dict[str, int] = Field(default_factory=dict)
    issue_counts: Dict[str, int] = Field(default_factory=dict)
    trace_paths: List[str] = Field(default_factory=list)


class BenchmarkMatrixReport(BaseModel):
    generated_at: datetime
    dataset_path: str
    cases_run: int
    benchmark_root: str
    fallback_budget_ms: int
    scenarios: List[BenchmarkScenarioSummary]
