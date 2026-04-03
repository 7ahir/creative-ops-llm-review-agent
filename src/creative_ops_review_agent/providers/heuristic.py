from __future__ import annotations

from typing import List

from creative_ops_review_agent.models import (
    BrandRuleSet,
    ChannelSpec,
    CreativeBrief,
    GeneratedVariant,
    PolicyRuleSet,
    ProviderOutput,
    UsageMetrics,
)
from creative_ops_review_agent.providers.base import CreativeProvider
from creative_ops_review_agent.utils import slug, truncate


class HeuristicCreativeProvider(CreativeProvider):
    name = "heuristic-provider"

    def generate(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
    ) -> ProviderOutput:
        if strategy not in {"baseline", "tool_aware"}:
            raise ValueError("Unknown strategy: %s" % strategy)

        variants = (
            self._build_baseline_variants(brief)
            if strategy == "baseline"
            else self._build_tool_aware_variants(brief, brand_rules, channel_spec)
        )

        tools = [] if strategy == "baseline" else ["brand_rules", "channel_specs", "policy_rules"]
        prompt_version = "heuristic-baseline-v1" if strategy == "baseline" else "heuristic-tool-aware-v2"
        prompt_preview = self._build_prompt_preview(brief, brand_rules, channel_spec, policy_rules, strategy)
        usage = self._estimate_usage(prompt_preview, variants)

        return ProviderOutput(
            provider_name=self.name,
            prompt_version=prompt_version,
            strategy=strategy,
            variants=variants,
            tools_consulted=tools,
            usage=usage,
            prompt_preview=prompt_preview,
        )

    def _build_baseline_variants(self, brief: CreativeBrief) -> List[GeneratedVariant]:
        templates = [
            (
                "Guaranteed better campaign results today",
                "%s helps %s unlock instant performance wins with less effort." % (
                    brief.product_name,
                    brief.audience,
                ),
                "Get results now",
            ),
            (
                "%s for faster growth across every channel" % brief.product_name,
                "Upgrade your workflow with a powerful solution that gives teams everything they need to move faster.",
                brief.call_to_action,
            ),
            (
                "The #1 way to launch creatives without delays",
                "%s gives busy teams a smarter path from brief to banner with premium performance and easy setup." % brief.product_name,
                "Start today",
            ),
        ]
        return [
            GeneratedVariant(
                variant_id="variant-" + slug("|".join(item)),
                headline=item[0],
                description=item[1],
                cta=item[2],
                rationale="Baseline variant that ignores external constraints and brand rules.",
            )
            for item in templates
        ]

    def _build_tool_aware_variants(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
    ) -> List[GeneratedVariant]:
        benefit = brief.key_message.rstrip(".")
        offer = (brief.offer_details or "").rstrip(".")
        brand_phrase = brand_rules.required_phrases[0] if brand_rules.required_phrases else ""
        cta = truncate(brief.call_to_action, channel_spec.max_cta_chars)

        headline_templates = [
            "%s: %s" % (brief.product_name, benefit),
            "%s %s" % (brand_phrase, brief.product_name),
            "Move from brief to banner with %s" % brief.product_name,
        ]
        description_templates = [
            "%s for %s. %s" % (benefit, brief.audience, offer or "Built for busy teams"),
            "Built for %s. %s" % (brief.audience, offer or benefit),
            "%s. %s" % (benefit, "Clear review signals before launch"),
        ]

        variants: List[GeneratedVariant] = []
        for index in range(3):
            headline = truncate(headline_templates[index], channel_spec.max_headline_chars)
            description = truncate(description_templates[index], channel_spec.max_description_chars)
            variants.append(
                GeneratedVariant(
                    variant_id="variant-" + slug(headline + description + cta),
                    headline=headline,
                    description=description,
                    cta=cta,
                    rationale=(
                        "Constraint-aware variant using brand phrase, placement limits, and concise CTA."
                    ),
                )
            )
        return variants

    def _build_prompt_preview(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
    ) -> str:
        lines = [
            "Campaign: %s" % brief.campaign_name,
            "Audience: %s" % brief.audience,
            "Objective: %s" % brief.objective,
            "Key message: %s" % brief.key_message,
            "Strategy: %s" % strategy,
        ]
        if strategy == "tool_aware":
            lines.extend(
                [
                    "Brand notes: %s" % ", ".join(brand_rules.style_notes[:2]),
                    "Avoid terms: %s" % ", ".join(brand_rules.avoid_terms),
                    "Placement caps: headline=%s description=%s cta=%s"
                    % (
                        channel_spec.max_headline_chars,
                        channel_spec.max_description_chars,
                        channel_spec.max_cta_chars,
                    ),
                    "Policy disallowed claims: %s" % ", ".join(policy_rules.disallowed_claims[:3]),
                ]
            )
        return "\n".join(lines)

    def _estimate_usage(self, prompt_preview: str, variants: List[GeneratedVariant]) -> UsageMetrics:
        prompt_tokens = max(40, int(len(prompt_preview.split()) * 1.4))
        completion_text = " ".join(
            [variant.headline + " " + variant.description + " " + variant.cta for variant in variants]
        )
        completion_tokens = max(60, int(len(completion_text.split()) * 1.4))
        estimated_cost_usd = round((prompt_tokens + completion_tokens) / 1000 * 0.0035, 4)
        return UsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
