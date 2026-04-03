from __future__ import annotations

from typing import Dict, List

from creative_ops_review_agent.models import (
    BrandRuleSet,
    ChannelSpec,
    CreativeBrief,
    GeneratedVariant,
    PolicyRuleSet,
    ReviewIssue,
    VariantReview,
    VariantScore,
)


SEVERITY_PENALTIES: Dict[str, float] = {"low": 0.04, "medium": 0.1, "high": 0.22}


def review_variant(
    brief: CreativeBrief,
    variant: GeneratedVariant,
    brand_rules: BrandRuleSet,
    channel_spec: ChannelSpec,
    policy_rules: PolicyRuleSet,
) -> VariantReview:
    issues: List[ReviewIssue] = []

    if len(variant.headline) > channel_spec.max_headline_chars:
        issues.append(
            ReviewIssue(
                severity="high",
                category="headline_length",
                message="Headline exceeds %s characters for %s."
                % (channel_spec.max_headline_chars, channel_spec.placement),
            )
        )
    if len(variant.description) > channel_spec.max_description_chars:
        issues.append(
            ReviewIssue(
                severity="high",
                category="description_length",
                message="Description exceeds %s characters for %s."
                % (channel_spec.max_description_chars, channel_spec.placement),
            )
        )
    if len(variant.cta) > channel_spec.max_cta_chars:
        issues.append(
            ReviewIssue(
                severity="medium",
                category="cta_length",
                message="CTA exceeds %s characters." % channel_spec.max_cta_chars,
            )
        )

    combined_text = " ".join([variant.headline, variant.description, variant.cta]).lower()

    for claim in policy_rules.disallowed_claims:
        if claim.lower() in combined_text:
            issues.append(
                ReviewIssue(
                    severity="high",
                    category="policy_claim",
                    message="Contains disallowed claim: %s." % claim,
                )
            )

    for term in list(brand_rules.avoid_terms) + list(brief.forbidden_terms):
        if term.lower() in combined_text:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    category="brand_term",
                    message="Contains avoided term: %s." % term,
                )
            )

    required_terms = list(brief.required_terms)
    if brand_rules.required_phrases:
        required_terms.append(brand_rules.required_phrases[0])
    for term in required_terms:
        if term and term.lower() not in combined_text:
            issues.append(
                ReviewIssue(
                    severity="low",
                    category="required_term",
                    message="Missing preferred phrase or required term: %s." % term,
                )
            )

    if channel_spec.requires_cta and brief.call_to_action.lower() not in variant.cta.lower():
        issues.append(
            ReviewIssue(
                severity="medium",
                category="cta_alignment",
                message="CTA does not align closely with the requested call to action.",
            )
        )

    if brief.key_message.lower().split()[0] not in combined_text:
        issues.append(
            ReviewIssue(
                severity="low",
                category="message_clarity",
                message="Output does not clearly echo the primary campaign message.",
            )
        )

    length_score = 1.0 - _penalty_for_categories(issues, ["headline_length", "description_length", "cta_length"])
    policy_score = 1.0 - _penalty_for_categories(issues, ["policy_claim"])
    brand_score = 1.0 - _penalty_for_categories(issues, ["brand_term", "required_term", "cta_alignment"])
    message_score = 1.0 - _penalty_for_categories(issues, ["message_clarity"])
    overall = max(0.0, round((length_score + policy_score + brand_score + message_score) / 4, 3))

    score = VariantScore(
        overall=overall,
        length_compliance=round(length_score, 3),
        policy_compliance=round(policy_score, 3),
        brand_alignment=round(brand_score, 3),
        message_clarity=round(message_score, 3),
    )
    high_issue_count = len([issue for issue in issues if issue.severity == "high"])
    passed = high_issue_count == 0 and overall >= 0.72

    return VariantReview(variant_id=variant.variant_id, passed=passed, score=score, issues=issues)


def _penalty_for_categories(issues: List[ReviewIssue], categories: List[str]) -> float:
    total = 0.0
    for issue in issues:
        if issue.category in categories:
            total += SEVERITY_PENALTIES[issue.severity]
    return min(total, 0.95)

