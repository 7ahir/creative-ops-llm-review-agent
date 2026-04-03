from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .models import (
    BrandRuleSet,
    ChannelSpec,
    CreativeBrief,
    GeneratedVariant,
    PolicyRuleSet,
    VariantReview,
)
from .scoring import review_variant
from .utils import truncate


def repair_variant_set(
    brief: CreativeBrief,
    variants: List[GeneratedVariant],
    reviews: List[VariantReview],
    brand_rules: BrandRuleSet,
    channel_spec: ChannelSpec,
    policy_rules: PolicyRuleSet,
) -> Tuple[List[GeneratedVariant], List[VariantReview], int]:
    repaired_variants = list(variants)
    repaired_reviews = list(reviews)
    repaired_count = 0

    for index, (variant, review) in enumerate(zip(variants, reviews)):
        if review.passed:
            continue
        candidate = repair_variant(
            brief=brief,
            variant=variant,
            review=review,
            brand_rules=brand_rules,
            channel_spec=channel_spec,
            policy_rules=policy_rules,
            index=index,
        )
        candidate_review = review_variant(
            brief=brief,
            variant=candidate,
            brand_rules=brand_rules,
            channel_spec=channel_spec,
            policy_rules=policy_rules,
        )
        if _is_better_review(review, candidate_review):
            repaired_variants[index] = candidate
            repaired_reviews[index] = candidate_review
            repaired_count += 1

    return repaired_variants, repaired_reviews, repaired_count


def repair_variant(
    brief: CreativeBrief,
    variant: GeneratedVariant,
    review: VariantReview,
    brand_rules: BrandRuleSet,
    channel_spec: ChannelSpec,
    policy_rules: PolicyRuleSet,
    index: int,
) -> GeneratedVariant:
    # Keep the repair step deterministic so I can explain exactly why a failing variant changed.
    disallowed_terms = _unique_terms(
        list(policy_rules.disallowed_claims)
        + list(policy_rules.sensitive_terms)
        + list(brand_rules.avoid_terms)
        + list(brief.forbidden_terms)
    )
    required_terms = _unique_terms(list(brief.required_terms) + list(brand_rules.required_phrases[:1]))
    primary_word = _primary_message_word(brief.key_message)
    brand_phrase = next((term for term in required_terms if "terms" not in term.lower()), "")

    headline = _repair_headline(
        index=index,
        current=variant.headline,
        primary_word=primary_word,
        brand_phrase=brand_phrase,
        max_chars=channel_spec.max_headline_chars,
        disallowed_terms=disallowed_terms,
    )
    description = _repair_description(
        index=index,
        current=variant.description,
        brief=brief,
        primary_word=primary_word,
        required_terms=required_terms,
        max_chars=channel_spec.max_description_chars,
        disallowed_terms=disallowed_terms,
    )
    cta = _repair_cta(brief.call_to_action, channel_spec.max_cta_chars, disallowed_terms)

    return GeneratedVariant(
        variant_id=variant.variant_id,
        headline=headline,
        description=description,
        cta=cta,
        rationale="Auto-repaired for policy, brand, and placement compliance.",
    )


def _repair_headline(
    index: int,
    current: str,
    primary_word: str,
    brand_phrase: str,
    max_chars: int,
    disallowed_terms: List[str],
) -> str:
    candidates = []
    if brand_phrase:
        candidates.extend(
            [
                "%s %s before launch" % (brand_phrase, primary_word.lower()),
                "%s banner review" % brand_phrase,
                "%s copy, %s" % (primary_word, brand_phrase),
                "%s copy check" % brand_phrase,
            ]
        )
    candidates.extend(
        [
            "%s copy before launch" % primary_word,
            "Review banner copy fast",
            "Check banner copy before launch",
        ]
    )
    base = _pick_candidate(candidates, index, max_chars)
    cleaned = _sanitize_text(base or current, disallowed_terms)
    return truncate(cleaned, max_chars)


def _repair_description(
    index: int,
    current: str,
    brief: CreativeBrief,
    primary_word: str,
    required_terms: List[str],
    max_chars: int,
    disallowed_terms: List[str],
) -> str:
    legal_terms = [term for term in required_terms if "terms" in term.lower()]
    required_suffix = " ".join(_ensure_period(term) for term in legal_terms)
    base_candidates = [
        brief.offer_details or "",
        "%s for busy teams." % brief.product_name,
        "%s copy with clear checks." % primary_word,
        "Built for busy teams.",
    ]
    rotated = _rotate(base_candidates, index)
    for candidate in rotated:
        cleaned = _sanitize_text(candidate, disallowed_terms)
        # Reattach required legal copy after cleaning so the sanitizer does not accidentally strip it away.
        if required_suffix and required_suffix.lower() not in cleaned.lower():
            cleaned = ("%s %s" % (_ensure_period(cleaned), required_suffix)).strip()
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) <= max_chars:
            return cleaned
    fallback = _sanitize_text(current, disallowed_terms)
    if required_suffix and required_suffix.lower() not in fallback.lower():
        fallback = ("%s %s" % (_ensure_period(fallback), required_suffix)).strip()
    return truncate(fallback or required_suffix or "Built for busy teams.", max_chars)


def _repair_cta(call_to_action: str, max_chars: int, disallowed_terms: List[str]) -> str:
    cleaned = _sanitize_text(call_to_action, disallowed_terms)
    return truncate(cleaned or "Learn more", max_chars)


def _is_better_review(original: VariantReview, candidate: VariantReview) -> bool:
    if candidate.passed and not original.passed:
        return True
    if candidate.score.overall > original.score.overall:
        return True
    return candidate.score.overall == original.score.overall and len(candidate.issues) < len(original.issues)


def _primary_message_word(key_message: str) -> str:
    for token in key_message.split():
        token = re.sub(r"[^A-Za-z0-9]+", "", token)
        if token:
            return token.capitalize()
    return "Review"


def _pick_candidate(candidates: List[str], index: int, max_chars: int) -> str:
    rotated = _rotate(candidates, index)
    for candidate in rotated:
        if len(candidate) <= max_chars:
            return candidate
    return truncate(rotated[0], max_chars) if rotated else ""


def _rotate(items: List[str], index: int) -> List[str]:
    if not items:
        return []
    start = index % len(items)
    return items[start:] + items[:start]


def _sanitize_text(text: str, disallowed_terms: List[str]) -> str:
    cleaned = text.strip()
    for term in disallowed_terms:
        if not term:
            continue
        cleaned = re.sub(re.escape(term), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.:;!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,:;])(?=[^\s])", r"\1 ", cleaned)
    cleaned = cleaned.replace("..", ".")
    return cleaned.strip(" ,;:-")


def _ensure_period(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.endswith((".", "!", "?")):
        return stripped
    return stripped + "."


def _unique_terms(terms: List[str]) -> List[str]:
    seen: Dict[str, str] = {}
    for term in terms:
        normalized = term.strip().lower()
        if normalized and normalized not in seen:
            seen[normalized] = term.strip()
    return list(seen.values())
