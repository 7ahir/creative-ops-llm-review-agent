from pathlib import Path

from creative_ops_review_agent.config import Settings
from creative_ops_review_agent.knowledge import KnowledgeStore
from creative_ops_review_agent.models import CreativeBrief, GeneratedVariant
from creative_ops_review_agent.repair import repair_variant_set
from creative_ops_review_agent.scoring import review_variant


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
        ollama_model="qwen3:4b",
        ollama_api_key="ollama",
        ollama_think=False,
        mcp_host="127.0.0.1",
        mcp_port=8002,
        mcp_path="/mcp",
    )


def sample_brief() -> CreativeBrief:
    return CreativeBrief.model_validate(
        {
            "brand_key": "acme",
            "campaign_name": "Spring Launch Push",
            "product_name": "Acme Creative Studio",
            "audience": "performance marketers at retail brands",
            "objective": "Generate compliant banner copy faster",
            "key_message": "Review banner copy against brand and placement constraints before launch.",
            "tone": "confident",
            "offer_details": "Built for busy teams. Terms apply.",
            "required_terms": ["Terms apply"],
            "forbidden_terms": ["guaranteed", "instant"],
            "call_to_action": "Book a demo",
            "channel": "display",
            "placement": "display_300x250",
        }
    )


def test_repair_loop_fixes_common_constraint_failures(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    knowledge = KnowledgeStore(settings)
    brief = sample_brief()
    brand_rules = knowledge.brand_rules(brief.brand_key)
    channel_spec = knowledge.channel_spec(brief.placement)
    policy_rules = knowledge.policy_rules()

    variants = [
        GeneratedVariant(
            variant_id="variant-1",
            headline="Review your banner copy against brand and placement constraints before launch",
            description="Built for busy teams. Terms apply.",
            cta="Book a demo",
            rationale="Raw local model output.",
        ),
        GeneratedVariant(
            variant_id="variant-2",
            headline="Ensure brand alignment before banner launch",
            description="Built for busy teams. Terms apply.",
            cta="Book a:demo",
            rationale="Raw local model output.",
        ),
        GeneratedVariant(
            variant_id="variant-3",
            headline="Check brand constraints before launch",
            description="Built for busy teams. Terms apply.",
            cta="Book a demo",
            rationale="Raw local model output.",
        ),
    ]
    reviews = [
        review_variant(brief, variant, brand_rules, channel_spec, policy_rules)
        for variant in variants
    ]

    repaired_variants, repaired_reviews, repaired_count = repair_variant_set(
        brief=brief,
        variants=variants,
        reviews=reviews,
        brand_rules=brand_rules,
        channel_spec=channel_spec,
        policy_rules=policy_rules,
    )

    assert repaired_count == 3
    assert all(review.passed for review in repaired_reviews)
    assert all("Teams first" in variant.headline or "Teams first" in variant.description for variant in repaired_variants)
    assert all(variant.cta == "Book a demo" for variant in repaired_variants)
