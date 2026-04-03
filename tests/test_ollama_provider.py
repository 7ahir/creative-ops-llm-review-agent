import json
from pathlib import Path

from creative_ops_llm_review_agent.config import Settings
from creative_ops_llm_review_agent.knowledge import KnowledgeStore
from creative_ops_llm_review_agent.models import CreativeBrief
from creative_ops_llm_review_agent.providers.ollama_chat import OllamaChatCreativeProvider
from creative_ops_llm_review_agent.tool_runtime import ConstraintToolRuntime


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


class FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self) -> "FakeStreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        for line in self.lines:
            yield line


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls = []
        self.lines = [
            json.dumps(
                {
                    "message": {
                        "content": (
                            '{"variants":['
                            '{"headline":"Teams first banner review","description":"Review banner copy before launch. Terms apply.","cta":"Book a demo","rationale":"Uses the placement constraints."},'
                            '{"headline":"Review copy with fewer surprises","description":"Built for busy teams. Terms apply.","cta":"Book a demo","rationale":"Keeps the value direct."},'
                            '{"headline":"Creative review for launch teams","description":"Check banner copy against placement rules. Terms apply.","cta":"Book a demo","rationale":"Uses the requested CTA."}'
                            ']}'
                        )
                    },
                    "done": True,
                    "prompt_eval_count": 90,
                    "eval_count": 70,
                }
            )
        ]

    def stream(self, method: str, url: str, json=None, timeout=None) -> FakeStreamResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeStreamResponse(self.lines)


def sample_brief() -> CreativeBrief:
    return CreativeBrief.model_validate(
        {
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
    )


def test_ollama_provider_uses_native_chat_with_server_context(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    runtime = ConstraintToolRuntime(KnowledgeStore(settings))
    fake_client = FakeHttpClient()
    provider = OllamaChatCreativeProvider(settings=settings, tool_runtime=runtime, client=fake_client)

    brief = sample_brief()
    brand_rules = runtime.knowledge.brand_rules(brief.brand_key)
    channel_spec = runtime.knowledge.channel_spec(brief.placement)
    policy_rules = runtime.knowledge.policy_rules()

    output = provider.generate(
        brief=brief,
        brand_rules=brand_rules,
        channel_spec=channel_spec,
        policy_rules=policy_rules,
        strategy="tool_aware",
    )

    assert output.provider_name == "ollama-chat"
    assert len(output.variants) == 3
    assert output.tools_consulted == ["get_brand_rules", "get_channel_spec", "get_policy_rules"]
    assert output.usage.prompt_tokens == 90
    assert fake_client.calls[0]["method"] == "POST"
    assert fake_client.calls[0]["url"] == "http://localhost:11434/api/chat"
    assert fake_client.calls[0]["json"]["think"] is False
    assert fake_client.calls[0]["json"]["format"]["properties"]["variants"]["maxItems"] == 3
    assert "Resolved constraint context" in fake_client.calls[0]["json"]["messages"][1]["content"]
