from pathlib import Path

from creative_ops_llm_review_agent.config import Settings
from creative_ops_llm_review_agent.knowledge import KnowledgeStore
from creative_ops_llm_review_agent.models import CreativeBrief
from creative_ops_llm_review_agent.providers.openai_responses import OpenAIResponsesCreativeProvider
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


class FakeUsage:
    input_tokens = 120
    output_tokens = 80


class FakeResponse:
    def __init__(self, response_id, output, output_text="", usage=None):
        self.id = response_id
        self.output = output
        self.output_text = output_text
        self.usage = usage or FakeUsage()


class FakeResponsesClient:
    def __init__(self):
        self.calls = []
        self._responses = [
            FakeResponse(
                "resp_1",
                [
                    {
                        "type": "function_call",
                        "name": "get_brand_rules",
                        "arguments": "{\"brand_key\": \"acme\"}",
                        "call_id": "call_1",
                    }
                ],
            ),
            FakeResponse(
                "resp_2",
                [],
                output_text=(
                    '{"variants":['
                    '{"headline":"Teams first reviews","description":"Review banner copy before launch. Terms apply.","cta":"Book a demo","rationale":"Uses brand phrase and clear CTA."},'
                    '{"headline":"Launch with fewer copy surprises","description":"Built for busy teams. Terms apply.","cta":"Book a demo","rationale":"Keeps the value practical."},'
                    '{"headline":"Creative review for busy teams","description":"Check banner copy against placement rules. Terms apply.","cta":"Book a demo","rationale":"Highlights constraint checks."}'
                    ']}'
                ),
            ),
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponsesClient()


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


def test_openai_provider_executes_local_tool_loop(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    runtime = ConstraintToolRuntime(KnowledgeStore(settings))
    provider = OpenAIResponsesCreativeProvider(settings=settings, tool_runtime=runtime)
    provider.client = FakeOpenAIClient()

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

    assert output.provider_name == "openai-responses"
    assert len(output.variants) == 3
    assert output.tools_consulted == ["get_brand_rules"]
    assert output.usage.prompt_tokens == 120
