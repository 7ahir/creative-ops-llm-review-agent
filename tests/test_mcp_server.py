from pathlib import Path

from fastapi.testclient import TestClient

from creative_ops_llm_review_agent.config import Settings
from creative_ops_llm_review_agent.mcp_server import create_mcp_app


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


def test_mcp_tools_list_and_call(tmp_path: Path) -> None:
    app = create_mcp_app(build_settings(tmp_path))
    client = TestClient(app)

    list_response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    tools = list_response.json()["result"]["tools"]
    assert any(tool["name"] == "get_brand_rules" for tool in tools)

    call_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_channel_spec", "arguments": {"placement": "display_300x250"}},
        },
    )
    payload = call_response.json()["result"]
    assert payload["isError"] is False
    assert payload["structuredContent"]["placement"] == "display_300x250"
