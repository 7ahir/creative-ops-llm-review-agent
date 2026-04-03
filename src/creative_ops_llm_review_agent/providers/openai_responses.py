from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from creative_ops_llm_review_agent.config import Settings
from creative_ops_llm_review_agent.models import (
    BrandRuleSet,
    ChannelSpec,
    CreativeBrief,
    GeneratedVariant,
    PolicyRuleSet,
    ProviderOutput,
    UsageMetrics,
)
from creative_ops_llm_review_agent.providers.base import CreativeProvider
from creative_ops_llm_review_agent.tool_runtime import ConstraintToolRuntime
from creative_ops_llm_review_agent.utils import slug


class OpenAIResponsesCreativeProvider(CreativeProvider):
    name = "openai-responses"

    def __init__(
        self,
        settings: Settings,
        tool_runtime: ConstraintToolRuntime,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.settings = settings
        self.tool_runtime = tool_runtime
        self.client = client

    def generate(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
    ) -> ProviderOutput:
        messages = self._build_messages(brief=brief, strategy=strategy)
        tools = self.tool_runtime.openai_tools() if strategy == "tool_aware" else []
        prompt_preview = self._build_prompt_preview(brief, strategy)
        client = self._get_client()
        response = client.responses.create(
            model=self.settings.openai_model,
            input=messages,
            tools=tools,
        )

        tool_events: List[Dict[str, Any]] = []
        while True:
            # Keep resolving local tools until the model stops asking for external context and emits final JSON.
            function_calls = [item for item in self._output_items(response) if self._item_type(item) == "function_call"]
            if not function_calls:
                break

            follow_up_input = []
            for item in function_calls:
                tool_name = self._item_attr(item, "name")
                arguments_json = self._item_attr(item, "arguments", "{}")
                tool_output = self.tool_runtime.execute_json(tool_name, arguments_json)
                tool_events.append(
                    {
                        "name": tool_name,
                        "arguments": json.loads(arguments_json or "{}"),
                        "output": json.loads(tool_output),
                    }
                )
                follow_up_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": self._item_attr(item, "call_id"),
                        "output": tool_output,
                    }
                )

            response = client.responses.create(
                model=self.settings.openai_model,
                previous_response_id=self._item_attr(response, "id"),
                input=follow_up_input,
                tools=tools,
            )

        raw_output_text = getattr(response, "output_text", "") or ""
        parsed = self._parse_output(raw_output_text)
        variants = self._hydrate_variants(parsed)
        usage = self._extract_usage(response)

        return ProviderOutput(
            provider_name=self.name,
            prompt_version="openai-%s-%s" % (self.settings.openai_model, strategy),
            strategy=strategy,
            variants=variants,
            tools_consulted=[event["name"] for event in tool_events],
            usage=usage,
            prompt_preview=prompt_preview,
            raw_output_text=raw_output_text,
            tool_events=tool_events,
        )

    def _build_messages(self, brief: CreativeBrief, strategy: str) -> List[Dict[str, Any]]:
        developer_text = (
            "You generate banner copy variants for creative operations teams.\n"
            "Return valid JSON only with this shape: "
            '{"variants":[{"headline":"...","description":"...","cta":"...","rationale":"..."}]}.\n'
            "Never wrap the JSON in markdown.\n"
            "Generate exactly 3 variants.\n"
            "Headlines and descriptions must fit the placement constraints.\n"
            "If strategy is tool_aware, use the available tools to fetch brand, policy, and channel constraints before finalizing outputs.\n"
            "Avoid hype and unsupported claims. Keep rationale short and practical."
        )
        user_text = (
            "Strategy: %s\n"
            "Creative brief:\n%s" % (strategy, json.dumps(brief.model_dump(mode="json"), indent=2))
        )
        return [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": developer_text}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        ]

    def _build_prompt_preview(self, brief: CreativeBrief, strategy: str) -> str:
        return "Model=%s Strategy=%s Campaign=%s Placement=%s" % (
            self.settings.openai_model,
            strategy,
            brief.campaign_name,
            brief.placement,
        )

    def _parse_output(self, raw_output_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_output_text)
        except json.JSONDecodeError:
            start = raw_output_text.find("{")
            end = raw_output_text.rfind("}")
            if start == -1 or end == -1:
                raise RuntimeError("Model output was not valid JSON.")
            return json.loads(raw_output_text[start : end + 1])

    def _hydrate_variants(self, payload: Dict[str, Any]) -> List[GeneratedVariant]:
        variants = payload.get("variants", [])
        if not isinstance(variants, list) or len(variants) != 3:
            raise RuntimeError("Model did not return exactly 3 variants.")
        hydrated: List[GeneratedVariant] = []
        for item in variants:
            headline = item["headline"].strip()
            description = item["description"].strip()
            cta = item["cta"].strip()
            hydrated.append(
                GeneratedVariant(
                    variant_id="variant-" + slug(headline + description + cta),
                    headline=headline,
                    description=description,
                    cta=cta,
                    rationale=item.get("rationale", "").strip() or "Generated via OpenAI Responses API.",
                )
            )
        return hydrated

    def _extract_usage(self, response: Any) -> UsageMetrics:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        estimated_cost_usd = round(
            (prompt_tokens / 1000 * self.settings.openai_input_cost_per_1k)
            + (completion_tokens / 1000 * self.settings.openai_output_cost_per_1k),
            4,
        )
        return UsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

    def _output_items(self, response: Any) -> List[Any]:
        output = getattr(response, "output", None)
        if output is None:
            return []
        return list(output)

    def _get_client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI()
        return self.client

    def _item_type(self, item: Any) -> Optional[str]:
        return self._item_attr(item, "type")

    def _item_attr(self, item: Any, field: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(field, default)
        return getattr(item, field, default)
