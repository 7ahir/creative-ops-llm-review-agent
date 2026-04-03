from __future__ import annotations

import json
from json import JSONDecodeError, JSONDecoder
from typing import Any, Dict, List, Optional, Tuple

import httpx

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


class OllamaChatCreativeProvider(CreativeProvider):
    name = "ollama-chat"

    def __init__(
        self,
        settings: Settings,
        tool_runtime: ConstraintToolRuntime,
        client: Optional[httpx.Client] = None,
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
        # I resolve deterministic constraints server-side so the local model spends tokens on copy, not tool loops.
        tool_events = self._build_server_context_events(
            brief=brief,
            brand_rules=brand_rules,
            channel_spec=channel_spec,
            policy_rules=policy_rules,
            strategy=strategy,
        )
        messages = self._build_messages(
            brief=brief,
            brand_rules=brand_rules,
            channel_spec=channel_spec,
            policy_rules=policy_rules,
            strategy=strategy,
        )
        payload, raw_output_text, usage = self._run_native_chat(messages)
        variants = self._hydrate_variants(payload)

        return ProviderOutput(
            provider_name=self.name,
            prompt_version="ollama-native-%s-%s-v2" % (self.settings.ollama_model, strategy),
            strategy=strategy,
            variants=variants,
            tools_consulted=[event["name"] for event in tool_events],
            usage=usage,
            prompt_preview=self._build_prompt_preview(brief, strategy),
            raw_output_text=raw_output_text,
            tool_events=tool_events,
        )

    def _build_messages(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
    ) -> List[Dict[str, Any]]:
        system_text = (
            "You generate banner copy variants for creative operations teams.\n"
            "Return a single JSON object that matches the provided schema exactly.\n"
            "Generate exactly 3 variants and do not include extra keys.\n"
            "Do not echo the brief or the constraint objects.\n"
            "Each variant must contain original banner copy only.\n"
            "Keep rationale under 12 words and keep copy concise."
        )
        user_sections = [
            "Strategy: %s" % strategy,
            "Creative brief:\n%s" % self._brief_context(brief),
        ]
        if strategy == "tool_aware":
            user_sections.append(
                "Resolved constraint context:\n%s"
                % self._constraint_context(brand_rules, channel_spec, policy_rules)
            )
            user_sections.append(
                "Use the resolved constraint context strictly. Headlines, descriptions, and CTAs must fit the limits."
            )
        else:
            user_sections.append("Use only the creative brief. Do not assume extra brand or policy context.")
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ]

    def _build_prompt_preview(self, brief: CreativeBrief, strategy: str) -> str:
        return "Model=%s Strategy=%s Context=%s Campaign=%s Placement=%s" % (
            self.settings.ollama_model,
            strategy,
            "server_injected" if strategy == "tool_aware" else "brief_only",
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

    def _parse_complete_output(self, raw_output_text: str) -> Optional[Dict[str, Any]]:
        text = raw_output_text.lstrip()
        if not text:
            return None
        try:
            # Streaming chunks can arrive mid-object, so only accept output once one full JSON object is closed.
            payload, end = JSONDecoder().raw_decode(text)
        except JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            raise RuntimeError("Model output was not a JSON object.")
        if text[end:].strip():
            return None
        return payload

    def _hydrate_variants(self, payload: Dict[str, Any]) -> List[GeneratedVariant]:
        variants = payload.get("variants", [])
        if not isinstance(variants, list) or len(variants) != 3:
            raise RuntimeError("Model did not return exactly 3 variants.")
        return [
            GeneratedVariant(
                variant_id="variant-" + slug(item["headline"].strip() + item["description"].strip() + item["cta"].strip()),
                headline=item["headline"].strip(),
                description=item["description"].strip(),
                cta=item["cta"].strip(),
                rationale=item.get("rationale", "").strip() or "Generated via native Ollama chat.",
            )
            for item in variants
        ]

    def _extract_usage(self, final_chunk: Optional[Dict[str, Any]]) -> UsageMetrics:
        prompt_tokens = int(final_chunk.get("prompt_eval_count", 0)) if final_chunk else 0
        completion_tokens = int(final_chunk.get("eval_count", 0)) if final_chunk else 0
        return UsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=0.0,
        )

    def _build_server_context_events(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
    ) -> List[Dict[str, Any]]:
        if strategy != "tool_aware":
            return []
        return [
            {
                "name": "get_brand_rules",
                "source": "server_context",
                "arguments": {"brand_key": brief.brand_key},
                "output": brand_rules.model_dump(mode="json"),
            },
            {
                "name": "get_channel_spec",
                "source": "server_context",
                "arguments": {"placement": brief.placement},
                "output": channel_spec.model_dump(mode="json"),
            },
            {
                "name": "get_policy_rules",
                "source": "server_context",
                "arguments": {},
                "output": policy_rules.model_dump(mode="json"),
            },
        ]

    def _brief_context(self, brief: CreativeBrief) -> str:
        payload = {
            "campaign_name": brief.campaign_name,
            "product_name": brief.product_name,
            "audience": brief.audience,
            "objective": brief.objective,
            "key_message": brief.key_message,
            "tone": brief.tone,
            "offer_details": brief.offer_details,
            "required_terms": brief.required_terms,
            "forbidden_terms": brief.forbidden_terms,
            "call_to_action": brief.call_to_action,
            "channel": brief.channel,
            "placement": brief.placement,
            "locale": brief.locale,
        }
        return json.dumps(payload, separators=(",", ":"))

    def _constraint_context(
        self,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
    ) -> str:
        lines = [
            "brand_tone=%s" % ", ".join(brand_rules.approved_tone_descriptors),
            "brand_required_phrase=%s" % ", ".join(brand_rules.required_phrases),
            "brand_avoid=%s" % ", ".join(brand_rules.avoid_terms),
            "brand_style=%s" % " | ".join(brand_rules.style_notes),
            "headline_max_chars=%s" % channel_spec.max_headline_chars,
            "description_max_chars=%s" % channel_spec.max_description_chars,
            "cta_max_chars=%s" % channel_spec.max_cta_chars,
            "cta_required=%s" % ("yes" if channel_spec.requires_cta else "no"),
            "placement_notes=%s" % " | ".join(channel_spec.notes),
            "policy_disallowed=%s" % ", ".join(policy_rules.disallowed_claims),
            "policy_required_disclaimer=%s" % ", ".join(policy_rules.required_disclaimer_terms),
            "policy_sensitive=%s" % ", ".join(policy_rules.sensitive_terms),
        ]
        return "\n".join(lines)

    def _run_native_chat(self, messages: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str, UsageMetrics]:
        request_payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "think": self.settings.ollama_think,
            "format": self._response_schema(),
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 220,
            },
        }
        raw_output_parts: List[str] = []
        parsed_payload: Optional[Dict[str, Any]] = None
        final_chunk: Optional[Dict[str, Any]] = None

        try:
            with self._get_client().stream(
                "POST",
                self._chat_url(),
                json=request_payload,
                timeout=httpx.Timeout(120.0, connect=5.0, read=120.0, write=10.0),
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    chunk = json.loads(raw_line)
                    if "error" in chunk:
                        raise RuntimeError("Ollama returned an error: %s" % chunk["error"])
                    final_chunk = chunk
                    content = (chunk.get("message") or {}).get("content", "")
                    if content:
                        raw_output_parts.append(content)
                        parsed_payload = self._parse_complete_output("".join(raw_output_parts))
                        if parsed_payload is not None:
                            # Stop once the first valid object is complete instead of waiting for extra tokens.
                            break
                    if chunk.get("done", False):
                        break
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ollama request failed: %s" % exc) from exc

        raw_output_text = "".join(raw_output_parts)
        if parsed_payload is None:
            parsed_payload = self._parse_output(raw_output_text)
        usage = self._extract_usage(final_chunk if final_chunk and final_chunk.get("done") else None)
        return parsed_payload, raw_output_text, usage

    def _response_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "variants": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string"},
                            "description": {"type": "string"},
                            "cta": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["headline", "description", "cta", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["variants"],
            "additionalProperties": False,
        }

    def _chat_url(self) -> str:
        base_url = self.settings.ollama_base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url + "/api/chat"

    def _get_client(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client()
        return self.client
