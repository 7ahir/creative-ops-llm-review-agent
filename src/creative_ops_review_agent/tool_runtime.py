from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .knowledge import KnowledgeStore


class ConstraintToolRuntime:
    def __init__(self, knowledge: KnowledgeStore) -> None:
        self.knowledge = knowledge
        # Keep one dispatch table so OpenAI tools, MCP tools, and direct execution all share the same truth.
        self._dispatch: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_brand_rules": self.get_brand_rules,
            "get_channel_spec": self.get_channel_spec,
            "get_policy_rules": self.get_policy_rules,
        }

    def get_brand_rules(self, brand_key: str) -> Dict[str, Any]:
        return self.knowledge.brand_rules(brand_key).model_dump(mode="json")

    def get_channel_spec(self, placement: str) -> Dict[str, Any]:
        return self.knowledge.channel_spec(placement).model_dump(mode="json")

    def get_policy_rules(self) -> Dict[str, Any]:
        return self.knowledge.policy_rules().model_dump(mode="json")

    def openai_tools(self) -> List[Dict[str, Any]]:
        # These schemas are strict on purpose so the provider loop fails loudly when a tool contract drifts.
        return [
            {
                "type": "function",
                "name": "get_brand_rules",
                "description": "Return the current brand constraints and required phrases for a given brand key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brand_key": {
                            "type": "string",
                            "description": "The brand identifier from the brief, such as 'acme'."
                        }
                    },
                    "required": ["brand_key"],
                    "additionalProperties": False
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_channel_spec",
                "description": "Return placement-specific character limits and CTA requirements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "placement": {
                            "type": "string",
                            "description": "The placement identifier from the brief, such as 'display_300x250'."
                        }
                    },
                    "required": ["placement"],
                    "additionalProperties": False
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_policy_rules",
                "description": "Return global policy constraints, disallowed claims, and disclaimer requirements.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },
                "strict": True,
            },
        ]

    def chat_completions_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_brand_rules",
                    "description": "Return the current brand constraints and required phrases for a given brand key.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "brand_key": {"type": "string"}
                        },
                        "required": ["brand_key"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_channel_spec",
                    "description": "Return placement-specific character limits and CTA requirements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "placement": {"type": "string"}
                        },
                        "required": ["placement"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_policy_rules",
                    "description": "Return global policy constraints, disallowed claims, and disclaimer requirements.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def mcp_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_brand_rules",
                "description": "Look up brand rules and required phrases by brand key.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "brand_key": {"type": "string"}
                    },
                    "required": ["brand_key"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_channel_spec",
                "description": "Look up placement-specific character and CTA constraints.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "placement": {"type": "string"}
                    },
                    "required": ["placement"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_policy_rules",
                "description": "Return global policy rules used during copy review.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if name not in self._dispatch:
            raise KeyError("Unknown tool: %s" % name)
        args = arguments or {}
        return self._dispatch[name](**args)

    def execute_json(self, name: str, arguments_json: Optional[str] = None) -> str:
        # The provider integrations already speak JSON, so keep the runtime boundary JSON-native too.
        arguments = json.loads(arguments_json) if arguments_json else {}
        return json.dumps(self.execute(name, arguments))
