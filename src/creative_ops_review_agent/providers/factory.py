from __future__ import annotations

from .base import CreativeProvider
from .heuristic import HeuristicCreativeProvider
from .ollama_chat import OllamaChatCreativeProvider
from .openai_responses import OpenAIResponsesCreativeProvider
from ..config import Settings
from ..tool_runtime import ConstraintToolRuntime


def get_provider(name: str, settings: Settings, tool_runtime: ConstraintToolRuntime) -> CreativeProvider:
    if name == "heuristic":
        return HeuristicCreativeProvider()
    if name == "openai":
        return OpenAIResponsesCreativeProvider(settings=settings, tool_runtime=tool_runtime)
    if name == "ollama":
        return OllamaChatCreativeProvider(settings=settings, tool_runtime=tool_runtime)
    raise ValueError("Unknown provider: %s" % name)
