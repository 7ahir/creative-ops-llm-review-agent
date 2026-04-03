from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from .config import Settings
from .models import BrandRuleSet, ChannelSpec, PolicyRuleSet


class KnowledgeStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @lru_cache(maxsize=1)
    def _load_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text())

    def brand_rules(self, brand_key: str) -> BrandRuleSet:
        payload = self._load_json(self.settings.data_dir / "brand_rules.json")
        if brand_key not in payload:
            raise KeyError("Unknown brand_key: %s" % brand_key)
        return BrandRuleSet.model_validate(payload[brand_key])

    def channel_spec(self, placement: str) -> ChannelSpec:
        payload = self._load_json(self.settings.data_dir / "channel_specs.json")
        if placement not in payload:
            raise KeyError("Unknown placement: %s" % placement)
        return ChannelSpec.model_validate(payload[placement])

    def policy_rules(self) -> PolicyRuleSet:
        payload = self._load_json(self.settings.data_dir / "policy_rules.json")
        return PolicyRuleSet.model_validate(payload["global"])

