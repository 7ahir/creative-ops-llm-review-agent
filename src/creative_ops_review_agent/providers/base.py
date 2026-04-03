from __future__ import annotations

from abc import ABC, abstractmethod

from creative_ops_review_agent.models import (
    BrandRuleSet,
    ChannelSpec,
    CreativeBrief,
    PolicyRuleSet,
    ProviderOutput,
)


class CreativeProvider(ABC):
    name = "abstract"

    @abstractmethod
    def generate(
        self,
        brief: CreativeBrief,
        brand_rules: BrandRuleSet,
        channel_spec: ChannelSpec,
        policy_rules: PolicyRuleSet,
        strategy: str,
    ) -> ProviderOutput:
        raise NotImplementedError

