from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class LLMResponse:
    text: str


class LLMProviderError(Exception):
    """Raised when an LLM provider cannot generate a response."""


class LLMProvider(Protocol):
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse: ...
