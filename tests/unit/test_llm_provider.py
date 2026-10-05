import pytest

from mcp_rag_chat.llm import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)


class FakeLLMProvider:
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        return LLMResponse(text=f"Generated: {request.user_prompt}")


@pytest.mark.asyncio
async def test_llm_provider_contract() -> None:
    provider: LLMProvider = FakeLLMProvider()

    request = LLMRequest(
        system_prompt="Answer using the provided context.",
        user_prompt="What is MCP?",
    )

    response = await provider.generate(request)

    assert response.text == "Generated: What is MCP?"
