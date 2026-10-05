from types import SimpleNamespace

import pytest

from mcp_rag_chat.llm import (
    GeminiLLMProvider,
    LLMProviderError,
    LLMRequest,
)


class FakeModels:
    def __init__(
        self,
        response_text: str | None = "Generated answer",
        error: Exception | None = None,
    ) -> None:
        self.response_text = response_text
        self.error = error

        self.model: str | None = None
        self.contents: str | None = None
        self.config = None

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config,
    ):
        if self.error is not None:
            raise self.error

        self.model = model
        self.contents = contents
        self.config = config

        return SimpleNamespace(
            text=self.response_text,
        )


class FakeGeminiClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


@pytest.mark.asyncio
async def test_generate_returns_llm_response() -> None:
    models = FakeModels(response_text="MCP is a protocol.")

    provider = GeminiLLMProvider(
        client=FakeGeminiClient(models),
        model="test-model",
    )

    response = await provider.generate(
        LLMRequest(
            system_prompt="Answer using supplied context.",
            user_prompt="What is MCP?",
        )
    )

    assert response.text == "MCP is a protocol."
    assert models.model == "test-model"
    assert models.contents == "What is MCP?"


@pytest.mark.asyncio
async def test_generate_maps_system_prompt() -> None:
    models = FakeModels()

    provider = GeminiLLMProvider(
        client=FakeGeminiClient(models),
        model="test-model",
    )

    await provider.generate(
        LLMRequest(
            system_prompt="Use only retrieved evidence.",
            user_prompt="Explain the architecture.",
        )
    )

    assert models.config.system_instruction == "Use only retrieved evidence."


@pytest.mark.asyncio
async def test_generate_translates_provider_error() -> None:
    models = FakeModels(error=RuntimeError("provider unavailable"))

    provider = GeminiLLMProvider(
        client=FakeGeminiClient(models),
        model="test-model",
    )

    with pytest.raises(
        LLMProviderError,
        match="Gemini failed to generate a response",
    ):
        await provider.generate(
            LLMRequest(
                system_prompt="System",
                user_prompt="Question",
            )
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_response() -> None:
    models = FakeModels(response_text="")

    provider = GeminiLLMProvider(
        client=FakeGeminiClient(models),
        model="test-model",
    )

    with pytest.raises(
        LLMProviderError,
        match="Gemini returned an empty response",
    ):
        await provider.generate(
            LLMRequest(
                system_prompt="System",
                user_prompt="Question",
            )
        )


def test_model_must_not_be_empty() -> None:
    models = FakeModels()

    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        GeminiLLMProvider(
            client=FakeGeminiClient(models),
            model=" ",
        )
