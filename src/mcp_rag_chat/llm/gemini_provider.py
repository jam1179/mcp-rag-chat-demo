import asyncio

from google import genai
from google.genai import types

from mcp_rag_chat.llm.provider import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
)


class GeminiLLMProvider:
    def __init__(
        self,
        client: genai.Client,
        model: str,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("model must not be empty")

        self._client = client
        self._model = model

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=request.user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=request.system_prompt,
                ),
            )
        except Exception as exc:
            raise LLMProviderError("Gemini failed to generate a response") from exc

        text = response.text

        if not text or not text.strip():
            raise LLMProviderError("Gemini returned an empty response")

        return LLMResponse(
            text=text.strip(),
        )
