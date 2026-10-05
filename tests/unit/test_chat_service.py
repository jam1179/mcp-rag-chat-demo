import pytest

from mcp_rag_chat.chat import (
    ChatService,
    ChatServiceError,
)
from mcp_rag_chat.client import RetrievedContext
from mcp_rag_chat.llm import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
)


class FakeRAGChatClient:
    def __init__(
        self,
        contexts: list[RetrievedContext] | None = None,
        prompt: str = "Use only retrieved evidence.",
        search_error: Exception | None = None,
        prompt_error: Exception | None = None,
    ) -> None:
        self.contexts = contexts or []
        self.prompt = prompt
        self.search_error = search_error
        self.prompt_error = prompt_error

        self.search_query: str | None = None
        self.search_top_k: int | None = None
        self.prompt_question: str | None = None

    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        if self.search_error is not None:
            raise self.search_error

        self.search_query = query
        self.search_top_k = top_k

        return self.contexts

    async def get_grounded_qa_prompt(
        self,
        question: str,
    ) -> str:
        if self.prompt_error is not None:
            raise self.prompt_error

        self.prompt_question = question

        return self.prompt


class FakeLLMProvider:
    def __init__(
        self,
        response_text: str = "Generated answer",
        error: Exception | None = None,
    ) -> None:
        self.response_text = response_text
        self.error = error
        self.request: LLMRequest | None = None

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        if self.error is not None:
            raise self.error

        self.request = request

        return LLMResponse(
            text=self.response_text,
        )


def create_context(
    *,
    document_id: str = "doc-1",
    document_name: str = "architecture.pdf",
    content: str = "The platform uses MCP.",
    page_start: int | None = 4,
    page_end: int | None = 5,
    relevance_score: float = 0.91,
) -> RetrievedContext:
    return RetrievedContext(
        document_id=document_id,
        document_name=document_name,
        content=content,
        page_start=page_start,
        page_end=page_end,
        relevance_score=relevance_score,
    )


@pytest.mark.asyncio
async def test_ask_returns_grounded_answer_with_citations() -> None:
    contexts = [
        create_context(
            document_id="doc-1",
            document_name="architecture.pdf",
            content="The platform uses MCP.",
            page_start=4,
            page_end=5,
            relevance_score=0.91,
        ),
        create_context(
            document_id="doc-2",
            document_name="security.pdf",
            content="Authentication uses OAuth.",
            page_start=8,
            page_end=8,
            relevance_score=0.87,
        ),
    ]

    rag_client = FakeRAGChatClient(
        contexts=contexts,
    )

    llm_provider = FakeLLMProvider(
        response_text=("The platform uses MCP [S1] and OAuth authentication [S2].")
    )

    service = ChatService(
        rag_client=rag_client,
        llm_provider=llm_provider,
        top_k=3,
    )

    result = await service.ask("Describe the architecture.")

    assert result.answer == (
        "The platform uses MCP [S1] and OAuth authentication [S2]."
    )

    # Verify retrieval invocation.
    assert rag_client.search_query == "Describe the architecture."
    assert rag_client.search_top_k == 3

    # Verify grounding prompt invocation.
    assert rag_client.prompt_question == "Describe the architecture."

    # Verify authoritative citation mapping.
    assert len(result.sources) == 2

    first_source = result.sources[0]

    assert first_source.citation_id == "S1"
    assert first_source.document_id == "doc-1"
    assert first_source.document_name == "architecture.pdf"
    assert first_source.page_start == 4
    assert first_source.page_end == 5
    assert first_source.relevance_score == 0.91

    second_source = result.sources[1]

    assert second_source.citation_id == "S2"
    assert second_source.document_id == "doc-2"
    assert second_source.document_name == "security.pdf"
    assert second_source.page_start == 8
    assert second_source.page_end == 8
    assert second_source.relevance_score == 0.87

    # Verify LLM request.
    assert llm_provider.request is not None

    assert llm_provider.request.system_prompt == "Use only retrieved evidence."

    user_prompt = llm_provider.request.user_prompt

    # Citation identifiers supplied to the LLM must match the
    # identifiers returned to the presentation layer.
    assert "[S1]" in user_prompt
    assert "[S2]" in user_prompt

    assert "architecture.pdf" in user_prompt
    assert "security.pdf" in user_prompt

    assert "Pages: 4-5" in user_prompt
    assert "Pages: 8" in user_prompt

    assert "The platform uses MCP." in user_prompt
    assert "Authentication uses OAuth." in user_prompt

    # Retrieval scores are diagnostic metadata, not evidence.
    assert "0.91" not in user_prompt
    assert "0.87" not in user_prompt


@pytest.mark.asyncio
async def test_ask_does_not_call_llm_when_no_context_found() -> None:
    rag_client = FakeRAGChatClient(
        contexts=[],
    )

    llm_provider = FakeLLMProvider()

    service = ChatService(
        rag_client=rag_client,
        llm_provider=llm_provider,
    )

    result = await service.ask("What is something completely unrelated?")

    assert result.sources == ()

    assert "could not find sufficient information" in result.answer

    # Critical grounding invariant:
    # no evidence -> no LLM generation.
    assert llm_provider.request is None

    # We should also avoid retrieving the grounding prompt because
    # generation will not occur.
    assert rag_client.prompt_question is None


@pytest.mark.asyncio
async def test_ask_rejects_empty_question() -> None:
    service = ChatService(
        rag_client=FakeRAGChatClient(),
        llm_provider=FakeLLMProvider(),
    )

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        await service.ask("   ")


def test_constructor_rejects_invalid_top_k() -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        ChatService(
            rag_client=FakeRAGChatClient(),
            llm_provider=FakeLLMProvider(),
            top_k=0,
        )


@pytest.mark.asyncio
async def test_ask_translates_retrieval_failure() -> None:
    rag_client = FakeRAGChatClient(search_error=RuntimeError("MCP unavailable"))

    service = ChatService(
        rag_client=rag_client,
        llm_provider=FakeLLMProvider(),
    )

    with pytest.raises(
        ChatServiceError,
        match="Failed to retrieve grounding context",
    ):
        await service.ask("What is MCP?")


@pytest.mark.asyncio
async def test_ask_translates_prompt_failure() -> None:
    rag_client = FakeRAGChatClient(
        contexts=[
            create_context(),
        ],
        prompt_error=RuntimeError("MCP prompt unavailable"),
    )

    service = ChatService(
        rag_client=rag_client,
        llm_provider=FakeLLMProvider(),
    )

    with pytest.raises(
        ChatServiceError,
        match="Failed to retrieve the grounded QA prompt",
    ):
        await service.ask("What is MCP?")


@pytest.mark.asyncio
async def test_ask_translates_llm_failure() -> None:
    rag_client = FakeRAGChatClient(
        contexts=[
            create_context(),
        ],
    )

    llm_provider = FakeLLMProvider(
        error=LLMProviderError("Gemini failed to generate a response")
    )

    service = ChatService(
        rag_client=rag_client,
        llm_provider=llm_provider,
    )

    with pytest.raises(
        ChatServiceError,
        match="Failed to generate grounded answer",
    ):
        await service.ask("What is MCP?")


@pytest.mark.asyncio
async def test_prompt_uses_single_page_reference() -> None:
    rag_client = FakeRAGChatClient(
        contexts=[
            create_context(
                page_start=8,
                page_end=8,
            ),
        ]
    )

    llm_provider = FakeLLMProvider()

    service = ChatService(
        rag_client=rag_client,
        llm_provider=llm_provider,
    )

    await service.ask("What is MCP?")

    assert llm_provider.request is not None
    assert "Pages: 8" in llm_provider.request.user_prompt


@pytest.mark.asyncio
async def test_prompt_uses_unknown_when_page_is_missing() -> None:
    rag_client = FakeRAGChatClient(
        contexts=[
            create_context(
                page_start=None,
                page_end=None,
            ),
        ]
    )

    llm_provider = FakeLLMProvider()

    service = ChatService(
        rag_client=rag_client,
        llm_provider=llm_provider,
    )

    await service.ask("What is MCP?")

    assert llm_provider.request is not None
    assert "Pages: unknown" in llm_provider.request.user_prompt


@pytest.mark.asyncio
async def test_citation_order_matches_retrieval_order() -> None:
    rag_client = FakeRAGChatClient(
        contexts=[
            create_context(
                document_id="doc-a",
                document_name="a.pdf",
                content="Evidence A",
            ),
            create_context(
                document_id="doc-b",
                document_name="b.pdf",
                content="Evidence B",
            ),
            create_context(
                document_id="doc-c",
                document_name="c.pdf",
                content="Evidence C",
            ),
        ]
    )

    llm_provider = FakeLLMProvider()

    service = ChatService(
        rag_client=rag_client,
        llm_provider=llm_provider,
    )

    result = await service.ask("Explain the system.")

    assert [source.citation_id for source in result.sources] == [
        "S1",
        "S2",
        "S3",
    ]

    assert [source.document_id for source in result.sources] == [
        "doc-a",
        "doc-b",
        "doc-c",
    ]

    assert llm_provider.request is not None

    user_prompt = llm_provider.request.user_prompt

    assert user_prompt.index("[S1]") < user_prompt.index("[S2]")
    assert user_prompt.index("[S2]") < user_prompt.index("[S3]")
