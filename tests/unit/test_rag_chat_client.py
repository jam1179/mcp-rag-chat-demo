import pytest

from mcp_rag_chat.client import (
    RAGChatClient,
    RAGChatClientError,
    RetrievedContext,
)


def test_map_search_result() -> None:
    structured_content = {
        "result": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "document_name": "architecture.pdf",
                "content": "The platform uses a microservice architecture.",
                "content_hash": "abc123",
                "page_start": 4,
                "page_end": 5,
                "relevance_score": 0.91,
            }
        ]
    }

    result = RAGChatClient._map_search_result(structured_content)

    assert result == [
        RetrievedContext(
            document_id="doc-1",
            document_name="architecture.pdf",
            content="The platform uses a microservice architecture.",
            page_start=4,
            page_end=5,
            relevance_score=0.91,
        )
    ]


def test_map_empty_search_result() -> None:
    result = RAGChatClient._map_search_result(
        {
            "result": [],
        }
    )

    assert result == []


def test_invalid_search_result_raises_client_error() -> None:
    structured_content = {
        "result": [
            {
                "document_id": "doc-1",
                # document_name intentionally missing
                "content": "content",
                "page_start": 1,
                "page_end": 1,
                "relevance_score": 0.8,
            }
        ]
    }

    with pytest.raises(
        RAGChatClientError,
        match="Invalid search_documents result",
    ):
        RAGChatClient._map_search_result(structured_content)


@pytest.mark.asyncio
async def test_search_requires_active_client() -> None:
    client = RAGChatClient("http://127.0.0.1:8000/mcp")

    with pytest.raises(
        RAGChatClientError,
        match="must be used as an async context manager",
    ):
        await client.search_documents("architecture")


@pytest.mark.parametrize(
    "server_url",
    [
        "",
        "   ",
    ],
)
def test_server_url_must_not_be_empty(
    server_url: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="server_url must not be empty",
    ):
        RAGChatClient(server_url)


def test_missing_result_field_raises_client_error() -> None:
    with pytest.raises(
        RAGChatClientError,
        match="response must contain a result list",
    ):
        RAGChatClient._map_search_result({})
