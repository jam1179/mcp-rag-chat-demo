from typing import Any

from mcp import Client

from mcp_rag_chat.client.models import RetrievedContext


class RAGChatClientError(Exception):
    """Raised when communication with the RAG MCP server fails."""


class RAGChatClient:
    """
    Application-facing client for chat-oriented RAG capabilities.

    This class intentionally hides MCP SDK and transport details from
    higher-level consumers such as the LLM host and Streamlit UI.
    """

    def __init__(self, server_url: str) -> None:
        if not server_url or not server_url.strip():
            raise ValueError("server_url must not be empty")

        self._server_url = server_url
        self._client: Client | None = None

    async def __aenter__(self) -> "RAGChatClient":
        try:
            self._client = Client(self._server_url)
            await self._client.__aenter__()
            return self
        except Exception as exc:
            self._client = None
            raise RAGChatClientError(
                f"Failed to connect to MCP server: {self._server_url}"
            ) from exc

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        if self._client is None:
            return

        try:
            await self._client.__aexit__(
                exc_type,
                exc_value,
                traceback,
            )
        finally:
            self._client = None

    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        client = self._require_client()

        try:
            result = await client.call_tool(
                "search_documents",
                {
                    "query": query,
                    "top_k": top_k,
                },
            )
        except Exception as exc:
            raise RAGChatClientError("Failed to search documents through MCP") from exc

        if result.is_error:
            raise RAGChatClientError("MCP search_documents tool returned an error")

        return self._map_search_result(result.structured_content)

    async def get_grounded_qa_prompt(
        self,
        question: str,
    ) -> str:
        client = self._require_client()

        try:
            result = await client.get_prompt(
                "grounded_qa",
                {
                    "question": question,
                },
            )
        except Exception as exc:
            raise RAGChatClientError(
                "Failed to retrieve grounded_qa prompt through MCP"
            ) from exc

        return self._extract_prompt_text(result)

    async def read_document(
        self,
        document_id: str,
    ) -> str:
        client = self._require_client()

        uri = f"document://{document_id}"

        try:
            result = await client.read_resource(uri)
        except Exception as exc:
            raise RAGChatClientError(
                f"Failed to read document through MCP: {document_id}"
            ) from exc

        if not result.contents:
            raise RAGChatClientError(f"MCP resource returned no content: {uri}")

        content = result.contents[0]

        text = getattr(content, "text", None)

        if text is None:
            raise RAGChatClientError(f"MCP resource did not return text content: {uri}")

        return text

    def _require_client(self) -> Client:
        if self._client is None:
            raise RAGChatClientError(
                "RAGChatClient must be used as an async context manager"
            )

        return self._client

    @staticmethod
    def _map_search_result(
        structured_content: Any,
    ) -> list[RetrievedContext]:
        if not isinstance(structured_content, dict):
            raise RAGChatClientError("Unexpected search_documents response structure")

        raw_results = structured_content.get("result")

        if not isinstance(raw_results, list):
            raise RAGChatClientError(
                "search_documents response must contain a result list"
            )

        contexts: list[RetrievedContext] = []

        for item in raw_results:
            if not isinstance(item, dict):
                raise RAGChatClientError("Unexpected search_documents result item")

            try:
                contexts.append(
                    RetrievedContext(
                        document_id=item["document_id"],
                        document_name=item["document_name"],
                        content=item["content"],
                        page_start=item.get("page_start"),
                        page_end=item.get("page_end"),
                        relevance_score=float(item["relevance_score"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RAGChatClientError("Invalid search_documents result") from exc

        return contexts

    @staticmethod
    def _extract_prompt_text(result: Any) -> str:
        messages = getattr(result, "messages", None)

        if not messages:
            raise RAGChatClientError("MCP prompt returned no messages")

        text_parts: list[str] = []

        for message in messages:
            content = getattr(message, "content", None)

            text = getattr(content, "text", None)

            if text:
                text_parts.append(text)

        if not text_parts:
            raise RAGChatClientError("MCP prompt returned no text content")

        return "\n".join(text_parts)
