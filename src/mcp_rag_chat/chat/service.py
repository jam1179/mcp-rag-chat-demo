from mcp_rag_chat.chat.models import (
    ChatAnswer,
    ChatSource,
)
from mcp_rag_chat.client import (
    RAGChatClient,
    RetrievedContext,
)
from mcp_rag_chat.llm import (
    LLMProvider,
    LLMRequest,
)


class ChatServiceError(Exception):
    """Raised when the RAG chat workflow cannot complete."""


class ChatService:
    def __init__(
        self,
        rag_client: RAGChatClient,
        llm_provider: LLMProvider,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self._rag_client = rag_client
        self._llm_provider = llm_provider
        self._top_k = top_k

    async def ask(
        self,
        question: str,
    ) -> ChatAnswer:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("question must not be empty")

        # ---------------------------------------------------------
        # 1. Retrieve grounding evidence through the MCP boundary.
        # ---------------------------------------------------------
        try:
            contexts = await self._rag_client.search_documents(
                query=normalized_question,
                top_k=self._top_k,
            )
        except Exception as exc:
            raise ChatServiceError("Failed to retrieve grounding context") from exc

        # ---------------------------------------------------------
        # 2. Grounding invariant:
        #    no retrieved evidence -> no LLM generation.
        # ---------------------------------------------------------
        if not contexts:
            return ChatAnswer(
                answer=(
                    "I could not find sufficient information "
                    "in the knowledge base to answer this question."
                ),
                sources=(),
            )

        # ---------------------------------------------------------
        # 3. Retrieve the grounding policy from the MCP server.
        # ---------------------------------------------------------
        try:
            system_prompt = await self._rag_client.get_grounded_qa_prompt(
                normalized_question
            )
        except Exception as exc:
            raise ChatServiceError("Failed to retrieve the grounded QA prompt") from exc

        # ---------------------------------------------------------
        # 4. Create the authoritative citation/provenance mapping.
        #
        #    S1 -> contexts[0]
        #    S2 -> contexts[1]
        #    ...
        #
        #    The same source objects are used for both the LLM
        #    context and the final ChatAnswer.
        # ---------------------------------------------------------
        sources = self._build_sources(contexts)

        # ---------------------------------------------------------
        # 5. Assemble the untrusted retrieved evidence into the
        #    user-level prompt.
        # ---------------------------------------------------------
        user_prompt = self._build_user_prompt(
            question=normalized_question,
            contexts=contexts,
            sources=sources,
        )

        # ---------------------------------------------------------
        # 6. Generate the grounded answer through the abstract LLM
        #    provider boundary.
        # ---------------------------------------------------------
        try:
            response = await self._llm_provider.generate(
                LLMRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            )
        except Exception as exc:
            raise ChatServiceError("Failed to generate grounded answer") from exc

        # ---------------------------------------------------------
        # 7. Return the generated answer together with the same
        #    authoritative citation/provenance mapping.
        # ---------------------------------------------------------
        return ChatAnswer(
            answer=response.text,
            sources=sources,
        )

    @staticmethod
    def _build_sources(
        contexts: list[RetrievedContext],
    ) -> tuple[ChatSource, ...]:
        """
        Build the authoritative mapping between retrieved evidence
        and application-level citation identifiers.

        Citation identifiers deliberately do not expose storage-level
        identifiers such as chunk_id or content_hash.
        """
        return tuple(
            ChatSource(
                citation_id=f"S{index}",
                document_id=context.document_id,
                document_name=context.document_name,
                page_start=context.page_start,
                page_end=context.page_end,
                relevance_score=context.relevance_score,
            )
            for index, context in enumerate(
                contexts,
                start=1,
            )
        )

    @staticmethod
    def _build_user_prompt(
        question: str,
        contexts: list[RetrievedContext],
        sources: tuple[ChatSource, ...],
    ) -> str:
        """
        Assemble retrieved evidence for the LLM.

        Retrieved document content is treated as untrusted evidence,
        not as system-level instructions.
        """
        context_sections: list[str] = []

        for context, source in zip(
            contexts,
            sources,
            strict=True,
        ):
            page_reference = ChatService._format_page_reference(
                context.page_start,
                context.page_end,
            )

            context_sections.append(
                "\n".join(
                    [
                        f"[{source.citation_id}]",
                        f"Document: {context.document_name}",
                        f"Pages: {page_reference}",
                        "Content:",
                        context.content,
                    ]
                )
            )

        context_text = "\n\n".join(context_sections)

        return (
            "Answer the question using only the retrieved "
            "context below.\n"
            "Cite supporting evidence using the source identifiers "
            "exactly as provided, for example [S1] or [S2].\n"
            "Do not invent source identifiers.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context_text}"
        )

    @staticmethod
    def _format_page_reference(
        page_start: int | None,
        page_end: int | None,
    ) -> str:
        if page_start is None:
            return "unknown"

        if page_end is None or page_end == page_start:
            return str(page_start)

        return f"{page_start}-{page_end}"
