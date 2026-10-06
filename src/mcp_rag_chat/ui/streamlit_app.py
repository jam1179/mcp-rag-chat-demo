import asyncio
from typing import Any

import streamlit as st
from google import genai

from mcp_rag_chat.chat import ChatAnswer, ChatService
from mcp_rag_chat.client import RAGChatClient
from mcp_rag_chat.config.settings import Settings
from mcp_rag_chat.llm import GeminiLLMProvider


def format_page_reference(
    page_start: int | None,
    page_end: int | None,
) -> str:
    if page_start is None:
        return "Page unavailable"

    if page_end is None or page_end == page_start:
        return f"Page {page_start}"

    return f"Pages {page_start}-{page_end}"


def to_ui_message(
    answer: ChatAnswer,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": answer.answer,
        "sources": [
            {
                "citation_id": source.citation_id,
                "document_name": source.document_name,
                "page_start": source.page_start,
                "page_end": source.page_end,
            }
            for source in answer.sources
        ],
    }


async def ask_question(
    question: str,
    settings: Settings,
) -> ChatAnswer:
    gemini_client = genai.Client(
        api_key=settings.gemini_api_key,
    )

    llm_provider = GeminiLLMProvider(
        client=gemini_client,
        model=settings.gemini_llm_model,
    )

    async with RAGChatClient(settings.mcp_server_url) as rag_client:
        chat_service = ChatService(
            rag_client=rag_client,
            llm_provider=llm_provider,
        )

        return await chat_service.ask(question)


def render_sources(
    sources: list[dict[str, Any]],
) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for source in sources:
            page_reference = format_page_reference(
                source["page_start"],
                source["page_end"],
            )

            st.markdown(
                f"**[{source['citation_id']}] "
                f"{source['document_name']}** — "
                f"{page_reference}"
            )


def render_message(
    message: dict[str, Any],
) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


def main() -> None:
    st.set_page_config(
        page_title="MCP RAG Chat",
        page_icon="💬",
    )

    st.title("MCP RAG Chat")
    st.caption("Ask questions grounded in the knowledge base.")

    settings = Settings()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Ask a question about the knowledge base")

    if not question:
        return

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                answer = asyncio.run(
                    ask_question(
                        question=question,
                        settings=settings,
                    )
                )
            except Exception:
                error_message = "I couldn't complete the request. Please try again."

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
                return

        assistant_message = to_ui_message(answer)

        st.markdown(assistant_message["content"])

        render_sources(assistant_message["sources"])

        st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
