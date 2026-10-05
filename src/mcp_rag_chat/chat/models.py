from dataclasses import dataclass


@dataclass(frozen=True)
class ChatSource:
    citation_id: str
    document_id: str
    document_name: str
    page_start: int | None
    page_end: int | None
    relevance_score: float


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    sources: tuple[ChatSource, ...]
