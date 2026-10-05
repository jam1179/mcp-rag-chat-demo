from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedContext:
    document_id: str
    document_name: str
    content: str
    page_start: int | None
    page_end: int | None
    relevance_score: float
