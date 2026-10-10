# syntax=docker/dockerfile:1

# ---------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency layer
COPY pyproject.toml uv.lock README.md ./

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project

# Application layer
COPY src ./src

# Install application as a real package.
# Do not leave an editable reference to /app/src.
RUN uv sync \
    --frozen \
    --no-dev \
    --no-editable


# ---------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------
FROM python:3.13-slim AS runtime

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --create-home app

# Application code and dependencies are owned by root.
# The non-root runtime user can read/execute but cannot modify them.
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 8501

CMD ["streamlit", "run", "/app/.venv/lib/python3.13/site-packages/mcp_rag_chat/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]