# MCP RAG Chat Demo

A Streamlit-based chat application that answers questions using documents indexed by a separate **MCP RAG Server**. The application communicates with the server over the **Model Context Protocol (MCP), Streamable HTTP transport**, and uses an LLM to formulate responses from retrieved context.

> **Architecture boundary:** This repository is an MCP **client**, not the document-ingestion or vector-storage service. The MCP RAG Server owns retrieval, ingestion, embeddings, and LanceDB persistence.

## Architecture

```text
Browser
   |
   v
Streamlit UI (mcp-rag-chat-demo)
   |
   +---- LLM provider (answer generation)
   |
   +---- MCP Client -- Streamable HTTP --> MCP RAG Server
                                           |
                                           +-- search_documents tool
                                           +-- document embeddings / retrieval
                                           +-- LanceDB persistent store
```

The two applications are independently deployable. The chat demo does not import server-side implementation modules.

## Features

- Interactive document-grounded question answering through Streamlit.
- MCP client integration over Streamable HTTP.
- Semantic retrieval through the MCP server's `search_documents` tool.
- Separate application configuration and container image.
- User-friendly failure handling when an upstream request cannot be completed.

## Prerequisites

- Python version compatible with this project's `pyproject.toml` (the Docker deployment uses Python 3.13).
- [`uv`](https://docs.astral.sh/uv/) for local dependency management.
- A running MCP RAG Server with indexed documents.
- Valid credentials for the configured LLM provider, supplied through the project's environment configuration.
- Docker Desktop and Docker Compose for containerized execution.

## Repository layout

```text
mcp-rag-chat-demo/
├── Dockerfile
├── .dockerignore
├── .env                 # Local configuration; do not commit secrets
├── pyproject.toml
├── uv.lock
└── src/
    └── mcp_rag_chat/
        ├── config/
        │   └── settings.py
        ├── client/
        │   ├── client.py
        │   └── models.py
        ├── chat/
        ├── llm/
        └── ui/
            └── streamlit_app.py
```

## Configuration

Create a local `.env` file based on the settings required by `src/mcp_rag_chat/config/settings.py`. Keep API keys out of version control.

The MCP endpoint differs by execution environment:

| Environment | `MCP_SERVER_URL` |
| --- | --- |
| Chat demo running on the host | `http://localhost:8000/mcp` |
| Chat demo running in Docker Compose | `http://mcp-rag-server:8000/mcp` |

The Compose deployment reads `./mcp-rag-chat-demo/.env` and overrides `MCP_SERVER_URL` to use Docker's internal service DNS. Other LLM-specific environment variable names and defaults should be checked against `config/settings.py`; they are deliberately not guessed here.

## Run locally

Start the MCP RAG Server first, and ensure that documents have been ingested. Then, from the `mcp-rag-chat-demo` directory:

```powershell
uv sync --locked
$env:MCP_SERVER_URL = "http://localhost:8000/mcp"
uv run --env-file .env streamlit run src/mcp_rag_chat/ui/streamlit_app.py
```

Open <http://localhost:8501>.

> If the application uses a configured Python module entry point or additional startup options, follow the definitions in `pyproject.toml` and `Dockerfile` instead of assuming the command above is authoritative for every environment.

## Run with Docker Compose

The sibling MCP server and chat demo are orchestrated from the parent `ai-projects` directory:

```text
ai-projects/
├── compose.yaml
├── mcp-rag-server/
└── mcp-rag-chat-demo/
```

From that parent directory:

```powershell
docker compose up -d
docker compose ps
```

Open:

- **Chat UI:** <http://localhost:8501>
- **MCP endpoint (host):** <http://localhost:8000/mcp>

Inspect logs:

```powershell
docker compose logs --tail 100 mcp-rag-chat-demo
docker compose logs --tail 100 mcp-rag-server
```

Stop the stack while **retaining indexed documents**:

```powershell
docker compose down
```

**Do not run `docker compose down -v`** unless you intentionally want to remove the named LanceDB volume and its indexed data.

## Example workflow

1. Start the MCP server and the Streamlit application.
2. Confirm the MCP server already contains indexed documents (for example, `Helm.pdf`).
3. Open the Streamlit UI.
4. Ask: **“According to Helm.pdf, what are the three important concepts in Helm?”**
5. Review the response against the document. The demonstrated Helm concepts are **chart, configuration, and release**.

Documents are ingested by the MCP server, not by the chat application's local filesystem.

## Operational behavior

### MCP server unavailable

The Streamlit UI remains accessible. A failed chat request can display:

> I couldn't complete the request. Please try again.

Once the MCP server is ready again, subsequent chat requests can recover without restarting Streamlit.

### Data persistence

LanceDB data is owned by the MCP server and stored in a Docker named volume (`ai-projects_rag-data` in the validated Compose environment). Recreating the MCP server container or restarting the full Compose stack does not require re-ingesting documents when that volume is retained.

### Response latency

End-to-end chat latency can vary independently of direct MCP retrieval latency. During Docker validation, direct MCP searches took approximately **0.35–0.63 seconds**, while some chat responses were much slower; later responses were observed around **2–3 seconds**. These are observations, **not service-level guarantees**. Further investigation is deferred to the observability milestone.

## Security notes

- Never commit `.env`, API keys, tokens, or credentials.
- The validated Compose deployment runs containers as a non-root user, drops Linux capabilities, enables `no-new-privileges`, and uses read-only root filesystems with narrowly scoped writable mounts.
- Production release is **not approved** solely on the basis of functional Docker validation. The MCP server image has outstanding OS vulnerability findings that require remediation or a formal, scoped risk decision.
- Authentication, API gateway controls, production observability, and scaling are separate roadmap items; do not assume they are already implemented.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Chat UI unavailable | `docker compose ps` and chat container logs; confirm port 8501 is published |
| Chat cannot reach MCP | `MCP_SERVER_URL`, MCP container readiness, Compose service DNS |
| Generic request failure | MCP availability, LLM credentials/provider connectivity, container logs |
| No document-grounded results | Confirm the document was ingested into the MCP server's LanceDB store |
| Data missing after stack reset | Check whether the named volume was removed or a different volume was mounted |
| Slow responses | Separate MCP retrieval time from LLM generation and chat orchestration; add tracing when observability is implemented |

## Development and validation

Check dependency lock consistency:

```powershell
uv lock --check
```

The chat demo's test commands, if present, should be taken from its actual test configuration. The full system's Docker integration validation has covered MCP protocol calls, service outages and recovery, container recreation, persistent retrieval, and full-stack restart.

## Roadmap context

This chat demo is part of a larger MCP RAG system. Docker functional validation is complete; upcoming work includes Kubernetes deployment, API gateway integration, OAuth2, observability, resilience, and CI/CD. Production readiness remains subject to security and operational release gates.
