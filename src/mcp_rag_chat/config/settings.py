from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # MCP Server Configuration
    # ---------------------------------------------------------
    mcp_server_url: str = "http://127.0.0.1:8000/mcp"

    # ---------------------------------------------------------
    # Gemini LLM Configuration
    # ---------------------------------------------------------
    gemini_api_key: str = Field(
        min_length=1,
        repr=False,
    )

    gemini_llm_model: str = "gemini-3.8-flash"

    # ---------------------------------------------------------
    # Field Validators
    # ---------------------------------------------------------
    @field_validator(
        "gemini_api_key",
        "gemini_llm_model",
    )
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Configuration value must not be blank")
        return value

    @field_validator("mcp_server_url")
    @classmethod
    def validate_mcp_url(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("MCP server URL must not be blank or padded")

        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError(
                "MCP server URL contains an invalid address or port"
            ) from exc

        if parsed.scheme not in ("http", "https"):
            raise ValueError("MCP server URL must use HTTP or HTTPS")

        if not parsed.hostname:
            raise ValueError("MCP server URL must contain a hostname")

        if port is not None and not 1 <= port <= 65535:
            raise ValueError("MCP server URL contains an invalid port")

        if parsed.username or parsed.password:
            raise ValueError("MCP server URL must not contain credentials")

        if parsed.fragment:
            raise ValueError("MCP server URL must not contain a fragment")

        if parsed.query:
            raise ValueError("MCP server URL must not contain query parameters")

        if not parsed.path.startswith("/"):
            raise ValueError("MCP server URL must contain a valid path")

        return value

    # ---------------------------------------------------------
    # Pydantic Settings Configuration
    # ---------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )
