from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    mcp_server_url: str = "http://127.0.0.1:8000/mcp"

    gemini_api_key: str = Field(
        min_length=1,
    )

    gemini_llm_model: str = "gemini-2.5-flash"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
