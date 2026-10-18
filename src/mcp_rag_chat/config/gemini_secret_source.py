import os
from pathlib import Path
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource


class GeminiSecretFileSource(PydanticBaseSettingsSource):
    """Resolve GEMINI_API_KEY_FILE without exposing secret contents."""

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        file_path = os.environ.get("GEMINI_API_KEY_FILE")

        if file_path is None:
            return {}

        if not file_path.strip():
            raise ValueError("GEMINI_API_KEY_FILE must not be blank")

        try:
            api_key = Path(file_path).read_text(encoding="utf-8").rstrip("\r\n")
        except (OSError, UnicodeError) as exc:
            raise ValueError("Unable to read configured Gemini API key file") from None

        if not api_key.strip():
            raise ValueError("Gemini API key file must not be empty")

        return {"gemini_api_key": api_key}
