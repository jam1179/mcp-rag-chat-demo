import pytest
from pydantic import ValidationError

from mcp_rag_chat.config.settings import Settings


@pytest.fixture
def make_settings(monkeypatch):
    """Create Settings without ambient environment or .env files."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)

    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)

    def factory(**overrides):
        values = {"gemini_api_key": "test-api-key"}
        values.update(overrides)

        return Settings(_env_file=None, **values)

    return factory


def test_valid_default_settings(make_settings):
    settings = make_settings()

    assert settings.mcp_server_url == "http://127.0.0.1:8000/mcp"
    assert settings.gemini_llm_model == "gemini-3.8-flash"


def test_missing_api_key(monkeypatch):
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)

    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "gemini_api_key" in str(exc_info.value)


@pytest.mark.parametrize(
    "api_key",
    ["", " ", "   ", "\t", "\n"],
)
def test_blank_api_key_rejected(monkeypatch, api_key):
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", api_key)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "invalid_model",
    ["", " ", "   "],
)
def test_blank_llm_model_rejected(make_settings, invalid_model):
    with pytest.raises(ValidationError):
        make_settings(gemini_llm_model=invalid_model)


@pytest.mark.parametrize(
    "invalid_url",
    [
        "",
        " ",
        "localhost:8000/mcp",
        "ftp://localhost:8000/mcp",
        "http://",
        "http://localhost:0/mcp",
        "http://localhost:65536/mcp",
        "http://user:pass@localhost:8000/mcp",
        "http://localhost:8000/mcp?query=1",
        "http://localhost:8000/mcp#fragment",
        " http://localhost:8000/mcp ",
    ],
)
def test_invalid_mcp_url_rejected(make_settings, invalid_url):
    with pytest.raises(ValidationError):
        make_settings(mcp_server_url=invalid_url)


@pytest.mark.parametrize(
    "valid_url",
    [
        "http://127.0.0.1:8000/mcp",
        "http://mcp-rag-server:8000/mcp",
        "https://rag.example.com/mcp",
    ],
)
def test_valid_mcp_urls_accepted(make_settings, valid_url):
    settings = make_settings(mcp_server_url=valid_url)

    assert settings.mcp_server_url == valid_url


def test_valid_docker_configuration(make_settings):
    settings = make_settings(
        mcp_server_url="http://mcp-rag-server:8000/mcp",
        gemini_llm_model="gemini-3.8-flash",
    )

    assert settings.mcp_server_url == "http://mcp-rag-server:8000/mcp"


def test_api_key_not_exposed_in_model_repr(make_settings):
    secret = "super-secret-test-key"

    settings = make_settings(gemini_api_key=secret)

    assert secret not in repr(settings)


def test_invalid_api_key_not_exposed_in_error(monkeypatch):
    invalid_secret = "   "

    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", invalid_secret)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    errors = exc_info.value.errors(
        include_input=False,
        include_context=False,
    )

    assert len(errors) == 1
    assert errors[0]["loc"] == ("gemini_api_key",)
    assert errors[0]["type"] == "value_error"
    assert "input" not in errors[0]
    assert "ctx" not in errors[0]


# ---------------------------------------------------------
# Gemini File-Based Credential Tests
# ---------------------------------------------------------


def test_api_key_loaded_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    secret_file = tmp_path / "gemini-api-key"
    secret_file.write_text("file-secret-key\n", encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(secret_file))

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key == "file-secret-key"


def test_file_takes_precedence_over_environment(monkeypatch, tmp_path):
    secret_file = tmp_path / "gemini-api-key"
    secret_file.write_text("file-secret-key", encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY", "environment-secret-key")
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(secret_file))

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key == "file-secret-key"


def test_constructor_takes_precedence_over_file(monkeypatch, tmp_path):
    secret_file = tmp_path / "gemini-api-key"
    secret_file.write_text("file-secret-key", encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(secret_file))

    settings = Settings(
        _env_file=None,
        gemini_api_key="constructor-secret-key",
    )

    assert settings.gemini_api_key == "constructor-secret-key"


def test_missing_secret_file_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fallback-secret-key")
    monkeypatch.setenv(
        "GEMINI_API_KEY_FILE",
        str(tmp_path / "missing-secret"),
    )

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_empty_secret_file_fails_closed(monkeypatch, tmp_path):
    secret_file = tmp_path / "gemini-api-key"
    secret_file.write_text(" \n ", encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(secret_file))

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_blank_secret_file_path_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_FILE", "   ")

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_file_secret_not_exposed_in_repr(monkeypatch, tmp_path):
    secret = "sensitive-file-secret"

    secret_file = tmp_path / "gemini-api-key"
    secret_file.write_text(secret, encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(secret_file))

    settings = Settings(_env_file=None)

    assert secret not in repr(settings)
