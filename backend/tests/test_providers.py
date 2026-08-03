import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from app.core.providers import (
    GenerationResult,
    LocalOllamaProvider,
    ModelInfo,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnreachableError,
    SpeedResult,
)


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.raise_for_status.return_value = None
    return response


def _provider() -> LocalOllamaProvider:
    return LocalOllamaProvider(base_url="http://localhost:11434")


def test_generate_response_success():
    ollama_json = {
        "model": "llama3:8b",
        "response": "hello world",
        "prompt_eval_count": 5,
        "eval_count": 3,
        "total_duration": 1_000_000_000,
        "eval_duration": 500_000_000,
    }
    with patch("httpx.AsyncClient.post", AsyncMock(return_value=_mock_response(200, ollama_json))):
        result = asyncio.run(_provider().generate_response("llama3:8b", "hi"))

    assert isinstance(result, GenerationResult)
    assert result.text == "hello world"
    assert result.completion_tokens == 3
    assert result.temperature == 0.0


def test_test_speed_success():
    ollama_json = {
        "model": "llama3:8b",
        "response": "...",
        "eval_count": 100,
        "eval_duration": 1_000_000_000,  # 1s -> 100 tok/s
    }
    with patch("httpx.AsyncClient.post", AsyncMock(return_value=_mock_response(200, ollama_json))):
        result = asyncio.run(_provider().test_speed("llama3:8b"))

    assert isinstance(result, SpeedResult)
    assert result.tokens_per_second == pytest.approx(100.0)


def test_generate_response_ollama_unreachable():
    mock_post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    with patch("httpx.AsyncClient.post", mock_post), pytest.raises(OllamaUnreachableError):
        asyncio.run(_provider().generate_response("llama3:8b", "hi"))


def test_generate_response_timeout_raises_distinct_error():
    mock_post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    with patch("httpx.AsyncClient.post", mock_post), pytest.raises(OllamaTimeoutError) as excinfo:
        asyncio.run(_provider().generate_response("deepseek-r1:1.5b", "hi"))

    assert "deepseek-r1:1.5b" in str(excinfo.value)
    assert "Is it running" not in str(excinfo.value)


def test_generate_response_model_not_found():
    err_json = {"error": "model 'nope:8b' not found, try pulling it first"}
    mock_post = AsyncMock(return_value=_mock_response(404, err_json))
    with patch("httpx.AsyncClient.post", mock_post), pytest.raises(OllamaModelNotFoundError):
        asyncio.run(_provider().generate_response("nope:8b", "hi"))


def test_get_model_info_success():
    # Real /api/show response shape, confirmed live against a locally pulled model
    # (docs/implementation-log.md) -- trimmed to the fields get_model_info actually reads.
    ollama_json = {
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "3.2B",
            "quantization_level": "Q4_K_M",
        },
    }
    with patch("httpx.AsyncClient.post", AsyncMock(return_value=_mock_response(200, ollama_json))):
        result = asyncio.run(_provider().get_model_info("llama3.2:latest"))

    assert isinstance(result, ModelInfo)
    assert result.quantization_level == "Q4_K_M"
    assert result.parameter_size == "3.2B"


def test_get_model_info_model_not_found():
    err_json = {"error": "model 'nope:8b' not found"}
    mock_post = AsyncMock(return_value=_mock_response(404, err_json))
    with patch("httpx.AsyncClient.post", mock_post), pytest.raises(OllamaModelNotFoundError):
        asyncio.run(_provider().get_model_info("nope:8b"))


def test_get_model_info_timeout_raises_distinct_error():
    mock_post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    with patch("httpx.AsyncClient.post", mock_post), pytest.raises(OllamaTimeoutError):
        asyncio.run(_provider().get_model_info("deepseek-r1:1.5b"))


def test_connect_error_still_raises_unreachable_not_timeout():
    # Regression check: a genuine connection failure (Ollama actually down) must stay
    # OllamaUnreachableError, not get swept up by the new OllamaTimeoutError handling.
    mock_post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    with patch("httpx.AsyncClient.post", mock_post), pytest.raises(OllamaUnreachableError):
        asyncio.run(_provider().generate_response("llama3:8b", "hi"))
