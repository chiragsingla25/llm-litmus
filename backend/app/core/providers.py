"""BaseLLMProvider and LocalOllamaProvider live here."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings

_DEFAULT_SPEED_TEST_PROMPT = "Write a two-sentence summary of why the sky is blue."


class OllamaProviderError(Exception):
    """Base class for LocalOllamaProvider boundary failures."""


class OllamaUnreachableError(OllamaProviderError):
    def __init__(self, base_url: str) -> None:
        msg = f"Could not reach Ollama at {base_url}. Is it running? Try `ollama serve`."
        super().__init__(msg)
        self.base_url = base_url


class OllamaTimeoutError(OllamaProviderError):
    def __init__(self, model: str, timeout: float) -> None:
        msg = (
            f"Ollama did not respond for model '{model}' within {timeout:.0f}s. It may still "
            "be generating a long response (reasoning models can take a while) rather than "
            "being down -- consider raising ollama_request_timeout_seconds in config.json."
        )
        super().__init__(msg)
        self.model = model
        self.timeout = timeout


class OllamaModelNotFoundError(OllamaProviderError):
    def __init__(self, model: str, detail: str = "") -> None:
        msg = f"Model '{model}' is not available in Ollama. Try `ollama pull {model}`."
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)
        self.model = model


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ns: int
    eval_duration_ns: int
    temperature: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeedResult:
    model: str
    tokens_per_second: float
    completion_tokens: int
    eval_duration_ns: int


@dataclass(frozen=True)
class ModelInfo:
    quantization_level: str
    parameter_size: str


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_response(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
    ) -> GenerationResult: ...

    @abstractmethod
    async def test_speed(
        self,
        model: str,
        *,
        prompt: str = _DEFAULT_SPEED_TEST_PROMPT,
    ) -> SpeedResult: ...

    @abstractmethod
    async def get_model_info(self, model: str) -> ModelInfo: ...


class LocalOllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        timeout: float = settings.OLLAMA_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def generate_response(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
    ) -> GenerationResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(f"{self._base_url}/api/generate", json=payload)
            except httpx.TimeoutException as exc:
                raise OllamaTimeoutError(model, self._timeout) from exc
            except httpx.TransportError as exc:
                raise OllamaUnreachableError(self._base_url) from exc

        if response.status_code == 404:
            detail = response.json().get("error", "")
            raise OllamaModelNotFoundError(model, detail)
        response.raise_for_status()

        data = response.json()
        return GenerationResult(
            text=data.get("response", ""),
            model=data.get("model", model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_duration_ns=data.get("total_duration", 0),
            eval_duration_ns=data.get("eval_duration", 0),
            temperature=temperature,
            raw=data,
        )

    async def test_speed(
        self,
        model: str,
        *,
        prompt: str = _DEFAULT_SPEED_TEST_PROMPT,
    ) -> SpeedResult:
        result = await self.generate_response(model, prompt, temperature=0.0)
        duration_s = result.eval_duration_ns / 1e9
        tokens_per_second = (result.completion_tokens / duration_s) if duration_s > 0 else 0.0
        return SpeedResult(
            model=model,
            tokens_per_second=tokens_per_second,
            completion_tokens=result.completion_tokens,
            eval_duration_ns=result.eval_duration_ns,
        )

    async def get_model_info(self, model: str) -> ModelInfo:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(f"{self._base_url}/api/show", json={"model": model})
            except httpx.TimeoutException as exc:
                raise OllamaTimeoutError(model, self._timeout) from exc
            except httpx.TransportError as exc:
                raise OllamaUnreachableError(self._base_url) from exc

        if response.status_code == 404:
            detail = response.json().get("error", "")
            raise OllamaModelNotFoundError(model, detail)
        response.raise_for_status()

        details = response.json().get("details", {})
        return ModelInfo(
            quantization_level=details.get("quantization_level", ""),
            parameter_size=details.get("parameter_size", ""),
        )
