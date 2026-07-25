from typing import Any, Dict, List
import httpx
from app.core.config import settings


class OllamaProvider:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL):
        self.base_url = base_url

    async def list_models(self) -> List[Dict[str, Any]]:
        """Fetch all locally available Ollama models."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])

    async def generate(self, model: str, prompt: str, system: str = "") -> str:
        """Send prompt to Ollama with temperature 0 for deterministic outputs."""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")

    async def unload_model(self, model: str) -> None:
        """Unload model from VRAM using keep_alive=0."""
        payload = {
            "model": model,
            "keep_alive": 0
        }
        async with httpx.AsyncClient() as client:
            await client.post(f"{self.base_url}/api/generate", json=payload)


ollama_provider = OllamaProvider()

