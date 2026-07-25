import time
from app.core.providers import ollama_provider


async def profile_hardware(model_name: str) -> float:
    """Benchmark target model to calculate tokens per second."""
    prompt = "Count from 1 to 20 as fast as possible."
    
    start_time = time.perf_counter()
    response_text = await ollama_provider.generate(model=model_name, prompt=prompt)
    elapsed_seconds = time.perf_counter() - start_time

    estimated_tokens = len(response_text.split())
    if elapsed_seconds <= 0:
        return 10.0

    return max(estimated_tokens / elapsed_seconds, 1.0)