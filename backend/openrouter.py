import httpx
from typing import List, Dict, Any, Optional
import logging
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL
import asyncio
import random

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_CONCURRENT_REQUESTS = 3

DEFAULT_TIMEOUT=360

_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
_client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = DEFAULT_TIMEOUT
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    async with _semaphore:
        for attempt_idx in range(MAX_ATTEMPTS):
            try:
                response = await _client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()

                data = response.json()
                message = data['choices'][0]['message']

                return {
                    'content': message.get('content'),
                    'reasoning_details': message.get('reasoning_details')
                }

            except Exception as e:
                # TODO: in the future use contextvars and implement logging with trace_id here
                logger.error(f"Error querying model {model} Attempt #{attempt_idx+1}: {e}")
                await asyncio.sleep(30 * (attempt_idx + 1) * random.uniform(1, 1.4))
    return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """

    async def wrapped_query(m):
        res = await query_model(m, messages, DEFAULT_TIMEOUT)
        return m, res

    tasks = [asyncio.create_task(wrapped_query(m)) for m in models]
    final_results = {}

    try:
        for next_task in asyncio.as_completed(tasks):
            model_name, result = await next_task
            
            if result is None:
                logger.warning(f"Model {model_name} failed. Cancelling all other tasks.")
                return None
            
            final_results[model_name] = result

        return final_results

    except Exception as e:
        # TODO: consider adding logging here
        return None

    finally:
        for t in tasks:
            if not t.done():
                t.cancel()