"""OpenRouter API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
import logging
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL
import asyncio


logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }
    for attempt_idx in range(MAX_ATTEMPTS):
        try:
            # Note that here a new client is created every time we try to repeat the request
            # Without it I got Too many requests error
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload
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
            await asyncio.sleep(10)
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
    import asyncio


    async def wrapped_query(m):
        res = await query_model(m, messages, 60)
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