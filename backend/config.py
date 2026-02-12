"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "gpt-oss-120b",
    "deepseek-v3.2",
    "mistral-large",
    "glm-4.7"
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "deepseek-v3.2"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://llm.ai.e-infra.cz/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
