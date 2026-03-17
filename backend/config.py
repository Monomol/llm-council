"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SUBMIT_DB_URL = os.getenv("SUBMIT_DB_URL")
SCHEMA_NAME = os.getenv("SCHEMA_NAME")
ALLOWED_TOKENS = set(
    t.strip() for t in os.getenv("API_ALLOWED_TOKENS", "").split(",") if t.strip()
)

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "deepseek-v3.2-thinking",
    "gpt-oss-120b",
    # "mistral-large", # seems to be broken?
    "glm-4.7"
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = COUNCIL_MODELS[0]

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://llm.ai.e-infra.cz/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
