import os
import json
from dotenv import load_dotenv

load_dotenv()  # loads .env file if present

def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"api_keys": {}}

def get_api_key(provider: str) -> str:
    # env var takes priority over config file
    # Supported env vars:
    #   CODI_GROQ_API_KEY
    #   CODI_OPENAI_API_KEY
    #   CODI_ANTHROPIC_API_KEY
    #   CODI_GEMINI_API_KEY
    env_key = f"CODI_{provider.upper()}_API_KEY"
    return os.environ.get(env_key) or load_config()["api_keys"].get(provider, "")