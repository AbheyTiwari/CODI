# llm_factory.py
from config import MODE, CLOUD_PROVIDER
from config import REFINER_MODEL_LOCAL, CODER_MODEL_LOCAL
from config import REFINER_MODEL_CLOUD, CODER_MODEL_CLOUD
from config import GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

def get_refiner_llm():
    if MODE == "local":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=REFINER_MODEL_LOCAL,
            temperature=0.2,
            num_ctx=4096,
            timeout=120
        )
    else:
        return _get_cloud_llm(REFINER_MODEL_CLOUD, temperature=0.2)

def get_coder_llm():
    if MODE == "local":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=CODER_MODEL_LOCAL,
            temperature=0.1,
            num_ctx=4096,
            timeout=120
        )
    else:
        return _get_cloud_llm(CODER_MODEL_CLOUD, temperature=0.1)

def _get_cloud_llm(model: str, temperature: float):
    if CLOUD_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=GROQ_API_KEY
        )
    elif CLOUD_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=OPENAI_API_KEY
        )
    elif CLOUD_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=ANTHROPIC_API_KEY
        )
    else:
        raise ValueError(f"Unknown cloud provider: {CLOUD_PROVIDER}")
