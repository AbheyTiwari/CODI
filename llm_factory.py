# llm_factory.py
from config import MODE, CLOUD_PROVIDER
from config import REFINER_MODEL_LOCAL, CODER_MODEL_LOCAL
from config import REFINER_MODEL_CLOUD, CODER_MODEL_CLOUD
from config import REFINER_MODEL_GEMINI, CODER_MODEL_GEMINI
from config_loader import get_api_key

# Set this to your Ubuntu machine's LAN IP when MODE = "remote"
# Run llm_server.py on Ubuntu first, it will print the correct IP
REMOTE_LLM_URL = "http://YOUR_UBUNTU_IP:8000"

def get_refiner_llm():
    if MODE == "local":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=REFINER_MODEL_LOCAL,
            temperature=0.2,
            num_ctx=4096,
            timeout=300,
        )
    elif MODE == "remote":
        return _get_remote_llm(mode="refiner")
    elif CLOUD_PROVIDER == "gemini":
        return _get_gemini_llm(REFINER_MODEL_GEMINI, temperature=0.2)
    else:
        return _get_cloud_llm(REFINER_MODEL_CLOUD, temperature=0.2)

def get_coder_llm():
    if MODE == "local":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=CODER_MODEL_LOCAL,
            temperature=0.1,
            num_ctx=8192,
            timeout=300,
        )
    elif MODE == "remote":
        return _get_remote_llm(mode="coder")
    elif CLOUD_PROVIDER == "gemini":
        return _get_gemini_llm(CODER_MODEL_GEMINI, temperature=0.1)
    else:
        return _get_cloud_llm(CODER_MODEL_CLOUD, temperature=0.1)

def _get_remote_llm(mode: str = "coder"):
    """
    Calls the FastAPI server running on Ubuntu (llm_server.py).
    Ubuntu does the heavy inference, Windows just sends/receives.
    """
    import requests
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, List, Optional
    from pydantic import Field

    base_url = REMOTE_LLM_URL
    llm_mode = mode

    class RemoteChatLLM(BaseChatModel):
        base_url: str = Field(default=base_url)
        llm_mode: str = Field(default=llm_mode)

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Any = None,
            **kwargs,
        ) -> ChatResult:
            system = ""
            user = ""
            for m in messages:
                if m.type == "system":
                    system = m.content
                elif m.type == "human":
                    user = m.content  # takes last human message

            try:
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={"system": system, "user": user, "mode": self.llm_mode},
                    timeout=300,
                )
                resp.raise_for_status()
                content = resp.json()["output"]
            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    f"Cannot reach Ubuntu LLM server at {self.base_url}. "
                    f"Make sure llm_server.py is running on Ubuntu."
                )
            except Exception as e:
                raise RuntimeError(f"Remote LLM error: {e}")

            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=content))]
            )

        @property
        def _llm_type(self) -> str:
            return "remote-codi"

    return RemoteChatLLM(base_url=base_url, llm_mode=llm_mode)

def _get_gemini_llm(model: str, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=get_api_key("gemini")
    )

def _get_cloud_llm(model: str, temperature: float):
    if CLOUD_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=get_api_key("groq")
        )
    elif CLOUD_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=get_api_key("openai")
        )
    elif CLOUD_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=get_api_key("anthropic")
        )
    else:
        raise ValueError(f"Unknown cloud provider: {CLOUD_PROVIDER}")
