"""
Configuração centralizada para os scripts de prompt engineering.
Carrega variáveis de ambiente e fornece função para criar instância do LLM.

Centralized configuration for prompt engineering scripts.
Loads environment variables and provides functions to create LLM instances.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()


@dataclass
class TokenUsage:
    """Classe para rastrear uso de tokens. / Class to track token usage."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int):
        """Adiciona tokens ao contador. / Add tokens to the counter."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def reset(self):
        """Reseta os contadores. / Reset the counters."""
        self.input_tokens = 0
        self.output_tokens = 0

    def __str__(self) -> str:
        return (
            f"📊 Tokens - Input: {self.input_tokens:,} | "
            f"Output: {self.output_tokens:,} | "
            f"Total: {self.total_tokens:,}"
        )


# Instância global para rastrear tokens
token_tracker = TokenUsage()


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Cria e retorna uma instância do ChatOpenAI configurada.
    Creates and returns a configured ChatOpenAI instance.

    Args:
        temperature: Controla a aleatoriedade das respostas (0.0 a 1.0)
                    Controls response randomness (0.0 to 1.0)

    Returns:
        Instância configurada do ChatOpenAI / Configured ChatOpenAI instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY não encontrada. "
            "Configure a variável de ambiente ou crie um arquivo .env\n"
            "OPENAI_API_KEY not found. "
            "Set the environment variable or create a .env file"
        )

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature
    )


def get_model_name() -> str:
    """Retorna o nome do modelo configurado. / Returns the configured model name."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_embeddings():
    """
    Retorna instância de embeddings configurada.
    Returns configured embeddings instance.

    Uses OpenAI embeddings by default, falls back to Ollama if configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return OpenAIEmbeddings(api_key=api_key)

    # Fallback para Ollama se não houver API key da OpenAI
    try:
        from langchain_ollama import OllamaEmbeddings
        ollama_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=ollama_model, base_url=ollama_base_url)
    except ImportError:
        raise ValueError(
            "Nenhuma configuração de embeddings disponível. "
            "Configure OPENAI_API_KEY ou instale langchain-ollama.\n"
            "No embeddings configuration available. "
            "Set OPENAI_API_KEY or install langchain-ollama."
        )


def get_ollama_llm(model: str = None, temperature: float = 0.7):
    """
    Cria e retorna uma instância do ChatOllama para modelos locais.
    Creates and returns a ChatOllama instance for local models.

    Args:
        model: Nome do modelo Ollama (ex: llama3.2, mistral, codellama)
               Ollama model name (e.g., llama3.2, mistral, codellama)
        temperature: Controla a aleatoriedade das respostas (0.0 a 1.0)
                    Controls response randomness (0.0 to 1.0)

    Returns:
        Instância configurada do ChatOllama / Configured ChatOllama instance
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama não instalado. Execute: pip install langchain-ollama\n"
            "langchain-ollama not installed. Run: pip install langchain-ollama"
        )

    if model is None:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature
    )


def get_ollama_embeddings(model: str = None):
    """
    Cria e retorna embeddings locais via Ollama.
    Creates and returns local embeddings via Ollama.

    Args:
        model: Nome do modelo de embeddings (ex: nomic-embed-text)
               Embedding model name (e.g., nomic-embed-text)

    Returns:
        Instância configurada do OllamaEmbeddings / Configured OllamaEmbeddings instance
    """
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        raise ImportError(
            "langchain-ollama não instalado. Execute: pip install langchain-ollama\n"
            "langchain-ollama not installed. Run: pip install langchain-ollama"
        )

    if model is None:
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return OllamaEmbeddings(model=model, base_url=base_url)


def is_ollama_available() -> bool:
    """
    Verifica se o Ollama está disponível e rodando.
    Checks if Ollama is available and running.

    Returns:
        True se Ollama estiver acessível / True if Ollama is accessible
    """
    import urllib.request
    import urllib.error

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def extract_tokens_from_response(response) -> tuple[int, int]:
    """
    Extrai contagem de tokens de uma resposta do LLM.
    Extracts token count from an LLM response.

    Args:
        response: Resposta do ChatOpenAI (AIMessage) / ChatOpenAI response (AIMessage)

    Returns:
        Tupla (input_tokens, output_tokens) / Tuple (input_tokens, output_tokens)
    """
    input_tokens = 0
    output_tokens = 0

    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if 'token_usage' in metadata:
            usage = metadata['token_usage']
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)

    return input_tokens, output_tokens


def print_token_usage(input_tokens: int, output_tokens: int, label: str = ""):
    """Imprime o uso de tokens formatado. / Prints formatted token usage."""
    total = input_tokens + output_tokens
    prefix = f"[{label}] " if label else ""
    print(f"   {prefix}📊 Tokens - Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {total:,}")


def print_total_usage(tracker: TokenUsage, label: str = "TOTAL DA SESSÃO"):
    """Imprime o total de tokens usados. / Prints total tokens used."""
    print(f"\n{'='*60}")
    print(f"📈 {label}")
    print(f"   Input:  {tracker.input_tokens:,} tokens")
    print(f"   Output: {tracker.output_tokens:,} tokens")
    print(f"   Total:  {tracker.total_tokens:,} tokens")
    print(f"{'='*60}")
