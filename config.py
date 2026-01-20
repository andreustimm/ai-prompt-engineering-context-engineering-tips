"""
Configuração centralizada para os scripts de prompt engineering.
Carrega variáveis de ambiente e fornece função para criar instância do LLM.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()


@dataclass
class TokenUsage:
    """Classe para rastrear uso de tokens."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int):
        """Adiciona tokens ao contador."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def reset(self):
        """Reseta os contadores."""
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

    Args:
        temperature: Controla a aleatoriedade das respostas (0.0 a 1.0)

    Returns:
        Instância configurada do ChatOpenAI
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY não encontrada. "
            "Configure a variável de ambiente ou crie um arquivo .env"
        )

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature
    )


def get_model_name() -> str:
    """Retorna o nome do modelo configurado."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def extract_tokens_from_response(response) -> tuple[int, int]:
    """
    Extrai contagem de tokens de uma resposta do LLM.

    Args:
        response: Resposta do ChatOpenAI (AIMessage)

    Returns:
        Tupla (input_tokens, output_tokens)
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
    """Imprime o uso de tokens formatado."""
    total = input_tokens + output_tokens
    prefix = f"[{label}] " if label else ""
    print(f"   {prefix}📊 Tokens - Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {total:,}")


def print_total_usage(tracker: TokenUsage, label: str = "TOTAL DA SESSÃO"):
    """Imprime o total de tokens usados."""
    print(f"\n{'='*60}")
    print(f"📈 {label}")
    print(f"   Input:  {tracker.input_tokens:,} tokens")
    print(f"   Output: {tracker.output_tokens:,} tokens")
    print(f"   Total:  {tracker.total_tokens:,} tokens")
    print(f"{'='*60}")
