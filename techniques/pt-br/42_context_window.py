"""
Técnica 42: Gerenciamento de Janela de Contexto

Demonstra gerenciamento inteligente da janela de contexto para prevenir
perda de informação e manter qualidade em conversas longas.

Conceitos-chave da FullCycle AI Tech Week (Aula 2):
- Janela de contexto tem um limite - quando cheia, info antiga é perdida
- Sumarização degrada qualidade (problema "resumo do resumo")
- Cada sumarização compõe a perda
- Subagentes ajudam por terem janelas de contexto separadas

Casos de uso:
- Agentes de codificação de longa duração
- Tarefas complexas com múltiplos passos
- Manter contexto em conversas estendidas
"""

import tiktoken
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Limites de janela de contexto (aproximados)
CONTEXT_LIMITS = {
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
}


@dataclass
class Message:
    """Uma mensagem na conversa."""
    role: str
    content: str
    tokens: int = 0
    is_summary: bool = False
    original_messages: int = 0  # Quantas mensagens isto sumariza


@dataclass
class ContextWindow:
    """
    Gerencia a janela de contexto com estratégias de sumarização inteligente.

    Insight chave da FullCycle: Sumarização é com perdas. Cada vez que você
    sumariza, perde informação. "Resumo do resumo" compõe isso.
    """
    max_tokens: int
    messages: list[Message] = field(default_factory=list)
    total_tokens: int = 0
    summarization_count: int = 0  # Rastreia quantas vezes sumarizamos
    encoding: any = field(default=None)

    def __post_init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(self, text: str) -> int:
        """Conta tokens no texto."""
        return len(self.encoding.encode(text))

    def add_message(self, role: str, content: str) -> bool:
        """
        Adiciona uma mensagem à janela de contexto.
        Retorna True se sumarização foi necessária.
        """
        tokens = self.count_tokens(content)
        message = Message(role=role, content=content, tokens=tokens)

        # Verifica se precisamos sumarizar
        if self.total_tokens + tokens > self.max_tokens * 0.8:  # Threshold de 80%
            self._smart_summarize()
            self.summarization_count += 1

        self.messages.append(message)
        self.total_tokens += tokens
        return self.summarization_count > 0

    def _smart_summarize(self):
        """
        Sumarização inteligente que preserva informações críticas.

        Estratégia:
        1. Mantém mensagem do sistema intacta
        2. Mantém mensagens recentes intactas
        3. Sumariza mensagens antigas com extração de info crítica
        """
        if len(self.messages) < 5:
            return

        # Encontra mensagens para sumarizar (mantém últimas 3)
        to_summarize = self.messages[:-3]
        to_keep = self.messages[-3:]

        if not to_summarize:
            return

        # Extrai conteúdo para sumarização
        summary_content = "\n".join([
            f"{m.role}: {m.content}" for m in to_summarize
            if not m.is_summary  # Não re-sumariza resumos (perde mais info)
        ])

        # Cria resumo com preservação de info crítica
        summary_prompt = f"""Resuma esta conversa, preservando:
1. Decisões-chave tomadas
2. Detalhes importantes de código/técnicos
3. Questões não resolvidas
4. Contexto/estado atual

Conversa para resumir:
{summary_content}

Crie um resumo conciso mas completo que mantenha todo o contexto crítico."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=500
        )

        summary_text = response.choices[0].message.content

        # Cria mensagem de resumo
        summary_msg = Message(
            role="system",
            content=f"[Resumo da conversa anterior - sumarização #{self.summarization_count + 1}]\n{summary_text}",
            tokens=self.count_tokens(summary_text),
            is_summary=True,
            original_messages=len(to_summarize)
        )

        # Substitui mensagens antigas pelo resumo
        self.messages = [summary_msg] + to_keep
        self.total_tokens = sum(m.tokens for m in self.messages)

    def get_messages_for_api(self) -> list[dict]:
        """Obtém mensagens formatadas para API OpenAI."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def get_stats(self) -> dict:
        """Obtém estatísticas da janela de contexto."""
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": (self.total_tokens / self.max_tokens) * 100,
            "message_count": len(self.messages),
            "summarization_count": self.summarization_count,
            "has_summaries": any(m.is_summary for m in self.messages),
        }


class ContextAwareAgent:
    """
    Um agente que monitora e gerencia sua janela de contexto.

    Insight chave: Em vez de deixar o contexto transbordar silenciosamente,
    gerencie-o ativamente para manter qualidade.
    """

    def __init__(self, max_tokens: int = 4000):
        self.context = ContextWindow(max_tokens=max_tokens)
        self.quality_warnings: list[str] = []

    def chat(self, user_message: str) -> str:
        """
        Processa uma mensagem de chat com gerenciamento de contexto.

        Monitora janela de contexto e avisa sobre potencial degradação de qualidade.
        """
        # Adiciona mensagem do usuário
        needed_summarization = self.context.add_message("user", user_message)

        if needed_summarization:
            self.quality_warnings.append(
                f"Contexto sumarizado (#{self.context.summarization_count}). "
                "Algumas informações podem ter sido perdidas."
            )

        # Gera resposta
        response = client.chat.completions.create(
            model=MODEL,
            messages=self.context.get_messages_for_api()
        )

        assistant_message = response.choices[0].message.content

        # Adiciona resposta do assistente
        self.context.add_message("assistant", assistant_message)

        return assistant_message

    def get_context_health(self) -> dict:
        """
        Avalia a saúde do contexto atual.

        Maior contagem de sumarização = menor qualidade de contexto.
        """
        stats = self.context.get_stats()

        # Calcula score de saúde (menor é pior)
        health_score = 100
        health_score -= stats["summarization_count"] * 15  # Cada sumarização prejudica
        health_score -= max(0, stats["usage_percent"] - 70)  # Alto uso prejudica

        return {
            **stats,
            "health_score": max(0, health_score),
            "quality_warnings": self.quality_warnings.copy(),
            "recommendation": self._get_recommendation(health_score)
        }

    def _get_recommendation(self, health_score: int) -> str:
        if health_score >= 80:
            return "Contexto está saudável. Continue normalmente."
        elif health_score >= 50:
            return "Contexto degradando. Considere iniciar nova sessão ou usar subagentes."
        else:
            return "Contexto severamente degradado. Inicie nova sessão ou faça checkpoint de info importante."


def demonstrate_context_degradation():
    """
    Demonstra como o contexto degrada ao longo do tempo com sumarização.

    Isso mostra o problema "resumo do resumo" mencionado na FullCycle.
    """
    print("=" * 60)
    print("Gerenciamento de Janela de Contexto - Demo de Degradação")
    print("=" * 60)

    # Usa janela de contexto pequena para demonstrar
    agent = ContextAwareAgent(max_tokens=2000)

    # Simula uma conversa longa
    messages = [
        "Vamos construir um sistema de gerenciamento de usuários. Comece com o modelo User.",
        "Agora adicione autenticação com tokens JWT.",
        "Implemente hash de senha com bcrypt.",
        "Adicione fluxo de verificação de email.",
        "Crie o endpoint de login.",
        "Adicione funcionalidade de refresh token.",
        "Implemente logout que invalida tokens.",
        "Adicione rate limiting para prevenir brute force.",
        "Crie fluxo de reset de senha.",
        "Adicione autenticação de dois fatores.",
    ]

    print("\nSimulando conversa longa com janela de contexto pequena...")
    print(f"Max tokens: {agent.context.max_tokens}")
    print()

    for i, msg in enumerate(messages, 1):
        print(f"Mensagem {i}: {msg[:50]}...")
        response = agent.chat(msg)
        stats = agent.context.get_stats()
        print(f"  Tokens: {stats['total_tokens']}/{stats['max_tokens']} ({stats['usage_percent']:.1f}%)")
        print(f"  Sumarizações: {stats['summarization_count']}")
        if stats['summarization_count'] > 0:
            print(f"  ⚠️ Contexto foi sumarizado - perda de informação possível")
        print()

    # Mostra saúde final do contexto
    health = agent.get_context_health()
    print("\n" + "=" * 60)
    print("Relatório Final de Saúde do Contexto")
    print("=" * 60)
    print(f"Score de Saúde: {health['health_score']}/100")
    print(f"Total de Sumarizações: {health['summarization_count']}")
    print(f"Recomendação: {health['recommendation']}")
    if health['quality_warnings']:
        print("\nAvisos:")
        for warning in health['quality_warnings']:
            print(f"  - {warning}")


def demonstrate_subagent_strategy():
    """
    Demonstra uso de subagentes para evitar degradação de contexto.

    Insight chave: Cada subagente tem sua própria janela de contexto.
    """
    print("\n" + "=" * 60)
    print("Estratégia de Subagentes - Evitando Degradação de Contexto")
    print("=" * 60)

    print("""
A estratégia de subagentes evita degradação de contexto por:

1. Agente principal tem sua própria janela de contexto
2. Cada tarefa de subagente roda com janela de contexto NOVA
3. Apenas RESULTADOS são passados de volta (não contexto completo)
4. Isso previne degradação "resumo do resumo"

Exemplo de fluxo:
┌─────────────────────────────────────────────────────────┐
│ Agente Principal (Janela de Contexto A)                 │
│                                                         │
│  Tarefa: "Construir sistema de usuários"               │
│    │                                                    │
│    ├─> Subagente 1 (Contexto Novo B)                   │
│    │     Tarefa: "Criar modelo User"                   │
│    │     Retorna: {model_code, decisions}              │
│    │                                                    │
│    ├─> Subagente 2 (Contexto Novo C)                   │
│    │     Tarefa: "Adicionar auth" + resultados Sub 1   │
│    │     Retorna: {auth_code, decisions}               │
│    │                                                    │
│    └─> Agente principal agrega resultados              │
│                                                         │
│  Cada subagente: Contexto novo = Sem degradação        │
│  Agente principal: Só armazena resultados = Contexto   │
│                    mínimo                               │
└─────────────────────────────────────────────────────────┘
""")

    # Simula padrão de subagentes
    class SubagentOrchestrator:
        def __init__(self):
            self.results = []

        def execute_subtask(self, task: str, context: str = "") -> dict:
            """Executa uma subtarefa com contexto novo (simulado)."""
            prompt = f"""Execute esta tarefa e retorne um resultado estruturado.

Tarefa: {task}
{"Contexto de passos anteriores:" + context if context else ""}

Retorne um JSON com: {{"result": "...", "key_decisions": ["..."]}}"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            return response.choices[0].message.content

    orchestrator = SubagentOrchestrator()

    subtasks = [
        "Projete o modelo de dados User com campos para nome, email, senha",
        "Implemente função de hash de senha",
        "Crie função de geração de token JWT",
    ]

    print("\nExecutando subtarefas com contextos novos...")
    accumulated_context = ""

    for i, task in enumerate(subtasks, 1):
        print(f"\nSubtarefa {i}: {task}")
        result = orchestrator.execute_subtask(task, accumulated_context)
        print(f"Resultado: {result[:200]}...")

        # Passa apenas info essencial adiante (não contexto completo)
        accumulated_context += f"\n- Completado: {task}"

    print("\n✅ Cada subtarefa rodou com contexto novo - sem degradação!")


def main():
    print("=" * 60)
    print("Gerenciamento de Janela de Contexto")
    print("Conceitos-Chave da FullCycle AI Tech Week")
    print("=" * 60)

    print("""
A janela de contexto é como um balde de memória:
- Quando cheio, informação mais antiga transborda (é perdida)
- Sumarização comprime mas PERDE informação
- "Resumo do resumo" compõe a perda exponencialmente

Ilustração do Wesley do vídeo:
"Eu te amo" -> [contexto enche] -> "te amo" (perdeu "Eu")
O significado muda completamente!

Estratégias para gerenciar:
1. Monitore uso do contexto proativamente
2. Use subagentes para tarefas isoladas
3. Faça checkpoint de informações críticas
4. Inicie sessões novas quando degradado
""")

    # Demo 1: Degradação de contexto
    demonstrate_context_degradation()

    # Demo 2: Estratégia de subagentes
    demonstrate_subagent_strategy()

    print("\n" + "=" * 60)
    print("Principais Aprendizados")
    print("=" * 60)
    print("""
1. Gerenciamento de janela de contexto é CRÍTICO para agentes de longa duração
2. Sumarização é com perdas - evite quando possível
3. Subagentes fornecem janelas de contexto novas
4. Monitore e avise sobre degradação de contexto
5. Faça checkpoint de informações importantes antes que sejam perdidas
""")


if __name__ == "__main__":
    main()
