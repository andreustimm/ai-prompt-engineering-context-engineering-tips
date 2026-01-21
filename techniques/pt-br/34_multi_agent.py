"""
Aplicações Multi-Agente com LLMs

Sistemas multi-agente permitem que múltiplos agentes de IA colaborem
para resolver tarefas complexas. Cada agente pode ter especialização,
ferramentas e responsabilidades diferentes.

Padrões de Multi-Agente:
- Orquestrador: Um agente central coordena outros agentes
- Pipeline: Agentes processam em sequência
- Debate: Agentes discutem para chegar a consenso
- Hierárquico: Agentes organizados em níveis de autoridade

Casos de uso:
- Desenvolvimento de software (planejador, codificador, revisor, testador)
- Pesquisa (coletor, analisador, sintetizador)
- Atendimento ao cliente (triagem, especialistas, supervisor)
- Análise de dados (ETL, análise, visualização)

Requisitos:
- pip install openai
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.openai_client import get_openai_client


class AgentRole(Enum):
    """Papéis possíveis para agentes."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"


@dataclass
class AgentMessage:
    """Mensagem trocada entre agentes."""
    sender: str
    receiver: str
    content: str
    message_type: str = "task"  # task, response, feedback, question
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.message_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class Agent:
    """Representa um agente individual no sistema."""
    name: str
    role: AgentRole
    system_prompt: str
    tools: list = field(default_factory=list)
    memory: list = field(default_factory=list)

    def add_to_memory(self, message: AgentMessage):
        """Adiciona mensagem à memória do agente."""
        self.memory.append(message)

    def get_context(self, max_messages: int = 10) -> str:
        """Retorna contexto recente da memória."""
        recent = self.memory[-max_messages:]
        return "\n".join([
            f"[{m.sender} -> {m.receiver}]: {m.content}"
            for m in recent
        ])


class MultiAgentSystem:
    """
    Sistema de múltiplos agentes colaborativos.

    Este sistema demonstra como coordenar múltiplos agentes de IA
    para resolver tarefas complexas de forma colaborativa.
    """

    def __init__(self, name: str = "multi-agent-system"):
        self.name = name
        self.agents: dict[str, Agent] = {}
        self.message_history: list[AgentMessage] = []
        self.client = get_openai_client()

    def add_agent(self, agent: Agent):
        """Adiciona um agente ao sistema."""
        self.agents[agent.name] = agent
        print(f"   Agente adicionado: {agent.name} ({agent.role.value})")

    def send_message(self, message: AgentMessage):
        """Envia mensagem entre agentes."""
        self.message_history.append(message)

        # Adiciona à memória do remetente e destinatário
        if message.sender in self.agents:
            self.agents[message.sender].add_to_memory(message)
        if message.receiver in self.agents:
            self.agents[message.receiver].add_to_memory(message)

    def agent_respond(self, agent_name: str, task: str, context: str = "") -> str:
        """Faz um agente responder a uma tarefa."""
        if agent_name not in self.agents:
            return f"Agente '{agent_name}' não encontrado"

        agent = self.agents[agent_name]

        # Constrói o prompt com contexto
        full_prompt = f"""
Contexto da conversa:
{context if context else agent.get_context()}

Tarefa atual:
{task}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content


def create_software_development_team() -> MultiAgentSystem:
    """Cria um time de desenvolvimento de software com múltiplos agentes."""

    system = MultiAgentSystem(name="dev-team")

    # Agente Planejador
    planner = Agent(
        name="Planejador",
        role=AgentRole.PLANNER,
        system_prompt="""Você é um arquiteto de software experiente.
Sua função é:
- Analisar requisitos de software
- Criar planos de implementação detalhados
- Dividir tarefas em subtarefas menores
- Identificar dependências e riscos

Sempre responda de forma estruturada com:
1. Análise do problema
2. Plano de implementação
3. Tarefas específicas para o time"""
    )

    # Agente Codificador
    coder = Agent(
        name="Codificador",
        role=AgentRole.EXECUTOR,
        system_prompt="""Você é um desenvolvedor Python experiente.
Sua função é:
- Implementar código baseado nas especificações
- Seguir boas práticas de programação
- Escrever código limpo e bem documentado
- Criar funções e classes reutilizáveis

Sempre inclua:
- Docstrings explicativas
- Type hints
- Tratamento de erros básico"""
    )

    # Agente Revisor
    reviewer = Agent(
        name="Revisor",
        role=AgentRole.REVIEWER,
        system_prompt="""Você é um engenheiro de qualidade de software.
Sua função é:
- Revisar código em busca de bugs
- Verificar boas práticas
- Sugerir melhorias de performance
- Avaliar segurança do código

Forneça feedback construtivo com:
1. Pontos positivos
2. Problemas encontrados
3. Sugestões de melhoria"""
    )

    # Agente Testador
    tester = Agent(
        name="Testador",
        role=AgentRole.SPECIALIST,
        system_prompt="""Você é um especialista em testes de software.
Sua função é:
- Criar casos de teste abrangentes
- Identificar edge cases
- Escrever testes unitários
- Validar a cobertura de testes

Sempre inclua:
- Testes para casos normais
- Testes para casos de borda
- Testes de erro"""
    )

    system.add_agent(planner)
    system.add_agent(coder)
    system.add_agent(reviewer)
    system.add_agent(tester)

    return system


def demonstrate_pipeline_pattern():
    """Demonstra o padrão Pipeline de multi-agentes."""

    print("\n" + "=" * 60)
    print("PADRÃO PIPELINE")
    print("=" * 60)

    print("""
    No padrão Pipeline, as tarefas fluem sequencialmente:

    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │Planejador│──▶│Codificador│──▶│ Revisor  │──▶│ Testador │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘

    Cada agente processa e passa para o próximo.
    """)

    system = create_software_development_team()
    task = "Criar uma função que valide endereços de email"

    print(f"\nTarefa: {task}")
    print("-" * 40)

    # 1. Planejador analisa
    print("\n1. PLANEJADOR analisa a tarefa:")
    plan = system.agent_respond("Planejador", task)
    print(f"   {plan[:500]}...")

    # Registra a mensagem
    system.send_message(AgentMessage(
        sender="Planejador",
        receiver="Codificador",
        content=plan,
        message_type="task"
    ))

    # 2. Codificador implementa
    print("\n2. CODIFICADOR implementa:")
    code = system.agent_respond(
        "Codificador",
        f"Implemente baseado neste plano:\n{plan}"
    )
    print(f"   {code[:500]}...")

    system.send_message(AgentMessage(
        sender="Codificador",
        receiver="Revisor",
        content=code,
        message_type="response"
    ))

    # 3. Revisor avalia
    print("\n3. REVISOR avalia o código:")
    review = system.agent_respond(
        "Revisor",
        f"Revise este código:\n{code}"
    )
    print(f"   {review[:500]}...")

    system.send_message(AgentMessage(
        sender="Revisor",
        receiver="Testador",
        content=f"Código:\n{code}\n\nRevisão:\n{review}",
        message_type="feedback"
    ))

    # 4. Testador cria testes
    print("\n4. TESTADOR cria testes:")
    tests = system.agent_respond(
        "Testador",
        f"Crie testes para:\n{code}"
    )
    print(f"   {tests[:500]}...")

    return system


def demonstrate_debate_pattern():
    """Demonstra o padrão Debate de multi-agentes."""

    print("\n" + "=" * 60)
    print("PADRÃO DEBATE")
    print("=" * 60)

    print("""
    No padrão Debate, agentes discutem para chegar a consenso:

         ┌──────────┐
         │Moderador │
         └────┬─────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Expert1│◀▶│Expert2│◀▶│Expert3│
└───────┘ └───────┘ └───────┘

    Agentes debatem até convergir em uma solução.
    """)

    system = MultiAgentSystem(name="debate-system")

    # Criar especialistas com perspectivas diferentes
    expert_performance = Agent(
        name="Expert_Performance",
        role=AgentRole.SPECIALIST,
        system_prompt="""Você é um especialista em performance de software.
Sempre priorize e argumente em favor de:
- Velocidade de execução
- Uso eficiente de memória
- Otimização de algoritmos
Defenda suas posições com argumentos técnicos."""
    )

    expert_maintainability = Agent(
        name="Expert_Manutenibilidade",
        role=AgentRole.SPECIALIST,
        system_prompt="""Você é um especialista em manutenibilidade de código.
Sempre priorize e argumente em favor de:
- Código legível e claro
- Documentação adequada
- Padrões de projeto
Defenda suas posições com argumentos técnicos."""
    )

    expert_security = Agent(
        name="Expert_Seguranca",
        role=AgentRole.SPECIALIST,
        system_prompt="""Você é um especialista em segurança de software.
Sempre priorize e argumente em favor de:
- Validação de entrada
- Proteção contra vulnerabilidades
- Princípio do menor privilégio
Defenda suas posições com argumentos técnicos."""
    )

    moderator = Agent(
        name="Moderador",
        role=AgentRole.ORCHESTRATOR,
        system_prompt="""Você é um moderador de discussões técnicas.
Sua função é:
- Sintetizar diferentes perspectivas
- Identificar pontos de consenso
- Propor soluções balanceadas
- Facilitar a tomada de decisão"""
    )

    system.add_agent(expert_performance)
    system.add_agent(expert_maintainability)
    system.add_agent(expert_security)
    system.add_agent(moderator)

    question = "Qual a melhor forma de implementar autenticação de usuários?"

    print(f"\nQuestão para debate: {question}")
    print("-" * 40)

    # Cada especialista dá sua opinião
    print("\n1. Expert Performance:")
    perf_opinion = system.agent_respond("Expert_Performance", question)
    print(f"   {perf_opinion[:400]}...")

    print("\n2. Expert Manutenibilidade:")
    maint_opinion = system.agent_respond("Expert_Manutenibilidade", question)
    print(f"   {maint_opinion[:400]}...")

    print("\n3. Expert Segurança:")
    sec_opinion = system.agent_respond("Expert_Seguranca", question)
    print(f"   {sec_opinion[:400]}...")

    # Moderador sintetiza
    print("\n4. MODERADOR sintetiza:")
    synthesis_prompt = f"""
Sintetize as seguintes opiniões sobre: {question}

Performance: {perf_opinion}

Manutenibilidade: {maint_opinion}

Segurança: {sec_opinion}

Proponha uma solução balanceada que considere todas as perspectivas.
"""
    synthesis = system.agent_respond("Moderador", synthesis_prompt)
    print(f"   {synthesis[:600]}...")

    return system


def demonstrate_hierarchical_pattern():
    """Demonstra o padrão Hierárquico de multi-agentes."""

    print("\n" + "=" * 60)
    print("PADRÃO HIERÁRQUICO")
    print("=" * 60)

    print("""
    No padrão Hierárquico, agentes são organizados em níveis:

                    ┌──────────────┐
                    │   Diretor    │
                    │  (Nível 1)   │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  Gerente   │  │  Gerente   │  │  Gerente   │
    │  Backend   │  │  Frontend  │  │   DevOps   │
    │ (Nível 2)  │  │ (Nível 2)  │  │ (Nível 2)  │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │               │               │
       ┌──┴──┐         ┌──┴──┐         ┌──┴──┐
       ▼     ▼         ▼     ▼         ▼     ▼
    Dev1   Dev2     Dev3   Dev4     Dev5   Dev6
    (N3)   (N3)     (N3)   (N3)     (N3)   (N3)

    Delegação top-down, relatórios bottom-up.
    """)

    system = MultiAgentSystem(name="hierarchical-system")

    # Diretor (Nível 1)
    director = Agent(
        name="Diretor",
        role=AgentRole.ORCHESTRATOR,
        system_prompt="""Você é o diretor técnico de uma empresa.
Sua função é:
- Definir a visão técnica geral
- Delegar tarefas para gerentes
- Tomar decisões estratégicas
- Resolver conflitos entre áreas

Sempre delegue tarefas específicas para cada área."""
    )

    # Gerentes (Nível 2)
    backend_manager = Agent(
        name="Gerente_Backend",
        role=AgentRole.PLANNER,
        system_prompt="""Você é o gerente de backend.
Responsável por:
- APIs e serviços
- Bancos de dados
- Integrações
- Performance do servidor

Receba tarefas do diretor e distribua para sua equipe."""
    )

    frontend_manager = Agent(
        name="Gerente_Frontend",
        role=AgentRole.PLANNER,
        system_prompt="""Você é o gerente de frontend.
Responsável por:
- Interface do usuário
- Experiência do usuário
- Performance do cliente
- Acessibilidade

Receba tarefas do diretor e distribua para sua equipe."""
    )

    system.add_agent(director)
    system.add_agent(backend_manager)
    system.add_agent(frontend_manager)

    project = "Desenvolver um sistema de e-commerce"

    print(f"\nProjeto: {project}")
    print("-" * 40)

    # Diretor define a estratégia
    print("\n1. DIRETOR define estratégia:")
    strategy = system.agent_respond(
        "Diretor",
        f"Defina a estratégia e delegue tarefas para: {project}"
    )
    print(f"   {strategy[:500]}...")

    # Gerentes recebem e planejam
    print("\n2. GERENTE BACKEND planeja:")
    backend_plan = system.agent_respond(
        "Gerente_Backend",
        f"Baseado na estratégia do diretor, planeje as tarefas de backend:\n{strategy}"
    )
    print(f"   {backend_plan[:400]}...")

    print("\n3. GERENTE FRONTEND planeja:")
    frontend_plan = system.agent_respond(
        "Gerente_Frontend",
        f"Baseado na estratégia do diretor, planeje as tarefas de frontend:\n{strategy}"
    )
    print(f"   {frontend_plan[:400]}...")

    return system


def main():
    print("=" * 60)
    print("APLICAÇÕES MULTI-AGENTE COM LLMs")
    print("=" * 60)

    print("""
    Sistemas multi-agente permitem que múltiplos agentes de IA
    colaborem para resolver tarefas complexas.

    Principais padrões:

    1. PIPELINE - Processamento sequencial
       Input → Agente1 → Agente2 → Agente3 → Output

    2. DEBATE - Discussão para consenso
       Agentes com perspectivas diferentes debatem

    3. HIERÁRQUICO - Organização em níveis
       Delegação top-down, relatórios bottom-up

    4. ORQUESTRADOR - Coordenação central
       Um agente central coordena os demais
    """)

    # Demonstrar cada padrão
    print("\n" + "=" * 60)
    print("CRIANDO TIME DE DESENVOLVIMENTO")
    print("=" * 60)

    system = create_software_development_team()

    # Demonstrar padrões (cada um faz chamadas à API)
    demonstrate_pipeline_pattern()
    demonstrate_debate_pattern()
    demonstrate_hierarchical_pattern()

    print("\n" + "=" * 60)
    print("BENEFÍCIOS DE MULTI-AGENTES")
    print("=" * 60)

    print("""
    1. Especialização
       - Cada agente foca em uma área específica
       - Prompts mais direcionados e efetivos

    2. Escalabilidade
       - Adicione novos agentes conforme necessário
       - Paralelize tarefas independentes

    3. Qualidade
       - Revisão cruzada entre agentes
       - Múltiplas perspectivas no problema

    4. Manutenibilidade
       - Agentes modulares e independentes
       - Fácil de atualizar ou substituir

    5. Rastreabilidade
       - Histórico de comunicação entre agentes
       - Auditoria de decisões
    """)

    print("\n" + "=" * 60)
    print("FRAMEWORKS POPULARES")
    print("=" * 60)

    print("""
    Para implementar multi-agentes em produção:

    1. LangGraph (LangChain)
       - Grafos de agentes com estados
       - Integração com LangChain
       - pip install langgraph

    2. AutoGen (Microsoft)
       - Conversação multi-agente
       - Execução de código
       - pip install pyautogen

    3. CrewAI
       - Equipes de agentes com papéis
       - Tarefas e processos
       - pip install crewai

    4. Swarm (OpenAI)
       - Framework experimental
       - Handoffs entre agentes
       - Leve e educacional
    """)

    print("\nFim do demo de Multi-Agentes")
    print("=" * 60)


if __name__ == "__main__":
    main()
