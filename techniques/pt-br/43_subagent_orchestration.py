"""
Técnica 43: Orquestração de Subagentes com Isolamento de Contexto

Demonstra orquestração de múltiplos subagentes, cada um com janelas de contexto
isoladas, para evitar degradação de contexto e permitir processamento paralelo.

Conceitos-chave da FullCycle AI Tech Week (Aula 2):
- Cada subagente tem sua PRÓPRIA janela de contexto (isolada)
- Agente principal só recebe RESULTADOS (não contexto completo)
- Isso amplifica o número de janelas de contexto disponíveis
- Permite desenvolvimento paralelo com múltiplos agentes

Casos de uso:
- Tarefas de desenvolvimento de longa duração
- Desenvolvimento paralelo de features
- Workflows complexos com múltiplos passos
- Evitar limites de janela de contexto
"""

import json
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class SubagentResult:
    """Resultado de uma execução de subagente."""
    agent_name: str
    task: str
    result: str
    key_decisions: list[str]
    files_created: list[str]
    execution_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class Subagent:
    """
    Um subagente especializado com seu próprio contexto isolado.

    Insight chave: Cada subagente começa com uma janela de contexto NOVA.
    Ele não herda o contexto completo do agente principal - apenas o que é
    explicitamente passado para ele. Isso previne degradação de contexto.
    """
    name: str
    specialty: str
    system_prompt: str
    messages: list[dict] = field(default_factory=list)

    def __post_init__(self):
        # Começa com contexto novo - apenas system prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def execute(self, task: str, context: str = "") -> SubagentResult:
        """
        Executa uma tarefa com contexto isolado.

        O subagente só sabe:
        1. Sua especialidade/system prompt
        2. A tarefa específica
        3. Contexto mínimo passado explicitamente

        Ele NÃO sabe o histórico completo da conversa!
        """
        import time
        start_time = time.time()

        # Constrói mensagem de tarefa com apenas contexto necessário
        task_content = f"""## Tarefa
{task}

## Contexto (de passos anteriores)
{context if context else "Esta é a primeira tarefa, sem contexto anterior."}

## Instruções
1. Complete a tarefa de acordo com sua especialidade
2. Seja específico e pronto para implementação
3. Documente decisões-chave tomadas
4. Liste arquivos que você criaria

Responda com JSON:
{{
    "result": "implementação/solução detalhada",
    "key_decisions": ["decisão 1", "decisão 2"],
    "files_created": ["arquivo1.py", "arquivo2.py"]
}}"""

        self.messages.append({"role": "user", "content": task_content})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                response_format={"type": "json_object"}
            )

            result_data = json.loads(response.choices[0].message.content)
            execution_time = time.time() - start_time

            return SubagentResult(
                agent_name=self.name,
                task=task,
                result=result_data.get("result", ""),
                key_decisions=result_data.get("key_decisions", []),
                files_created=result_data.get("files_created", []),
                execution_time=execution_time,
                success=True
            )
        except Exception as e:
            return SubagentResult(
                agent_name=self.name,
                task=task,
                result="",
                key_decisions=[],
                files_created=[],
                execution_time=time.time() - start_time,
                success=False,
                error=str(e)
            )


class SubagentOrchestrator:
    """
    Orquestra múltiplos subagentes para tarefas complexas.

    Padrão chave da FullCycle:
    - Orquestrador principal tem contexto mínimo
    - Cada subagente recebe contexto novo
    - Apenas resultados são agregados
    - Pode rodar múltiplos agentes em paralelo
    """

    def __init__(self):
        self.subagents: dict[str, Subagent] = {}
        self.results: list[SubagentResult] = []
        self.accumulated_context: str = ""

    def register_subagent(self, name: str, specialty: str, system_prompt: str):
        """Registra um subagente especializado."""
        self.subagents[name] = Subagent(
            name=name,
            specialty=specialty,
            system_prompt=system_prompt
        )

    def execute_sequential(self, tasks: list[tuple[str, str]]) -> list[SubagentResult]:
        """
        Executa tarefas sequencialmente, passando resultados entre passos.

        Args:
            tasks: Lista de tuplas (agent_name, task_description)
        """
        results = []

        for agent_name, task in tasks:
            agent = self.subagents.get(agent_name)
            if not agent:
                print(f"Aviso: Agente '{agent_name}' não encontrado")
                continue

            print(f"\n🤖 Executando com {agent_name}: {task[:50]}...")

            # Passa contexto acumulado (apenas resultados, não histórico completo)
            result = agent.execute(task, self.accumulated_context)
            results.append(result)

            if result.success:
                # Adiciona apenas info chave ao contexto (não tudo!)
                self.accumulated_context += f"\n\n## Completado: {task}\n"
                self.accumulated_context += f"Decisões chave: {', '.join(result.key_decisions)}\n"
                self.accumulated_context += f"Arquivos: {', '.join(result.files_created)}\n"
                print(f"   ✅ Completado em {result.execution_time:.2f}s")
            else:
                print(f"   ❌ Falhou: {result.error}")

        return results

    def execute_parallel(self, tasks: list[tuple[str, str, str]]) -> list[SubagentResult]:
        """
        Executa tarefas independentes em paralelo.

        Cada tarefa recebe sua própria instância de subagente = janela de contexto nova.

        Args:
            tasks: Lista de tuplas (agent_type, task_description, shared_context)
        """
        print("\n🚀 Executando tarefas em paralelo (janelas de contexto separadas)...")

        results = []
        for agent_name, task, context in tasks:
            # Cria instância nova de agente para cada tarefa
            template = self.subagents.get(agent_name)
            if not template:
                continue

            fresh_agent = Subagent(
                name=f"{template.name}-{len(results)}",
                specialty=template.specialty,
                system_prompt=template.system_prompt
            )

            result = fresh_agent.execute(task, context)
            results.append(result)

            status = "✅" if result.success else "❌"
            print(f"   {status} {fresh_agent.name}: {task[:40]}...")

        return results


def create_development_agents() -> SubagentOrchestrator:
    """Cria um conjunto de subagentes de desenvolvimento especializados."""
    orchestrator = SubagentOrchestrator()

    # Agente Backend
    orchestrator.register_subagent(
        name="backend",
        specialty="Desenvolvimento Backend",
        system_prompt="""Você é um especialista em desenvolvimento backend.
Foque em:
- Padrões de arquitetura limpa
- Boas práticas de design de API
- Modelagem de banco de dados
- Tratamento de erros
- Considerações de segurança

Seja específico e pronto para implementação em suas respostas."""
    )

    # Agente Frontend
    orchestrator.register_subagent(
        name="frontend",
        specialty="Desenvolvimento Frontend",
        system_prompt="""Você é um especialista em desenvolvimento frontend.
Foque em:
- Arquitetura de componentes
- Gerenciamento de estado
- Experiência do usuário
- Acessibilidade
- Otimização de performance

Seja específico e pronto para implementação em suas respostas."""
    )

    # Agente de Testes
    orchestrator.register_subagent(
        name="testing",
        specialty="Testes & Qualidade",
        system_prompt="""Você é um especialista em testes e garantia de qualidade.
Foque em:
- Design de testes unitários
- Testes de integração
- Cobertura de testes
- Casos extremos
- Gerenciamento de dados de teste

Seja específico e pronto para implementação em suas respostas."""
    )

    # Agente DevOps
    orchestrator.register_subagent(
        name="devops",
        specialty="DevOps & Infraestrutura",
        system_prompt="""Você é um especialista em DevOps e infraestrutura.
Foque em:
- Pipelines CI/CD
- Configuração de containers
- Monitoramento & logging
- Hardening de segurança
- Estratégias de deployment

Seja específico e pronto para implementação em suas respostas."""
    )

    return orchestrator


def demonstrate_sequential():
    """Demonstra execução sequencial com passagem de contexto."""
    print("=" * 60)
    print("Execução Sequencial com Passagem de Contexto")
    print("=" * 60)

    print("""
Fluxo:
┌──────────────────────────────────────────────────────────┐
│ Tarefa 1 (Backend) ─────► Resultado 1                    │
│                              │                           │
│                              ▼                           │
│ Tarefa 2 (Frontend) ◄──── Contexto do Resultado 1       │
│         │                                                │
│         ▼                                                │
│ Tarefa 3 (Testes) ◄──── Contexto dos Resultados 1,2     │
└──────────────────────────────────────────────────────────┘

Cada agente tem contexto NOVO, recebendo apenas info essencial.
""")

    orchestrator = create_development_agents()

    tasks = [
        ("backend", "Crie um modelo User com campos: id, email, name, created_at"),
        ("backend", "Crie um endpoint POST /users para criar novos usuários"),
        ("testing", "Escreva testes unitários para o modelo User e endpoint"),
    ]

    results = orchestrator.execute_sequential(tasks)

    print("\n" + "=" * 60)
    print("Resumo dos Resultados")
    print("=" * 60)

    for result in results:
        print(f"\n📦 {result.agent_name}: {result.task[:40]}...")
        print(f"   Sucesso: {result.success}")
        print(f"   Tempo: {result.execution_time:.2f}s")
        print(f"   Decisões: {result.key_decisions[:2] if result.key_decisions else 'Nenhuma'}")
        print(f"   Arquivos: {result.files_created[:3] if result.files_created else 'Nenhum'}")


def demonstrate_parallel():
    """Demonstra execução paralela para tarefas independentes."""
    print("\n" + "=" * 60)
    print("Execução Paralela com Contextos Isolados")
    print("=" * 60)

    print("""
Fluxo:
┌──────────────────────────────────────────────────────────┐
│             Fundação Completa                            │
│                    │                                     │
│        ┌──────────┼──────────┬──────────┐               │
│        ▼          ▼          ▼          ▼               │
│    Feature A  Feature B  Feature C  Feature D           │
│   (Backend)  (Backend)  (Frontend) (DevOps)             │
│        │          │          │          │               │
│        └──────────┴──────────┴──────────┘               │
│                    │                                     │
│              Agregação                                   │
└──────────────────────────────────────────────────────────┘

Todas as features executam com janelas de contexto SEPARADAS!
10 agentes = 10 janelas de contexto = Sem degradação
""")

    orchestrator = create_development_agents()

    # Contexto compartilhado da fundação
    foundation_context = """
## Fundação Completa
- Modelo User criado com id, email, name, created_at
- Schema do banco de dados inicializado
- Estrutura base da API pronta
"""

    # Tarefas independentes que podem rodar em paralelo
    parallel_tasks = [
        ("backend", "Implemente autenticação de usuário com JWT", foundation_context),
        ("backend", "Implemente endpoint de atualização de perfil de usuário", foundation_context),
        ("frontend", "Crie componente de formulário de login", foundation_context),
        ("devops", "Configure Docker para a API", foundation_context),
    ]

    results = orchestrator.execute_parallel(parallel_tasks)

    print("\n" + "=" * 60)
    print("Resumo dos Resultados Paralelos")
    print("=" * 60)

    total_time = sum(r.execution_time for r in results)
    print(f"\nTempo total de execução: {total_time:.2f}s")
    print(f"Se sequencial: ~{total_time:.2f}s")
    print(f"Vantagem do paralelo: Cada agente teve contexto novo!")

    for result in results:
        print(f"\n📦 {result.agent_name}")
        print(f"   Tarefa: {result.task[:40]}...")
        print(f"   Sucesso: {result.success}")


def demonstrate_feature_breakdown():
    """Demonstra o padrão de divisão de features da FullCycle."""
    print("\n" + "=" * 60)
    print("Padrão de Divisão de Features")
    print("(Da FullCycle: Planejamento permite paralelização)")
    print("=" * 60)

    print("""
Insight do Wesley: Planeje suas features para identificar dependências.

Exemplo de Divisão de Features:
┌─────────────────────────────────────────────────────────┐
│ Feature 1: Fundação ──► DEVE ser primeiro (dependência) │
│     │                                                   │
│     ▼                                                   │
│ Feature 2: Auth ──► Depende da Feature 1               │
│     │                                                   │
│     ▼                                                   │
│ Feature 3: UI Base ──► Depende da Feature 2            │
│     │                                                   │
│ ════════════ ZONA PARALELA ════════════                │
│     │                                                   │
│     ├──► Feature 4: Perfil de Usuário (independente)   │
│     ├──► Feature 5: Configurações (independente)       │
│     ├──► Feature 6: Dashboard (independente)           │
│     └──► Feature 7: Relatórios (independente)          │
│                                                         │
│ Após Feature 3, todas as outras podem rodar PARALELO!  │
└─────────────────────────────────────────────────────────┘

"Se eu tenho 10 agentes, posso rodar 10 features simultaneamente"
- Wesley Williams, FullCycle AI Tech Week
""")

    orchestrator = create_development_agents()

    # Fase 1: Dependências sequenciais
    print("\n📍 Fase 1: Sequencial (Dependências)")
    sequential_tasks = [
        ("backend", "Crie schema do banco de dados e modelo User"),
        ("backend", "Implemente sistema de autenticação"),
    ]
    orchestrator.execute_sequential(sequential_tasks)

    # Fase 2: Features independentes paralelas
    print("\n📍 Fase 2: Paralelo (Features Independentes)")
    foundation = orchestrator.accumulated_context

    parallel_tasks = [
        ("backend", "Implemente gerenciamento de perfil de usuário", foundation),
        ("backend", "Implemente configurações/preferências", foundation),
        ("frontend", "Crie layout do dashboard", foundation),
        ("testing", "Escreva testes de integração para auth", foundation),
    ]
    orchestrator.execute_parallel(parallel_tasks)


def main():
    print("=" * 60)
    print("Orquestração de Subagentes com Isolamento de Contexto")
    print("Conceitos-Chave da FullCycle AI Tech Week")
    print("=" * 60)

    print("""
Por que Subagentes?

1. Cada subagente tem sua PRÓPRIA janela de contexto
   - Agente principal: Janela de Contexto A
   - Subagente 1: Janela de Contexto B (nova!)
   - Subagente 2: Janela de Contexto C (nova!)

2. Isso MULTIPLICA sua capacidade de contexto
   - Em vez de 1 janela ficando cheia e sumarizada
   - Você tem N janelas novas

3. Permite paralelização
   - 10 branches em paralelo = 10 agentes trabalhando
   - Cada um com contexto novo = sem degradação

4. Agregação de resultados
   - Agente principal só recebe RESULTADOS
   - Não o histórico completo da conversa
   - Mantém contexto principal limpo
""")

    # Demo 1: Execução sequencial
    demonstrate_sequential()

    # Demo 2: Execução paralela
    demonstrate_parallel()

    # Demo 3: Padrão de divisão de features
    demonstrate_feature_breakdown()

    print("\n" + "=" * 60)
    print("Principais Aprendizados")
    print("=" * 60)
    print("""
1. Subagentes têm janelas de contexto ISOLADAS
2. Passe apenas RESULTADOS, não contexto completo
3. Planeje features para identificar oportunidades de paralelização
4. Sequencial para dependências, paralelo para independentes
5. É assim que você escala sem degradação de contexto

"Você pode ter 10 sessões do Cloud Code trabalhando em 10 branches"
- Wesley Williams, FullCycle AI Tech Week
""")


if __name__ == "__main__":
    main()
