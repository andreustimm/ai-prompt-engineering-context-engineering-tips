"""
Técnica 44: Memória Compartilhada Entre Agentes

Demonstra um padrão de Memory DB onde múltiplos agentes podem compartilhar
conhecimento, incluindo erros/problemas encontrados, para aprendizado
cruzado e feedback entre agentes.

Conceitos-chave da FullCycle AI Tech Week (Aula 2):
- Memory DB permite que agentes compartilhem conhecimento
- Categorias de memória de curto, médio e longo prazo
- Loop de feedback: agente A encontra problema → armazena → agente B evita
- Busca semântica sobre memórias compartilhadas

Casos de uso:
- Equipes de desenvolvimento multi-agente
- Reconhecimento de padrões de erro
- Aprendizado entre projetos
- Base de conhecimento organizacional
"""

import json
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class Memory:
    """Uma única entrada de memória no sistema de memória compartilhada."""
    id: str
    content: str
    memory_type: str  # "short", "medium", "long"
    category: str  # "error", "solution", "pattern", "decision", "context"
    source_agent: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    access_count: int = 0
    relevance_score: float = 1.0

    def is_expired(self) -> bool:
        """Verifica se a memória expirou."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        """Converte para dicionário para armazenamento."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "category": self.category,
            "source_agent": self.source_agent,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "relevance_score": self.relevance_score,
        }


class MemoryDB:
    """
    Banco de dados de memória compartilhada para sistemas multi-agente.

    Insight chave da FullCycle: Memória é categorizada por duração:
    - Curto prazo: Contexto da tarefa atual (expira rapidamente)
    - Médio prazo: Contexto de sessão/projeto (dias)
    - Longo prazo: Conhecimento organizacional (permanente)
    """

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.embeddings_cache: dict[str, list[float]] = {}

    def _generate_id(self, content: str) -> str:
        """Gera ID único para memória."""
        return hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    def _get_expiration(self, memory_type: str) -> Optional[datetime]:
        """Obtém tempo de expiração baseado no tipo de memória."""
        now = datetime.now()
        if memory_type == "short":
            return now + timedelta(hours=1)
        elif memory_type == "medium":
            return now + timedelta(days=7)
        else:  # longo prazo
            return None  # Nunca expira

    def store(
        self,
        content: str,
        memory_type: str,
        category: str,
        source_agent: str,
        metadata: dict = None
    ) -> Memory:
        """
        Armazena uma nova memória.

        Args:
            content: O conteúdo da memória
            memory_type: "short", "medium" ou "long"
            category: "error", "solution", "pattern", "decision", "context"
            source_agent: Nome do agente armazenando a memória
            metadata: Metadados adicionais
        """
        memory = Memory(
            id=self._generate_id(content),
            content=content,
            memory_type=memory_type,
            category=category,
            source_agent=source_agent,
            created_at=datetime.now(),
            expires_at=self._get_expiration(memory_type),
            metadata=metadata or {},
        )
        self.memories[memory.id] = memory
        return memory

    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 5
    ) -> list[Memory]:
        """
        Recupera memórias baseado em query e filtros.

        Usa matching semântico via LLM para encontrar memórias relevantes.
        """
        # Filtra por categoria e tipo
        candidates = [
            m for m in self.memories.values()
            if not m.is_expired()
            and (category is None or m.category == category)
            and (memory_type is None or m.memory_type == memory_type)
        ]

        if not candidates:
            return []

        # Usa LLM para ranquear relevância
        memories_text = "\n".join([
            f"[{m.id}] ({m.category}): {m.content[:200]}"
            for m in candidates
        ])

        ranking_prompt = f"""Dada a query e memórias, ranqueie as memórias por relevância.
Retorne um objeto JSON com IDs de memória e scores de relevância (0-1).

Query: {query}

Memórias:
{memories_text}

Retorne: {{"rankings": [{{"id": "...", "score": 0.9}}, ...]}}
Inclua apenas memórias com score > 0.3"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": ranking_prompt}],
            response_format={"type": "json_object"}
        )

        rankings = json.loads(response.choices[0].message.content)

        # Atualiza contadores de acesso e retorna memórias ranqueadas
        results = []
        for ranking in rankings.get("rankings", [])[:limit]:
            memory_id = ranking.get("id")
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.access_count += 1
                memory.relevance_score = ranking.get("score", 0.5)
                results.append(memory)

        return results

    def get_recent(self, limit: int = 10) -> list[Memory]:
        """Obtém memórias mais recentes."""
        valid = [m for m in self.memories.values() if not m.is_expired()]
        return sorted(valid, key=lambda m: m.created_at, reverse=True)[:limit]

    def get_by_category(self, category: str) -> list[Memory]:
        """Obtém todas as memórias em uma categoria."""
        return [
            m for m in self.memories.values()
            if m.category == category and not m.is_expired()
        ]

    def cleanup_expired(self) -> int:
        """Remove memórias expiradas."""
        expired_ids = [
            m.id for m in self.memories.values() if m.is_expired()
        ]
        for mid in expired_ids:
            del self.memories[mid]
        return len(expired_ids)

    def get_stats(self) -> dict:
        """Obtém estatísticas de memória."""
        valid = [m for m in self.memories.values() if not m.is_expired()]
        return {
            "total_memories": len(valid),
            "by_type": {
                "short": len([m for m in valid if m.memory_type == "short"]),
                "medium": len([m for m in valid if m.memory_type == "medium"]),
                "long": len([m for m in valid if m.memory_type == "long"]),
            },
            "by_category": {
                cat: len([m for m in valid if m.category == cat])
                for cat in ["error", "solution", "pattern", "decision", "context"]
            },
            "total_accesses": sum(m.access_count for m in valid),
        }


class MemoryAwareAgent:
    """
    Um agente que usa memória compartilhada para aprendizado e feedback.

    Insight chave: Agentes podem aprender das experiências uns dos outros
    sem compartilhar contexto completo - apenas armazenando aprendizados chave.
    """

    def __init__(self, name: str, memory_db: MemoryDB, specialty: str = "geral"):
        self.name = name
        self.memory_db = memory_db
        self.specialty = specialty

    def execute_with_memory(self, task: str) -> dict:
        """
        Executa uma tarefa enquanto utiliza e contribui para memória compartilhada.

        Fluxo:
        1. Recupera memórias relevantes antes de começar
        2. Executa tarefa com contexto de memória
        3. Armazena aprendizados de volta na memória
        """
        # Passo 1: Recupera memórias relevantes
        relevant_memories = self.memory_db.recall(
            query=task,
            limit=5
        )

        memory_context = ""
        if relevant_memories:
            memory_context = "\n\n## Memórias Relevantes de Outros Agentes:\n"
            for mem in relevant_memories:
                memory_context += f"- [{mem.category}] {mem.content}\n"
                memory_context += f"  (de {mem.source_agent}, acessado {mem.access_count}x)\n"

        # Passo 2: Executa tarefa
        execution_prompt = f"""Você é {self.name}, um especialista em {self.specialty}.

Execute esta tarefa e aprenda das experiências anteriores de agentes.

## Tarefa:
{task}
{memory_context}

## Instruções:
1. Considere memórias relevantes ao tomar decisões
2. Execute a tarefa
3. Anote erros encontrados ou soluções encontradas
4. Documente decisões-chave e padrões

Retorne JSON:
{{
    "result": "seu resultado de execução",
    "errors_found": ["erro 1", ...],
    "solutions_applied": ["solução 1", ...],
    "patterns_identified": ["padrão 1", ...],
    "key_decisions": ["decisão 1", ...]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": execution_prompt}],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Passo 3: Armazena aprendizados de volta na memória
        self._store_learnings(task, result)

        return result

    def _store_learnings(self, task: str, result: dict):
        """Armazena aprendizados da execução de tarefa na memória compartilhada."""

        # Armazena erros como curto prazo (pode ser específico da tarefa)
        for error in result.get("errors_found", []):
            self.memory_db.store(
                content=f"Erro na tarefa '{task[:50]}': {error}",
                memory_type="short",
                category="error",
                source_agent=self.name,
                metadata={"task": task}
            )

        # Armazena soluções como médio prazo (útil para tarefas similares)
        for solution in result.get("solutions_applied", []):
            self.memory_db.store(
                content=solution,
                memory_type="medium",
                category="solution",
                source_agent=self.name,
                metadata={"task": task}
            )

        # Armazena padrões como longo prazo (conhecimento organizacional)
        for pattern in result.get("patterns_identified", []):
            self.memory_db.store(
                content=pattern,
                memory_type="long",
                category="pattern",
                source_agent=self.name,
                metadata={"task": task}
            )

        # Armazena decisões-chave como médio prazo
        for decision in result.get("key_decisions", []):
            self.memory_db.store(
                content=decision,
                memory_type="medium",
                category="decision",
                source_agent=self.name,
                metadata={"task": task}
            )


def demonstrate_shared_memory():
    """Demonstra o padrão de memória compartilhada com múltiplos agentes."""
    print("=" * 60)
    print("Memória Compartilhada Entre Agentes")
    print("=" * 60)

    print("""
Fluxo:
┌─────────────────────────────────────────────────────────────┐
│                      Memory DB                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ Curto   │  │ Médio   │  │ Longo   │                     │
│  │ (1hr)   │  │ (7dias) │  │ (perm)  │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │                           │
│       └────────────┴────────────┘                           │
│                    │                                         │
│      ┌─────────────┼─────────────┐                          │
│      │             │             │                          │
│      ▼             ▼             ▼                          │
│  ┌───────┐    ┌───────┐    ┌───────┐                       │
│  │Agente │    │Agente │    │Agente │                       │
│  │Backend│    │Frontend│   │Testes │                       │
│  └───────┘    └───────┘    └───────┘                       │
│                                                              │
│  Agente A encontra erro → armazena → Agente B evita        │
└─────────────────────────────────────────────────────────────┘
""")

    # Cria memória compartilhada
    memory_db = MemoryDB()

    # Cria agentes que compartilham memória
    backend_agent = MemoryAwareAgent("AgenteBackend", memory_db, "desenvolvimento backend")
    frontend_agent = MemoryAwareAgent("AgenteFrontend", memory_db, "desenvolvimento frontend")
    testing_agent = MemoryAwareAgent("AgenteTestes", memory_db, "testes e QA")

    # Tarefa 1: Agente backend trabalha na API
    print("\n" + "=" * 60)
    print("Tarefa 1: Agente Backend - Criar API de Usuário")
    print("=" * 60)

    result1 = backend_agent.execute_with_memory(
        "Crie um endpoint de API REST para cadastro de usuário com validação de email"
    )
    print(f"\nResultado: {result1.get('result', '')[:200]}...")
    print(f"Erros encontrados: {result1.get('errors_found', [])}")
    print(f"Soluções: {result1.get('solutions_applied', [])}")

    # Tarefa 2: Agente frontend - pode aprender da experiência do backend
    print("\n" + "=" * 60)
    print("Tarefa 2: Agente Frontend - Criar Formulário de Cadastro")
    print("(Deve se beneficiar dos aprendizados do Backend)")
    print("=" * 60)

    result2 = frontend_agent.execute_with_memory(
        "Crie um formulário de cadastro de usuário que chama a API de cadastro"
    )
    print(f"\nResultado: {result2.get('result', '')[:200]}...")
    print(f"Padrões identificados: {result2.get('patterns_identified', [])}")

    # Tarefa 3: Agente de testes - aprende de ambos
    print("\n" + "=" * 60)
    print("Tarefa 3: Agente de Testes - Escrever Testes de Cadastro")
    print("(Deve se beneficiar dos aprendizados do Backend e Frontend)")
    print("=" * 60)

    result3 = testing_agent.execute_with_memory(
        "Escreva testes de integração para o fluxo de cadastro de usuário"
    )
    print(f"\nResultado: {result3.get('result', '')[:200]}...")
    print(f"Decisões-chave: {result3.get('key_decisions', [])}")

    # Mostra estatísticas de memória
    print("\n" + "=" * 60)
    print("Estatísticas do Memory DB")
    print("=" * 60)

    stats = memory_db.get_stats()
    print(f"\nTotal de memórias: {stats['total_memories']}")
    print(f"\nPor tipo:")
    for mtype, count in stats['by_type'].items():
        print(f"  {mtype}: {count}")
    print(f"\nPor categoria:")
    for cat, count in stats['by_category'].items():
        if count > 0:
            print(f"  {cat}: {count}")

    # Consulta memórias
    print("\n" + "=" * 60)
    print("Consultando Memória Compartilhada")
    print("=" * 60)

    patterns = memory_db.get_by_category("pattern")
    print(f"\nPadrões de longo prazo aprendidos ({len(patterns)}):")
    for p in patterns[:5]:
        print(f"  - {p.content[:100]}...")
        print(f"    (de {p.source_agent})")


def demonstrate_feedback_loop():
    """Demonstra o padrão de loop de feedback."""
    print("\n" + "=" * 60)
    print("Padrão de Loop de Feedback")
    print("=" * 60)

    print("""
Loop de Feedback:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Agente A executa tarefa                                 │
│      │                                                   │
│      ▼                                                   │
│  Encontra erro/problema                                  │
│      │                                                   │
│      ▼                                                   │
│  Armazena no Memory DB                                   │
│      │                                                   │
│      ▼                                                   │
│  Agente B começa tarefa similar                          │
│      │                                                   │
│      ▼                                                   │
│  Recupera memórias relevantes                            │
│      │                                                   │
│      ▼                                                   │
│  EVITA o mesmo erro!                                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
""")

    memory_db = MemoryDB()

    # Semeia com alguns erros conhecidos
    memory_db.store(
        content="TypeError ao comparar None com string na validação - sempre verifique None primeiro",
        memory_type="long",
        category="error",
        source_agent="AgenteAnterior"
    )

    memory_db.store(
        content="Rate limiting deve ser implementado no API gateway, não no código da aplicação",
        memory_type="long",
        category="pattern",
        source_agent="AgenteArquiteto"
    )

    memory_db.store(
        content="Use Pydantic para validação de request - captura erros cedo com mensagens claras",
        memory_type="long",
        category="solution",
        source_agent="AgenteBackend"
    )

    # Novo agente se beneficia do conhecimento histórico
    new_agent = MemoryAwareAgent("NovoDesenvolvedor", memory_db, "desenvolvimento full-stack")

    print("\nNovo agente executando tarefa com memórias históricas...")
    result = new_agent.execute_with_memory(
        "Implemente validação de entrada para o endpoint de atualização de perfil de usuário"
    )

    print(f"\nResultado se beneficiou de {len(memory_db.recall('validação', limit=10))} memórias anteriores")
    print(f"Novos padrões descobertos: {result.get('patterns_identified', [])}")


def main():
    print("=" * 60)
    print("Memória Compartilhada Entre Agentes")
    print("Conceitos-Chave da FullCycle AI Tech Week")
    print("=" * 60)

    print("""
Por que Memória Compartilhada?

1. Acumulação de Aprendizado
   - Cada agente contribui para conhecimento coletivo
   - Erros encontrados uma vez são evitados por todos

2. Categorias de Memória
   - Curto prazo: Contexto da tarefa atual (1 hora)
   - Médio prazo: Contexto do projeto (7 dias)
   - Longo prazo: Conhecimento organizacional (permanente)

3. Loop de Feedback
   - Agente A encontra problema → Memory DB
   - Agente B recupera → Evita problema
   - Melhoria contínua

4. Comunicação Entre Agentes
   - Sem compartilhar contexto completo
   - Apenas aprendizados-chave e padrões
""")

    # Demo 1: Memória compartilhada com múltiplos agentes
    demonstrate_shared_memory()

    # Demo 2: Padrão de loop de feedback
    demonstrate_feedback_loop()

    print("\n" + "=" * 60)
    print("Principais Aprendizados")
    print("=" * 60)
    print("""
1. Memory DB permite aprendizado entre agentes
2. Categorize por duração: curto/médio/longo prazo
3. Categorize por tipo: erro/solução/padrão/decisão
4. Busca semântica permite recuperação relevante
5. Loops de feedback melhoram performance coletiva

"Quanto mais agentes usam o sistema, mais inteligente ele se torna"
- FullCycle AI Tech Week
""")


if __name__ == "__main__":
    main()
