"""
Técnica 41: Sistema de Skills para Agentes

Demonstra carregamento de conhecimento sob demanda para agentes de IA através
de um sistema de skills. Skills são carregadas apenas quando necessárias,
economizando tokens e melhorando a precisão.

Conceitos-chave da FullCycle AI Tech Week (Aula 2):
- Skills NÃO são carregadas automaticamente - são sob demanda
- Descrições de skills são como SEO - devem ser otimizadas para a IA encontrá-las
- "Skills invisíveis" = skills que nunca são usadas por causa de descrições ruins
- Skills podem incluir: diretrizes, material de referência, scripts, hooks

Casos de uso:
- Agentes de desenvolvimento que precisam de conhecimento específico por contexto
- Agentes especializados que carregam apenas skills relevantes
- Otimização de tokens através de carregamento lazy
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class Skill:
    """
    Uma skill representa conhecimento sob demanda para um agente de IA.

    A descrição é CRÍTICA - é como SEO para IA.
    Se a descrição é ruim, a skill se torna "invisível" e nunca é usada.
    """
    name: str
    description: str  # CRÍTICO: Isso é como SEO - deve ser otimizado!
    keywords: list[str]  # Palavras-chave adicionais para matching
    content: str  # O conhecimento/diretrizes reais
    scripts: dict[str, str] = field(default_factory=dict)  # Scripts opcionais

    def to_context(self) -> str:
        """Converte skill para string de contexto para a IA."""
        context = f"# Skill: {self.name}\n\n{self.content}"
        if self.scripts:
            context += "\n\n## Scripts Disponíveis:\n"
            for script_name, script_desc in self.scripts.items():
                context += f"- `{script_name}`: {script_desc}\n"
        return context


class SkillRegistry:
    """
    Registro de skills disponíveis com matching semântico.

    Insight chave: A IA decide quais skills carregar baseado nas descrições,
    então descrições devem ser otimizadas como palavras-chave de SEO.
    """

    def __init__(self):
        self.skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """Registra uma nova skill."""
        self.skills[skill.name] = skill

    def get_skill_catalog(self) -> str:
        """Obtém um catálogo de todas as skills (apenas nomes e descrições)."""
        catalog = "# Skills Disponíveis\n\n"
        for name, skill in self.skills.items():
            catalog += f"## {name}\n"
            catalog += f"**Descrição**: {skill.description}\n"
            catalog += f"**Palavras-chave**: {', '.join(skill.keywords)}\n\n"
        return catalog

    def load_skill(self, name: str) -> Optional[Skill]:
        """Carrega uma skill específica pelo nome."""
        return self.skills.get(name)

    def search_skills(self, query: str) -> list[Skill]:
        """Busca skills por matching de palavras-chave."""
        query_lower = query.lower()
        matches = []
        for skill in self.skills.values():
            # Verifica nome, descrição e palavras-chave
            if (query_lower in skill.name.lower() or
                query_lower in skill.description.lower() or
                any(query_lower in kw.lower() for kw in skill.keywords)):
                matches.append(skill)
        return matches


def create_sample_skills() -> SkillRegistry:
    """Cria skills de exemplo demonstrando o conceito."""
    registry = SkillRegistry()

    # Skill 1: Testes Python
    # Nota: Descrição otimizada como SEO - inclui termos-chave que a IA pode buscar
    testing_skill = Skill(
        name="python-testing",
        description=(
            "Diretrizes de testes Python com pytest, padrões testify, testes unitários, "
            "testes de integração, mocking, fixtures, TDD, cobertura de testes, asserções, "
            "testes parametrizados, organização de testes, integração CI/CD de testes"
        ),
        keywords=[
            "teste", "pytest", "unittest", "mock", "fixture", "TDD",
            "cobertura", "assert", "integração", "teste unitário"
        ],
        content="""
## Diretrizes de Testes Python

### Organização de Testes
- Use diretório `tests/` na raiz do projeto
- Espelhe estrutura do código: `tests/unit/`, `tests/integration/`
- Nomeie arquivos de teste: `test_<modulo>.py`
- Nomeie funções de teste: `test_<o_que_testa>`

### Boas Práticas Pytest
1. Use fixtures para setup/teardown
2. Parametrize testes para múltiplas entradas
3. Use markers para categorização (@pytest.mark.slow)
4. Mantenha testes isolados e independentes

### Exemplo de Estrutura
```python
# tests/unit/test_calculator.py
import pytest
from calculator import add

@pytest.fixture
def calculator():
    return Calculator()

@pytest.mark.parametrize("a,b,expected", [(1,2,3), (0,0,0), (-1,1,0)])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

### Requisitos de Cobertura
- Mínimo 80% de cobertura de código
- 100% de cobertura para caminhos críticos
- Execute: `pytest --cov=src --cov-report=html`
""",
        scripts={
            "run_tests.sh": "Executa todos os testes com cobertura",
            "validate_tests.sh": "Valida convenções de nomenclatura de testes",
        }
    )
    registry.register(testing_skill)

    # Skill 2: Desenvolvimento de API
    api_skill = Skill(
        name="api-development",
        description=(
            "Desenvolvimento de API REST com FastAPI, endpoints, rotas, métodos HTTP, "
            "modelos request/response, autenticação, JWT, rate limiting, "
            "documentação OpenAPI, modelos Pydantic, injeção de dependência"
        ),
        keywords=[
            "api", "rest", "fastapi", "endpoint", "rota", "http",
            "jwt", "autenticação", "pydantic", "openapi", "swagger"
        ],
        content="""
## Diretrizes de Desenvolvimento de API

### Estrutura FastAPI
```
app/
  ├── main.py           # Entrada da aplicação
  ├── api/
  │   ├── routes/       # Handlers de rotas
  │   └── deps.py       # Dependências
  ├── models/           # Modelos Pydantic
  ├── services/         # Lógica de negócio
  └── core/             # Config, segurança
```

### Convenções de Endpoints
- Use substantivos no plural: `/users`, `/products`
- Use verbos HTTP corretamente: GET, POST, PUT, DELETE
- Retorne códigos de status apropriados

### Autenticação
- Use JWT para auth stateless
- Implemente refresh tokens
- Rate limit endpoints sensíveis

### Exemplo de Endpoint
```python
from fastapi import APIRouter, Depends, HTTPException
from models import User, UserCreate

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=User, status_code=201)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    return await user_service.create(db, user)
```
""",
        scripts={
            "generate_openapi.sh": "Gera spec OpenAPI",
            "validate_api.sh": "Valida API contra spec",
        }
    )
    registry.register(api_skill)

    # Skill 3: Tratamento de Erros
    error_skill = Skill(
        name="error-handling",
        description=(
            "Padrões de tratamento de erros, hierarquia de exceções, blocos try/except, "
            "exceções customizadas, logging de erros, degradação graciosa, "
            "recuperação de erros, erros de validação, erros de lógica de negócio"
        ),
        keywords=[
            "erro", "exceção", "try", "except", "catch", "throw",
            "logging", "validação", "recuperação", "gracioso"
        ],
        content="""
## Diretrizes de Tratamento de Erros

### Hierarquia de Exceções
```python
class AppError(Exception):
    '''Erro base da aplicação'''
    pass

class ValidationError(AppError):
    '''Validação de entrada falhou'''
    pass

class NotFoundError(AppError):
    '''Recurso não encontrado'''
    pass

class BusinessRuleError(AppError):
    '''Violação de regra de negócio'''
    pass
```

### Boas Práticas
1. Seja específico - capture exceções específicas
2. Faça log antes de tratar - preserve contexto
3. Falhe rápido - valide cedo
4. Forneça contexto - inclua dados relevantes

### Padrão de Logging
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = risky_operation()
except ValidationError as e:
    logger.warning(f"Validação falhou: {e}", extra={"input": data})
    raise
except Exception as e:
    logger.exception(f"Erro inesperado: {e}")
    raise AppError("Ocorreu um erro inesperado") from e
```
""",
        scripts={
            "check_error_handling.sh": "Analisa cobertura de tratamento de erros",
        }
    )
    registry.register(error_skill)

    return registry


class SkillAwareAgent:
    """
    Um agente de IA que pode descobrir e carregar skills sob demanda.

    Insight chave: O agente primeiro vê o catálogo de skills (apenas descrições),
    depois decide quais skills carregar baseado na tarefa.
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.loaded_skills: list[str] = []

    def process_task(self, task: str) -> str:
        """
        Processa uma tarefa com carregamento inteligente de skills.

        Fluxo:
        1. Mostra tarefa + catálogo de skills para IA
        2. IA decide quais skills são relevantes
        3. Carrega apenas essas skills
        4. Executa tarefa com skills carregadas
        """
        # Passo 1: Pergunta à IA quais skills são necessárias
        skill_catalog = self.registry.get_skill_catalog()

        selection_prompt = f"""Você é um agente de IA com acesso a skills especializadas.

Dada uma tarefa, analise quais skills seriam úteis e selecione-as.

## Skills Disponíveis (apenas descrições - conteúdo não carregado ainda):
{skill_catalog}

## Tarefa:
{task}

## Instruções:
1. Analise o que a tarefa requer
2. Selecione apenas as skills DIRETAMENTE relevantes
3. Retorne um objeto JSON com: {{"skills": ["skill-name-1", "skill-name-2"]}}

Selecione apenas skills que genuinamente ajudarão. Não carregue skills desnecessárias (desperdiça tokens).
"""

        selection_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": selection_prompt}],
            response_format={"type": "json_object"}
        )

        selected = json.loads(selection_response.choices[0].message.content)
        skill_names = selected.get("skills", [])

        # Passo 2: Carrega skills selecionadas
        loaded_content = []
        for name in skill_names:
            skill = self.registry.load_skill(name)
            if skill:
                loaded_content.append(skill.to_context())
                self.loaded_skills.append(name)

        # Passo 3: Executa tarefa com skills carregadas
        skills_context = "\n\n---\n\n".join(loaded_content) if loaded_content else "Nenhuma skill carregada."

        execution_prompt = f"""Você é um agente de desenvolvimento de IA.

## Skills Carregadas:
{skills_context}

## Tarefa:
{task}

## Instruções:
Execute a tarefa usando as diretrizes das skills carregadas.
Seja específico e siga os padrões/convenções definidos nas skills.
"""

        execution_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": execution_prompt}]
        )

        return execution_response.choices[0].message.content


def main():
    print("=" * 60)
    print("Sistema de Skills para Agentes - Carregamento de Conhecimento Sob Demanda")
    print("=" * 60)

    # Cria registry de skills
    registry = create_sample_skills()

    print("\n📚 Skills Registradas:")
    for name in registry.skills:
        print(f"  - {name}")

    # Cria agente com awareness de skills
    agent = SkillAwareAgent(registry)

    # Exemplo 1: Tarefa que precisa de skill de testes
    print("\n" + "=" * 60)
    print("Exemplo 1: Tarefa relacionada a testes")
    print("=" * 60)

    task1 = "Escreva testes unitários para uma função que calcula números fibonacci"
    print(f"\nTarefa: {task1}")

    response1 = agent.process_task(task1)
    print(f"\nSkills carregadas: {agent.loaded_skills}")
    print(f"\nResposta:\n{response1[:1000]}...")

    # Exemplo 2: Tarefa que precisa de skill de API
    print("\n" + "=" * 60)
    print("Exemplo 2: Tarefa relacionada a API")
    print("=" * 60)

    agent.loaded_skills = []  # Reset
    task2 = "Crie um endpoint REST para gerenciar perfis de usuário com autenticação"
    print(f"\nTarefa: {task2}")

    response2 = agent.process_task(task2)
    print(f"\nSkills carregadas: {agent.loaded_skills}")
    print(f"\nResposta:\n{response2[:1000]}...")

    # Exemplo 3: Tarefa que precisa de múltiplas skills
    print("\n" + "=" * 60)
    print("Exemplo 3: Tarefa requerendo múltiplas skills")
    print("=" * 60)

    agent.loaded_skills = []  # Reset
    task3 = "Crie um endpoint de API com tratamento de erros adequado e testes"
    print(f"\nTarefa: {task3}")

    response3 = agent.process_task(task3)
    print(f"\nSkills carregadas: {agent.loaded_skills}")
    print(f"\nResposta:\n{response3[:1000]}...")

    print("\n" + "=" * 60)
    print("Insights Chave da FullCycle AI Tech Week:")
    print("=" * 60)
    print("""
1. Skills NÃO são carregadas automaticamente - são sob demanda
2. Descrições de skills são como SEO - otimize-as!
3. "Skills invisíveis" = skills nunca usadas devido a descrições ruins
4. Skills economizam tokens carregando apenas o necessário
5. Skills podem incluir: diretrizes, material de referência, scripts
    """)


if __name__ == "__main__":
    main()
