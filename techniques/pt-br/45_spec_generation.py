"""
Técnica 45: Desenvolvimento Orientado a Especificação

Demonstra geração de especificações técnicas abrangentes antes de escrever código.
Isso garante alinhamento entre requisitos e implementação.

Conceitos-chave da FullCycle AI Tech Week (Aula 2):
- Gere especificação ANTES do código
- Spec inclui: arquitetura, interfaces, casos de uso, restrições
- Valide consistência da spec antes da implementação
- Divida em tarefas implementáveis

Casos de uso:
- Desenvolvimento de novas features
- Redesign de sistemas
- Design de API
- Decisões de arquitetura
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class UseCase:
    """Um caso de uso na especificação."""
    name: str
    actor: str
    description: str
    preconditions: list[str]
    postconditions: list[str]
    main_flow: list[str]
    alternative_flows: list[str] = field(default_factory=list)


@dataclass
class Interface:
    """Uma interface/contrato na especificação."""
    name: str
    type: str  # "api", "event", "function", "class"
    description: str
    inputs: dict
    outputs: dict
    constraints: list[str] = field(default_factory=list)


@dataclass
class TechnicalSpec:
    """Especificação técnica completa para uma feature."""
    title: str
    version: str
    created_at: datetime
    summary: str
    goals: list[str]
    non_goals: list[str]
    architecture: dict
    use_cases: list[UseCase]
    interfaces: list[Interface]
    data_models: list[dict]
    constraints: list[str]
    dependencies: list[str]
    risks: list[dict]
    implementation_phases: list[dict]
    success_metrics: list[str]

    def to_markdown(self) -> str:
        """Converte especificação para formato markdown."""
        md = f"# Especificação Técnica: {self.title}\n\n"
        md += f"**Versão**: {self.version}\n"
        md += f"**Criado**: {self.created_at.strftime('%Y-%m-%d')}\n\n"

        md += f"## Resumo\n{self.summary}\n\n"

        md += "## Objetivos\n"
        for goal in self.goals:
            md += f"- {goal}\n"
        md += "\n"

        md += "## Não-Objetivos\n"
        for ng in self.non_goals:
            md += f"- {ng}\n"
        md += "\n"

        md += "## Arquitetura\n"
        md += f"**Padrão**: {self.architecture.get('pattern', 'N/A')}\n"
        md += f"**Componentes**:\n"
        for comp in self.architecture.get('components', []):
            md += f"- {comp}\n"
        md += "\n"

        md += "## Casos de Uso\n"
        for uc in self.use_cases:
            md += f"### {uc.name}\n"
            md += f"**Ator**: {uc.actor}\n"
            md += f"**Descrição**: {uc.description}\n"
            md += f"**Pré-condições**: {', '.join(uc.preconditions)}\n"
            md += f"**Fluxo**: {' → '.join(uc.main_flow)}\n\n"

        md += "## Interfaces\n"
        for iface in self.interfaces:
            md += f"### {iface.name} ({iface.type})\n"
            md += f"{iface.description}\n"
            md += f"**Entradas**: {json.dumps(iface.inputs, indent=2)}\n"
            md += f"**Saídas**: {json.dumps(iface.outputs, indent=2)}\n\n"

        md += "## Modelos de Dados\n"
        for model in self.data_models:
            md += f"### {model.get('name', 'Desconhecido')}\n"
            md += f"**Campos**: {json.dumps(model.get('fields', {}), indent=2)}\n\n"

        md += "## Restrições\n"
        for constraint in self.constraints:
            md += f"- {constraint}\n"
        md += "\n"

        md += "## Fases de Implementação\n"
        for i, phase in enumerate(self.implementation_phases, 1):
            md += f"### Fase {i}: {phase.get('name', 'Desconhecida')}\n"
            md += f"**Tarefas**:\n"
            for task in phase.get('tasks', []):
                md += f"- {task}\n"
            md += "\n"

        md += "## Métricas de Sucesso\n"
        for metric in self.success_metrics:
            md += f"- {metric}\n"

        return md


class SpecGenerator:
    """
    Gera especificações técnicas abrangentes a partir de requisitos.

    Insight chave da FullCycle: Spec antes do código previne esforço desperdiçado
    e garante que todos os stakeholders estejam alinhados.
    """

    def __init__(self):
        self.specs: dict[str, TechnicalSpec] = {}

    def generate_spec(self, requirements: str, project_context: str = "") -> TechnicalSpec:
        """
        Gera uma especificação técnica completa a partir de requisitos.

        Fluxo:
        1. Analisa requisitos
        2. Define objetivos e não-objetivos
        3. Projeta arquitetura
        4. Define casos de uso
        5. Especifica interfaces
        6. Planeja implementação
        """
        print("Gerando especificação técnica...")

        # Passo 1: Análise inicial
        analysis = self._analyze_requirements(requirements, project_context)

        # Passo 2: Gera arquitetura
        architecture = self._design_architecture(requirements, analysis)

        # Passo 3: Define casos de uso
        use_cases = self._define_use_cases(requirements, analysis)

        # Passo 4: Especifica interfaces
        interfaces = self._specify_interfaces(requirements, architecture)

        # Passo 5: Define modelos de dados
        data_models = self._define_data_models(requirements, interfaces)

        # Passo 6: Planeja implementação
        phases = self._plan_implementation(use_cases, interfaces)

        # Constrói spec completa
        spec = TechnicalSpec(
            title=analysis.get("title", "Feature Sem Título"),
            version="1.0.0",
            created_at=datetime.now(),
            summary=analysis.get("summary", ""),
            goals=analysis.get("goals", []),
            non_goals=analysis.get("non_goals", []),
            architecture=architecture,
            use_cases=[UseCase(**uc) for uc in use_cases],
            interfaces=[Interface(**iface) for iface in interfaces],
            data_models=data_models,
            constraints=analysis.get("constraints", []),
            dependencies=analysis.get("dependencies", []),
            risks=analysis.get("risks", []),
            implementation_phases=phases,
            success_metrics=analysis.get("success_metrics", [])
        )

        self.specs[spec.title] = spec
        return spec

    def _analyze_requirements(self, requirements: str, context: str) -> dict:
        """Analisa requisitos para extrair informações chave."""
        prompt = f"""Analise estes requisitos e extraia informações chave.

Requisitos:
{requirements}

Contexto:
{context if context else "Nenhum contexto adicional fornecido."}

Retorne JSON com:
{{
    "title": "Título da Feature",
    "summary": "Resumo em um parágrafo",
    "goals": ["objetivo 1", "objetivo 2"],
    "non_goals": ["explicitamente fora do escopo 1"],
    "constraints": ["restrição técnica 1"],
    "dependencies": ["dependência externa 1"],
    "risks": [{{"risk": "...", "mitigation": "..."}}],
    "success_metrics": ["métrica mensurável 1"]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _design_architecture(self, requirements: str, analysis: dict) -> dict:
        """Projeta a arquitetura do sistema."""
        prompt = f"""Projete a arquitetura para esta feature.

Requisitos: {requirements}
Objetivos: {analysis.get('goals', [])}
Restrições: {analysis.get('constraints', [])}

Retorne JSON com:
{{
    "pattern": "padrão de arquitetura (ex: Clean Architecture, MVC)",
    "components": ["componente 1", "componente 2"],
    "layers": ["camada 1", "camada 2"],
    "data_flow": "descrição do fluxo de dados",
    "integration_points": ["integração 1"]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _define_use_cases(self, requirements: str, analysis: dict) -> list[dict]:
        """Define casos de uso detalhados."""
        prompt = f"""Defina casos de uso para esta feature.

Requisitos: {requirements}
Objetivos: {analysis.get('goals', [])}

Retorne JSON com:
{{
    "use_cases": [
        {{
            "name": "UC001: Nome",
            "actor": "Usuário/Sistema",
            "description": "...",
            "preconditions": ["condição 1"],
            "postconditions": ["condição 1"],
            "main_flow": ["passo 1", "passo 2"],
            "alternative_flows": ["fluxo alt 1"]
        }}
    ]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("use_cases", [])

    def _specify_interfaces(self, requirements: str, architecture: dict) -> list[dict]:
        """Especifica todas as interfaces/contratos."""
        prompt = f"""Especifique as interfaces para esta feature.

Requisitos: {requirements}
Arquitetura: {json.dumps(architecture)}

Retorne JSON com:
{{
    "interfaces": [
        {{
            "name": "NomeDaInterface",
            "type": "api/event/function/class",
            "description": "...",
            "inputs": {{"param": "tipo"}},
            "outputs": {{"campo": "tipo"}},
            "constraints": ["restrição 1"]
        }}
    ]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("interfaces", [])

    def _define_data_models(self, requirements: str, interfaces: list) -> list[dict]:
        """Define modelos de dados."""
        prompt = f"""Defina modelos de dados para esta feature.

Requisitos: {requirements}
Interfaces: {json.dumps(interfaces)}

Retorne JSON com:
{{
    "models": [
        {{
            "name": "NomeDoModelo",
            "description": "...",
            "fields": {{"nome_campo": "tipo"}},
            "validations": ["validação 1"],
            "relationships": ["relacionamento 1"]
        }}
    ]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("models", [])

    def _plan_implementation(self, use_cases: list, interfaces: list) -> list[dict]:
        """Planeja fases de implementação."""
        prompt = f"""Planeje as fases de implementação.

Casos de Uso: {json.dumps(use_cases)}
Interfaces: {json.dumps(interfaces)}

Retorne JSON com:
{{
    "phases": [
        {{
            "name": "Nome da fase",
            "description": "...",
            "tasks": ["tarefa 1", "tarefa 2"],
            "deliverables": ["entregável 1"],
            "dependencies": ["dependência da fase X"]
        }}
    ]
}}

Ordene fases por dependências (fundação primeiro, depois features)."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("phases", [])


class SpecValidator:
    """
    Valida especificações técnicas quanto a consistência e completude.

    Insight chave: Valide ANTES de codificar para capturar problemas cedo.
    """

    def validate(self, spec: TechnicalSpec) -> dict:
        """
        Valida uma especificação para problemas.

        Retorna resultados de validação com quaisquer problemas encontrados.
        """
        issues = []
        warnings = []

        # Verifica completude
        if not spec.goals:
            issues.append("Nenhum objetivo definido")
        if not spec.use_cases:
            issues.append("Nenhum caso de uso definido")
        if not spec.interfaces:
            warnings.append("Nenhuma interface definida")
        if not spec.implementation_phases:
            issues.append("Nenhuma fase de implementação definida")

        # Verifica consistência
        if spec.goals and spec.use_cases:
            # Verifica se cada objetivo tem pelo menos um caso de uso
            for goal in spec.goals:
                has_uc = any(
                    goal.lower() in uc.description.lower()
                    for uc in spec.use_cases
                )
                if not has_uc:
                    warnings.append(f"Objetivo '{goal[:50]}...' pode não ter um caso de uso cobrindo")

        # Verifica se interfaces têm definições adequadas
        for iface in spec.interfaces:
            if not iface.inputs and not iface.outputs:
                warnings.append(f"Interface '{iface.name}' não tem entradas ou saídas")

        # Usa LLM para validação mais profunda
        spec_summary = f"""
Título: {spec.title}
Objetivos: {spec.goals}
Casos de Uso: {[uc.name for uc in spec.use_cases]}
Interfaces: {[i.name for i in spec.interfaces]}
Fases: {[p.get('name') for p in spec.implementation_phases]}
"""

        validation_prompt = f"""Valide esta especificação técnica para problemas.

{spec_summary}

Verifique:
1. Componentes críticos faltando
2. Inconsistências lógicas
3. Requisitos pouco claros
4. Tratamento de erros faltando
5. Considerações de segurança

Retorne JSON:
{{
    "is_valid": true/false,
    "issues": ["problema crítico 1"],
    "warnings": ["aviso 1"],
    "suggestions": ["melhoria 1"]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": validation_prompt}],
            response_format={"type": "json_object"}
        )

        llm_validation = json.loads(response.choices[0].message.content)

        return {
            "is_valid": len(issues) == 0 and llm_validation.get("is_valid", False),
            "issues": issues + llm_validation.get("issues", []),
            "warnings": warnings + llm_validation.get("warnings", []),
            "suggestions": llm_validation.get("suggestions", [])
        }


class TaskBreakdown:
    """
    Divide especificações em tarefas implementáveis.

    Insight chave: Tarefas devem ser atômicas e verificáveis independentemente.
    """

    def breakdown(self, spec: TechnicalSpec) -> list[dict]:
        """Divide uma especificação em tarefas."""
        all_tasks = []

        for i, phase in enumerate(spec.implementation_phases):
            phase_tasks = self._breakdown_phase(phase, i + 1, spec)
            all_tasks.extend(phase_tasks)

        return all_tasks

    def _breakdown_phase(self, phase: dict, phase_num: int, spec: TechnicalSpec) -> list[dict]:
        """Divide uma única fase em tarefas."""
        prompt = f"""Divida esta fase de implementação em tarefas atômicas.

Fase: {phase.get('name')}
Esboço de tarefas: {phase.get('tasks', [])}
Contexto da spec completa: {spec.title}

Retorne JSON:
{{
    "tasks": [
        {{
            "id": "T{phase_num}.1",
            "name": "Nome da tarefa",
            "description": "Descrição detalhada",
            "type": "code/test/config/docs",
            "acceptance_criteria": ["critério 1"],
            "dependencies": ["T1.1"],
            "estimated_complexity": "low/medium/high"
        }}
    ]
}}

Tarefas devem ser:
1. Atômicas (um entregável claro)
2. Testáveis independentemente
3. Com critérios de aceite claros"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content).get("tasks", [])


def demonstrate_spec_generation():
    """Demonstra o workflow de desenvolvimento orientado a especificação."""
    print("=" * 60)
    print("Desenvolvimento Orientado a Especificação")
    print("=" * 60)

    print("""
Workflow:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Requisitos                                                 │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────┐                                   │
│  │ Geração de Spec     │                                   │
│  │  - Objetivos        │                                   │
│  │  - Arquitetura      │                                   │
│  │  - Casos de Uso     │                                   │
│  │  - Interfaces       │                                   │
│  │  - Modelos de Dados │                                   │
│  └──────────┬──────────┘                                   │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                   │
│  │ Validação da Spec   │  ◄─── Corrige problemas           │
│  │  - Completude       │                                   │
│  │  - Consistência     │                                   │
│  │  - Segurança        │                                   │
│  └──────────┬──────────┘                                   │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                   │
│  │ Divisão de Tarefas  │                                   │
│  │  - Tarefas atômicas │                                   │
│  │  - Dependências     │                                   │
│  │  - Aceite           │                                   │
│  └──────────┬──────────┘                                   │
│             │                                               │
│             ▼                                               │
│  Implementação (com guia claro!)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

    # Requisitos de exemplo
    requirements = """
    Construir um sistema de autenticação de usuários com as seguintes features:
    - Cadastro de usuário com verificação de email
    - Login com email/senha
    - Funcionalidade de reset de senha
    - Gerenciamento de sessão baseado em JWT
    - Rate limiting para segurança
    - Logging de auditoria para compliance
    """

    project_context = """
    Stack tecnológica: Python/FastAPI, PostgreSQL, Redis
    Padrões existentes: Clean Architecture
    Compliance: LGPD, SOC2
    """

    # Gera especificação
    generator = SpecGenerator()
    print("\n📋 Gerando especificação...")
    spec = generator.generate_spec(requirements, project_context)

    # Valida especificação
    validator = SpecValidator()
    print("\n✅ Validando especificação...")
    validation = validator.validate(spec)

    print(f"\nResultado da Validação: {'PASSOU' if validation['is_valid'] else 'FALHOU'}")
    if validation['issues']:
        print("Problemas:")
        for issue in validation['issues']:
            print(f"  ❌ {issue}")
    if validation['warnings']:
        print("Avisos:")
        for warning in validation['warnings']:
            print(f"  ⚠️ {warning}")
    if validation['suggestions']:
        print("Sugestões:")
        for suggestion in validation['suggestions']:
            print(f"  💡 {suggestion}")

    # Divide em tarefas
    print("\n📝 Dividindo em tarefas...")
    breakdown = TaskBreakdown()
    tasks = breakdown.breakdown(spec)

    print(f"\nGeradas {len(tasks)} tarefas:")
    for task in tasks[:10]:  # Mostra primeiras 10
        print(f"\n  [{task.get('id')}] {task.get('name')}")
        print(f"      Tipo: {task.get('type')}")
        print(f"      Complexidade: {task.get('estimated_complexity')}")
        if task.get('dependencies'):
            print(f"      Depende de: {task.get('dependencies')}")

    # Gera spec em markdown
    print("\n" + "=" * 60)
    print("Especificação Gerada (Markdown)")
    print("=" * 60)
    print(spec.to_markdown()[:2000] + "...")


def main():
    print("=" * 60)
    print("Desenvolvimento Orientado a Especificação")
    print("Conceitos-Chave da FullCycle AI Tech Week")
    print("=" * 60)

    print("""
Por que Desenvolvimento Orientado a Especificação?

1. Alinhamento ANTES do Código
   - Todos concordam no que construir
   - Previne esforço de implementação desperdiçado
   - Captura problemas quando é barato corrigir

2. Especificação Completa Inclui:
   - Objetivos e Não-Objetivos
   - Decisões de arquitetura
   - Casos de uso com fluxos
   - Contratos de interface
   - Modelos de dados
   - Fases de implementação

3. Validação Antes da Implementação
   - Verifica completude
   - Verifica consistência
   - Identifica riscos cedo

4. Divisão de Tarefas
   - Tarefas atômicas, testáveis
   - Dependências claras
   - Critérios de aceite

"A spec É o blueprint do código - não construa sem ela"
- FullCycle AI Tech Week
""")

    # Executa demonstração
    demonstrate_spec_generation()

    print("\n" + "=" * 60)
    print("Principais Aprendizados")
    print("=" * 60)
    print("""
1. Gere spec ANTES do código
2. Valide spec para completude e consistência
3. Divida em tarefas atômicas e testáveis
4. Use spec como guia de implementação
5. Atualize spec conforme requisitos evoluem

Desenvolvimento orientado a especificação previne:
- Desalinhamento entre stakeholders
- Esforço de implementação desperdiçado
- Casos extremos faltando
- Features incompletas
""")


if __name__ == "__main__":
    main()
