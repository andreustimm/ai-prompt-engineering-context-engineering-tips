# Roadmap de Estudos para Desenvolvimento Assistido por IA

Um guia completo para desenvolvedores que querem dominar o desenvolvimento assistido por IA, conectando as 45 técnicas deste projeto com ferramentas modernas como Claude Code.

> **Language / Idioma:** Português Brasileiro | [English](ROADMAP.md)

---

## Índice

1. [Visão Geral: O Futuro do Desenvolvimento](#1-visão-geral-o-futuro-do-desenvolvimento)
2. [Mapa de Conexões entre Técnicas](#2-mapa-de-conexões-entre-técnicas)
3. [Trilhas de Aprendizado](#3-trilhas-de-aprendizado)
4. [Desenvolvimento com IA - Organização de Projetos](#4-desenvolvimento-com-ia---organização-de-projetos)
5. [Ferramentas Modernas - Claude Code](#5-ferramentas-modernas---claude-code)
6. [Templates e Exemplos](#6-templates-e-exemplos)
7. [Recursos Adicionais](#7-recursos-adicionais)

---

## 1. Visão Geral: O Futuro do Desenvolvimento

```
┌─────────────────────────────────────────────────────────────┐
│                    AI-ASSISTED DEVELOPMENT                  │
├─────────────────────────────────────────────────────────────┤
│  PROMPT ENGINEERING    │    CONTEXT ENGINEERING             │
│  (Como falar com IA)   │    (Como dar contexto à IA)        │
├────────────────────────┴────────────────────────────────────┤
│                     TOOLING MODERNO                         │
│  Claude Code │ Agents │ Skills │ MCP │ Hooks                │
└─────────────────────────────────────────────────────────────┘
```

### O que é Desenvolvimento Assistido por IA?

Desenvolvimento assistido por IA é um paradigma onde desenvolvedores trabalham junto com ferramentas de IA para aumentar produtividade, qualidade de código e velocidade de aprendizado. Combina:

- **Prompt Engineering**: A arte de se comunicar efetivamente com modelos de IA
- **Context Engineering**: A ciência de fornecer a informação certa para a IA
- **Tooling Moderno**: Integração de IA no fluxo de desenvolvimento

### Por que Aprender Isso?

| Benefício | Descrição |
|-----------|-----------|
| **Produtividade** | Automatizar tarefas repetitivas, gerar boilerplate, acelerar debugging |
| **Qualidade de Código** | Code review assistido por IA, sugestões de melhores práticas, verificações de segurança |
| **Aprendizado** | Explicações instantâneas, geração de documentação, exploração de conceitos |
| **Resolução de Problemas** | Múltiplas abordagens, análise de trade-offs, decisões de arquitetura |

---

## 2. Mapa de Conexões entre Técnicas

Entender como as 45 técnicas se relacionam ajuda a escolher a ferramenta certa para cada situação.

```
FUNDAÇÃO (01-06)
    │
    ├── 01 Zero-Shot ──────────────────────────────┐
    │       │                                      │
    ├── 02 Chain of Thought ───┐                   │
    │       │                  │                   │
    ├── 03 Few-Shot ──────────┐│                   │
    │       │                 ││                   │
    ├── 04 Tree of Thoughts ──┼┼─── 07 Self-Consistency
    │       │                 ││          │
    ├── 05 Skeleton of Thought┼┼─── 08 Least-to-Most
    │       │                 ││          │
    └── 06 ReAct ─────────────┼┴─── 09 Self-Refine
            │                 │           │
            │                 └──── 10 Prompt Chaining
            │                             │
            ▼                             ▼
    ┌───────────────────────────────────────────────┐
    │          CONTEXT ENGINEERING (11-30)          │
    ├───────────────────────────────────────────────┤
    │                                               │
    │   11 RAG Basic ◄─────────────────────────┐    │
    │       │                                  │    │
    │       ├── 12 RAG + Reranking             │    │
    │       │                                  │    │
    │       └── 13 RAG Conversational          │    │
    │               │                          │    │
    │               ▼                          │    │
    │   ┌─────────────────────────────────┐    │    │
    │   │  21 Advanced Chunking ◄─────────┼────┘    │
    │   │      │                          │         │
    │   │      ├── 22 Hybrid Search       │         │
    │   │      │                          │         │
    │   │      ├── 23 Query Transform     │         │
    │   │      │                          │         │
    │   │      ├── 24 Contextual Compress │         │
    │   │      │                          │         │
    │   │      └── 25 Self-Query          │         │
    │   │              │                  │         │
    │   └──────────────┼──────────────────┘         │
    │                  ▼                            │
    │   ┌─────────────────────────────────┐         │
    │   │  26 Parent-Document             │         │
    │   │  27 Multi-Vector                │         │
    │   │  28 Ensemble Retrieval          │         │
    │   │  29 Long Context                │         │
    │   │  30 Time-Weighted               │         │
    │   └─────────────────────────────────┘         │
    └───────────────────────────────────────────────┘
```

### Categorias de Técnicas

| Categoria | Técnicas | Quando Usar |
|-----------|----------|-------------|
| **Prompting Básico** | 01-06 | Começando, tarefas diretas |
| **Raciocínio Avançado** | 07-10 | Problemas complexos, múltiplos passos |
| **Fundamentos de RAG** | 11-13 | Q&A baseado em documentos |
| **Modelos Locais** | 14-15 | Privacidade, redução de custos, offline |
| **Saída Estruturada** | 16-17 | Integrações de API, uso de ferramentas |
| **Features Avançadas** | 18-20 | Visão, memória, otimização de prompts |
| **Chunking & Retrieval** | 21-25 | Otimização de RAG |
| **Gerenciamento de Contexto** | 26-30 | Padrões avançados de RAG |
| **MCP & Multi-Agent** | 31-35 | Integrações de protocolo, sistemas multi-agente |
| **Produção & Segurança** | 36-40 | Segurança, cache, custos, testes, fine-tuning |
| **IA Agêntica Avançado** | 41-45 | Skills, janela de contexto, orquestração, memória, specs |

---

## 3. Trilhas de Aprendizado

Escolha sua trilha com base no seu nível atual e objetivos.

### Trilha Iniciante (4 semanas)

Perfeita para desenvolvedores novos em desenvolvimento assistido por IA.

```
Semana 1: Fundamentos
├── 01 Zero-Shot
│   └── Aprenda a escrever prompts claros e diretos
├── 02 Chain of Thought
│   └── Guie a IA através de raciocínio passo a passo
└── 03 Few-Shot
    └── Use exemplos para moldar respostas

Semana 2: Raciocínio Avançado
├── 04 Tree of Thoughts
│   └── Explore múltiplos caminhos de solução
├── 05 Skeleton of Thought
│   └── Estrutura primeiro, detalhes depois
└── 06 ReAct
    └── Combine raciocínio com ações

Semana 3: RAG Básico
├── 11 RAG Basic
│   └── Fundamentos de retrieval de documentos
├── 12 RAG + Reranking
│   └── Melhore relevância de retrieval
└── 13 RAG Conversacional
    └── Q&A de documentos multi-turno

Semana 4: Projeto Prático
├── 16 Structured Output
│   └── JSON mode + modelos Pydantic
├── 17 Tool Calling
│   └── Ferramentas de função customizadas
└── Projeto: Construa um Chatbot com RAG
    └── Combine técnicas 11-13, 16-17
```

### Trilha Intermediária (4 semanas)

Para desenvolvedores familiarizados com prompting básico que querem dominar RAG.

```
Semana 1: Técnicas Avançadas de Prompt
├── 07 Self-Consistency
│   └── Múltiplas amostras + votação
├── 08 Least-to-Most
│   └── Decomposição progressiva
├── 09 Self-Refine
│   └── Melhoria iterativa
└── 10 Prompt Chaining
    └── Pipelines de prompts conectados

Semana 2: Chunking & Retrieval
├── 21 Advanced Chunking
│   └── Estratégias semânticas, recursivas, baseadas em tokens
├── 22 Hybrid Search
│   └── BM25 + Vector com RRF
└── 23 Query Transformation
    └── HyDE, Multi-Query, Step-Back

Semana 3: Otimização de Contexto
├── 24 Contextual Compression
│   └── Extrair apenas partes relevantes
├── 25 Self-Query
│   └── Auto-gerar filtros de metadados
└── 26 Parent-Document
    └── Busca em chunks pequenos, retorna contexto grande

Semana 4: Projeto de Produção
└── Construa um Sistema RAG de Produção
    ├── Combine estratégias de chunking
    ├── Implemente busca híbrida
    ├── Adicione transformação de query
    └── Deploy com monitoramento
```

### Trilha Avançada (4 semanas)

Para desenvolvedores construindo sistemas de IA para produção.

```
Semana 1: Multi-Vector & Ensemble
├── 27 Multi-Vector
│   └── Múltiplas representações de documentos
├── 28 Ensemble Retrieval
│   └── Combine retrievers com RRF ponderado
└── 29 Long Context
    └── Map-Reduce, Refine, Map-Rerank

Semana 2: Features Avançadas
├── 18 Vision/Multimodal
│   └── Análise de imagens com GPT-4o
├── 19 Memory/Conversation
│   └── Contexto de conversa persistente
├── 20 Meta-Prompting
│   └── LLM gerando/otimizando prompts
└── 30 Time-Weighted
    └── Viés de recência em retrieval

Semana 3: Modelos Locais
├── 14 Ollama Basic
│   └── LLMs locais sem custos de API
├── 15 Ollama + RAG
│   └── RAG 100% offline
└── Análise de Custos
    └── Compare custos local vs. API

Semana 4: Setup de Produção
├── Configuração Claude Code
│   └── CLAUDE.md, settings, hooks
├── Agents & Skills
│   └── Subagentes e workflows customizados
└── Sistema Completo
    └── Ambiente completo de desenvolvimento assistido por IA
```

---

## 4. Desenvolvimento com IA - Organização de Projetos

### Estrutura de Projeto Recomendada

```
meu-projeto/
├── src/                          # Código da aplicação
├── tests/                        # Testes (unit, integration)
├── docs/                         # Documentação
│   └── adr/                      # Architecture Decision Records
│       ├── 0000-template.md
│       ├── 0001-escolha-llm.md
│       └── 0002-estrategia-rag.md
├── CLAUDE.md                     # Contexto do projeto para IA
├── .claude/                      # Configuração Claude Code
│   ├── agents/                   # Subagentes customizados
│   │   ├── code-reviewer.md
│   │   └── debugger.md
│   ├── skills/                   # Skills do projeto
│   │   ├── commit/SKILL.md
│   │   └── deploy/SKILL.md
│   └── rules/                    # Regras por path
├── .mcp.json                     # Servidores MCP
└── scripts/                      # Scripts de automação
```

### Framework de Decomposição de Features

Ao trabalhar em features com assistência de IA, siga esta abordagem estruturada:

```
FEATURE REQUEST
      │
      ▼
┌─────────────────────────────────────────┐
│  1. ENTENDIMENTO (Use 02 CoT)           │
│  - O que o usuário quer?                │
│  - Quais são os requisitos?             │
│  - Quais são as restrições?             │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  2. DECOMPOSIÇÃO (Use 08 Least-to-Most) │
│  - Quebre em sub-tarefas                │
│  - Identifique dependências             │
│  - Priorize por valor                   │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  3. EXPLORAÇÃO (Use 04 ToT)             │
│  - Considere alternativas               │
│  - Avalie trade-offs                    │
│  - Documente em ADR                     │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  4. IMPLEMENTAÇÃO                       │
│  - Use Claude Code com contexto         │
│  - Itere com Self-Refine (09)           │
│  - Valide com testes                    │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  5. REVISÃO                             │
│  - Code review com subagente            │
│  - Documentação atualizada              │
│  - ADR se houve decisão arquitetural    │
└─────────────────────────────────────────┘
```

---

## 5. Ferramentas Modernas - Claude Code

Claude Code é uma ferramenta CLI que traz assistência de IA diretamente para seu fluxo de desenvolvimento.

### Principais Features

| Feature | Descrição |
|---------|-----------|
| **Agents** | Subprocessos especializados para tarefas específicas |
| **Skills** | Comandos reutilizáveis para workflows comuns |
| **MCP** | Model Context Protocol para integrações externas |
| **Hooks** | Ações automatizadas disparadas por eventos |

### Subagentes

Crie agentes customizados para tarefas especializadas:

```yaml
# .claude/agents/code-reviewer.md
---
name: code-reviewer
description: Revisor de qualidade e segurança de código
tools: Read, Grep, Glob
model: sonnet
---

Você é um revisor de código expert. Analise:
1. Qualidade e legibilidade do código
2. Vulnerabilidades de segurança
3. Problemas de performance
4. Aderência às convenções do projeto

Forneça feedback acionável com referências específicas de linha.
```

### Skills

Defina workflows reutilizáveis:

```yaml
# .claude/skills/commit/SKILL.md
---
name: smart-commit
description: Cria commits semânticos com contexto
---

Para criar um commit:
1. Analise mudanças com `git diff --staged`
2. Identifique tipo: feat|fix|docs|refactor|test
3. Escreva mensagem seguindo Conventional Commits
4. Inclua Co-Authored-By
```

### Servidores MCP

Conecte-se a serviços externos:

```json
// .mcp.json
{
  "servers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@bytebase/dbhub", "--dsn", "${DATABASE_URL}"]
    }
  }
}
```

### Hooks

Automatize ações no uso de ferramentas:

```json
// Em settings.json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "npx prettier --write \"$file_path\""
      }]
    }]
  }
}
```

### Agentic Workflows - Loops e Orquestração

Agentes de IA que rodam em loops são um padrão fundamental para construir sistemas autônomos. Entender esses padrões é essencial para desenvolver workflows sofisticados de assistência por IA.

#### O que são Agentic Loops?

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENTIC LOOP                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌───────────┐    ┌──────────┐    ┌──────────┐            │
│    │  PERCEIVE │───▶│  REASON  │───▶│   ACT    │            │
│    │  (Input)  │    │  (Think) │    │ (Output) │            │
│    └───────────┘    └──────────┘    └────┬─────┘            │
│         ▲                                │                  │
│         │         ┌──────────┐           │                  │
│         └─────────│ OBSERVE  │◀──────────┘                  │
│                   │(Feedback)│                              │
│                   └──────────┘                              │
│                                                             │
│    Loop continua até: objetivo atingido OU max iterações    │
└─────────────────────────────────────────────────────────────┘
```

#### Padrões de Agentic Workflows

**Padrão 1: ReAct Loop (Técnica 06)**

```python
while not done:
    thought = llm.think(observation)      # Raciocínio
    action = llm.decide_action(thought)   # Decisão
    observation = execute(action)         # Execução
    done = check_completion(observation)  # Verificação
```

**Padrão 2: Plan-Execute Loop**

```python
plan = llm.create_plan(goal)              # Planejamento inicial
for step in plan:
    result = execute(step)                # Executa passo
    if needs_replan(result):
        plan = llm.replan(goal, result)   # Replanejar se necessário
```

**Padrão 3: Self-Refine Loop (Técnica 09)**

```python
response = llm.generate(task)
while not satisfactory:
    critique = llm.critique(response)     # Auto-crítica
    response = llm.improve(response, critique)  # Melhoria
    satisfactory = evaluate(response)
```

**Padrão 4: Multi-Agent Orchestration**

```
┌─────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                    │
│                    (Coordinator)                    │
├─────────────────────────────────────────────────────┤
│         │             │             │               │
│    ┌────▼─────┐  ┌────▼────┐   ┌────▼────┐          │
│    │  Agent 1 │  │ Agent 2 │   │ Agent 3 │          │
│    │(Research)│  │ (Code)  │   │(Review) │          │
│    └────┬─── ─┘  └────┬────┘   └────┬────┘          │
│         │             │             │               │
│         └─────────────┴─────────────┘               │
│                       │                             │
│                  ┌────▼────┐                        │
│                  │ COMBINE │                        │
│                  │ RESULTS │                        │
│                  └─────────┘                        │
└─────────────────────────────────────────────────────┘
```

#### Implementação Prática com LangChain/LangGraph

```python
# Exemplo: ReAct Loop com LangChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_tool, calculator_tool]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,      # Limite de iterações do loop
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

# Loop acontece automaticamente dentro do executor
result = agent_executor.invoke({"input": "Pesquise e calcule..."})
```

#### Workflow Recomendado para Desenvolvimento

```
┌─────────────────────────────────────────────────────────────┐
│              WORKFLOW DE DESENVOLVIMENTO COM IA             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DEFINIÇÃO                                               │
│     └── Defina objetivo claro e critérios de sucesso        │
│                                                             │
│  2. DECOMPOSIÇÃO                                            │
│     └── Quebre em sub-tarefas (use Least-to-Most)           │
│                                                             │
│  3. ESCOLHA DO PADRÃO                                       │
│     ├── Tarefa simples → Single Agent (ReAct)               │
│     ├── Tarefa complexa → Plan-Execute                      │
│     ├── Qualidade crítica → Self-Refine Loop                │
│     └── Multi-domínio → Multi-Agent Orchestration           │
│                                                             │
│  4. IMPLEMENTAÇÃO                                           │
│     ├── Configure max_iterations (evita loops infinitos)    │
│     ├── Defina critérios de parada claros                   │
│     └── Adicione logging/observability                      │
│                                                             │
│  5. VALIDAÇÃO                                               │
│     ├── Teste com casos edge                                │
│     ├── Monitore custos (tokens por loop)                   │
│     └── Ajuste parâmetros conforme necessário               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Tabela de Decisão: Qual Padrão Usar?

| Cenário | Padrão Recomendado | Técnica Base |
|---------|-------------------|--------------|
| Busca + Raciocínio | ReAct Loop | 06 ReAct |
| Tarefas multi-passo | Plan-Execute | 08 Least-to-Most |
| Geração de conteúdo | Self-Refine | 09 Self-Refine |
| Pipeline de dados | Prompt Chaining | 10 Prompt Chaining |
| Múltiplas perspectivas | Multi-Agent | 07 Self-Consistency |
| Decisões complexas | Tree of Thoughts | 04 ToT |

#### Dicas Práticas

- **Sempre defina `max_iterations`** - Evita loops infinitos e custos explosivos
- **Logging é essencial** - Registre cada iteração para debug
- **Critérios de parada claros** - O agente precisa saber quando terminar
- **Fallbacks** - Tenha plano B se o loop não convergir
- **Custos** - Cada iteração = mais tokens = mais custo

---

## 6. Templates e Exemplos

### Template CLAUDE.md

```markdown
# [Nome do Projeto]

## Visão Geral
[Descrição em 2-3 frases]

## Stack Tecnológica
- **Backend**: Python 3.11, FastAPI
- **LLM**: OpenAI GPT-4o-mini via LangChain
- **Vector DB**: ChromaDB
- **Testes**: pytest

## Arquitetura
[Diagrama ou descrição da arquitetura]

## Convenções
- Código em inglês, documentação bilíngue
- Docstrings Google Style
- Type hints obrigatórios
- Testes para toda feature nova

## Comandos Frequentes
```bash
# Rodar testes
pytest tests/ -v

# Servidor de desenvolvimento
uvicorn src.main:app --reload

# Build para produção
docker build -t app .
```

## Áreas Sensíveis
- `.env` - Nunca commitar
- `src/auth/` - Revisar mudanças de segurança
- `migrations/` - Testar em staging primeiro
```

### Template ADR (Architecture Decision Record)

```markdown
# ADR [Número]: [Título]

## Status
[Proposto | Aceito | Deprecated | Substituído por ADR-XXX]

## Contexto
[Situação técnica que requer uma decisão]

## Decisão
[A decisão tomada e seu racional]

## Alternativas Consideradas
1. **[Alternativa 1]**: [Prós/Contras]
2. **[Alternativa 2]**: [Prós/Contras]

## Consequências
### Positivas
- [Benefício 1]
- [Benefício 2]

### Negativas
- [Trade-off 1]
- [Trade-off 2]

## Referências
- [Links relevantes]
```

### Guia de Seleção de Técnicas

| Situação | Técnica Recomendada | Por quê |
|----------|---------------------|---------|
| Classificação simples | 01 Zero-Shot | Direto, sem exemplos necessários |
| Problemas de matemática/lógica | 02 Chain of Thought | Raciocínio passo a passo |
| Formato de saída específico | 03 Few-Shot | Exemplos guiam formato |
| Decisões complexas | 04 Tree of Thoughts | Explora múltiplos caminhos |
| Conteúdo longo | 05 Skeleton of Thought | Estrutura primeiro |
| Precisa de dados externos | 06 ReAct | Raciocínio + ações |
| Alta precisão necessária | 07 Self-Consistency | Múltiplas amostras + votação |
| Problema complexo | 08 Least-to-Most | Decomposição progressiva |
| Melhoria de qualidade | 09 Self-Refine | Crítica iterativa |
| Workflow multi-passo | 10 Prompt Chaining | Pipeline conectado |
| Q&A de documentos | 11-13 RAG | Retrieval-augmented |
| Privacidade/offline | 14-15 Ollama | Modelos locais |
| Integração de API | 16 Structured Output | Schema garantido |
| Ferramentas externas | 17 Tool Calling | Execução de funções |
| Análise de imagens | 18 Vision | Multimodal |
| Contexto de chat | 19 Memory | Histórico de conversa |
| Otimização de prompts | 20 Meta-Prompting | LLM cria prompts |
| Chunks melhores | 21 Advanced Chunking | Seleção de estratégia |
| Keyword + semântico | 22 Hybrid Search | Retrieval combinado |
| Retrieval ruim | 23 Query Transform | Melhora queries |
| Redução de tokens | 24 Compression | Extrai relevante |
| Filtros de metadados | 25 Self-Query | Auto-gera filtros |
| Precisa mais contexto | 26 Parent-Document | Pais grandes |
| Múltiplas visões | 27 Multi-Vector | Resumo + perguntas |
| Múltiplos métodos | 28 Ensemble | Combina retrievers |
| Documentos grandes | 29 Long Context | Map-Reduce/Refine |
| Sensível ao tempo | 30 Time-Weighted | Viés de recência |
| Integrações externas | 31 MCP Basics | Base do protocolo |
| Servidores CLI | 32 MCP Server STDIO | Standard I/O |
| Servidores web | 33 MCP Server HTTP | Transporte HTTP/SSE |
| Múltiplos especialistas | 34 Multi-Agent | Colaboração paralela |
| Qualidade de prompts | 35 Prompt Evaluation | Métricas & testes |
| Preocupações de segurança | 36 LLM Security | Validação de input |
| Velocidade de resposta | 37 Caching | Reduzir latência |
| Restrições de orçamento | 38 Cost Optimization | Gerenciamento de tokens |
| Testes de LLM | 39 AI Testing | Estratégias de teste |
| Comportamento customizado | 40 Fine-Tuning | Treinamento de modelo |
| Conhecimento dinâmico | 41 Agent Skills | Carregamento sob demanda |
| Conversas longas | 42 Context Window | Sumarização inteligente |
| Tarefas complexas | 43 Subagent Orchestration | Contextos isolados |
| Aprendizado cross-agente | 44 Shared Memory | Memory DB |
| Geração de código | 45 Spec Generation | Abordagem spec-first |

---

## 7. Recursos Adicionais

### Este Projeto

- [README.md](README.md) - Documentação completa das técnicas (inglês)
- [README.pt-BR.md](README.pt-BR.md) - Documentação completa das técnicas (português)
- [techniques/en/](techniques/en/) - Implementações em inglês
- [techniques/pt-br/](techniques/pt-br/) - Implementações em português
- [sample_data/](sample_data/) - Dados de exemplo para testes

### Recursos Externos

#### Documentação
- [Documentação LangChain](https://python.langchain.com/docs/)
- [Referência da API OpenAI](https://platform.openai.com/docs/api-reference)
- [Documentação Claude Code](https://docs.anthropic.com/claude-code)
- [Documentação Ollama](https://ollama.ai/docs)

#### Aprendizado
- [Guia de Prompt Engineering](https://www.promptingguide.ai/)
- [LangChain Academy](https://academy.langchain.com/)
- [Curso de Prompt Engineering da Anthropic](https://docs.anthropic.com/claude/docs/prompt-engineering)

#### Comunidades
- [Discord LangChain](https://discord.gg/langchain)
- [Discord Ollama](https://discord.gg/ollama)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

### Próximos Passos

1. **Escolha Sua Trilha**: Comece com a trilha que corresponde ao seu nível
2. **Execute Exemplos**: Execute os scripts neste projeto
3. **Construa Projetos**: Aplique técnicas em projetos reais
4. **Compartilhe Conhecimento**: Contribua de volta para a comunidade

---

## Cartão de Referência Rápida

```
┌─────────────────────────────────────────────────────────────┐
│                    REFERÊNCIA RÁPIDA DE TÉCNICAS            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PROMPT ENGINEERING                                         │
│  ├── Zero-Shot (01)      → Prompts diretos                  │
│  ├── Chain of Thought (02) → Passo a passo                  │
│  ├── Few-Shot (03)       → Exemplos guiam                   │
│  ├── Tree of Thoughts (04) → Múltiplos caminhos             │
│  ├── Skeleton (05)       → Estrutura primeiro               │
│  ├── ReAct (06)          → Raciocínio + Ações               │
│  ├── Self-Consistency (07) → Vota em respostas              │
│  ├── Least-to-Most (08)  → Decompõe problemas               │
│  ├── Self-Refine (09)    → Itera & melhora                  │
│  └── Prompt Chaining (10) → Pipeline de prompts             │
│                                                             │
│  CONTEXT ENGINEERING                                        │
│  ├── RAG Basic (11)      → Retrieval de documentos          │
│  ├── RAG Reranking (12)  → Melhor relevância                │
│  ├── RAG Conversational (13) → Chat + docs                  │
│  ├── Ollama Basic (14)   → LLMs locais                      │
│  ├── Ollama RAG (15)     → RAG offline                      │
│  ├── Structured (16)     → JSON/Pydantic                    │
│  ├── Tool Calling (17)   → Funções                          │
│  ├── Vision (18)         → Imagens                          │
│  ├── Memory (19)         → Conversação                      │
│  ├── Meta-Prompting (20) → Otimiza prompts                  │
│  ├── Chunking (21)       → Estratégias de split             │
│  ├── Hybrid Search (22)  → BM25 + Vector                    │
│  ├── Query Transform (23) → Melhora queries                 │
│  ├── Compression (24)    → Reduz tokens                     │
│  ├── Self-Query (25)     → Auto filtros                     │
│  ├── Parent-Doc (26)     → Mais contexto                    │
│  ├── Multi-Vector (27)   → Múltiplas representações         │
│  ├── Ensemble (28)       → Combina retrievers               │
│  ├── Long Context (29)   → Docs grandes                     │
│  └── Time-Weighted (30)  → Viés de recência                 │
│                                                             │
│  MCP & MULTI-AGENT                                          │
│  ├── MCP Basics (31)     → Fundamentos do protocolo         │
│  ├── MCP Server STDIO (32) → Servidores Standard I/O        │
│  ├── MCP Server HTTP (33) → Servidores HTTP/SSE             │
│  ├── Multi-Agent (34)    → Colaboração entre agentes        │
│  └── Prompt Evaluation (35) → Métricas de qualidade         │
│                                                             │
│  PRODUÇÃO & SEGURANÇA                                       │
│  ├── LLM Security (36)   → Validação de input, guardrails   │
│  ├── Caching (37)        → Estratégias de cache             │
│  ├── Cost Optimization (38) → Gerenciamento de tokens       │
│  ├── AI Testing (39)     → Estratégias de teste para LLMs   │
│  └── Fine-Tuning (40)    → Customização de modelos          │
│                                                             │
│  IA AGÊNTICA AVANÇADO                                       │
│  ├── Agent Skills (41)   → Carregamento sob demanda         │
│  ├── Context Window (42) → Sumarização inteligente          │
│  ├── Subagent Orch. (43) → Contextos isolados               │
│  ├── Shared Memory (44)  → Memory DB para agentes           │
│  └── Spec Generation (45) → Desenvolvimento spec-first      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

*Este roadmap é um documento vivo. Conforme novas técnicas e ferramentas surgem, ele será atualizado para refletir as melhores práticas em desenvolvimento assistido por IA.*
