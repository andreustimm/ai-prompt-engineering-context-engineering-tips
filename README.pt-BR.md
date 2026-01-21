# Prompt Engineering & Context Engineering com LangChain e OpenAI

Scripts demonstrando 45 técnicas de Prompt Engineering, Context Engineering e IA Agêntica usando LangChain e a API da OpenAI.

> **Language / Idioma:** Português Brasileiro | [English](README.md)

> **Roadmap de Estudos:** Quer dominar desenvolvimento assistido por IA? Confira o [ROADMAP.pt-BR.md](ROADMAP.pt-BR.md) - um guia completo com trilhas de aprendizado (iniciante a avançado), mapa de conexões entre técnicas, templates de organização de projetos e integração com ferramentas modernas como Claude Code.

## Técnicas Implementadas

### Prompt Engineering - Básico (01-06)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `01_zero_shot.py` | Zero-Shot | Prompts diretos sem exemplos prévios |
| `02_chain_of_thought.py` | Chain of Thought (CoT) | Raciocínio passo a passo |
| `03_few_shot.py` | Few-Shot | Exemplos para guiar o modelo |
| `04_tree_of_thoughts.py` | Tree of Thoughts (ToT) | Múltiplos caminhos de raciocínio |
| `05_skeleton_of_thought.py` | Skeleton of Thought (SoT) | Estrutura primeiro, detalhes depois |
| `06_react_agent.py` | ReAct | Raciocínio + Ações com ferramentas |

### Prompt Engineering - Avançado (07-10)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `07_self_consistency.py` | Self-Consistency | Gera N respostas, vota na mais consistente |
| `08_least_to_most.py` | Least-to-Most | Decomposição progressiva em sub-problemas |
| `09_self_refine.py` | Self-Refine | Crítica e melhoria iterativa |
| `10_prompt_chaining.py` | Prompt Chaining | Pipeline de prompts conectados |

### Context Engineering - RAG (11-13)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `11_rag_basic.py` | RAG Básico | ChromaDB + busca semântica + chunking |
| `12_rag_reranking.py` | RAG + Reranking | Reordenação para melhor relevância |
| `13_rag_conversational.py` | RAG Conversacional | RAG com memória de chat |

### Modelos Locais - Ollama (14-15)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `14_ollama_basic.py` | Ollama Básico | LLMs locais (Llama 3, Mistral) |
| `15_ollama_rag.py` | Ollama + RAG | RAG 100% offline |

### Saída Estruturada & Ferramentas (16-17)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `16_structured_output.py` | Structured Output | Modo JSON + modelos Pydantic |
| `17_tool_calling.py` | Tool Calling | Ferramentas/funções customizadas |

### Recursos Avançados (18-20)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `18_vision_multimodal.py` | Vision/Multimodal | Análise de imagens com GPT-4o |
| `19_memory_conversation.py` | Memory/Conversation | Contexto persistente de conversa |
| `20_meta_prompting.py` | Meta-Prompting | LLM gerando/otimizando prompts |

### Context Engineering - Chunking & Recuperação (21-25)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `21_advanced_chunking.py` | Chunking Avançado | Estratégias semântica, recursiva, por tokens, janela deslizante |
| `22_hybrid_search.py` | Busca Híbrida | BM25 (palavra-chave) + Vetor (semântico) com fusão RRF |
| `23_query_transformation.py` | Transformação de Query | HyDE, Multi-Query, Step-Back, Decomposição |
| `24_contextual_compression.py` | Compressão Contextual | Extrair apenas partes relevantes dos documentos |
| `25_self_query.py` | Self-Query Retrieval | LLM gera filtros de metadados automaticamente |

### Context Engineering - Gerenciamento de Contexto (26-30)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `26_parent_document.py` | Recuperação Documento-Pai | Chunks pequenos para busca, pais grandes para contexto |
| `27_multi_vector.py` | Recuperação Multi-Vetor | Múltiplas representações (resumo + perguntas + conteúdo) |
| `28_ensemble_retrieval.py` | Recuperação Ensemble | Combina múltiplos retrievers com RRF ponderado |
| `29_long_context.py` | Estratégias de Contexto Longo | Map-Reduce, Refine, Map-Rerank para documentos grandes |
| `30_time_weighted.py` | Recuperação Ponderada por Tempo | Viés de recência com decaimento exponencial |

### MCP & IA Agêntica (31-35)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `31_mcp_basics.py` | Fundamentos MCP | Model Context Protocol (recursos, ferramentas, prompts) |
| `32_mcp_server_stdio.py` | Servidor MCP STDIO | Servidor MCP local com transporte entrada/saída padrão |
| `33_mcp_server_http.py` | Servidor MCP HTTP/SSE | Servidor MCP remoto com HTTP e Server-Sent Events |
| `34_multi_agent.py` | Multi-Agente | Agentes de IA colaborativos (pipeline, debate, hierárquico) |
| `35_prompt_evaluation.py` | Avaliação de Prompts | Avaliar qualidade de prompts, testes A/B, observabilidade |

### Enterprise & Produção (36-40)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `36_llm_security.py` | Segurança LLM | OWASP Top 10, detecção de prompt injection, guardrails, rate limiting |
| `37_caching_strategies.py` | Estratégias de Cache | Cache de resposta, cache semântico, cache de embedding, cache de conversa |
| `38_cost_optimization.py` | Otimização de Custos | Contagem de tokens, seleção de modelo, rastreamento de uso, gestão de orçamento |
| `39_ai_testing.py` | Testes de IA | Testes para saídas não-determinísticas, validadores, mocking, snapshots |
| `40_fine_tuning.py` | Fine-tuning | Preparação de dataset, validação, workflow de fine-tuning, boas práticas |

### IA Agêntica - Avançado (41-45)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `41_agent_skills.py` | Skills para Agentes | Carregamento de conhecimento sob demanda, descrições SEO-like, carregamento lazy |
| `42_context_window.py` | Gerenciamento de Janela de Contexto | Monitorar uso de contexto, sumarização inteligente, saúde do contexto |
| `43_subagent_orchestration.py` | Orquestração de Subagentes | Janelas de contexto isoladas, execução paralela, agregação de resultados |
| `44_shared_memory.py` | Memória Compartilhada | Memory DB para agentes, memória curto/médio/longo prazo, loops de feedback |
| `45_spec_generation.py` | Desenvolvimento Orientado a Spec | Geração de especificação técnica, validação, divisão de tarefas |

## Requisitos

- Python 3.10+
- Chave de API da OpenAI
- (Opcional) Ollama para modelos locais
- (Opcional) Chave de API Cohere para reranking

## Instalação

1. **Clone ou navegue até o diretório do projeto:**

```bash
cd /caminho/para/projeto
```

2. **Crie e ative um ambiente virtual:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

4. **Configure as credenciais:**

```bash
cp .env.example .env
```

Edite o arquivo `.env` e adicione suas chaves:

```
OPENAI_API_KEY=sk-sua-chave-aqui
OPENAI_MODEL=gpt-4o-mini

# Opcional - para Ollama (modelos locais)
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Opcional - para reranking com Cohere
COHERE_API_KEY=sua-chave-cohere-aqui
```

5. **(Opcional) Instale o Ollama para modelos locais:**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Inicie o serviço Ollama
ollama serve

# Baixe um modelo
ollama pull llama3.2
ollama pull nomic-embed-text  # Para embeddings
```

## Uso

Execute qualquer script da pasta `techniques/`:

**Exemplos em Português:**
```bash
# Prompting Básico (01-06)
python techniques/pt-br/01_zero_shot.py
python techniques/pt-br/02_chain_of_thought.py
python techniques/pt-br/03_few_shot.py
python techniques/pt-br/04_tree_of_thoughts.py
python techniques/pt-br/05_skeleton_of_thought.py
python techniques/pt-br/06_react_agent.py

# Prompting Avançado (07-10)
python techniques/pt-br/07_self_consistency.py
python techniques/pt-br/08_least_to_most.py
python techniques/pt-br/09_self_refine.py
python techniques/pt-br/10_prompt_chaining.py

# RAG (11-13) - Requer sample_data/
python techniques/pt-br/11_rag_basic.py
python techniques/pt-br/12_rag_reranking.py
python techniques/pt-br/13_rag_conversational.py

# Ollama (14-15) - Requer Ollama rodando
python techniques/pt-br/14_ollama_basic.py
python techniques/pt-br/15_ollama_rag.py

# Saída Estruturada & Ferramentas (16-17)
python techniques/pt-br/16_structured_output.py
python techniques/pt-br/17_tool_calling.py

# Recursos Avançados (18-20)
python techniques/pt-br/18_vision_multimodal.py
python techniques/pt-br/19_memory_conversation.py
python techniques/pt-br/20_meta_prompting.py

# Context Engineering - Chunking & Recuperação (21-25)
python techniques/pt-br/21_advanced_chunking.py
python techniques/pt-br/22_hybrid_search.py
python techniques/pt-br/23_query_transformation.py
python techniques/pt-br/24_contextual_compression.py
python techniques/pt-br/25_self_query.py

# Context Engineering - Gerenciamento de Contexto (26-30)
python techniques/pt-br/26_parent_document.py
python techniques/pt-br/27_multi_vector.py
python techniques/pt-br/28_ensemble_retrieval.py
python techniques/pt-br/29_long_context.py
python techniques/pt-br/30_time_weighted.py

# MCP & IA Agêntica (31-35)
python techniques/pt-br/31_mcp_basics.py
python techniques/pt-br/32_mcp_server_stdio.py
python techniques/pt-br/33_mcp_server_http.py
python techniques/pt-br/34_multi_agent.py
python techniques/pt-br/35_prompt_evaluation.py

# Enterprise & Produção (36-40)
python techniques/pt-br/36_llm_security.py
python techniques/pt-br/37_caching_strategies.py
python techniques/pt-br/38_cost_optimization.py
python techniques/pt-br/39_ai_testing.py
python techniques/pt-br/40_fine_tuning.py

# IA Agêntica - Avançado (41-45)
python techniques/pt-br/41_agent_skills.py
python techniques/pt-br/42_context_window.py
python techniques/pt-br/43_subagent_orchestration.py
python techniques/pt-br/44_shared_memory.py
python techniques/pt-br/45_spec_generation.py
```

**Exemplos em Inglês:**
```bash
python techniques/en/01_zero_shot.py
# ... (mesmo padrão com en/)
```

## Descrição das Técnicas

### 1. Zero-Shot Prompting

Técnica onde o modelo recebe uma tarefa sem exemplos prévios, usando apenas seu conhecimento pré-treinado.

**Funções disponíveis:**
- `classificar_sentimento(texto)` - Classifica sentimento como POSITIVO, NEGATIVO ou NEUTRO
- `traduzir_texto(texto, idioma_destino)` - Traduz texto para o idioma especificado
- `extrair_entidades(texto)` - Extrai pessoas, locais, organizações e datas
- `resumir_texto(texto)` - Resume texto em poucas frases

---

### 2. Chain of Thought (CoT)

Instrui o modelo a "pensar passo a passo" antes de chegar à resposta final, melhorando o desempenho em tarefas de raciocínio.

**Funções disponíveis:**
- `resolver_problema_matematico(problema)` - Resolve problemas matemáticos mostrando cada passo
- `raciocinio_logico(puzzle)` - Resolve puzzles lógicos com deduções
- `analisar_decisao(situacao)` - Analisa cenários para tomada de decisão
- `debug_codigo(codigo, erro)` - Analisa código e erro para encontrar solução

---

### 3. Few-Shot Prompting

Fornece exemplos ao modelo antes da tarefa, ajudando-o a entender o formato e tipo de resposta esperada.

**Funções disponíveis:**
- `classificar_ticket_suporte(ticket)` - Classifica tickets com categoria, prioridade e ação
- `converter_para_sql(descricao)` - Converte linguagem natural para SQL
- `gerar_docstring(codigo)` - Gera docstrings no padrão Google Style
- `extrair_dados_estruturados(texto)` - Extrai dados em formato JSON

---

### 4. Tree of Thoughts (ToT)

Explora múltiplos caminhos de raciocínio em paralelo, avalia cada um e seleciona o mais promissor.

**Funções disponíveis:**
- `tree_of_thoughts(problema, profundidade)` - Executa algoritmo ToT completo
- `gerar_pensamentos(problema, num)` - Gera múltiplas abordagens iniciais
- `avaliar_pensamento(problema, pensamento)` - Avalia viabilidade de uma abordagem
- `expandir_pensamento(problema, pensamento, proximo_passo)` - Desenvolve uma abordagem

---

### 5. Skeleton of Thought (SoT)

Gera primeiro um "esqueleto" (estrutura/tópicos) e depois expande cada parte, permitindo paralelização.

**Funções disponíveis:**
- `skeleton_of_thought_sync(tema, contexto)` - Versão síncrona
- `skeleton_of_thought_async(tema, contexto)` - Versão assíncrona (paralela)
- `gerar_esqueleto(tema, contexto)` - Gera lista de tópicos
- `expandir_topico(tema, topico, contexto)` - Expande um tópico específico

---

### 6. ReAct Agent

Combina raciocínio (Thought) com ações (Action) e observações (Observation) em um loop iterativo, usando ferramentas externas.

**Ferramentas disponíveis:**
- `busca_web` - Busca na internet via DuckDuckGo
- `wikipedia` - Consulta à Wikipedia
- `calculadora` - Cálculos matemáticos

---

### 7. Self-Consistency

Gera múltiplas respostas para o mesmo problema, depois usa votação majoritária para selecionar a resposta mais consistente. Melhora a precisão em tarefas de raciocínio.

**Funções disponíveis:**
- `resolver_com_consistencia(problema, num_amostras)` - Resolve com múltiplas amostras e votação
- `resolver_com_votacao(problema, num_amostras)` - Alternativa com votação explícita
- `extrair_resposta(resposta)` - Extrai resposta final da resposta

**Exemplo:**
```python
from techniques.pt_br.self_consistency import resolver_com_consistencia

resultado = resolver_com_consistencia(
    "Se um trem viaja 120 km em 2 horas, qual é sua velocidade?",
    num_amostras=5
)
print(resultado["resposta_final"])  # 60 km/h
```

---

### 8. Least-to-Most Prompting

Decompõe problemas complexos em sub-problemas menores, resolve-os progressivamente do mais simples ao mais complexo, construindo sobre respostas anteriores.

**Funções disponíveis:**
- `resolver_progressivamente(problema)` - Decomposição e solução completa
- `decompor_problema(problema)` - Quebra em sub-problemas
- `resolver_subproblema(subproblema, contexto)` - Resolve com contexto anterior

**Exemplo:**
```python
from techniques.pt_br.least_to_most import resolver_progressivamente

resultado = resolver_progressivamente(
    "Como construir um modelo de machine learning para prever preços de casas?"
)
print(resultado["resposta_final"])
```

---

### 9. Self-Refine

Gera uma resposta inicial, depois iterativamente critica e melhora até atingir padrões de qualidade.

**Funções disponíveis:**
- `auto_refinar(tarefa, max_iteracoes)` - Loop completo de refinamento
- `gerar_inicial(tarefa)` - Cria primeiro rascunho
- `criticar(tarefa, resposta)` - Avalia e identifica problemas
- `refinar(tarefa, resposta, feedback)` - Melhora baseado na crítica

**Exemplo:**
```python
from techniques.pt_br.self_refine import auto_refinar

resultado = auto_refinar(
    "Escreva uma função para verificar se uma string é um palíndromo",
    max_iteracoes=3
)
print(resultado["resposta_final"])
```

---

### 10. Prompt Chaining

Conecta múltiplos prompts em um pipeline onde a saída de um se torna a entrada do próximo, permitindo workflows complexos de múltiplos passos.

**Funções disponíveis:**
- `encadear_prompts(entrada_inicial, cadeia_prompts)` - Executa pipeline de prompts
- `cadeia_pesquisa(topico)` - Pesquisa → Análise → Resumo
- `cadeia_conteudo(topico)` - Esboço → Rascunho → Edição → Formatação

**Exemplo:**
```python
from techniques.pt_br.prompt_chaining import cadeia_conteudo

resultado = cadeia_conteudo("Benefícios do Trabalho Remoto")
print(resultado["saida_final"])
```

---

### 11. RAG Básico

Geração Aumentada por Recuperação com ChromaDB para armazenamento de documentos, busca semântica e chunking de texto.

**Funções disponíveis:**
- `criar_vectorstore(documentos)` - Cria vector store ChromaDB
- `consulta_rag(pergunta, vectorstore)` - Consulta com RAG
- `carregar_e_dividir_documentos(caminho)` - Carrega e faz chunking de documentos

**Recursos principais:**
- Chunking recursivo de texto (1000 chars, 200 overlap)
- Embeddings OpenAI para busca semântica
- Recuperação top-k com scores de relevância

**Exemplo:**
```python
from techniques.pt_br.rag_basic import criar_vectorstore, consulta_rag

# Carrega documentos e cria vector store
vectorstore = criar_vectorstore(documentos)

# Consulta com RAG
resultado = consulta_rag("O que é machine learning?", vectorstore)
print(resultado["resposta"])
```

---

### 12. RAG + Reranking

Aprimora o RAG básico com reranking para melhorar a relevância da recuperação. Suporta múltiplos métodos de reranking.

**Métodos de reranking:**
- Reranking baseado em LLM (usa GPT para pontuar relevância)
- Cohere Rerank (requer chave de API)
- CrossEncoder (modelo transformer local)

**Funções disponíveis:**
- `rag_com_reranking(pergunta, vectorstore, metodo)` - RAG com reranking
- `rerank_llm(pergunta, documentos)` - Reranking baseado em LLM
- `rerank_cohere(pergunta, documentos)` - Reranking via API Cohere

---

### 13. RAG Conversacional

RAG com memória de conversa para diálogos de múltiplos turnos. Mantém contexto entre perguntas.

**Tipos de memória:**
- Buffer Memory - Histórico completo de conversa
- Summary Memory - Resumo comprimido

**Funções disponíveis:**
- `criar_rag_conversacional(vectorstore)` - Cria cadeia conversacional
- `conversar(pergunta)` - Chat com memória
- `obter_historico_chat()` - Recupera histórico de conversa

---

### 14. Ollama Básico

Use LLMs locais via Ollama sem custos de API ou dependência de internet.

**Modelos suportados:**
- `llama3.2` - Llama 3 da Meta
- `mistral` - Mistral 7B
- `codellama` - Llama especializado em código
- `phi3` - Phi-3 da Microsoft

**Funções disponíveis:**
- `ollama_chat(mensagem)` - Chat com modelo local
- `ollama_gerar(prompt)` - Geração de texto
- `listar_modelos_locais()` - Lista modelos disponíveis

**Exemplo:**
```python
from techniques.pt_br.ollama_basic import ollama_chat

resposta = ollama_chat("Explique computação quântica em termos simples")
print(resposta)
```

---

### 15. Ollama + RAG

RAG 100% offline usando Ollama tanto para embeddings quanto para geração.

**Componentes:**
- Embeddings locais: `nomic-embed-text`
- LLM local: `llama3.2` ou `mistral`
- ChromaDB para armazenamento de vetores

**Funções disponíveis:**
- `criar_vectorstore_local(documentos)` - Cria store com embeddings locais
- `consulta_rag_local(pergunta, vectorstore)` - Consulta com RAG local

---

### 16. Structured Output

Força saídas do LLM a seguir esquemas específicos usando modelos Pydantic ou modo JSON.

**Funções disponíveis:**
- `extrair_pessoa(texto)` - Extrai info de pessoa como modelo Pydantic
- `extrair_nota_fiscal(texto)` - Extrai dados de nota fiscal
- `extrair_json_modo(texto, esquema)` - Extração JSON genérica

**Exemplo:**
```python
from techniques.pt_br.structured_output import extrair_pessoa
from pydantic import BaseModel

class Pessoa(BaseModel):
    nome: str
    idade: int
    profissao: str

resultado = extrair_pessoa("João é um engenheiro de software de 30 anos")
print(resultado.nome)  # João
print(resultado.idade)  # 30
```

---

### 17. Tool Calling

Permite que LLMs chamem funções/ferramentas customizadas para executar ações ou recuperar informações.

**Ferramentas disponíveis:**
- `obter_clima(cidade)` - Obtém informação do clima
- `calcular(expressao)` - Realiza cálculos
- `buscar_banco_dados(consulta)` - Busca em banco de dados mock

**Exemplo:**
```python
from techniques.pt_br.tool_calling import agente_com_ferramentas

resposta = agente_com_ferramentas(
    "Qual o clima em Tóquio e calcule 15% de gorjeta em R$85"
)
print(resposta)
```

---

### 18. Vision/Multimodal

Analisa imagens usando modelos com capacidade de visão como GPT-4o.

**Funções disponíveis:**
- `analisar_imagem(caminho_imagem, prompt)` - Analisa imagem com prompt customizado
- `descrever_imagem(caminho_imagem)` - Gera descrição detalhada
- `extrair_texto_da_imagem(caminho_imagem)` - Extração de texto tipo OCR
- `analisar_grafico(caminho_imagem)` - Analisa gráficos e charts
- `comparar_imagens(imagem1, imagem2)` - Compara duas imagens

**Exemplo:**
```python
from techniques.pt_br.vision_multimodal import analisar_grafico

resultado = analisar_grafico("sample_data/images/chart.png")
print(resultado)  # Tipo de gráfico, dados, insights
```

---

### 19. Memory/Conversation

Mantém contexto de conversa através de múltiplas interações usando diferentes estratégias de memória.

**Tipos de memória:**
- `MemoriaBuffer` - Armazena histórico completo de conversa
- `MemoriaJanela` - Armazena últimas N trocas
- `MemoriaResumo` - Mantém resumo comprimido
- `MemoriaEntidades` - Rastreia entidades mencionadas

**Exemplo:**
```python
from techniques.pt_br.memory_conversation import CadeiaConversa

cadeia = CadeiaConversa(tipo_memoria="buffer")
resposta1 = cadeia.conversar("Meu nome é Alice")
resposta2 = cadeia.conversar("Qual é meu nome?")  # Lembra "Alice"
```

---

### 20. Meta-Prompting

Usa um LLM para gerar, otimizar e melhorar prompts para outras tarefas de LLM.

**Funções disponíveis:**
- `gerar_prompt(descricao_tarefa)` - Gera prompt otimizado
- `otimizar_prompt(prompt_original, problemas)` - Melhora prompt existente
- `avaliar_prompt(prompt, tarefa)` - Pontua e critica um prompt
- `gerar_variacoes_prompt(prompt_base)` - Variações para teste A/B
- `auto_melhorar_prompt(prompt, tarefa, entrada_teste)` - Melhoria iterativa

**Exemplo:**
```python
from techniques.pt_br.meta_prompting import gerar_prompt

prompt = gerar_prompt(
    descricao_tarefa="Extrair informações chave de emails de clientes",
    contexto="Suporte de empresa SaaS",
    restricoes=["Saída JSON", "Incluir nível de urgência"]
)
print(prompt)
```

---

### 21. Chunking Avançado

Múltiplas estratégias de divisão de texto otimizadas para diferentes tipos de conteúdo e cenários de recuperação.

**Estratégias de chunking:**
- `RecursivoCaractere` - Divisão hierárquica por separadores
- `BaseadoToken` - Divisão por contagem de tokens (model-aware)
- `MarkdownConsciente` - Respeita estrutura markdown
- `Semantico` - Agrupa por similaridade semântica
- `JanelaDeslizante` - Janelas de tamanho fixo com sobreposição
- `BaseadoSentenca` - Limites naturais de sentença

**Funções disponíveis:**
- `chunking_recursivo_caractere(texto, tamanho_chunk, sobreposicao)` - Divisão recursiva padrão
- `chunking_baseado_token(texto, tamanho_chunk)` - Divisão consciente de tokens
- `chunking_markdown_consciente(texto)` - Preserva estrutura de markdown
- `chunking_semantico(texto, limiar)` - Agrupamento baseado em similaridade
- `chunking_janela_deslizante(texto, tamanho_janela, passo)` - Janelas sobrepostas
- `chunking_baseado_sentenca(texto, sentencas_por_chunk)` - Agrupamento de sentenças

**Exemplo:**
```python
from techniques.pt_br.advanced_chunking import chunking_semantico

chunks = chunking_semantico(documento_longo, limiar=0.75)
for chunk in chunks:
    print(f"Chunk: {len(chunk)} caracteres")
```

---

### 22. Busca Híbrida

Combina busca baseada em palavras-chave (BM25) e semântica (vetor) usando Fusão de Ranking Recíproco.

**Componentes:**
- `RetrieverBM25` - Correspondência tradicional de palavras-chave
- `RetrieverVetor` - Busca por similaridade semântica
- `RetrieverHibrido` - Combinação ponderada

**Funções disponíveis:**
- `criar_retriever_hibrido(documentos, peso_bm25, peso_vetor)` - Criar retriever híbrido
- `fusao_rank_reciproco(lista_resultados, k)` - Combinar resultados ranqueados
- `busca_hibrida(consulta, k)` - Buscar com ambos os métodos

**Exemplo:**
```python
from techniques.pt_br.hybrid_search import RetrieverHibrido

retriever = RetrieverHibrido(documentos, peso_bm25=0.4, peso_vetor=0.6)
resultados = retriever.buscar("algoritmos de machine learning", k=5)
```

---

### 23. Transformação de Query

Transforma consultas para melhorar a eficácia da recuperação usando várias técnicas.

**Métodos de transformação:**
- `HyDE` - Embeddings de Documento Hipotético (gera resposta hipotética, busca com ela)
- `Multi-Query` - Gera múltiplas variações da consulta
- `Step-Back` - Abstrai consulta para conceito mais amplo
- `Decomposição` - Divide consulta complexa em sub-consultas

**Funções disponíveis:**
- `transformar_hyde(consulta)` - Gerar documento hipotético
- `transformar_multi_query(consulta, num_queries)` - Gerar variações de consulta
- `transformar_step_back(consulta)` - Abstrair para pergunta mais ampla
- `decompor_query(consulta)` - Dividir em sub-perguntas

**Exemplo:**
```python
from techniques.pt_br.query_transformation import transformar_multi_query

queries = transformar_multi_query(
    "Quais são as melhores práticas para microsserviços?",
    num_queries=3
)
# Retorna variações como:
# - "melhores práticas arquitetura microsserviços"
# - "como projetar microsserviços efetivamente"
# - "padrões recomendados para desenvolvimento de microsserviços"
```

---

### 24. Compressão Contextual

Extrai apenas as porções relevantes dos documentos recuperados para reduzir ruído e uso de tokens.

**Métodos de compressão:**
- `ExtratorLLM` - Usa LLM para extrair sentenças relevantes
- `FiltroEmbeddings` - Filtra por similaridade semântica
- `ExtratorSentencas` - Extrai sentenças relevantes por pontuação

**Funções disponíveis:**
- `criar_retriever_compressao(retriever_base, compressor)` - Encapsular retriever com compressão
- `comprimir_llm(documentos, consulta)` - Compressão baseada em LLM
- `filtro_embeddings(documentos, consulta, limiar)` - Filtragem baseada em similaridade

**Exemplo:**
```python
from techniques.pt_br.contextual_compression import RetrieverCompressaoContextual

retriever_compressao = RetrieverCompressaoContextual(
    retriever_base=retriever_vetor,
    compressor=CompressorExtratorLLM()
)
# Retorna apenas trechos relevantes em vez de documentos completos
resultados = retriever_compressao.recuperar("O que é RAG?")
```

---

### 25. Self-Query Retrieval

LLM gera automaticamente filtros de metadados a partir de consultas em linguagem natural.

**Recursos:**
- Extração automática de filtros de consultas
- Suporte para operadores de comparação (=, >, <, >=, <=)
- Combina busca semântica com filtragem estruturada

**Funções disponíveis:**
- `criar_retriever_self_query(vectorstore, info_metadados)` - Criar retriever self-query
- `parsear_consulta(consulta)` - Extrair consulta semântica e filtros
- `aplicar_filtros(documentos, filtros)` - Aplicar filtros de metadados

**Exemplo:**
```python
from techniques.pt_br.self_query import SelfQueryRetriever

retriever = SelfQueryRetriever(
    vectorstore=vectorstore,
    campos_metadados=[
        {"nome": "categoria", "tipo": "string"},
        {"nome": "preco", "tipo": "float"},
        {"nome": "ano", "tipo": "integer"}
    ]
)

# Consulta: "eletrônicos baratos de 2024"
# Auto-gera: categoria="eletrônicos", preco<100, ano=2024
resultados = retriever.recuperar("eletrônicos baratos de 2024")
```

---

### 26. Recuperação Documento-Pai

Busca com chunks pequenos para precisão, mas recupera documentos-pai maiores para contexto.

**Conceito:**
- Chunks filhos: Pequenos (ex: 400 caracteres) para correspondência precisa
- Documentos pais: Maiores (ex: 2000 caracteres) para contexto completo
- Mapeia filho → pai para recuperação

**Funções disponíveis:**
- `criar_retriever_documento_pai(documentos, tamanho_filho, tamanho_pai)` - Criar retriever
- `adicionar_documentos(documentos)` - Indexar documentos com relação pai-filho
- `recuperar(consulta, k)` - Buscar filhos, retornar pais

**Exemplo:**
```python
from techniques.pt_br.parent_document import RetrieverDocumentoPai

retriever = RetrieverDocumentoPai(
    tamanho_chunk_filho=400,
    tamanho_chunk_pai=2000
)
retriever.adicionar_documentos(documentos)

# Busca chunks pequenos, retorna contexto pai completo
resultados = retriever.recuperar("arquitetura de rede neural", k=3)
```

---

### 27. Recuperação Multi-Vetor

Armazena múltiplas representações de documentos para recuperação aprimorada.

**Tipos de representação:**
- Conteúdo original do documento
- Resumos gerados
- Perguntas hipotéticas que o documento responde

**Funções disponíveis:**
- `criar_retriever_multi_vetor(documentos)` - Criar retriever com múltiplos vetores
- `gerar_resumo(documento)` - Gerar resumo do documento
- `gerar_perguntas(documento)` - Gerar perguntas hipotéticas
- `recuperar(consulta, k)` - Buscar todas as representações

**Exemplo:**
```python
from techniques.pt_br.multi_vector import RetrieverMultiVetor

retriever = RetrieverMultiVetor()
retriever.adicionar_documentos(documentos)  # Cria vetores de resumo + perguntas

# Pode corresponder consulta a resumo, perguntas ou conteúdo original
resultados = retriever.recuperar("Como funciona backpropagation?", k=3)
```

---

### 28. Recuperação Ensemble

Combina múltiplos retrievers usando Fusão de Ranking Recíproco com pesos configuráveis.

**Componentes:**
- Múltiplos retrievers base (BM25, Vetor, etc.)
- Pesos configuráveis por retriever
- Algoritmo RRF para combinação de scores

**Funções disponíveis:**
- `criar_retriever_ensemble(retrievers, pesos)` - Criar ensemble
- `fusao_rank_reciproco(lista_resultados, pesos, k)` - Combinar com RRF

**Exemplo:**
```python
from techniques.pt_br.ensemble_retrieval import RetrieverEnsemble

ensemble = RetrieverEnsemble(
    retrievers=[retriever_bm25, retriever_vetor, retriever_esparso],
    pesos=[0.3, 0.5, 0.2]
)
resultados = ensemble.recuperar("otimização de machine learning", k=5)
```

---

### 29. Estratégias de Contexto Longo

Processa documentos que excedem janelas de contexto típicas usando várias estratégias.

**Estratégias:**
- `Map-Reduce` - Processa chunks separadamente, combina resultados
- `Refine` - Constrói resposta iterativamente com cada chunk
- `Map-Rerank` - Pontua cada chunk, usa os melhores
- `Stuffing` - Encaixa conteúdo mais relevante no contexto

**Funções disponíveis:**
- `sumarizar_map_reduce(chunks)` - Sumarizar com map-reduce
- `sumarizar_refine(chunks)` - Refinamento iterativo
- `responder_map_rerank(chunks, pergunta)` - Pontuar e selecionar melhores chunks
- `stuffing_com_priorizacao(chunks, pergunta, max_contexto)` - Stuffing baseado em prioridade

**Exemplo:**
```python
from techniques.pt_br.long_context import sumarizar_map_reduce, responder_map_rerank

# Sumarizar um documento de 50 páginas
resumo = sumarizar_map_reduce(chunks_documento)

# Responder pergunta usando melhores chunks
resposta = responder_map_rerank(
    chunks=chunks_documento,
    pergunta="Quais são as principais conclusões?"
)
```

---

### 30. Recuperação Ponderada por Tempo

Incorpora relevância temporal com decaimento exponencial para preferir documentos recentes.

**Recursos:**
- Função de decaimento exponencial para ponderação temporal
- Taxa de decaimento e unidades de tempo configuráveis
- Combina similaridade semântica com recência

**Funções disponíveis:**
- `criar_retriever_ponderado_tempo(documentos, taxa_decaimento)` - Criar retriever
- `calcular_peso_temporal(timestamp, taxa_decaimento, unidade_tempo)` - Calcular peso de decaimento
- `recuperar(consulta, k, fator_peso_tempo)` - Buscar com ponderação temporal

**Exemplo:**
```python
from techniques.pt_br.time_weighted import RetrieverPonderadoPorTempo

retriever = RetrieverPonderadoPorTempo(
    documentos=artigos_noticias,
    taxa_decaimento=0.05,  # Por dia
    unidade_tempo="dias"
)

# Artigos recentes têm pontuação maior
resultados = retriever.recuperar(
    "desenvolvimentos em IA",
    k=5,
    fator_peso_tempo=0.4  # 40% tempo, 60% semântico
)
```

**Guia de configuração:**
| Caso de Uso | Taxa Decaimento | Unidade Tempo | Fator Peso |
|-------------|-----------------|---------------|------------|
| Notícias/Atual | 0.1-0.5 | horas | 0.5-0.7 |
| Histórico chat | 0.05-0.1 | horas | 0.3-0.5 |
| Documentação | 0.01-0.05 | dias | 0.2-0.4 |
| Artigos pesquisa | 0.001-0.01 | dias | 0.1-0.3 |

---

### 31. Fundamentos MCP

Model Context Protocol (MCP) é um protocolo aberto criado pela Anthropic para conectar assistentes de IA a fontes de dados e ferramentas externas de forma padronizada.

**Conceitos principais:**
- Resources - Dados expostos pelo servidor (arquivos, bancos de dados, APIs)
- Tools - Funções que o LLM pode invocar
- Prompts - Templates de prompts reutilizáveis

**Exemplo:**
```python
from techniques.pt_br.mcp_basics import MCPServerSimulator

servidor = MCPServerSimulator(name="servidor-demo")
servidor.add_tool("buscar_banco_dados", description="Busca dados", ...)
servidor.add_resource("file:///config.json", name="Config", ...)
```

---

### 32. Servidor MCP STDIO

STDIO é o método de transporte mais comum para servidores MCP locais. A comunicação ocorre através de stdin/stdout.

**Casos de uso:**
- Integração com Claude Desktop
- Ferramentas de linha de comando
- Acesso a arquivos locais
- Execução de scripts

**Exemplo:**
```python
# Em produção, use:
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("meu-servidor")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)
```

---

### 33. Servidor MCP HTTP/SSE

HTTP/SSE é o método de transporte para servidores MCP remotos. Permite comunicação através da rede.

**Casos de uso:**
- APIs de IA como serviço
- Integrações empresariais
- Servidores centralizados
- Microserviços de IA

**Exemplo:**
```python
# Em produção, use FastAPI com MCP SDK:
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.sse import SseServerTransport

app = FastAPI()
server = Server("meu-servidor-http")
```

---

### 34. Aplicações Multi-Agente

Sistemas multi-agente permitem que múltiplos agentes de IA colaborem para resolver tarefas complexas.

**Padrões:**
- `Pipeline` - Processamento sequencial (Agente1 → Agente2 → Agente3)
- `Debate` - Agentes com perspectivas diferentes debatem para chegar a consenso
- `Hierárquico` - Agentes organizados em níveis de autoridade (Diretor → Gerentes → Trabalhadores)
- `Orquestrador` - Agente central coordena os demais

**Frameworks:**
- LangGraph (LangChain)
- AutoGen (Microsoft)
- CrewAI
- Swarm (OpenAI)

**Exemplo:**
```python
from techniques.pt_br.multi_agent import MultiAgentSystem, Agent, AgentRole

sistema = MultiAgentSystem(name="time-dev")
sistema.add_agent(Agent(name="Planejador", role=AgentRole.PLANNER, ...))
sistema.add_agent(Agent(name="Codificador", role=AgentRole.EXECUTOR, ...))
sistema.add_agent(Agent(name="Revisor", role=AgentRole.REVIEWER, ...))
```

---

### 35. Avaliação de Prompts

A avaliação de prompts é essencial para garantir qualidade, consistência e melhoria contínua em aplicações com LLMs.

**Métricas:**
- Relevância - A resposta aborda a pergunta?
- Coerência - O texto tem lógica e fluxo?
- Fundamentação - Baseado em fatos/contexto?
- Precisão - Informações corretas?
- Segurança - Conteúdo apropriado?

**Ferramentas:**
- LangSmith (LangChain)
- LangFuse (Open Source)
- Weights & Biases
- Promptfoo (CLI)
- Phoenix (Arize)

**Exemplo:**
```python
from techniques.pt_br.prompt_evaluation import PromptEvaluator

avaliador = PromptEvaluator()
resultado = avaliador.evaluate_relevance(pergunta, resposta)
print(f"Relevância: {resultado.score:.2f}")
```

---

### 36. Segurança LLM

Medidas de segurança para aplicações LLM seguindo as diretrizes OWASP Top 10 para LLMs.

**Componentes de segurança:**
- `PromptInjectionDetector` - Detecta tentativas de injeção de prompt
- `OutputValidator` - Valida e sanitiza saídas do LLM
- `ContentGuardrail` - Aplica políticas de conteúdo
- `RateLimiter` - Previne abuso com algoritmo token bucket
- `SecureLLMWrapper` - Combina todas as medidas de segurança

**Exemplo:**
```python
from techniques.pt_br.llm_security import SecureLLMWrapper

llm_seguro = SecureLLMWrapper(
    enable_injection_detection=True,
    enable_output_validation=True,
    rate_limit_rpm=60
)

resposta = llm_seguro.chat("Entrada do usuário aqui")
```

---

### 37. Estratégias de Cache

Múltiplas abordagens de cache para reduzir custos de API e melhorar tempos de resposta.

**Tipos de cache:**
- `ResponseCache` - Cache de correspondência exata para prompts idênticos
- `EmbeddingCache` - Cache de embeddings para evitar recomputação
- `SemanticCache` - Encontra queries similares usando similaridade de embeddings
- `ConversationCache` - Cache de contextos de conversa

**Exemplo:**
```python
from techniques.pt_br.caching_strategies import SemanticCache

cache = SemanticCache(similarity_threshold=0.95)
cache.set("O que é Python?", "Python é uma linguagem de programação...")

# Query similar retornará resposta em cache
resposta = cache.get("Me fale sobre Python")
```

---

### 38. Otimização de Custos

Ferramentas e estratégias para monitorar e reduzir custos de API de LLM.

**Componentes:**
- `TokenCounter` - Conta tokens antes de chamadas à API
- `UsageTracker` - Rastreia uso e custos ao longo do tempo
- `ModelSelector` - Escolhe modelo ideal baseado na complexidade da tarefa

**Exemplo:**
```python
from techniques.pt_br.cost_optimization import TokenCounter, UsageTracker

contador = TokenCounter()
tokens = contador.count_tokens("Seu prompt aqui")
custo_estimado = contador.estimate_cost(tokens, 500, "gpt-4o-mini")

tracker = UsageTracker()
tracker.daily_budget = 1.00
```

---

### 39. Testes de IA

Frameworks e estratégias de teste para saídas não-determinísticas de LLMs.

**Abordagens de teste:**
- Validadores baseados em propriedades (contém, tamanho, regex, JSON)
- Testes de similaridade semântica com embeddings
- Avaliação LLM-as-Judge
- Clientes mock para CI/CD
- Testes de snapshot para detecção de regressão

**Exemplo:**
```python
from techniques.pt_br.ai_testing import LLMTestRunner, TestCase, ContainsValidator

teste = TestCase(
    name="Teste de saudação",
    prompt="Diga olá",
    validators=[ContainsValidator(["olá"])]
)

runner = LLMTestRunner()
resultado = runner.run_test(teste)
```

---

### 40. Fine-tuning

Workflow completo para fine-tuning de modelos OpenAI em datasets customizados.

**Componentes:**
- `DatasetGenerator` - Cria exemplos de treinamento a partir de templates
- `DatasetValidator` - Valida formato e qualidade do dataset
- `FineTuningManager` - Upload, treino e uso de modelos fine-tuned

**Exemplo:**
```python
from techniques.pt_br.fine_tuning import DatasetGenerator, DatasetValidator

generator = DatasetGenerator(system_prompt="Classifique o sentimento...")
generator.add_examples_from_pairs([
    ("Produto excelente!", "POSITIVO"),
    ("Experiência terrível", "NEGATIVO")
])
generator.export_jsonl("dados_treinamento.jsonl")
```

## Monitoramento de Tokens

Todos os scripts incluem **contagem automática de tokens** para ajudar a monitorar custos e uso da API.

### Saída de Exemplo

Cada chamada ao LLM exibe os tokens utilizados:

```
Texto: Este produto é incrível! Superou todas as...
   Tokens - Input: 52 | Output: 3 | Total: 55
Sentimento: POSITIVO
```

Ao final de cada script, é exibido um resumo total:

```
============================================================
TOTAL - Zero-Shot Prompting
   Input:  1,234 tokens
   Output: 456 tokens
   Total:  1,690 tokens
============================================================
```

## Estrutura do Projeto

```
.
├── .env.example              # Template de configuração
├── .gitignore                # Arquivos ignorados pelo Git
├── README.md                 # Documentação em inglês
├── README.pt-BR.md           # Documentação em português
├── requirements.txt          # Dependências do projeto
├── config.py                 # Configuração centralizada + Token tracking
├── sample_data/              # Dados de exemplo para RAG, Vision e Context Engineering
│   ├── documents/            # Documentos de texto para RAG e Context Engineering
│   │   ├── ai_handbook.txt
│   │   ├── company_faq.txt
│   │   ├── technical_docs.md
│   │   ├── products_catalog.json   # Dados de produtos com metadados (Self-Query)
│   │   ├── news_articles.txt       # Artigos com datas (Time-Weighted)
│   │   └── long_document.txt       # Documento grande (Long Context)
│   └── images/               # Imagens para demos de Vision
│       ├── chart.png
│       ├── diagram.png
│       └── photo.jpg
└── techniques/
    ├── en/                   # Exemplos em inglês (45 scripts)
    │   ├── 01_zero_shot.py
    │   ├── ...
    │   ├── 20_meta_prompting.py
    │   ├── 21_advanced_chunking.py
    │   ├── ...
    │   ├── 30_time_weighted.py
    │   ├── 31_mcp_basics.py
    │   ├── ...
    │   ├── 40_fine_tuning.py
    │   ├── 41_agent_skills.py
    │   ├── 42_context_window.py
    │   ├── 43_subagent_orchestration.py
    │   ├── 44_shared_memory.py
    │   └── 45_spec_generation.py
    └── pt-br/                # Exemplos em português (45 scripts)
        ├── 01_zero_shot.py
        ├── ...
        ├── 20_meta_prompting.py
        ├── 21_advanced_chunking.py
        ├── ...
        ├── 30_time_weighted.py
        ├── 31_mcp_basics.py
        ├── ...
        ├── 40_fine_tuning.py
        ├── 41_agent_skills.py
        ├── 42_context_window.py
        ├── 43_subagent_orchestration.py
        ├── 44_shared_memory.py
        └── 45_spec_generation.py
```

## Configuração

O arquivo `config.py` fornece funções utilitárias:

```python
from config import get_llm, get_model_name, TokenUsage

# Criar instância do LLM com temperatura personalizada
llm = get_llm(temperature=0.7)

# Obter nome do modelo configurado
modelo = get_model_name()  # ex: "gpt-4o-mini"

# Criar tracker de tokens
tracker = TokenUsage()

# Para Ollama (modelos locais)
from config import get_ollama_llm, get_ollama_embeddings, is_ollama_available

if is_ollama_available():
    local_llm = get_ollama_llm(model="llama3.2")
    local_embeddings = get_ollama_embeddings()

# Para embeddings
from config import get_embeddings
embeddings = get_embeddings()  # Embeddings OpenAI
```

## Entendendo a Temperatura

A temperatura é um dos parâmetros mais importantes ao trabalhar com LLMs. Ela controla a **aleatoriedade** e **criatividade** das respostas do modelo.

### O que é Temperatura?

- **Intervalo:** 0.0 a 2.0 (o uso mais comum é de 0.0 a 1.0)
- **Valores baixos (0.0-0.3):** Respostas mais determinísticas, focadas e consistentes
- **Valores altos (0.7-1.0+):** Respostas mais criativas, diversas e imprevisíveis

### Temperatura por Técnica

| Técnica | Temperatura | Motivo |
|---------|-------------|--------|
| Classificação Zero-Shot | 0.0 | Resultados consistentes |
| Chain of Thought | 0.0 | Raciocínio preciso |
| Few-Shot | 0.0 - 0.3 | Seguir exemplos |
| Tree of Thoughts | 0.3 - 0.8 | Pensamentos diversos |
| Self-Consistency | 0.7 - 0.9 | Precisa de variação |
| Self-Refine | 0.3 - 0.5 | Crítica equilibrada |
| RAG | 0.0 - 0.3 | Respostas factuais |
| Structured Output | 0.0 | Schema consistente |
| Tool Calling | 0.0 | Uso confiável de ferramentas |
| Meta-Prompting | 0.5 - 0.7 | Prompts criativos |
| Transformação de Query | 0.3 - 0.7 | Variações criativas |
| Compressão Contextual | 0.0 | Extração precisa |
| Self-Query (geração filtro) | 0.0 | Filtros precisos |
| Multi-Vetor (resumos) | 0.3 | Resumos equilibrados |
| Contexto Longo | 0.0 - 0.3 | Síntese precisa |

## Modelos Suportados

### Modelos OpenAI

- `gpt-4o` - Mais capaz, mais caro
- `gpt-4o-mini` - Bom equilíbrio custo/performance (recomendado)
- `gpt-4-turbo` - Versão turbo do GPT-4
- `gpt-3.5-turbo` - Mais barato, menos capaz

### Modelos Ollama (Local)

- `llama3.2` - Llama 3 da Meta (recomendado)
- `mistral` - Mistral 7B
- `codellama` - Especializado em código
- `phi3` - Phi-3 da Microsoft

## Dependências

### Dependências Core
- `langchain` - Framework para LLMs
- `langchain-openai` - Integração com OpenAI
- `openai` - Cliente da API OpenAI
- `python-dotenv` - Variáveis de ambiente

### Dependências RAG
- `chromadb` - Banco de dados vetorial
- `sentence-transformers` - Embeddings locais
- `pypdf` - Processamento de PDF
- `unstructured` - Parsing de documentos

### Dependências Context Engineering
- `rank-bm25` - Busca por palavras-chave BM25 para Busca Híbrida

### Dependências Ollama
- `langchain-ollama` - Integração com Ollama

### Dependências Opcionais
- `cohere` - Reranking Cohere
- `pillow` - Processamento de imagens

## Dicas de Uso

### Prompt Engineering

1. **Comece com Zero-Shot** - É a técnica mais simples e funciona bem para tarefas diretas.

2. **Use CoT para raciocínio** - Problemas matemáticos, lógicos ou que requerem análise se beneficiam do "pense passo a passo".

3. **Few-Shot para formatos específicos** - Quando precisa de saída em formato específico (JSON, SQL, etc.), forneça exemplos.

4. **Self-Consistency para precisão** - Quando precisa de alta precisão em tarefas de raciocínio, gere múltiplas respostas e vote.

5. **Ollama para privacidade** - Use modelos locais quando privacidade de dados é importante ou quer evitar custos de API.

6. **Structured Output para APIs** - Ao construir integrações, use modelos Pydantic para garantir saída consistente.

### Context Engineering

7. **RAG para conhecimento** - Use RAG quando precisa que o modelo responda baseado em documentos específicos.

8. **Busca Híbrida para precisão** - Combine BM25 + Vetor quando consultas contêm termos específicos ou palavras-chave.

9. **Chunking Avançado para qualidade** - Escolha estratégia de chunking baseada no tipo de conteúdo (semântico para artigos, markdown-aware para docs).

10. **Transformação de Query para recall** - Use HyDE ou Multi-Query quando a qualidade inicial de recuperação está baixa.

11. **Compressão Contextual para tokens** - Comprima documentos recuperados para reduzir uso de tokens mantendo relevância.

12. **Self-Query para dados estruturados** - Use quando documentos têm metadados ricos que podem filtrar resultados.

13. **Documento-Pai para contexto** - Quando chunks recuperados faltam contexto ao redor, use recuperação documento-pai.

14. **Contexto Longo para docs grandes** - Use Map-Reduce para sumarização, Map-Rerank para Q&A em documentos grandes.

15. **Ponderação Temporal para atualidade** - Use quando recência do documento importa (notícias, logs, histórico de chat).

## Custos

Os scripts fazem chamadas à API da OpenAI, que cobra por tokens.

### Monitorando Custos

Cada script exibe automaticamente:
- Tokens de entrada (input) e saída (output) por chamada
- Total de tokens ao final da execução

Preços aproximados (janeiro 2025):
| Modelo | Input (1M tokens) | Output (1M tokens) |
|--------|-------------------|-------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-3.5-turbo | $0.50 | $1.50 |

### Minimizando Custos

- Use `gpt-4o-mini` (padrão) em vez de `gpt-4o`
- Use Ollama para inferência local (grátis)
- Reduza a quantidade de exemplos nos testes
- Monitore os totais de tokens exibidos ao final de cada script

## Licença

MIT
