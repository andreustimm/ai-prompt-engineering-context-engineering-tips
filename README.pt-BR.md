# Prompt Engineering com LangChain e OpenAI

Scripts demonstrando 20 técnicas de Prompt Engineering usando LangChain e a API da OpenAI.

> **Language / Idioma:** Português Brasileiro | [English](README.md)

## Técnicas Implementadas

### Prompting Básico (01-06)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `01_zero_shot.py` | Zero-Shot | Prompts diretos sem exemplos prévios |
| `02_chain_of_thought.py` | Chain of Thought (CoT) | Raciocínio passo a passo |
| `03_few_shot.py` | Few-Shot | Exemplos para guiar o modelo |
| `04_tree_of_thoughts.py` | Tree of Thoughts (ToT) | Múltiplos caminhos de raciocínio |
| `05_skeleton_of_thought.py` | Skeleton of Thought (SoT) | Estrutura primeiro, detalhes depois |
| `06_react_agent.py` | ReAct | Raciocínio + Ações com ferramentas |

### Prompting Avançado (07-10)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `07_self_consistency.py` | Self-Consistency | Gera N respostas, vota na mais consistente |
| `08_least_to_most.py` | Least-to-Most | Decomposição progressiva em sub-problemas |
| `09_self_refine.py` | Self-Refine | Crítica e melhoria iterativa |
| `10_prompt_chaining.py` | Prompt Chaining | Pipeline de prompts conectados |

### RAG - Geração Aumentada por Recuperação (11-13)

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `11_rag_basic.py` | RAG Básico | ChromaDB + busca semântica + chunking |
| `12_rag_reranking.py` | RAG + Reranking | Reordenação para melhor relevância |
| `13_rag_conversational.py` | RAG Conversacional | RAG com memória de chat |

### Ollama - Modelos Locais (14-15)

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
├── sample_data/              # Dados de exemplo para RAG e Vision
│   ├── documents/            # Documentos de texto para RAG
│   │   ├── ai_handbook.txt
│   │   ├── company_faq.txt
│   │   └── technical_docs.md
│   └── images/               # Imagens para demos de Vision
│       ├── chart.png
│       ├── diagram.png
│       └── photo.jpg
└── techniques/
    ├── en/                   # Exemplos em inglês (20 scripts)
    │   ├── 01_zero_shot.py
    │   ├── ...
    │   └── 20_meta_prompting.py
    └── pt-br/                # Exemplos em português (20 scripts)
        ├── 01_zero_shot.py
        ├── ...
        └── 20_meta_prompting.py
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

### Dependências Ollama
- `langchain-ollama` - Integração com Ollama

### Dependências Opcionais
- `cohere` - Reranking Cohere
- `pillow` - Processamento de imagens

## Dicas de Uso

1. **Comece com Zero-Shot** - É a técnica mais simples e funciona bem para tarefas diretas.

2. **Use CoT para raciocínio** - Problemas matemáticos, lógicos ou que requerem análise se beneficiam do "pense passo a passo".

3. **Few-Shot para formatos específicos** - Quando precisa de saída em formato específico (JSON, SQL, etc.), forneça exemplos.

4. **Self-Consistency para precisão** - Quando precisa de alta precisão em tarefas de raciocínio, gere múltiplas respostas e vote.

5. **RAG para conhecimento** - Use RAG quando precisa que o modelo responda baseado em documentos específicos.

6. **Ollama para privacidade** - Use modelos locais quando privacidade de dados é importante ou quer evitar custos de API.

7. **Structured Output para APIs** - Ao construir integrações, use modelos Pydantic para garantir saída consistente.

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
