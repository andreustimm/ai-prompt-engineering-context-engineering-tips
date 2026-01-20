# Prompt Engineering com LangChain e OpenAI

Scripts demonstrando 6 técnicas de Prompt Engineering usando LangChain e a API da OpenAI.

> **🌐 Language / Idioma:** Português Brasileiro | [English](README.md)

## Técnicas Implementadas

| Script | Técnica | Descrição |
|--------|---------|-----------|
| `01_zero_shot.py` | Zero-Shot | Prompts diretos sem exemplos prévios |
| `02_chain_of_thought.py` | Chain of Thought (CoT) | Raciocínio passo a passo |
| `03_few_shot.py` | Few-Shot | Exemplos para guiar o modelo |
| `04_tree_of_thoughts.py` | Tree of Thoughts (ToT) | Múltiplos caminhos de raciocínio |
| `05_skeleton_of_thought.py` | Skeleton of Thought (SoT) | Estrutura primeiro, detalhes depois |
| `06_react_agent.py` | ReAct | Raciocínio + Ações com ferramentas |

## Requisitos

- Python 3.10+
- Chave de API da OpenAI

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

Edite o arquivo `.env` e adicione sua chave da OpenAI:

```
OPENAI_API_KEY=sk-sua-chave-aqui
OPENAI_MODEL=gpt-4o-mini
```

## Uso

Execute qualquer script da pasta `techniques/`:

**Exemplos em Português:**
```bash
python techniques/pt-br/01_zero_shot.py
python techniques/pt-br/02_chain_of_thought.py
python techniques/pt-br/03_few_shot.py
python techniques/pt-br/04_tree_of_thoughts.py
python techniques/pt-br/05_skeleton_of_thought.py
python techniques/pt-br/06_react_agent.py
```

**Exemplos em Inglês:**
```bash
python techniques/en/01_zero_shot.py
python techniques/en/02_chain_of_thought.py
python techniques/en/03_few_shot.py
python techniques/en/04_tree_of_thoughts.py
python techniques/en/05_skeleton_of_thought.py
python techniques/en/06_react_agent.py
```

## Descrição das Técnicas

### 1. Zero-Shot Prompting

Técnica onde o modelo recebe uma tarefa sem exemplos prévios, usando apenas seu conhecimento pré-treinado.

**Funções disponíveis:**
- `classificar_sentimento(texto)` - Classifica sentimento como POSITIVO, NEGATIVO ou NEUTRO
- `traduzir_texto(texto, idioma_destino)` - Traduz texto para o idioma especificado
- `extrair_entidades(texto)` - Extrai pessoas, locais, organizações e datas
- `resumir_texto(texto)` - Resume texto em poucas frases

**Exemplo:**
```python
from techniques.pt_br.zero_shot import classificar_sentimento

resultado = classificar_sentimento("Este produto é incrível!")
print(resultado)  # POSITIVO
```

---

### 2. Chain of Thought (CoT)

Instrui o modelo a "pensar passo a passo" antes de chegar à resposta final, melhorando o desempenho em tarefas de raciocínio.

**Funções disponíveis:**
- `resolver_problema_matematico(problema)` - Resolve problemas matemáticos mostrando cada passo
- `raciocinio_logico(puzzle)` - Resolve puzzles lógicos com deduções
- `analisar_decisao(situacao)` - Analisa cenários para tomada de decisão
- `debug_codigo(codigo, erro)` - Analisa código e erro para encontrar solução

**Exemplo:**
```python
from techniques.pt_br.chain_of_thought import resolver_problema_matematico

problema = "João comprou 5 camisetas por R$ 45 cada com 15% de desconto. Quanto pagou?"
solucao = resolver_problema_matematico(problema)
print(solucao)
```

---

### 3. Few-Shot Prompting

Fornece exemplos ao modelo antes da tarefa, ajudando-o a entender o formato e tipo de resposta esperada.

**Funções disponíveis:**
- `classificar_ticket_suporte(ticket)` - Classifica tickets com categoria, prioridade e ação
- `converter_para_sql(descricao)` - Converte linguagem natural para SQL
- `gerar_docstring(codigo)` - Gera docstrings no padrão Google Style
- `extrair_dados_estruturados(texto)` - Extrai dados em formato JSON

**Exemplo:**
```python
from techniques.pt_br.few_shot import converter_para_sql

sql = converter_para_sql("Listar todos os clientes do Brasil")
print(sql)  # SELECT * FROM clientes WHERE pais = 'Brasil';
```

---

### 4. Tree of Thoughts (ToT)

Explora múltiplos caminhos de raciocínio em paralelo, avalia cada um e seleciona o mais promissor.

**Funções disponíveis:**
- `tree_of_thoughts(problema, profundidade)` - Executa algoritmo ToT completo
- `gerar_pensamentos(problema, num)` - Gera múltiplas abordagens iniciais
- `avaliar_pensamento(problema, pensamento)` - Avalia viabilidade de uma abordagem
- `expandir_pensamento(problema, pensamento, proximo_passo)` - Desenvolve uma abordagem

**Exemplo:**
```python
from techniques.pt_br.tree_of_thoughts import tree_of_thoughts

problema = "Como triplicar o faturamento da startup em 18 meses?"
solucao = tree_of_thoughts(problema, profundidade=2)
print(solucao)
```

---

### 5. Skeleton of Thought (SoT)

Gera primeiro um "esqueleto" (estrutura/tópicos) e depois expande cada parte, permitindo paralelização.

**Funções disponíveis:**
- `skeleton_of_thought_sync(tema, contexto)` - Versão síncrona
- `skeleton_of_thought_async(tema, contexto)` - Versão assíncrona (paralela)
- `gerar_esqueleto(tema, contexto)` - Gera lista de tópicos
- `expandir_topico(tema, topico, contexto)` - Expande um tópico específico

**Exemplo:**
```python
from techniques.pt_br.skeleton_of_thought import skeleton_of_thought_sync

documento = skeleton_of_thought_sync(
    tema="Inteligência Artificial na Medicina",
    contexto="Foco em aplicações práticas"
)
print(documento)
```

**Versão assíncrona (mais rápida):**
```python
import asyncio
from techniques.pt_br.skeleton_of_thought import skeleton_of_thought_async

documento = asyncio.run(skeleton_of_thought_async("Segurança em APIs REST"))
print(documento)
```

---

### 6. ReAct Agent

Combina raciocínio (Thought) com ações (Action) e observações (Observation) em um loop iterativo, usando ferramentas externas.

**Ferramentas disponíveis:**
- `busca_web` - Busca na internet via DuckDuckGo
- `wikipedia` - Consulta à Wikipedia
- `calculadora` - Cálculos matemáticos

**Funções disponíveis:**
- `executar_agente(pergunta)` - Executa agente ReAct para responder perguntas
- `criar_agente_react()` - Cria instância do agente configurado

**Exemplo:**
```python
from techniques.pt_br.react_agent import executar_agente

resposta = executar_agente(
    "Quem ganhou a última Copa do Mundo e em que país foi?"
)
print(resposta)
```

## Monitoramento de Tokens

Todos os scripts incluem **contagem automática de tokens** para ajudar a monitorar custos e uso da API.

### Saída de Exemplo

Cada chamada ao LLM exibe os tokens utilizados:

```
Texto: Este produto é incrível! Superou todas as...
   📊 Tokens - Input: 52 | Output: 3 | Total: 55
Sentimento: POSITIVO
```

Ao final de cada script, é exibido um resumo total:

```
============================================================
📈 TOTAL - Zero-Shot Prompting
   Input:  1,234 tokens
   Output: 456 tokens
   Total:  1,690 tokens
============================================================
```

### Usando o Token Tracker em Seu Código

```python
from config import TokenUsage, extract_tokens_from_response, print_token_usage

# Criar um tracker
tracker = TokenUsage()

# Após uma chamada ao LLM
response = chain.invoke({"input": "texto"})
input_tokens, output_tokens = extract_tokens_from_response(response)

# Registrar e exibir
tracker.add(input_tokens, output_tokens)
print_token_usage(input_tokens, output_tokens, "minha_funcao")

# Ver totais
print(f"Total usado: {tracker.total_tokens} tokens")
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
└── techniques/
    ├── en/                   # Exemplos em inglês
    │   ├── 01_zero_shot.py
    │   ├── 02_chain_of_thought.py
    │   ├── 03_few_shot.py
    │   ├── 04_tree_of_thoughts.py
    │   ├── 05_skeleton_of_thought.py
    │   └── 06_react_agent.py
    └── pt-br/                # Exemplos em português
        ├── 01_zero_shot.py
        ├── 02_chain_of_thought.py
        ├── 03_few_shot.py
        ├── 04_tree_of_thoughts.py
        ├── 05_skeleton_of_thought.py
        └── 06_react_agent.py
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
```

## Entendendo a Temperatura

A temperatura é um dos parâmetros mais importantes ao trabalhar com LLMs. Ela controla a **aleatoriedade** e **criatividade** das respostas do modelo.

### O que é Temperatura?

- **Intervalo:** 0.0 a 2.0 (o uso mais comum é de 0.0 a 1.0)
- **Valores baixos (0.0-0.3):** Respostas mais determinísticas, focadas e consistentes
- **Valores altos (0.7-1.0+):** Respostas mais criativas, diversas e imprevisíveis

### Quando Usar Temperatura Baixa (0.0 - 0.3)

Use temperatura baixa quando precisar de **precisão, consistência e previsibilidade**:

| Caso de Uso | Temperatura Recomendada |
|-------------|------------------------|
| Tarefas de classificação | 0.0 |
| Extração de entidades | 0.0 |
| Geração de código | 0.0 - 0.2 |
| Cálculos matemáticos | 0.0 |
| Perguntas e respostas factuais | 0.0 - 0.2 |
| Parsing/formatação de dados | 0.0 |
| Geração de queries SQL | 0.0 |

**Exemplo:**
```python
# Para classificação - sempre use temperature=0
llm = get_llm(temperature=0)
```

### Quando Usar Temperatura Média (0.3 - 0.7)

Use temperatura média para um **equilíbrio entre consistência e criatividade**:

| Caso de Uso | Temperatura Recomendada |
|-------------|------------------------|
| Resumo de textos | 0.3 - 0.5 |
| Tradução | 0.3 |
| Escrita de conteúdo geral | 0.5 - 0.7 |
| Explicação de conceitos | 0.5 |
| Redação de emails | 0.5 - 0.7 |

**Exemplo:**
```python
# Para geração de conteúdo - criatividade moderada
llm = get_llm(temperature=0.5)
```

### Quando Usar Temperatura Alta (0.7 - 1.0+)

Use temperatura alta quando precisar de **criatividade e diversidade**:

| Caso de Uso | Temperatura Recomendada |
|-------------|------------------------|
| Escrita criativa | 0.7 - 0.9 |
| Brainstorming de ideias | 0.8 - 1.0 |
| Poesia/narrativas | 0.8 - 1.0 |
| Geração de alternativas | 0.7 - 0.9 |
| Cenários de role-playing | 0.7 - 0.9 |

**Exemplo:**
```python
# Para brainstorming - alta criatividade
llm = get_llm(temperature=0.8)
```

### Temperatura Usada em Cada Técnica

| Técnica | Função | Temperatura | Motivo |
|---------|--------|-------------|--------|
| Zero-Shot | `classificar_sentimento` | 0.0 | Classificação consistente |
| Zero-Shot | `traduzir_texto` | 0.3 | Pequena variação na forma |
| Zero-Shot | `resumir_texto` | 0.5 | Resumo equilibrado |
| CoT | `resolver_problema_matematico` | 0.0 | Cálculos precisos |
| CoT | `analisar_decisao` | 0.3 | Estruturado mas flexível |
| Few-Shot | `converter_para_sql` | 0.0 | Sintaxe SQL exata |
| Few-Shot | `gerar_docstring` | 0.3 | Estilo consistente |
| ToT | `gerar_pensamentos` | 0.8 | Abordagens diversas |
| ToT | `avaliar_pensamento` | 0.3 | Avaliação consistente |
| SoT | `gerar_esqueleto` | 0.5 | Estrutura equilibrada |
| SoT | `expandir_topico` | 0.6 | Conteúdo criativo |
| ReAct | Agente | 0.0 | Uso confiável de ferramentas |

### Dicas sobre Temperatura

1. **Comece baixo, aumente se necessário** - Inicie com temperature=0 e aumente apenas se as respostas forem muito repetitivas ou sem criatividade.

2. **Mesma entrada, saídas diferentes** - Temperaturas mais altas significam que o mesmo prompt pode produzir resultados diferentes a cada vez.

3. **Produção vs Desenvolvimento** - Use temperaturas mais baixas em produção para consistência; mais altas em desenvolvimento para exploração.

4. **Combine com outros parâmetros** - A temperatura funciona com `top_p` (amostragem de núcleo). Geralmente, ajuste um ou outro, não ambos.

5. **Ajuste específico por tarefa** - A temperatura ideal depende do seu caso de uso específico. Teste diferentes valores.

## Modelos Suportados

Você pode usar qualquer modelo da OpenAI alterando a variável `OPENAI_MODEL` no `.env`:

- `gpt-4o` - Mais capaz, mais caro
- `gpt-4o-mini` - Bom equilíbrio custo/performance (recomendado)
- `gpt-4-turbo` - Versão turbo do GPT-4
- `gpt-3.5-turbo` - Mais barato, menos capaz

## Dicas de Uso

1. **Comece com Zero-Shot** - É a técnica mais simples e funciona bem para tarefas diretas.

2. **Use CoT para raciocínio** - Problemas matemáticos, lógicos ou que requerem análise se beneficiam do "pense passo a passo".

3. **Few-Shot para formatos específicos** - Quando precisa de saída em formato específico (JSON, SQL, etc.), forneça exemplos.

4. **ToT para problemas complexos** - Use quando há múltiplas soluções possíveis e precisa avaliar trade-offs.

5. **SoT para conteúdo longo** - Ideal para gerar artigos, documentação ou respostas estruturadas.

6. **ReAct para informações externas** - Use quando precisa de dados atualizados ou cálculos.

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
- Reduza a quantidade de exemplos nos testes
- Comente as demonstrações que não precisa executar
- Monitore os totais de tokens exibidos ao final de cada script

## Licença

MIT
