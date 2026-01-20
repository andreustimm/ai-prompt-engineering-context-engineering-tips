# Prompt Engineering with LangChain and OpenAI

Scripts demonstrating 20 Prompt Engineering techniques using LangChain and the OpenAI API.

> **Language / Idioma:** [Português Brasileiro](README.pt-BR.md) | English

## Implemented Techniques

### Basic Prompting (01-06)

| Script | Technique | Description |
|--------|-----------|-------------|
| `01_zero_shot.py` | Zero-Shot | Direct prompts without prior examples |
| `02_chain_of_thought.py` | Chain of Thought (CoT) | Step-by-step reasoning |
| `03_few_shot.py` | Few-Shot | Examples to guide the model |
| `04_tree_of_thoughts.py` | Tree of Thoughts (ToT) | Multiple reasoning paths |
| `05_skeleton_of_thought.py` | Skeleton of Thought (SoT) | Structure first, details later |
| `06_react_agent.py` | ReAct | Reasoning + Actions with tools |

### Advanced Prompting (07-10)

| Script | Technique | Description |
|--------|-----------|-------------|
| `07_self_consistency.py` | Self-Consistency | Generate N responses, vote on most consistent |
| `08_least_to_most.py` | Least-to-Most | Progressive decomposition into sub-problems |
| `09_self_refine.py` | Self-Refine | Iterative critique and improvement |
| `10_prompt_chaining.py` | Prompt Chaining | Pipeline of connected prompts |

### RAG - Retrieval-Augmented Generation (11-13)

| Script | Technique | Description |
|--------|-----------|-------------|
| `11_rag_basic.py` | RAG Basic | ChromaDB + semantic search + chunking |
| `12_rag_reranking.py` | RAG + Reranking | Reordering for better relevance |
| `13_rag_conversational.py` | Conversational RAG | RAG with chat memory |

### Ollama - Local Models (14-15)

| Script | Technique | Description |
|--------|-----------|-------------|
| `14_ollama_basic.py` | Ollama Basic | Local LLMs (Llama 3, Mistral) |
| `15_ollama_rag.py` | Ollama + RAG | 100% offline RAG |

### Structured Output & Tools (16-17)

| Script | Technique | Description |
|--------|-----------|-------------|
| `16_structured_output.py` | Structured Output | JSON mode + Pydantic models |
| `17_tool_calling.py` | Tool Calling | Custom function tools |

### Advanced Features (18-20)

| Script | Technique | Description |
|--------|-----------|-------------|
| `18_vision_multimodal.py` | Vision/Multimodal | Image analysis with GPT-4o |
| `19_memory_conversation.py` | Memory/Conversation | Persistent conversation context |
| `20_meta_prompting.py` | Meta-Prompting | LLM generating/optimizing prompts |

## Requirements

- Python 3.10+
- OpenAI API key
- (Optional) Ollama for local models
- (Optional) Cohere API key for reranking

## Installation

1. **Clone or navigate to the project directory:**

```bash
cd /path/to/project
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure credentials:**

```bash
cp .env.example .env
```

Edit the `.env` file and add your keys:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Optional - for Ollama (local models)
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Optional - for Cohere reranking
COHERE_API_KEY=your-cohere-key-here
```

5. **(Optional) Install Ollama for local models:**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model
ollama pull llama3.2
ollama pull nomic-embed-text  # For embeddings
```

## Usage

Execute any script from the `techniques/` folder:

**English examples:**
```bash
# Basic Prompting (01-06)
python techniques/en/01_zero_shot.py
python techniques/en/02_chain_of_thought.py
python techniques/en/03_few_shot.py
python techniques/en/04_tree_of_thoughts.py
python techniques/en/05_skeleton_of_thought.py
python techniques/en/06_react_agent.py

# Advanced Prompting (07-10)
python techniques/en/07_self_consistency.py
python techniques/en/08_least_to_most.py
python techniques/en/09_self_refine.py
python techniques/en/10_prompt_chaining.py

# RAG (11-13) - Requires sample_data/
python techniques/en/11_rag_basic.py
python techniques/en/12_rag_reranking.py
python techniques/en/13_rag_conversational.py

# Ollama (14-15) - Requires Ollama running
python techniques/en/14_ollama_basic.py
python techniques/en/15_ollama_rag.py

# Structured Output & Tools (16-17)
python techniques/en/16_structured_output.py
python techniques/en/17_tool_calling.py

# Advanced Features (18-20)
python techniques/en/18_vision_multimodal.py
python techniques/en/19_memory_conversation.py
python techniques/en/20_meta_prompting.py
```

**Portuguese examples:**
```bash
python techniques/pt-br/01_zero_shot.py
# ... (same pattern with pt-br/)
```

## Technique Descriptions

### 1. Zero-Shot Prompting

Technique where the model receives a task without prior examples, using only its pre-trained knowledge.

**Available functions:**
- `classify_sentiment(text)` - Classifies sentiment as POSITIVE, NEGATIVE, or NEUTRAL
- `translate_text(text, target_language)` - Translates text to specified language
- `extract_entities(text)` - Extracts people, locations, organizations, and dates
- `summarize_text(text)` - Summarizes text in a few sentences

---

### 2. Chain of Thought (CoT)

Instructs the model to "think step by step" before reaching the final answer, improving performance on reasoning tasks.

**Available functions:**
- `solve_math_problem(problem)` - Solves math problems showing each step
- `logical_reasoning(puzzle)` - Solves logic puzzles with deductions
- `analyze_decision(situation)` - Analyzes scenarios for decision making
- `debug_code(code, error)` - Analyzes code and error to find solution

---

### 3. Few-Shot Prompting

Provides examples to the model before the task, helping it understand the format and expected response type.

**Available functions:**
- `classify_support_ticket(ticket)` - Classifies tickets with category, priority, and action
- `convert_to_sql(description)` - Converts natural language to SQL
- `generate_docstring(code)` - Generates Google Style docstrings
- `extract_structured_data(text)` - Extracts data in JSON format

---

### 4. Tree of Thoughts (ToT)

Explores multiple reasoning paths in parallel, evaluates each one, and selects the most promising.

**Available functions:**
- `tree_of_thoughts(problem, depth)` - Executes complete ToT algorithm
- `generate_thoughts(problem, num)` - Generates multiple initial approaches
- `evaluate_thought(problem, thought)` - Evaluates viability of an approach
- `expand_thought(problem, thought, next_step)` - Develops an approach

---

### 5. Skeleton of Thought (SoT)

First generates a "skeleton" (structure/topics) and then expands each part, allowing parallelization.

**Available functions:**
- `skeleton_of_thought_sync(topic, context)` - Synchronous version
- `skeleton_of_thought_async(topic, context)` - Asynchronous version (parallel)
- `generate_skeleton(topic, context)` - Generates list of topics
- `expand_topic(main_topic, topic, context)` - Expands a specific topic

---

### 6. ReAct Agent

Combines reasoning (Thought) with actions (Action) and observations (Observation) in an iterative loop, using external tools.

**Available tools:**
- `web_search` - Internet search via DuckDuckGo
- `wikipedia` - Wikipedia queries
- `calculator` - Mathematical calculations

---

### 7. Self-Consistency

Generates multiple responses to the same problem, then uses majority voting to select the most consistent answer. Improves accuracy on reasoning tasks.

**Available functions:**
- `self_consistency_solve(problem, num_samples)` - Solves with multiple samples and voting
- `solve_with_voting(problem, num_samples)` - Alternative with explicit voting
- `extract_answer(response)` - Extracts final answer from response

**Example:**
```python
from techniques.en.self_consistency import self_consistency_solve

result = self_consistency_solve(
    "If a train travels 120 km in 2 hours, what is its speed?",
    num_samples=5
)
print(result["final_answer"])  # 60 km/h
```

---

### 8. Least-to-Most Prompting

Decomposes complex problems into smaller sub-problems, solves them progressively from simplest to most complex, building on previous answers.

**Available functions:**
- `least_to_most_solve(problem)` - Complete decomposition and solution
- `decompose_problem(problem)` - Breaks down into sub-problems
- `solve_subproblem(subproblem, context)` - Solves with previous context

**Example:**
```python
from techniques.en.least_to_most import least_to_most_solve

result = least_to_most_solve(
    "How do I build a machine learning model to predict house prices?"
)
print(result["final_answer"])
```

---

### 9. Self-Refine

Generates an initial response, then iteratively critiques and improves it until it meets quality standards.

**Available functions:**
- `self_refine(task, max_iterations)` - Complete refinement loop
- `generate_initial(task)` - Creates first draft
- `critique(task, response)` - Evaluates and identifies issues
- `refine(task, response, feedback)` - Improves based on critique

**Example:**
```python
from techniques.en.self_refine import self_refine

result = self_refine(
    "Write a function to check if a string is a palindrome",
    max_iterations=3
)
print(result["final_response"])
```

---

### 10. Prompt Chaining

Connects multiple prompts in a pipeline where the output of one becomes the input of the next, enabling complex multi-step workflows.

**Available functions:**
- `chain_prompts(initial_input, prompt_chain)` - Executes prompt pipeline
- `research_chain(topic)` - Research → Analysis → Summary
- `content_chain(topic)` - Outline → Draft → Edit → Format

**Example:**
```python
from techniques.en.prompt_chaining import content_chain

result = content_chain("Benefits of Remote Work")
print(result["final_output"])
```

---

### 11. RAG Basic

Retrieval-Augmented Generation with ChromaDB for document storage, semantic search, and text chunking.

**Available functions:**
- `create_vectorstore(documents)` - Creates ChromaDB vector store
- `rag_query(question, vectorstore)` - Queries with RAG
- `load_and_split_documents(path)` - Loads and chunks documents

**Key features:**
- Recursive text chunking (1000 chars, 200 overlap)
- OpenAI embeddings for semantic search
- Top-k retrieval with relevance scores

**Example:**
```python
from techniques.en.rag_basic import create_vectorstore, rag_query

# Load documents and create vector store
vectorstore = create_vectorstore(documents)

# Query with RAG
result = rag_query("What is machine learning?", vectorstore)
print(result["answer"])
```

---

### 12. RAG + Reranking

Enhances basic RAG with reranking to improve retrieval relevance. Supports multiple reranking methods.

**Reranking methods:**
- LLM-based reranking (uses GPT to score relevance)
- Cohere Rerank (requires API key)
- CrossEncoder (local transformer model)

**Available functions:**
- `rag_with_reranking(question, vectorstore, method)` - RAG with reranking
- `llm_rerank(question, documents)` - LLM-based reranking
- `cohere_rerank(question, documents)` - Cohere API reranking

---

### 13. Conversational RAG

RAG with conversation memory for multi-turn dialogues. Maintains context across questions.

**Memory types:**
- Buffer Memory - Full conversation history
- Summary Memory - Compressed summary

**Available functions:**
- `create_conversational_rag(vectorstore)` - Creates conversational chain
- `chat(question)` - Chat with memory
- `get_chat_history()` - Retrieve conversation history

---

### 14. Ollama Basic

Use local LLMs via Ollama without API costs or internet dependency.

**Supported models:**
- `llama3.2` - Meta's Llama 3
- `mistral` - Mistral 7B
- `codellama` - Code-specialized Llama
- `phi3` - Microsoft's Phi-3

**Available functions:**
- `ollama_chat(message)` - Chat with local model
- `ollama_generate(prompt)` - Text generation
- `list_local_models()` - List available models

**Example:**
```python
from techniques.en.ollama_basic import ollama_chat

response = ollama_chat("Explain quantum computing in simple terms")
print(response)
```

---

### 15. Ollama + RAG

100% offline RAG using Ollama for both embeddings and generation.

**Components:**
- Local embeddings: `nomic-embed-text`
- Local LLM: `llama3.2` or `mistral`
- ChromaDB for vector storage

**Available functions:**
- `create_local_vectorstore(documents)` - Creates store with local embeddings
- `local_rag_query(question, vectorstore)` - Query with local RAG

---

### 16. Structured Output

Force LLM outputs to follow specific schemas using Pydantic models or JSON mode.

**Available functions:**
- `extract_person(text)` - Extract person info as Pydantic model
- `extract_invoice(text)` - Extract invoice data
- `json_mode_extract(text, schema)` - Generic JSON extraction

**Example:**
```python
from techniques.en.structured_output import extract_person
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

result = extract_person("John is a 30-year-old software engineer")
print(result.name)  # John
print(result.age)   # 30
```

---

### 17. Tool Calling

Enable LLMs to call custom functions/tools to perform actions or retrieve information.

**Available tools:**
- `get_weather(city)` - Get weather information
- `calculate(expression)` - Perform calculations
- `search_database(query)` - Search mock database

**Example:**
```python
from techniques.en.tool_calling import agent_with_tools

response = agent_with_tools(
    "What's the weather in Tokyo and calculate 15% tip on $85"
)
print(response)
```

---

### 18. Vision/Multimodal

Analyze images using vision-enabled models like GPT-4o.

**Available functions:**
- `analyze_image(image_path, prompt)` - Analyze image with custom prompt
- `describe_image(image_path)` - Generate detailed description
- `extract_text_from_image(image_path)` - OCR-like text extraction
- `analyze_chart(image_path)` - Analyze charts and graphs
- `compare_images(image1, image2)` - Compare two images

**Example:**
```python
from techniques.en.vision_multimodal import analyze_chart

result = analyze_chart("sample_data/images/chart.png")
print(result)  # Chart type, data, insights
```

---

### 19. Memory/Conversation

Maintain conversation context across multiple interactions using different memory strategies.

**Memory types:**
- `BufferMemory` - Stores complete conversation history
- `WindowMemory` - Stores last N exchanges
- `SummaryMemory` - Maintains compressed summary
- `EntityMemory` - Tracks mentioned entities

**Example:**
```python
from techniques.en.memory_conversation import ConversationChain

chain = ConversationChain(memory_type="buffer")
response1 = chain.chat("My name is Alice")
response2 = chain.chat("What's my name?")  # Remembers "Alice"
```

---

### 20. Meta-Prompting

Use an LLM to generate, optimize, and improve prompts for other LLM tasks.

**Available functions:**
- `generate_prompt(task_description)` - Generate optimized prompt
- `optimize_prompt(original_prompt, issues)` - Improve existing prompt
- `evaluate_prompt(prompt, task)` - Score and critique a prompt
- `generate_prompt_variations(base_prompt)` - A/B testing variations
- `auto_improve_prompt(prompt, task, test_input)` - Iterative improvement

**Example:**
```python
from techniques.en.meta_prompting import generate_prompt

prompt = generate_prompt(
    task_description="Extract key information from customer emails",
    context="SaaS company support",
    constraints=["JSON output", "Include urgency level"]
)
print(prompt)
```

## Token Monitoring

All scripts include **automatic token counting** to help monitor costs and API usage.

### Sample Output

Each LLM call displays the tokens used:

```
Text: This product is amazing! It exceeded all...
   Tokens - Input: 52 | Output: 3 | Total: 55
Sentiment: POSITIVE
```

At the end of each script, a total summary is displayed:

```
============================================================
TOTAL - Zero-Shot Prompting
   Input:  1,234 tokens
   Output: 456 tokens
   Total:  1,690 tokens
============================================================
```

## Project Structure

```
.
├── .env.example              # Configuration template
├── .gitignore                # Files ignored by Git
├── README.md                 # English documentation
├── README.pt-BR.md           # Portuguese documentation
├── requirements.txt          # Project dependencies
├── config.py                 # Centralized config + Token tracking
├── sample_data/              # Sample data for RAG and Vision
│   ├── documents/            # Text documents for RAG
│   │   ├── ai_handbook.txt
│   │   ├── company_faq.txt
│   │   └── technical_docs.md
│   └── images/               # Images for Vision demos
│       ├── chart.png
│       ├── diagram.png
│       └── photo.jpg
└── techniques/
    ├── en/                   # English examples (20 scripts)
    │   ├── 01_zero_shot.py
    │   ├── ...
    │   └── 20_meta_prompting.py
    └── pt-br/                # Portuguese examples (20 scripts)
        ├── 01_zero_shot.py
        ├── ...
        └── 20_meta_prompting.py
```

## Configuration

The `config.py` file provides utility functions:

```python
from config import get_llm, get_model_name, TokenUsage

# Create LLM instance with custom temperature
llm = get_llm(temperature=0.7)

# Get configured model name
model = get_model_name()  # e.g., "gpt-4o-mini"

# Create token tracker
tracker = TokenUsage()

# For Ollama (local models)
from config import get_ollama_llm, get_ollama_embeddings, is_ollama_available

if is_ollama_available():
    local_llm = get_ollama_llm(model="llama3.2")
    local_embeddings = get_ollama_embeddings()

# For embeddings
from config import get_embeddings
embeddings = get_embeddings()  # OpenAI embeddings
```

## Understanding Temperature

Temperature is one of the most important parameters when working with LLMs. It controls the **randomness** and **creativity** of the model's responses.

### What is Temperature?

- **Range:** 0.0 to 2.0 (most common usage is 0.0 to 1.0)
- **Low values (0.0-0.3):** More deterministic, focused, and consistent responses
- **High values (0.7-1.0+):** More creative, diverse, and unpredictable responses

### Temperature by Technique

| Technique | Temperature | Reason |
|-----------|-------------|--------|
| Zero-Shot Classification | 0.0 | Consistent results |
| Chain of Thought | 0.0 | Accurate reasoning |
| Few-Shot | 0.0 - 0.3 | Follow examples |
| Tree of Thoughts | 0.3 - 0.8 | Diverse thoughts |
| Self-Consistency | 0.7 - 0.9 | Need variation |
| Self-Refine | 0.3 - 0.5 | Balanced critique |
| RAG | 0.0 - 0.3 | Factual answers |
| Structured Output | 0.0 | Consistent schema |
| Tool Calling | 0.0 | Reliable tool use |
| Meta-Prompting | 0.5 - 0.7 | Creative prompts |

## Supported Models

### OpenAI Models

- `gpt-4o` - Most capable, most expensive
- `gpt-4o-mini` - Good cost/performance balance (recommended)
- `gpt-4-turbo` - Turbo version of GPT-4
- `gpt-3.5-turbo` - Cheaper, less capable

### Ollama Models (Local)

- `llama3.2` - Meta's Llama 3 (recommended)
- `mistral` - Mistral 7B
- `codellama` - Code-specialized
- `phi3` - Microsoft Phi-3

## Dependencies

### Core Dependencies
- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `openai` - OpenAI API client
- `python-dotenv` - Environment variables

### RAG Dependencies
- `chromadb` - Vector database
- `sentence-transformers` - Local embeddings
- `pypdf` - PDF processing
- `unstructured` - Document parsing

### Ollama Dependencies
- `langchain-ollama` - Ollama integration

### Optional Dependencies
- `cohere` - Cohere reranking
- `pillow` - Image processing

## Usage Tips

1. **Start with Zero-Shot** - It's the simplest technique and works well for direct tasks.

2. **Use CoT for reasoning** - Mathematical, logical problems, or those requiring analysis benefit from "think step by step".

3. **Few-Shot for specific formats** - When you need output in a specific format (JSON, SQL, etc.), provide examples.

4. **Self-Consistency for accuracy** - When you need high accuracy on reasoning tasks, generate multiple responses and vote.

5. **RAG for knowledge** - Use RAG when you need the model to answer based on specific documents.

6. **Ollama for privacy** - Use local models when data privacy is important or you want to avoid API costs.

7. **Structured Output for APIs** - When building integrations, use Pydantic models to ensure consistent output.

## Costs

The scripts make calls to the OpenAI API, which charges per token.

### Monitoring Costs

Each script automatically displays:
- Input and output tokens per call
- Total tokens at the end of execution

Approximate prices (January 2025):
| Model | Input (1M tokens) | Output (1M tokens) |
|-------|-------------------|-------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-3.5-turbo | $0.50 | $1.50 |

### Minimizing Costs

- Use `gpt-4o-mini` (default) instead of `gpt-4o`
- Use Ollama for local inference (free)
- Reduce the number of examples in tests
- Monitor token totals displayed at the end of each script

## License

MIT
