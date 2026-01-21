# Prompt Engineering & Context Engineering with LangChain and OpenAI

Scripts demonstrating 35 Prompt Engineering, Context Engineering, and Agentic AI techniques using LangChain and the OpenAI API.

> **Language / Idioma:** [Português Brasileiro](README.pt-BR.md) | English

> **Study Roadmap:** Want to master AI-assisted development? Check out the [ROADMAP.md](ROADMAP.md) - a comprehensive guide with learning tracks (beginner to advanced), technique connection maps, project organization templates, and modern tooling integration with Claude Code.

## Implemented Techniques

### Prompt Engineering - Basic (01-06)

| Script | Technique | Description |
|--------|-----------|-------------|
| `01_zero_shot.py` | Zero-Shot | Direct prompts without prior examples |
| `02_chain_of_thought.py` | Chain of Thought (CoT) | Step-by-step reasoning |
| `03_few_shot.py` | Few-Shot | Examples to guide the model |
| `04_tree_of_thoughts.py` | Tree of Thoughts (ToT) | Multiple reasoning paths |
| `05_skeleton_of_thought.py` | Skeleton of Thought (SoT) | Structure first, details later |
| `06_react_agent.py` | ReAct | Reasoning + Actions with tools |

### Prompt Engineering - Advanced (07-10)

| Script | Technique | Description |
|--------|-----------|-------------|
| `07_self_consistency.py` | Self-Consistency | Generate N responses, vote on most consistent |
| `08_least_to_most.py` | Least-to-Most | Progressive decomposition into sub-problems |
| `09_self_refine.py` | Self-Refine | Iterative critique and improvement |
| `10_prompt_chaining.py` | Prompt Chaining | Pipeline of connected prompts |

### Context Engineering - RAG (11-13)

| Script | Technique | Description |
|--------|-----------|-------------|
| `11_rag_basic.py` | RAG Basic | ChromaDB + semantic search + chunking |
| `12_rag_reranking.py` | RAG + Reranking | Reordering for better relevance |
| `13_rag_conversational.py` | Conversational RAG | RAG with chat memory |

### Local Models - Ollama (14-15)

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

### Context Engineering - Chunking & Retrieval (21-25)

| Script | Technique | Description |
|--------|-----------|-------------|
| `21_advanced_chunking.py` | Advanced Chunking | Semantic, recursive, token-based, sliding window strategies |
| `22_hybrid_search.py` | Hybrid Search | BM25 (keyword) + Vector (semantic) with RRF fusion |
| `23_query_transformation.py` | Query Transformation | HyDE, Multi-Query, Step-Back, Decomposition |
| `24_contextual_compression.py` | Contextual Compression | Extract only relevant parts from documents |
| `25_self_query.py` | Self-Query Retrieval | LLM auto-generates metadata filters |

### Context Engineering - Context Management (26-30)

| Script | Technique | Description |
|--------|-----------|-------------|
| `26_parent_document.py` | Parent-Document Retrieval | Small chunks for search, large parents for context |
| `27_multi_vector.py` | Multi-Vector Retrieval | Multiple representations (summary + questions + content) |
| `28_ensemble_retrieval.py` | Ensemble Retrieval | Combine multiple retrievers with weighted RRF |
| `29_long_context.py` | Long Context Strategies | Map-Reduce, Refine, Map-Rerank for large documents |
| `30_time_weighted.py` | Time-Weighted Retrieval | Recency bias in retrieval with exponential decay |

### MCP & Agentic AI (31-35)

| Script | Technique | Description |
|--------|-----------|-------------|
| `31_mcp_basics.py` | MCP Basics | Model Context Protocol fundamentals (resources, tools, prompts) |
| `32_mcp_server_stdio.py` | MCP Server STDIO | Local MCP server with standard input/output transport |
| `33_mcp_server_http.py` | MCP Server HTTP/SSE | Remote MCP server with HTTP and Server-Sent Events |
| `34_multi_agent.py` | Multi-Agent | Collaborative AI agents (pipeline, debate, hierarchical patterns) |
| `35_prompt_evaluation.py` | Prompt Evaluation | Evaluate prompt quality, A/B testing, observability |

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

# Context Engineering - Chunking & Retrieval (21-25)
python techniques/en/21_advanced_chunking.py
python techniques/en/22_hybrid_search.py
python techniques/en/23_query_transformation.py
python techniques/en/24_contextual_compression.py
python techniques/en/25_self_query.py

# Context Engineering - Context Management (26-30)
python techniques/en/26_parent_document.py
python techniques/en/27_multi_vector.py
python techniques/en/28_ensemble_retrieval.py
python techniques/en/29_long_context.py
python techniques/en/30_time_weighted.py

# MCP & Agentic AI (31-35)
python techniques/en/31_mcp_basics.py
python techniques/en/32_mcp_server_stdio.py
python techniques/en/33_mcp_server_http.py
python techniques/en/34_multi_agent.py
python techniques/en/35_prompt_evaluation.py
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

---

### 21. Advanced Chunking

Multiple text splitting strategies optimized for different content types and retrieval scenarios.

**Chunking strategies:**
- `RecursiveCharacter` - Hierarchical splitting by separators
- `TokenBased` - Split by token count (model-aware)
- `MarkdownAware` - Respects markdown structure
- `Semantic` - Groups by semantic similarity
- `SlidingWindow` - Overlapping fixed-size windows
- `SentenceBased` - Natural sentence boundaries

**Available functions:**
- `recursive_character_chunking(text, chunk_size, overlap)` - Standard recursive splitting
- `token_based_chunking(text, chunk_size)` - Token-aware splitting
- `markdown_aware_chunking(text)` - Structure-preserving for markdown
- `semantic_chunking(text, threshold)` - Similarity-based grouping
- `sliding_window_chunking(text, window_size, step)` - Overlapping windows
- `sentence_based_chunking(text, sentences_per_chunk)` - Sentence grouping

**Example:**
```python
from techniques.en.advanced_chunking import semantic_chunking

chunks = semantic_chunking(long_document, threshold=0.75)
for chunk in chunks:
    print(f"Chunk: {len(chunk)} chars")
```

---

### 22. Hybrid Search

Combines keyword-based (BM25) and semantic (vector) search using Reciprocal Rank Fusion.

**Components:**
- `BM25Retriever` - Traditional keyword matching
- `VectorRetriever` - Semantic similarity search
- `HybridRetriever` - Weighted combination

**Available functions:**
- `create_hybrid_retriever(documents, bm25_weight, vector_weight)` - Create hybrid retriever
- `reciprocal_rank_fusion(results_list, k)` - Combine ranked results
- `hybrid_search(query, k)` - Search with both methods

**Example:**
```python
from techniques.en.hybrid_search import HybridRetriever

retriever = HybridRetriever(documents, bm25_weight=0.4, vector_weight=0.6)
results = retriever.search("machine learning algorithms", k=5)
```

---

### 23. Query Transformation

Transform queries to improve retrieval effectiveness using various techniques.

**Transformation methods:**
- `HyDE` - Hypothetical Document Embeddings (generate hypothetical answer, search with that)
- `Multi-Query` - Generate multiple query variations
- `Step-Back` - Abstract query to broader concept
- `Decomposition` - Break complex query into sub-queries

**Available functions:**
- `hyde_transform(query)` - Generate hypothetical document
- `multi_query_transform(query, num_queries)` - Generate query variations
- `step_back_transform(query)` - Abstract to broader question
- `decompose_query(query)` - Split into sub-questions

**Example:**
```python
from techniques.en.query_transformation import multi_query_transform

queries = multi_query_transform(
    "What are the best practices for microservices?",
    num_queries=3
)
# Returns variations like:
# - "microservices architecture best practices"
# - "how to design microservices effectively"
# - "recommended patterns for microservice development"
```

---

### 24. Contextual Compression

Extract only the relevant portions of retrieved documents to reduce noise and token usage.

**Compression methods:**
- `LLMExtractor` - Use LLM to extract relevant sentences
- `EmbeddingsFilter` - Filter by semantic similarity
- `SentenceExtractor` - Extract relevant sentences by scoring

**Available functions:**
- `create_compression_retriever(base_retriever, compressor)` - Wrap retriever with compression
- `llm_compress(documents, query)` - LLM-based compression
- `embeddings_filter(documents, query, threshold)` - Similarity-based filtering

**Example:**
```python
from techniques.en.contextual_compression import ContextualCompressionRetriever

compression_retriever = ContextualCompressionRetriever(
    base_retriever=vector_retriever,
    compressor=LLMExtractorCompressor()
)
# Returns only relevant excerpts instead of full documents
results = compression_retriever.retrieve("What is RAG?")
```

---

### 25. Self-Query Retrieval

LLM automatically generates metadata filters from natural language queries.

**Features:**
- Automatic filter extraction from queries
- Support for comparison operators (=, >, <, >=, <=)
- Combines semantic search with structured filtering

**Available functions:**
- `create_self_query_retriever(vectorstore, metadata_info)` - Create self-query retriever
- `parse_query(query)` - Extract semantic query and filters
- `apply_filters(documents, filters)` - Apply metadata filters

**Example:**
```python
from techniques.en.self_query import SelfQueryRetriever

retriever = SelfQueryRetriever(
    vectorstore=vectorstore,
    metadata_fields=[
        {"name": "category", "type": "string"},
        {"name": "price", "type": "float"},
        {"name": "year", "type": "integer"}
    ]
)

# Query: "cheap electronics from 2024"
# Auto-generates: category="electronics", price<100, year=2024
results = retriever.retrieve("cheap electronics from 2024")
```

---

### 26. Parent-Document Retrieval

Search with small chunks for precision, but retrieve larger parent documents for context.

**Concept:**
- Child chunks: Small (e.g., 400 chars) for precise matching
- Parent documents: Larger (e.g., 2000 chars) for complete context
- Map child → parent for retrieval

**Available functions:**
- `create_parent_document_retriever(documents, child_size, parent_size)` - Create retriever
- `add_documents(documents)` - Index documents with parent-child relationship
- `retrieve(query, k)` - Search children, return parents

**Example:**
```python
from techniques.en.parent_document import ParentDocumentRetriever

retriever = ParentDocumentRetriever(
    child_chunk_size=400,
    parent_chunk_size=2000
)
retriever.add_documents(documents)

# Searches small chunks, returns full parent context
results = retriever.retrieve("neural network architecture", k=3)
```

---

### 27. Multi-Vector Retrieval

Store multiple representations of documents for improved retrieval.

**Representation types:**
- Original document content
- Generated summaries
- Hypothetical questions the document answers

**Available functions:**
- `create_multi_vector_retriever(documents)` - Create retriever with multiple vectors
- `generate_summary(document)` - Generate document summary
- `generate_questions(document)` - Generate hypothetical questions
- `retrieve(query, k)` - Search all representations

**Example:**
```python
from techniques.en.multi_vector import MultiVectorRetriever

retriever = MultiVectorRetriever()
retriever.add_documents(documents)  # Creates summary + question vectors

# Can match query to summary, questions, or original content
results = retriever.retrieve("How does backpropagation work?", k=3)
```

---

### 28. Ensemble Retrieval

Combine multiple retrievers using Reciprocal Rank Fusion with configurable weights.

**Components:**
- Multiple base retrievers (BM25, Vector, etc.)
- Configurable weights per retriever
- RRF algorithm for score combination

**Available functions:**
- `create_ensemble_retriever(retrievers, weights)` - Create ensemble
- `reciprocal_rank_fusion(results_list, weights, k)` - Combine with RRF

**Example:**
```python
from techniques.en.ensemble_retrieval import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever, sparse_retriever],
    weights=[0.3, 0.5, 0.2]
)
results = ensemble.retrieve("machine learning optimization", k=5)
```

---

### 29. Long Context Strategies

Process documents that exceed typical context windows using various strategies.

**Strategies:**
- `Map-Reduce` - Process chunks separately, combine results
- `Refine` - Iteratively build answer with each chunk
- `Map-Rerank` - Score each chunk, use best ones
- `Stuffing` - Fit most relevant content into context

**Available functions:**
- `map_reduce_summarize(chunks)` - Summarize with map-reduce
- `refine_summarize(chunks)` - Iterative refinement
- `map_rerank_answer(chunks, question)` - Score and select best chunks
- `stuffing_with_prioritization(chunks, question, max_context)` - Priority-based stuffing

**Example:**
```python
from techniques.en.long_context import map_reduce_summarize, map_rerank_answer

# Summarize a 50-page document
summary = map_reduce_summarize(document_chunks)

# Answer question using best chunks
answer = map_rerank_answer(
    chunks=document_chunks,
    question="What are the main conclusions?"
)
```

---

### 30. Time-Weighted Retrieval

Incorporate temporal relevance with exponential decay to prefer recent documents.

**Features:**
- Exponential decay function for time weighting
- Configurable decay rate and time units
- Combines semantic similarity with recency

**Available functions:**
- `create_time_weighted_retriever(documents, decay_rate)` - Create retriever
- `calculate_time_weight(timestamp, decay_rate, time_unit)` - Calculate decay weight
- `retrieve(query, k, time_weight_factor)` - Search with time weighting

**Example:**
```python
from techniques.en.time_weighted import TimeWeightedRetriever

retriever = TimeWeightedRetriever(
    documents=news_articles,
    decay_rate=0.05,  # Per day
    time_unit="days"
)

# Recent articles score higher
results = retriever.retrieve(
    "AI developments",
    k=5,
    time_weight_factor=0.4  # 40% time, 60% semantic
)
```

**Configuration guide:**
| Use Case | Decay Rate | Time Unit | Weight Factor |
|----------|------------|-----------|---------------|
| News/Current | 0.1-0.5 | hours | 0.5-0.7 |
| Chat history | 0.05-0.1 | hours | 0.3-0.5 |
| Documentation | 0.01-0.05 | days | 0.2-0.4 |
| Research papers | 0.001-0.01 | days | 0.1-0.3 |

---

### 31. MCP Basics

Model Context Protocol (MCP) is an open protocol created by Anthropic to connect AI assistants to external data sources and tools in a standardized way.

**Key concepts:**
- Resources - Data exposed by the server (files, databases, APIs)
- Tools - Functions that the LLM can invoke
- Prompts - Reusable prompt templates

**Example:**
```python
from techniques.en.mcp_basics import MCPServerSimulator

server = MCPServerSimulator(name="demo-server")
server.add_tool("search_database", description="Search data", ...)
server.add_resource("file:///config.json", name="Config", ...)
```

---

### 32. MCP Server STDIO

STDIO is the most common transport method for local MCP servers. Communication occurs through stdin/stdout.

**Use cases:**
- Claude Desktop integration
- Command line tools
- Local file access
- Script execution

**Example:**
```python
# In production, use:
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("my-server")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)
```

---

### 33. MCP Server HTTP/SSE

HTTP/SSE is the transport method for remote MCP servers. Enables communication over the network.

**Use cases:**
- AI-as-a-service APIs
- Enterprise integrations
- Centralized servers
- AI microservices

**Example:**
```python
# In production, use FastAPI with MCP SDK:
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.sse import SseServerTransport

app = FastAPI()
server = Server("my-http-server")
```

---

### 34. Multi-Agent Applications

Multi-agent systems allow multiple AI agents to collaborate to solve complex tasks.

**Patterns:**
- `Pipeline` - Sequential processing (Agent1 → Agent2 → Agent3)
- `Debate` - Agents with different perspectives discuss to reach consensus
- `Hierarchical` - Agents organized in authority levels (Director → Managers → Workers)
- `Orchestrator` - Central agent coordinates the others

**Frameworks:**
- LangGraph (LangChain)
- AutoGen (Microsoft)
- CrewAI
- Swarm (OpenAI)

**Example:**
```python
from techniques.en.multi_agent import MultiAgentSystem, Agent, AgentRole

system = MultiAgentSystem(name="dev-team")
system.add_agent(Agent(name="Planner", role=AgentRole.PLANNER, ...))
system.add_agent(Agent(name="Coder", role=AgentRole.EXECUTOR, ...))
system.add_agent(Agent(name="Reviewer", role=AgentRole.REVIEWER, ...))
```

---

### 35. Prompt Evaluation

Prompt evaluation is essential to ensure quality, consistency, and continuous improvement in LLM applications.

**Metrics:**
- Relevance - Does response address the question?
- Coherence - Does text have logic and flow?
- Groundedness - Based on facts/context?
- Accuracy - Correct information?
- Safety - Appropriate content?

**Tools:**
- LangSmith (LangChain)
- LangFuse (Open Source)
- Weights & Biases
- Promptfoo (CLI)
- Phoenix (Arize)

**Example:**
```python
from techniques.en.prompt_evaluation import PromptEvaluator

evaluator = PromptEvaluator()
result = evaluator.evaluate_relevance(question, answer)
print(f"Relevance: {result.score:.2f}")
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
├── sample_data/              # Sample data for RAG, Vision, and Context Engineering
│   ├── documents/            # Text documents for RAG and Context Engineering
│   │   ├── ai_handbook.txt
│   │   ├── company_faq.txt
│   │   ├── technical_docs.md
│   │   ├── products_catalog.json   # Product data with metadata (Self-Query)
│   │   ├── news_articles.txt       # Dated articles (Time-Weighted)
│   │   └── long_document.txt       # Large document (Long Context)
│   └── images/               # Images for Vision demos
│       ├── chart.png
│       ├── diagram.png
│       └── photo.jpg
└── techniques/
    ├── en/                   # English examples (35 scripts)
    │   ├── 01_zero_shot.py
    │   ├── ...
    │   ├── 20_meta_prompting.py
    │   ├── 21_advanced_chunking.py
    │   ├── ...
    │   ├── 30_time_weighted.py
    │   ├── 31_mcp_basics.py
    │   ├── 32_mcp_server_stdio.py
    │   ├── 33_mcp_server_http.py
    │   ├── 34_multi_agent.py
    │   └── 35_prompt_evaluation.py
    └── pt-br/                # Portuguese examples (35 scripts)
        ├── 01_zero_shot.py
        ├── ...
        ├── 20_meta_prompting.py
        ├── 21_advanced_chunking.py
        ├── ...
        ├── 30_time_weighted.py
        ├── 31_mcp_basics.py
        ├── 32_mcp_server_stdio.py
        ├── 33_mcp_server_http.py
        ├── 34_multi_agent.py
        └── 35_prompt_evaluation.py
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
| Query Transformation | 0.3 - 0.7 | Creative variations |
| Contextual Compression | 0.0 | Accurate extraction |
| Self-Query (filter gen) | 0.0 | Precise filters |
| Multi-Vector (summaries) | 0.3 | Balanced summaries |
| Long Context | 0.0 - 0.3 | Accurate synthesis |

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

### Context Engineering Dependencies
- `rank-bm25` - BM25 keyword search for Hybrid Search

### Ollama Dependencies
- `langchain-ollama` - Ollama integration

### Optional Dependencies
- `cohere` - Cohere reranking
- `pillow` - Image processing

## Usage Tips

### Prompt Engineering

1. **Start with Zero-Shot** - It's the simplest technique and works well for direct tasks.

2. **Use CoT for reasoning** - Mathematical, logical problems, or those requiring analysis benefit from "think step by step".

3. **Few-Shot for specific formats** - When you need output in a specific format (JSON, SQL, etc.), provide examples.

4. **Self-Consistency for accuracy** - When you need high accuracy on reasoning tasks, generate multiple responses and vote.

5. **Ollama for privacy** - Use local models when data privacy is important or you want to avoid API costs.

6. **Structured Output for APIs** - When building integrations, use Pydantic models to ensure consistent output.

### Context Engineering

7. **RAG for knowledge** - Use RAG when you need the model to answer based on specific documents.

8. **Hybrid Search for precision** - Combine BM25 + Vector search when queries contain specific terms or keywords.

9. **Advanced Chunking for quality** - Choose chunking strategy based on content type (semantic for articles, markdown-aware for docs).

10. **Query Transformation for recall** - Use HyDE or Multi-Query when initial retrieval quality is low.

11. **Contextual Compression for tokens** - Compress retrieved documents to reduce token usage while maintaining relevance.

12. **Self-Query for structured data** - Use when documents have rich metadata that can filter results.

13. **Parent-Document for context** - When retrieved chunks lack surrounding context, use parent-document retrieval.

14. **Long Context for large docs** - Use Map-Reduce for summarization, Map-Rerank for Q&A on large documents.

15. **Time-Weighted for freshness** - Use when document recency matters (news, logs, chat history).

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
