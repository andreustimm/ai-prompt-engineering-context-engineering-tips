# Prompt Engineering with LangChain and OpenAI

Scripts demonstrating 6 Prompt Engineering techniques using LangChain and the OpenAI API.

> **🌐 Language / Idioma:** [Português Brasileiro](README.pt-BR.md) | English

## Implemented Techniques

| Script | Technique | Description |
|--------|-----------|-------------|
| `01_zero_shot.py` | Zero-Shot | Direct prompts without prior examples |
| `02_chain_of_thought.py` | Chain of Thought (CoT) | Step-by-step reasoning |
| `03_few_shot.py` | Few-Shot | Examples to guide the model |
| `04_tree_of_thoughts.py` | Tree of Thoughts (ToT) | Multiple reasoning paths |
| `05_skeleton_of_thought.py` | Skeleton of Thought (SoT) | Structure first, details later |
| `06_react_agent.py` | ReAct | Reasoning + Actions with tools |

## Requirements

- Python 3.10+
- OpenAI API key

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

Edit the `.env` file and add your OpenAI key:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

## Usage

Execute any script from the `techniques/` folder:

**English examples:**
```bash
python techniques/en/01_zero_shot.py
python techniques/en/02_chain_of_thought.py
python techniques/en/03_few_shot.py
python techniques/en/04_tree_of_thoughts.py
python techniques/en/05_skeleton_of_thought.py
python techniques/en/06_react_agent.py
```

**Portuguese examples:**
```bash
python techniques/pt-br/01_zero_shot.py
python techniques/pt-br/02_chain_of_thought.py
python techniques/pt-br/03_few_shot.py
python techniques/pt-br/04_tree_of_thoughts.py
python techniques/pt-br/05_skeleton_of_thought.py
python techniques/pt-br/06_react_agent.py
```

## Technique Descriptions

### 1. Zero-Shot Prompting

Technique where the model receives a task without prior examples, using only its pre-trained knowledge.

**Available functions:**
- `classify_sentiment(text)` - Classifies sentiment as POSITIVE, NEGATIVE, or NEUTRAL
- `translate_text(text, target_language)` - Translates text to specified language
- `extract_entities(text)` - Extracts people, locations, organizations, and dates
- `summarize_text(text)` - Summarizes text in a few sentences

**Example:**
```python
from techniques.en.zero_shot import classify_sentiment

result = classify_sentiment("This product is amazing!")
print(result)  # POSITIVE
```

---

### 2. Chain of Thought (CoT)

Instructs the model to "think step by step" before reaching the final answer, improving performance on reasoning tasks.

**Available functions:**
- `solve_math_problem(problem)` - Solves math problems showing each step
- `logical_reasoning(puzzle)` - Solves logic puzzles with deductions
- `analyze_decision(situation)` - Analyzes scenarios for decision making
- `debug_code(code, error)` - Analyzes code and error to find solution

**Example:**
```python
from techniques.en.chain_of_thought import solve_math_problem

problem = "John bought 5 t-shirts for $45 each with a 15% discount. How much did he pay?"
solution = solve_math_problem(problem)
print(solution)
```

---

### 3. Few-Shot Prompting

Provides examples to the model before the task, helping it understand the format and expected response type.

**Available functions:**
- `classify_support_ticket(ticket)` - Classifies tickets with category, priority, and action
- `convert_to_sql(description)` - Converts natural language to SQL
- `generate_docstring(code)` - Generates Google Style docstrings
- `extract_structured_data(text)` - Extracts data in JSON format

**Example:**
```python
from techniques.en.few_shot import convert_to_sql

sql = convert_to_sql("List all customers from Brazil")
print(sql)  # SELECT * FROM customers WHERE country = 'Brazil';
```

---

### 4. Tree of Thoughts (ToT)

Explores multiple reasoning paths in parallel, evaluates each one, and selects the most promising.

**Available functions:**
- `tree_of_thoughts(problem, depth)` - Executes complete ToT algorithm
- `generate_thoughts(problem, num)` - Generates multiple initial approaches
- `evaluate_thought(problem, thought)` - Evaluates viability of an approach
- `expand_thought(problem, thought, next_step)` - Develops an approach

**Example:**
```python
from techniques.en.tree_of_thoughts import tree_of_thoughts

problem = "How to triple the startup's revenue in 18 months?"
solution = tree_of_thoughts(problem, depth=2)
print(solution)
```

---

### 5. Skeleton of Thought (SoT)

First generates a "skeleton" (structure/topics) and then expands each part, allowing parallelization.

**Available functions:**
- `skeleton_of_thought_sync(topic, context)` - Synchronous version
- `skeleton_of_thought_async(topic, context)` - Asynchronous version (parallel)
- `generate_skeleton(topic, context)` - Generates list of topics
- `expand_topic(main_topic, topic, context)` - Expands a specific topic

**Example:**
```python
from techniques.en.skeleton_of_thought import skeleton_of_thought_sync

document = skeleton_of_thought_sync(
    topic="Artificial Intelligence in Medicine",
    context="Focus on practical applications"
)
print(document)
```

**Async version (faster):**
```python
import asyncio
from techniques.en.skeleton_of_thought import skeleton_of_thought_async

document = asyncio.run(skeleton_of_thought_async("REST API Security"))
print(document)
```

---

### 6. ReAct Agent

Combines reasoning (Thought) with actions (Action) and observations (Observation) in an iterative loop, using external tools.

**Available tools:**
- `web_search` - Internet search via DuckDuckGo
- `wikipedia` - Wikipedia queries
- `calculator` - Mathematical calculations

**Available functions:**
- `execute_agent(question)` - Executes ReAct agent to answer questions
- `create_react_agent_instance()` - Creates configured agent instance

**Example:**
```python
from techniques.en.react_agent import execute_agent

response = execute_agent(
    "Who won the last World Cup and in which country was it held?"
)
print(response)
```

## Token Monitoring

All scripts include **automatic token counting** to help monitor costs and API usage.

### Sample Output

Each LLM call displays the tokens used:

```
Text: This product is amazing! It exceeded all...
   📊 Tokens - Input: 52 | Output: 3 | Total: 55
Sentiment: POSITIVE
```

At the end of each script, a total summary is displayed:

```
============================================================
📈 TOTAL - Zero-Shot Prompting
   Input:  1,234 tokens
   Output: 456 tokens
   Total:  1,690 tokens
============================================================
```

### Using the Token Tracker in Your Code

```python
from config import TokenUsage, extract_tokens_from_response, print_token_usage

# Create a tracker
tracker = TokenUsage()

# After an LLM call
response = chain.invoke({"input": "text"})
input_tokens, output_tokens = extract_tokens_from_response(response)

# Record and display
tracker.add(input_tokens, output_tokens)
print_token_usage(input_tokens, output_tokens, "my_function")

# View totals
print(f"Total used: {tracker.total_tokens} tokens")
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
└── techniques/
    ├── en/                   # English examples
    │   ├── 01_zero_shot.py
    │   ├── 02_chain_of_thought.py
    │   ├── 03_few_shot.py
    │   ├── 04_tree_of_thoughts.py
    │   ├── 05_skeleton_of_thought.py
    │   └── 06_react_agent.py
    └── pt-br/                # Portuguese examples
        ├── 01_zero_shot.py
        ├── 02_chain_of_thought.py
        ├── 03_few_shot.py
        ├── 04_tree_of_thoughts.py
        ├── 05_skeleton_of_thought.py
        └── 06_react_agent.py
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
```

## Understanding Temperature

Temperature is one of the most important parameters when working with LLMs. It controls the **randomness** and **creativity** of the model's responses.

### What is Temperature?

- **Range:** 0.0 to 2.0 (most common usage is 0.0 to 1.0)
- **Low values (0.0-0.3):** More deterministic, focused, and consistent responses
- **High values (0.7-1.0+):** More creative, diverse, and unpredictable responses

### When to Use Low Temperature (0.0 - 0.3)

Use low temperature when you need **accuracy, consistency, and predictability**:

| Use Case | Recommended Temperature |
|----------|------------------------|
| Classification tasks | 0.0 |
| Entity extraction | 0.0 |
| Code generation | 0.0 - 0.2 |
| Mathematical calculations | 0.0 |
| Factual Q&A | 0.0 - 0.2 |
| Data parsing/formatting | 0.0 |
| SQL query generation | 0.0 |

**Example:**
```python
# For classification - always use temperature=0
llm = get_llm(temperature=0)
```

### When to Use Medium Temperature (0.3 - 0.7)

Use medium temperature for a **balance between consistency and creativity**:

| Use Case | Recommended Temperature |
|----------|------------------------|
| Text summarization | 0.3 - 0.5 |
| Translation | 0.3 |
| General content writing | 0.5 - 0.7 |
| Explaining concepts | 0.5 |
| Email drafting | 0.5 - 0.7 |

**Example:**
```python
# For content generation - moderate creativity
llm = get_llm(temperature=0.5)
```

### When to Use High Temperature (0.7 - 1.0+)

Use high temperature when you need **creativity and diversity**:

| Use Case | Recommended Temperature |
|----------|------------------------|
| Creative writing | 0.7 - 0.9 |
| Brainstorming ideas | 0.8 - 1.0 |
| Poetry/storytelling | 0.8 - 1.0 |
| Generating alternatives | 0.7 - 0.9 |
| Role-playing scenarios | 0.7 - 0.9 |

**Example:**
```python
# For brainstorming - high creativity
llm = get_llm(temperature=0.8)
```

### Temperature Used in Each Technique

| Technique | Function | Temperature | Reason |
|-----------|----------|-------------|--------|
| Zero-Shot | `classify_sentiment` | 0.0 | Consistent classification |
| Zero-Shot | `translate_text` | 0.3 | Slight variation in phrasing |
| Zero-Shot | `summarize_text` | 0.5 | Balanced summary |
| CoT | `solve_math_problem` | 0.0 | Accurate calculations |
| CoT | `analyze_decision` | 0.3 | Structured but flexible |
| Few-Shot | `convert_to_sql` | 0.0 | Exact SQL syntax |
| Few-Shot | `generate_docstring` | 0.3 | Consistent style |
| ToT | `generate_thoughts` | 0.8 | Diverse approaches |
| ToT | `evaluate_thought` | 0.3 | Consistent evaluation |
| SoT | `generate_skeleton` | 0.5 | Balanced structure |
| SoT | `expand_topic` | 0.6 | Creative content |
| ReAct | Agent | 0.0 | Reliable tool usage |

### Temperature Tips

1. **Start low, increase if needed** - Begin with temperature=0 and increase only if responses are too repetitive or lack creativity.

2. **Same input, different outputs** - Higher temperatures mean the same prompt can produce different results each time.

3. **Production vs Development** - Use lower temperatures in production for consistency; higher in development for exploration.

4. **Combine with other parameters** - Temperature works with `top_p` (nucleus sampling). Generally, adjust one or the other, not both.

5. **Task-specific tuning** - The optimal temperature depends on your specific use case. Test different values.

## Supported Models

You can use any OpenAI model by changing the `OPENAI_MODEL` variable in `.env`:

- `gpt-4o` - Most capable, most expensive
- `gpt-4o-mini` - Good cost/performance balance (recommended)
- `gpt-4-turbo` - Turbo version of GPT-4
- `gpt-3.5-turbo` - Cheaper, less capable

## Usage Tips

1. **Start with Zero-Shot** - It's the simplest technique and works well for direct tasks.

2. **Use CoT for reasoning** - Mathematical, logical problems, or those requiring analysis benefit from "think step by step".

3. **Few-Shot for specific formats** - When you need output in a specific format (JSON, SQL, etc.), provide examples.

4. **ToT for complex problems** - Use when there are multiple possible solutions and you need to evaluate trade-offs.

5. **SoT for long content** - Ideal for generating articles, documentation, or structured responses.

6. **ReAct for external information** - Use when you need updated data or calculations.

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
- Reduce the number of examples in tests
- Comment out demonstrations you don't need to run
- Monitor token totals displayed at the end of each script

## License

MIT
