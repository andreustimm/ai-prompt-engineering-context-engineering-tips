"""
Meta-Prompting

Technique where an LLM is used to generate, optimize, or improve
prompts for other LLM tasks. The LLM becomes a prompt engineer.

Features:
- Automatic prompt generation
- Prompt optimization and refinement
- Task-specific prompt creation
- A/B testing of prompts

Use cases:
- Automated prompt engineering
- Prompt optimization pipelines
- Dynamic prompt adaptation
- Prompt template generation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker
token_tracker = TokenUsage()


def generate_prompt(task_description: str, context: str = "", constraints: list[str] = None) -> str:
    """
    Generate an optimized prompt for a given task.

    Args:
        task_description: What the prompt should accomplish
        context: Additional context about the use case
        constraints: Any constraints or requirements

    Returns:
        Generated prompt text
    """
    llm = get_llm(temperature=0.7)

    constraints_text = "\n".join([f"- {c}" for c in (constraints or [])])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert prompt engineer. Your task is to create highly effective prompts for LLMs.

When creating prompts, follow these best practices:
1. Be clear and specific about the task
2. Include relevant context and examples if helpful
3. Specify the desired output format
4. Add constraints or guardrails as needed
5. Use structured formatting (headers, bullets) for clarity
6. Include a persona/role if appropriate

Create prompts that are:
- Unambiguous and well-structured
- Focused on the specific task
- Designed to elicit high-quality responses"""),
        ("user", """Create an optimized prompt for the following task:

Task Description: {task_description}

Additional Context: {context}

Constraints:
{constraints}

Generate the complete prompt:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "task_description": task_description,
        "context": context or "None provided",
        "constraints": constraints_text or "None specified"
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Generate")

    return response.content


def optimize_prompt(original_prompt: str, issues: list[str] = None, goals: list[str] = None) -> str:
    """
    Optimize an existing prompt based on identified issues or goals.

    Args:
        original_prompt: The prompt to optimize
        issues: Known issues with the prompt
        goals: Optimization goals

    Returns:
        Optimized prompt
    """
    llm = get_llm(temperature=0.5)

    issues_text = "\n".join([f"- {i}" for i in (issues or [])])
    goals_text = "\n".join([f"- {g}" for g in (goals or [])])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at optimizing prompts for LLMs.
Analyze the given prompt and improve it while preserving its core purpose.

Consider:
- Clarity and specificity
- Structure and formatting
- Missing context or constraints
- Potential ambiguities
- Output format specification"""),
        ("user", """Original Prompt:
{original_prompt}

Known Issues:
{issues}

Optimization Goals:
{goals}

Provide the optimized prompt:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "original_prompt": original_prompt,
        "issues": issues_text or "None identified",
        "goals": goals_text or "General improvement"
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Optimize")

    return response.content


def evaluate_prompt(prompt: str, task_description: str) -> dict:
    """
    Evaluate a prompt's quality and provide feedback.

    Args:
        prompt: The prompt to evaluate
        task_description: What the prompt should accomplish

    Returns:
        Dictionary with scores and feedback
    """
    llm = get_llm(temperature=0.3)

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a prompt evaluation expert. Analyze prompts and provide detailed feedback.

Rate each aspect from 1-10 and explain your reasoning:
1. Clarity: How clear and unambiguous is the prompt?
2. Specificity: How well does it define the task?
3. Structure: Is it well-organized?
4. Completeness: Does it include necessary context?
5. Output Guidance: Does it specify the desired format?

Provide actionable improvement suggestions."""),
        ("user", """Task the prompt should accomplish:
{task_description}

Prompt to evaluate:
{prompt}

Provide your evaluation in this format:
CLARITY: [score]/10 - [explanation]
SPECIFICITY: [score]/10 - [explanation]
STRUCTURE: [score]/10 - [explanation]
COMPLETENESS: [score]/10 - [explanation]
OUTPUT GUIDANCE: [score]/10 - [explanation]

OVERALL SCORE: [average]/10

IMPROVEMENT SUGGESTIONS:
[list of specific suggestions]""")
    ])

    chain = eval_prompt | llm
    response = chain.invoke({
        "prompt": prompt,
        "task_description": task_description
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Evaluate")

    return {"evaluation": response.content}


def generate_prompt_variations(base_prompt: str, num_variations: int = 3) -> list[str]:
    """
    Generate variations of a prompt for A/B testing.

    Args:
        base_prompt: The base prompt to vary
        num_variations: Number of variations to generate

    Returns:
        List of prompt variations
    """
    llm = get_llm(temperature=0.8)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a creative prompt engineer. Generate distinct variations
of a given prompt while maintaining its core purpose.

Each variation should:
- Take a different approach to the task
- Use different phrasing or structure
- Potentially use different techniques (few-shot, chain of thought, etc.)
- Be clearly distinct from other variations"""),
        ("user", """Base Prompt:
{base_prompt}

Generate {num_variations} distinct variations of this prompt.
Separate each variation with "---VARIATION---"

Variations:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "base_prompt": base_prompt,
        "num_variations": num_variations
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Variations")

    # Parse variations
    content = response.content
    variations = [v.strip() for v in content.split("---VARIATION---") if v.strip()]

    return variations[:num_variations]


def create_prompt_template(task_type: str, variables: list[str]) -> str:
    """
    Create a reusable prompt template with variables.

    Args:
        task_type: Type of task (summarization, classification, etc.)
        variables: List of variable names to include

    Returns:
        Prompt template with {variable} placeholders
    """
    llm = get_llm(temperature=0.5)

    variables_text = ", ".join([f"{{{v}}}" for v in variables])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a prompt template designer. Create reusable prompt templates
with variable placeholders.

Use {variable_name} syntax for placeholders.
The template should be:
- Flexible enough to handle different inputs
- Specific enough to produce consistent outputs
- Well-structured and clear"""),
        ("user", """Create a prompt template for the following:

Task Type: {task_type}
Required Variables: {variables}

The template should include these variables as placeholders: {variables_text}

Prompt Template:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "task_type": task_type,
        "variables": ", ".join(variables),
        "variables_text": variables_text
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Template")

    return response.content


def auto_improve_prompt(prompt: str, task: str, test_input: str, max_iterations: int = 3) -> dict:
    """
    Automatically improve a prompt through iterative testing and refinement.

    Args:
        prompt: Initial prompt
        task: Task description
        test_input: Sample input to test with
        max_iterations: Maximum improvement iterations

    Returns:
        Dictionary with final prompt and improvement history
    """
    llm = get_llm(temperature=0.3)
    task_llm = get_llm(temperature=0.7)

    history = []
    current_prompt = prompt

    for i in range(max_iterations):
        print(f"\n   Iteration {i+1}...")

        # Test the current prompt
        test_prompt = ChatPromptTemplate.from_messages([
            ("system", current_prompt),
            ("user", "{input}")
        ])

        test_chain = test_prompt | task_llm
        test_response = test_chain.invoke({"input": test_input})

        input_tokens, output_tokens = extract_tokens_from_response(test_response)
        token_tracker.add(input_tokens, output_tokens)

        # Evaluate and improve
        improve_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a prompt improvement expert. Analyze how well a prompt performed
and suggest improvements.

Evaluate:
1. Did the output match the expected task?
2. Was the response well-structured?
3. Were there any issues or gaps?

Provide an improved version of the prompt."""),
            ("user", """Task: {task}

Current Prompt:
{current_prompt}

Test Input: {test_input}

Output Generated:
{output}

Analysis and Improved Prompt:""")
        ])

        improve_chain = improve_prompt | llm
        improve_response = improve_chain.invoke({
            "task": task,
            "current_prompt": current_prompt,
            "test_input": test_input,
            "output": test_response.content
        })

        input_tokens, output_tokens = extract_tokens_from_response(improve_response)
        token_tracker.add(input_tokens, output_tokens)

        history.append({
            "iteration": i + 1,
            "prompt": current_prompt,
            "output": test_response.content[:200] + "...",
            "feedback": improve_response.content[:200] + "..."
        })

        # Extract improved prompt (simple extraction)
        current_prompt = improve_response.content

    return {
        "initial_prompt": prompt,
        "final_prompt": current_prompt,
        "iterations": len(history),
        "history": history
    }


def main():
    print("=" * 60)
    print("META-PROMPTING - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Example 1: Generate a Prompt
    print("\n📝 PROMPT GENERATION")
    print("-" * 40)

    task = "Extract key information from customer support emails and categorize them"
    context = "For a SaaS company, emails may contain bug reports, feature requests, billing questions"
    constraints = ["Output should be JSON format", "Include urgency level", "Support multiple languages"]

    print(f"\nTask: {task}")
    generated = generate_prompt(task, context, constraints)
    print(f"\n📋 Generated Prompt:\n{generated}")

    # Example 2: Optimize a Prompt
    print("\n\n🔧 PROMPT OPTIMIZATION")
    print("-" * 40)

    original = "Summarize this article."
    issues = ["Too vague", "No length guidance", "No format specified"]
    goals = ["Make it specific", "Add output format", "Include key points extraction"]

    print(f"\nOriginal: {original}")
    optimized = optimize_prompt(original, issues, goals)
    print(f"\n📋 Optimized Prompt:\n{optimized}")

    # Example 3: Evaluate a Prompt
    print("\n\n📊 PROMPT EVALUATION")
    print("-" * 40)

    prompt_to_eval = """You are a helpful assistant. Answer the user's question."""
    task_desc = "Create a customer support chatbot that handles technical issues"

    print(f"\nPrompt: {prompt_to_eval}")
    print(f"Task: {task_desc}")
    evaluation = evaluate_prompt(prompt_to_eval, task_desc)
    print(f"\n📋 Evaluation:\n{evaluation['evaluation']}")

    # Example 4: Generate Variations
    print("\n\n🔀 PROMPT VARIATIONS")
    print("-" * 40)

    base = "Explain {concept} to a {audience}."
    print(f"\nBase: {base}")
    variations = generate_prompt_variations(base, num_variations=3)
    print(f"\n📋 Variations:")
    for i, v in enumerate(variations, 1):
        print(f"\n--- Variation {i} ---")
        print(v[:300] + "..." if len(v) > 300 else v)

    # Example 5: Create Template
    print("\n\n📄 TEMPLATE CREATION")
    print("-" * 40)

    task_type = "sentiment analysis"
    variables = ["text", "language", "output_format"]

    print(f"\nTask Type: {task_type}")
    print(f"Variables: {variables}")
    template = create_prompt_template(task_type, variables)
    print(f"\n📋 Template:\n{template}")

    print_total_usage(token_tracker, "TOTAL - Meta-Prompting")

    print("\n\n" + "=" * 60)
    print("Meta-Prompting allows LLMs to engineer prompts automatically,")
    print("enabling automated optimization and adaptation of prompts.")
    print("=" * 60)

    print("\nEnd of Meta-Prompting demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
