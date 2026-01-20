"""
Self-Refine Prompting

Technique where the model generates an initial response, critiques it,
and iteratively improves based on its own feedback until satisfactory.

Use cases:
- Writing and content improvement
- Code optimization
- Detailed explanations
- Creative content generation
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

# Global token tracker for this script
token_tracker = TokenUsage()


def generate_initial_response(task: str, context: str = "") -> str:
    """Generate an initial response to the task."""
    llm = get_llm(temperature=0.7)

    if context:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a skilled assistant. Complete the given task to the best of your ability.
Consider the provided context in your response."""),
            ("user", """Context: {context}

Task: {task}

Response:""")
        ])
        inputs = {"task": task, "context": context}
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a skilled assistant. Complete the given task to the best of your ability."),
            ("user", """Task: {task}

Response:""")
        ])
        inputs = {"task": task}

    chain = prompt | llm
    response = chain.invoke(inputs)

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Initial")

    return response.content


def critique_response(task: str, response: str, criteria: list[str] = None) -> str:
    """Critique a response and identify areas for improvement."""
    llm = get_llm(temperature=0.3)

    default_criteria = [
        "Accuracy and correctness",
        "Clarity and readability",
        "Completeness",
        "Structure and organization",
        "Relevance to the task"
    ]

    criteria = criteria or default_criteria
    criteria_text = "\n".join([f"- {c}" for c in criteria])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a critical reviewer. Analyze the given response and provide specific, actionable feedback.
Be constructive but thorough in identifying weaknesses and areas for improvement.

Evaluate based on these criteria:
{criteria}

Format your critique as:
STRENGTHS:
- [list strengths]

WEAKNESSES:
- [list weaknesses]

SPECIFIC IMPROVEMENTS:
- [list specific changes to make]"""),
        ("user", """Original Task: {task}

Response to Critique:
{response}

Provide your detailed critique:""")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "task": task,
        "response": response,
        "criteria": criteria_text
    })

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(result)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Critique")

    return result.content


def refine_response(task: str, response: str, critique: str) -> str:
    """Refine a response based on the critique."""
    llm = get_llm(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a skilled assistant improving your previous work.
Based on the critique provided, create an improved version of the response.
Address all the weaknesses and incorporate the suggested improvements."""),
        ("user", """Original Task: {task}

Previous Response:
{response}

Critique and Feedback:
{critique}

Improved Response:""")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "task": task,
        "response": response,
        "critique": critique
    })

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(result)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Refinement")

    return result.content


def check_if_satisfactory(task: str, response: str, min_score: int = 8) -> tuple[bool, int, str]:
    """Check if the response is satisfactory (score >= min_score)."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality evaluator. Rate the response on a scale of 1-10.
Be strict and objective in your evaluation.

Output format (exactly):
SCORE: [number 1-10]
REASON: [brief explanation]"""),
        ("user", """Task: {task}

Response to Evaluate:
{response}

Evaluation:""")
    ])

    chain = prompt | llm
    result = chain.invoke({"task": task, "response": response})

    # Extract and record tokens
    input_tokens, output_tokens = extract_tokens_from_response(result)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Evaluation")

    # Parse score
    content = result.content
    score = 5  # Default
    reason = ""

    for line in content.split('\n'):
        if 'SCORE:' in line.upper():
            try:
                score_str = line.split(':')[1].strip()
                score = int(''.join(filter(str.isdigit, score_str[:3])))
            except (ValueError, IndexError):
                pass
        elif 'REASON:' in line.upper():
            reason = line.split(':', 1)[1].strip() if ':' in line else ""

    return score >= min_score, score, reason


def self_refine(task: str, max_iterations: int = 3, min_score: int = 8, criteria: list[str] = None) -> dict:
    """
    Execute the self-refine loop.

    Args:
        task: The task to complete
        max_iterations: Maximum refinement iterations
        min_score: Minimum acceptable score (1-10)
        criteria: Custom evaluation criteria

    Returns:
        Dictionary with iterations, final response, and score
    """
    iterations = []

    print("\n   Generating initial response...")
    current_response = generate_initial_response(task)
    iterations.append({"response": current_response, "critique": None, "score": None})

    for i in range(max_iterations):
        print(f"\n   Checking quality (iteration {i+1})...")
        is_satisfactory, score, reason = check_if_satisfactory(task, current_response, min_score)
        iterations[-1]["score"] = score
        iterations[-1]["score_reason"] = reason

        print(f"      Score: {score}/10 - {reason}")

        if is_satisfactory:
            print(f"   ✓ Response satisfactory (score >= {min_score})")
            break

        if i < max_iterations - 1:
            print(f"\n   Critiquing response...")
            critique = critique_response(task, current_response, criteria)
            iterations[-1]["critique"] = critique

            print(f"\n   Refining based on critique...")
            current_response = refine_response(task, current_response, critique)
            iterations.append({"response": current_response, "critique": None, "score": None})

    # Final evaluation if loop completed
    if iterations[-1]["score"] is None:
        _, score, reason = check_if_satisfactory(task, current_response, min_score)
        iterations[-1]["score"] = score
        iterations[-1]["score_reason"] = reason

    return {
        "iterations": iterations,
        "final_response": current_response,
        "final_score": iterations[-1]["score"],
        "num_iterations": len(iterations)
    }


def improve_writing(text: str, style: str = "professional") -> dict:
    """Improve a piece of writing using self-refine."""
    task = f"Rewrite the following text in a {style} style while maintaining the original meaning:\n\n{text}"
    criteria = [
        "Grammar and spelling",
        "Tone matches requested style",
        "Flow and readability",
        "Conciseness",
        "Engagement"
    ]
    return self_refine(task, max_iterations=3, criteria=criteria)


def optimize_code(code: str, language: str = "Python") -> dict:
    """Optimize code using self-refine."""
    task = f"Optimize this {language} code for readability, efficiency, and best practices:\n\n{code}"
    criteria = [
        "Code correctness",
        "Readability and clarity",
        "Efficiency",
        "Follows best practices",
        "Proper error handling"
    ]
    return self_refine(task, max_iterations=3, criteria=criteria)


def improve_explanation(topic: str, audience: str = "general") -> dict:
    """Create and improve an explanation using self-refine."""
    task = f"Explain {topic} to a {audience} audience in a clear and engaging way"
    criteria = [
        "Accuracy",
        "Appropriate for target audience",
        "Uses helpful examples",
        "Clear structure",
        "Engaging presentation"
    ]
    return self_refine(task, max_iterations=3, criteria=criteria)


def main():
    print("=" * 60)
    print("SELF-REFINE PROMPTING - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Example 1: Writing Improvement
    print("\n✍️ WRITING IMPROVEMENT")
    print("-" * 40)

    original_text = """
    The meeting was good. We talked about stuff and made some decisions.
    Everyone seemed to agree mostly. We will do things differently now.
    The project should be done soon hopefully.
    """

    print(f"\nOriginal Text:\n{original_text.strip()}")
    print("\nImproving to professional style...")

    result = improve_writing(original_text, style="professional business")

    print(f"\n📋 FINAL VERSION (Score: {result['final_score']}/10):")
    print("-" * 40)
    print(result["final_response"])
    print(f"\n   Iterations needed: {result['num_iterations']}")

    # Example 2: Code Optimization
    print("\n\n💻 CODE OPTIMIZATION")
    print("-" * 40)

    original_code = '''
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j]:
                if lst[i] not in duplicates:
                    duplicates.append(lst[i])
    return duplicates
'''

    print(f"\nOriginal Code:{original_code}")
    print("Optimizing...")

    result = optimize_code(original_code, language="Python")

    print(f"\n📋 OPTIMIZED CODE (Score: {result['final_score']}/10):")
    print("-" * 40)
    print(result["final_response"])
    print(f"\n   Iterations needed: {result['num_iterations']}")

    # Example 3: Explanation Improvement
    print("\n\n📚 EXPLANATION IMPROVEMENT")
    print("-" * 40)

    topic = "how blockchain technology works"
    audience = "non-technical business executives"

    print(f"\nTopic: {topic}")
    print(f"Target Audience: {audience}")
    print("\nGenerating and refining explanation...")

    result = improve_explanation(topic, audience)

    print(f"\n📋 FINAL EXPLANATION (Score: {result['final_score']}/10):")
    print("-" * 40)
    print(result["final_response"])
    print(f"\n   Iterations needed: {result['num_iterations']}")

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Self-Refine Prompting")

    print("\nEnd of Self-Refine demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
