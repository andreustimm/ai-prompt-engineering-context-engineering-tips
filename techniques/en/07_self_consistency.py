"""
Self-Consistency Prompting

Technique that generates multiple responses with high temperature,
then uses majority voting to select the most consistent answer.

Use cases:
- Math problems with definitive answers
- Multiple choice questions
- Fact-based queries
- Logical reasoning problems
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collections import Counter
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


def generate_multiple_responses(prompt_template: ChatPromptTemplate, inputs: dict, n_samples: int = 5) -> list[str]:
    """Generate multiple responses with high temperature for diversity."""
    llm = get_llm(temperature=0.8)  # Higher temperature for diverse responses
    chain = prompt_template | llm

    responses = []
    for i in range(n_samples):
        response = chain.invoke(inputs)

        # Extract and record tokens
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        responses.append(response.content)

    return responses


def extract_final_answer(response: str) -> str:
    """Extract the final answer from a response."""
    # Look for common answer patterns
    lines = response.strip().split('\n')

    # Try to find explicit final answer
    for line in reversed(lines):
        line_lower = line.lower()
        if 'final answer' in line_lower or 'answer:' in line_lower or 'result:' in line_lower:
            # Extract the answer part
            if ':' in line:
                return line.split(':', 1)[1].strip()
            return line.strip()

    # If no explicit answer, return the last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()

    return response.strip()


def majority_vote(answers: list[str]) -> tuple[str, int, int]:
    """
    Perform majority voting on a list of answers.

    Returns:
        Tuple of (winning_answer, vote_count, total_votes)
    """
    # Normalize answers for comparison
    normalized = [a.strip().lower() for a in answers]
    counter = Counter(normalized)

    # Get most common
    most_common = counter.most_common(1)[0]

    # Return original (non-normalized) answer
    for i, norm in enumerate(normalized):
        if norm == most_common[0]:
            return answers[i], most_common[1], len(answers)

    return answers[0], 1, len(answers)


def solve_math_with_consistency(problem: str, n_samples: int = 5) -> dict:
    """Solve a math problem using self-consistency."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a math expert. Solve the given problem step by step.
At the end, clearly state your final answer in the format:
Final Answer: [your numerical answer]"""),
        ("user", "{problem}")
    ])

    print(f"   Generating {n_samples} independent solutions...")
    responses = generate_multiple_responses(prompt, {"problem": problem}, n_samples)

    # Extract answers from each response
    answers = [extract_final_answer(r) for r in responses]

    # Perform majority vote
    winner, votes, total = majority_vote(answers)

    return {
        "responses": responses,
        "extracted_answers": answers,
        "final_answer": winner,
        "confidence": f"{votes}/{total} ({votes/total*100:.0f}%)"
    }


def answer_multiple_choice(question: str, options: list[str], n_samples: int = 5) -> dict:
    """Answer a multiple choice question using self-consistency."""

    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at answering multiple choice questions.
Analyze the question carefully, consider each option, and select the best answer.
At the end, clearly state your final answer in the format:
Final Answer: [letter]"""),
        ("user", """Question: {question}

Options:
{options}

Think through this step by step and select the best answer.""")
    ])

    print(f"   Generating {n_samples} independent analyses...")
    responses = generate_multiple_responses(
        prompt,
        {"question": question, "options": options_text},
        n_samples
    )

    # Extract answers
    answers = []
    for r in responses:
        ans = extract_final_answer(r)
        # Try to extract just the letter
        for char in ans.upper():
            if char in 'ABCDEFGH':
                answers.append(char)
                break
        else:
            answers.append(ans)

    # Perform majority vote
    winner, votes, total = majority_vote(answers)

    return {
        "responses": responses,
        "extracted_answers": answers,
        "final_answer": winner,
        "confidence": f"{votes}/{total} ({votes/total*100:.0f}%)"
    }


def verify_fact_with_consistency(claim: str, n_samples: int = 5) -> dict:
    """Verify a factual claim using self-consistency."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fact-checker. Analyze the given claim and determine if it is TRUE or FALSE.
Explain your reasoning, then clearly state your verdict.
Final Answer: TRUE or FALSE"""),
        ("user", "Claim: {claim}")
    ])

    print(f"   Generating {n_samples} independent verifications...")
    responses = generate_multiple_responses(prompt, {"claim": claim}, n_samples)

    # Extract verdicts
    answers = []
    for r in responses:
        ans = extract_final_answer(r).upper()
        if 'TRUE' in ans:
            answers.append('TRUE')
        elif 'FALSE' in ans:
            answers.append('FALSE')
        else:
            answers.append(ans)

    # Perform majority vote
    winner, votes, total = majority_vote(answers)

    return {
        "responses": responses,
        "extracted_answers": answers,
        "final_answer": winner,
        "confidence": f"{votes}/{total} ({votes/total*100:.0f}%)"
    }


def solve_logic_puzzle(puzzle: str, n_samples: int = 5) -> dict:
    """Solve a logic puzzle using self-consistency."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at solving logic puzzles.
Work through the puzzle systematically, considering all constraints.
At the end, clearly state your solution.
Final Answer: [your answer]"""),
        ("user", "{puzzle}")
    ])

    print(f"   Generating {n_samples} independent solutions...")
    responses = generate_multiple_responses(prompt, {"puzzle": puzzle}, n_samples)

    # Extract answers
    answers = [extract_final_answer(r) for r in responses]

    # Perform majority vote
    winner, votes, total = majority_vote(answers)

    return {
        "responses": responses,
        "extracted_answers": answers,
        "final_answer": winner,
        "confidence": f"{votes}/{total} ({votes/total*100:.0f}%)"
    }


def main():
    print("=" * 60)
    print("SELF-CONSISTENCY PROMPTING - Demo")
    print("=" * 60)

    # Reset tracker
    token_tracker.reset()

    # Example 1: Math Problem
    print("\n🔢 MATH PROBLEM WITH SELF-CONSISTENCY")
    print("-" * 40)

    math_problem = """
    A store sells apples at $2 each and oranges at $3 each.
    John buys 5 apples and some oranges for a total of $22.
    How many oranges did John buy?
    """

    print(f"\nProblem: {math_problem.strip()}")
    result = solve_math_with_consistency(math_problem, n_samples=5)

    print(f"\n   Extracted answers: {result['extracted_answers']}")
    print(f"   Final Answer: {result['final_answer']}")
    print(f"   Confidence: {result['confidence']}")

    # Example 2: Multiple Choice
    print("\n\n📝 MULTIPLE CHOICE WITH SELF-CONSISTENCY")
    print("-" * 40)

    question = "What is the primary function of mitochondria in a cell?"
    options = [
        "Store genetic information",
        "Produce energy (ATP)",
        "Synthesize proteins",
        "Control cell division"
    ]

    print(f"\nQuestion: {question}")
    for i, opt in enumerate(options):
        print(f"   {chr(65+i)}. {opt}")

    result = answer_multiple_choice(question, options, n_samples=5)

    print(f"\n   Extracted answers: {result['extracted_answers']}")
    print(f"   Final Answer: {result['final_answer']}")
    print(f"   Confidence: {result['confidence']}")

    # Example 3: Fact Verification
    print("\n\n✓ FACT VERIFICATION WITH SELF-CONSISTENCY")
    print("-" * 40)

    claim = "The Great Wall of China is visible from space with the naked eye."

    print(f"\nClaim: {claim}")
    result = verify_fact_with_consistency(claim, n_samples=5)

    print(f"\n   Extracted answers: {result['extracted_answers']}")
    print(f"   Final Answer: {result['final_answer']}")
    print(f"   Confidence: {result['confidence']}")

    # Example 4: Logic Puzzle
    print("\n\n🧩 LOGIC PUZZLE WITH SELF-CONSISTENCY")
    print("-" * 40)

    puzzle = """
    Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, or a fish.
    - Alice does not have the dog.
    - Bob does not have the cat.
    - Carol has the fish.
    Who has the dog?
    """

    print(f"\nPuzzle: {puzzle.strip()}")
    result = solve_logic_puzzle(puzzle, n_samples=5)

    print(f"\n   Extracted answers: {result['extracted_answers']}")
    print(f"   Final Answer: {result['final_answer']}")
    print(f"   Confidence: {result['confidence']}")

    # Display total tokens
    print_total_usage(token_tracker, "TOTAL - Self-Consistency Prompting")

    print("\nEnd of Self-Consistency demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
