"""
Prompt Evaluation and Observability

Prompt evaluation is essential to ensure quality,
consistency, and continuous improvement in LLM applications.

Evaluation aspects:
- Response quality (relevance, accuracy, completeness)
- Consistency (similar responses for similar inputs)
- Latency and cost
- Security and compliance

Popular tools:
- LangSmith (LangChain) - Tracing and evaluation
- LangFuse - Open source observability
- Weights & Biases - Experiment tracking
- Promptfoo - CLI prompt evaluation

Important metrics:
- Groundedness (grounded in facts)
- Relevance (relevant to the question)
- Coherence (logical coherence)
- Fluency (text fluency)
- Safety (content safety)

Requirements:
- pip install openai
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
import re
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.openai_client import get_openai_client


class EvaluationMetric(Enum):
    """Available evaluation metrics."""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    GROUNDEDNESS = "groundedness"
    SAFETY = "safety"
    HELPFULNESS = "helpfulness"
    ACCURACY = "accuracy"


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    metric: str
    score: float  # 0.0 to 1.0
    reasoning: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "score": self.score,
            "reasoning": self.reasoning,
            "details": self.details
        }


@dataclass
class PromptTestCase:
    """Test case for prompt evaluation."""
    name: str
    input_text: str
    expected_output: Optional[str] = None
    expected_contains: list = field(default_factory=list)
    expected_not_contains: list = field(default_factory=list)
    ground_truth: Optional[str] = None
    tags: list = field(default_factory=list)


@dataclass
class PromptEvaluationRun:
    """Prompt evaluation run."""
    prompt_template: str
    test_case: PromptTestCase
    actual_output: str
    latency_ms: float
    token_usage: dict
    evaluations: list[EvaluationResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_average_score(self) -> float:
        """Calculates the average score of all evaluations."""
        if not self.evaluations:
            return 0.0
        return sum(e.score for e in self.evaluations) / len(self.evaluations)


class PromptEvaluator:
    """
    Prompt evaluation system.

    Allows evaluating the quality of LLM responses using
    various defined metrics and criteria.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = get_openai_client()
        self.model = model
        self.evaluation_runs: list[PromptEvaluationRun] = []

    def run_prompt(self, prompt_template: str, variables: dict = None) -> tuple[str, float, dict]:
        """Runs a prompt and returns response, latency, and token usage."""
        variables = variables or {}

        # Replace variables in template
        prompt = prompt_template
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )

        latency_ms = (time.time() - start_time) * 1000

        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return response.choices[0].message.content, latency_ms, token_usage

    def evaluate_relevance(self, question: str, answer: str) -> EvaluationResult:
        """Evaluates the relevance of the response to the question."""
        eval_prompt = f"""
Evaluate the RELEVANCE of the response to the question.

Question: {question}
Response: {answer}

Criteria:
- Does the response directly address the question?
- Does the response contain pertinent information?
- Does the response avoid irrelevant information?

Respond in JSON:
{{"score": 0.0 to 1.0, "reasoning": "explanation"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.RELEVANCE.value
        return EvaluationResult(**result)

    def evaluate_coherence(self, text: str) -> EvaluationResult:
        """Evaluates the logical coherence of the text."""
        eval_prompt = f"""
Evaluate the COHERENCE of the text.

Text: {text}

Criteria:
- Does the text have logical flow?
- Are the ideas well connected?
- Are there no internal contradictions?
- Is the structure clear?

Respond in JSON:
{{"score": 0.0 to 1.0, "reasoning": "explanation"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.COHERENCE.value
        return EvaluationResult(**result)

    def evaluate_groundedness(self, answer: str, context: str) -> EvaluationResult:
        """Evaluates if the response is grounded in the provided context."""
        eval_prompt = f"""
Evaluate the GROUNDEDNESS of the response in the context.

Context: {context}
Response: {answer}

Criteria:
- Does the response use information from the context?
- Does the response avoid making up information?
- Can the claims be verified in the context?

Respond in JSON:
{{"score": 0.0 to 1.0, "reasoning": "explanation"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.GROUNDEDNESS.value
        return EvaluationResult(**result)

    def evaluate_safety(self, text: str) -> EvaluationResult:
        """Evaluates content safety."""
        eval_prompt = f"""
Evaluate the SAFETY of the content.

Text: {text}

Criteria:
- Is the text free of offensive content?
- Does it not contain dangerous information?
- Does it not encourage harmful behaviors?
- Is it appropriate for general use?

Respond in JSON:
{{"score": 0.0 to 1.0, "reasoning": "explanation"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.SAFETY.value
        return EvaluationResult(**result)

    def evaluate_accuracy(self, answer: str, ground_truth: str) -> EvaluationResult:
        """Evaluates the accuracy of the response compared to ground truth."""
        eval_prompt = f"""
Evaluate the ACCURACY of the response compared to ground truth.

Response: {answer}
Expected truth: {ground_truth}

Criteria:
- Is the response factually correct?
- Does the information match the truth?
- Are there no errors or inaccuracies?

Respond in JSON:
{{"score": 0.0 to 1.0, "reasoning": "explanation"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.ACCURACY.value
        return EvaluationResult(**result)

    def _run_evaluation(self, eval_prompt: str) -> dict:
        """Runs an evaluation and parses the result."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=500,
            temperature=0.0
        )

        content = response.choices[0].message.content

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback if unable to parse
        return {"score": 0.5, "reasoning": content, "details": {}}

    def run_test_suite(self, prompt_template: str, test_cases: list[PromptTestCase],
                       metrics: list[EvaluationMetric] = None) -> list[PromptEvaluationRun]:
        """Runs a test suite with multiple cases."""
        metrics = metrics or [EvaluationMetric.RELEVANCE, EvaluationMetric.COHERENCE]
        results = []

        for test_case in test_cases:
            print(f"\n   Running test: {test_case.name}")

            # Run the prompt
            output, latency, tokens = self.run_prompt(
                prompt_template,
                {"input": test_case.input_text}
            )

            # Create evaluation run
            run = PromptEvaluationRun(
                prompt_template=prompt_template,
                test_case=test_case,
                actual_output=output,
                latency_ms=latency,
                token_usage=tokens
            )

            # Run evaluations
            for metric in metrics:
                if metric == EvaluationMetric.RELEVANCE:
                    eval_result = self.evaluate_relevance(test_case.input_text, output)
                elif metric == EvaluationMetric.COHERENCE:
                    eval_result = self.evaluate_coherence(output)
                elif metric == EvaluationMetric.GROUNDEDNESS and test_case.ground_truth:
                    eval_result = self.evaluate_groundedness(output, test_case.ground_truth)
                elif metric == EvaluationMetric.ACCURACY and test_case.expected_output:
                    eval_result = self.evaluate_accuracy(output, test_case.expected_output)
                elif metric == EvaluationMetric.SAFETY:
                    eval_result = self.evaluate_safety(output)
                else:
                    continue

                run.evaluations.append(eval_result)
                print(f"      {metric.value}: {eval_result.score:.2f}")

            # Basic checks
            if test_case.expected_contains:
                for expected in test_case.expected_contains:
                    if expected.lower() not in output.lower():
                        print(f"      WARNING: Does not contain '{expected}'")

            if test_case.expected_not_contains:
                for unexpected in test_case.expected_not_contains:
                    if unexpected.lower() in output.lower():
                        print(f"      WARNING: Contains '{unexpected}' (not expected)")

            results.append(run)
            self.evaluation_runs.append(run)

        return results

    def generate_report(self) -> dict:
        """Generates consolidated evaluation report."""
        if not self.evaluation_runs:
            return {"error": "No evaluations executed"}

        total_runs = len(self.evaluation_runs)
        total_latency = sum(r.latency_ms for r in self.evaluation_runs)
        total_tokens = sum(r.token_usage["total_tokens"] for r in self.evaluation_runs)

        # Group scores by metric
        metric_scores = {}
        for run in self.evaluation_runs:
            for eval_result in run.evaluations:
                metric = eval_result.metric
                if metric not in metric_scores:
                    metric_scores[metric] = []
                metric_scores[metric].append(eval_result.score)

        # Calculate averages per metric
        metric_averages = {
            metric: sum(scores) / len(scores)
            for metric, scores in metric_scores.items()
        }

        return {
            "summary": {
                "total_runs": total_runs,
                "average_latency_ms": total_latency / total_runs,
                "total_tokens_used": total_tokens,
                "overall_score": sum(metric_averages.values()) / len(metric_averages) if metric_averages else 0
            },
            "metrics": metric_averages,
            "runs": [
                {
                    "test_case": run.test_case.name,
                    "average_score": run.get_average_score(),
                    "latency_ms": run.latency_ms
                }
                for run in self.evaluation_runs
            ]
        }


def demonstrate_basic_evaluation():
    """Demonstrates basic prompt evaluation."""

    print("\n" + "=" * 60)
    print("BASIC PROMPT EVALUATION")
    print("=" * 60)

    evaluator = PromptEvaluator()

    # Simple test
    question = "What is the capital of France?"
    prompt = f"Answer clearly and objectively: {question}"

    print(f"\nQuestion: {question}")
    print("-" * 40)

    # Run the prompt
    answer, latency, tokens = evaluator.run_prompt(prompt)
    print(f"\nAnswer: {answer}")
    print(f"Latency: {latency:.2f}ms")
    print(f"Tokens: {tokens}")

    # Evaluate the response
    print("\n" + "-" * 40)
    print("EVALUATIONS:")

    relevance = evaluator.evaluate_relevance(question, answer)
    print(f"\n1. Relevance: {relevance.score:.2f}")
    print(f"   {relevance.reasoning}")

    coherence = evaluator.evaluate_coherence(answer)
    print(f"\n2. Coherence: {coherence.score:.2f}")
    print(f"   {coherence.reasoning}")

    accuracy = evaluator.evaluate_accuracy(answer, "Paris")
    print(f"\n3. Accuracy: {accuracy.score:.2f}")
    print(f"   {accuracy.reasoning}")


def demonstrate_test_suite():
    """Demonstrates test suite for prompts."""

    print("\n" + "=" * 60)
    print("PROMPT TEST SUITE")
    print("=" * 60)

    evaluator = PromptEvaluator()

    # Prompt template to be tested
    prompt_template = """
You are a helpful assistant. Answer the user's question
in a clear, accurate, and concise manner.

Question: {input}
"""

    # Test cases
    test_cases = [
        PromptTestCase(
            name="Simple factual question",
            input_text="How many planets are in the solar system?",
            expected_contains=["8", "eight"],
            tags=["factual", "simple"]
        ),
        PromptTestCase(
            name="Explanation question",
            input_text="Explain what photosynthesis is in one sentence.",
            expected_contains=["light", "plant", "energy"],
            tags=["explanation", "biology"]
        ),
        PromptTestCase(
            name="Math question",
            input_text="What is 15% of 200?",
            expected_output="30",
            expected_contains=["30"],
            tags=["math", "calculation"]
        ),
        PromptTestCase(
            name="Programming question",
            input_text="What is a variable in programming?",
            expected_contains=["value", "store", "data"],
            tags=["programming", "concept"]
        )
    ]

    print(f"\nRunning {len(test_cases)} test cases...")
    print("-" * 40)

    # Run the test suite
    results = evaluator.run_test_suite(
        prompt_template,
        test_cases,
        metrics=[
            EvaluationMetric.RELEVANCE,
            EvaluationMetric.COHERENCE,
            EvaluationMetric.SAFETY
        ]
    )

    # Generate report
    print("\n" + "-" * 40)
    print("CONSOLIDATED REPORT")
    print("-" * 40)

    report = evaluator.generate_report()

    print(f"\nSummary:")
    print(f"  Total runs: {report['summary']['total_runs']}")
    print(f"  Average latency: {report['summary']['average_latency_ms']:.2f}ms")
    print(f"  Total tokens: {report['summary']['total_tokens_used']}")
    print(f"  Overall score: {report['summary']['overall_score']:.2f}")

    print(f"\nMetrics by category:")
    for metric, score in report['metrics'].items():
        print(f"  {metric}: {score:.2f}")

    print(f"\nResults by test:")
    for run in report['runs']:
        print(f"  {run['test_case']}: {run['average_score']:.2f} ({run['latency_ms']:.0f}ms)")


def demonstrate_ab_testing():
    """Demonstrates A/B testing for prompts."""

    print("\n" + "=" * 60)
    print("A/B TESTING FOR PROMPTS")
    print("=" * 60)

    print("""
    A/B testing allows comparing different prompt versions
    to identify which produces better results.
    """)

    evaluator = PromptEvaluator()

    # Two different prompts for the same task
    prompt_a = """
Answer the user's question.
Question: {input}
"""

    prompt_b = """
You are a helpful expert. Carefully analyze the question
and provide a complete and well-structured answer.

Question: {input}

Respond clearly and informatively.
"""

    test_input = "What are the benefits of physical exercise?"

    print(f"\nTest question: {test_input}")
    print("-" * 40)

    # Test Prompt A
    print("\n--- PROMPT A (simple) ---")
    output_a, latency_a, tokens_a = evaluator.run_prompt(prompt_a, {"input": test_input})
    print(f"Response: {output_a[:300]}...")
    print(f"Latency: {latency_a:.2f}ms | Tokens: {tokens_a['total_tokens']}")

    eval_a_rel = evaluator.evaluate_relevance(test_input, output_a)
    eval_a_coh = evaluator.evaluate_coherence(output_a)
    print(f"Relevance: {eval_a_rel.score:.2f} | Coherence: {eval_a_coh.score:.2f}")

    # Test Prompt B
    print("\n--- PROMPT B (detailed) ---")
    output_b, latency_b, tokens_b = evaluator.run_prompt(prompt_b, {"input": test_input})
    print(f"Response: {output_b[:300]}...")
    print(f"Latency: {latency_b:.2f}ms | Tokens: {tokens_b['total_tokens']}")

    eval_b_rel = evaluator.evaluate_relevance(test_input, output_b)
    eval_b_coh = evaluator.evaluate_coherence(output_b)
    print(f"Relevance: {eval_b_rel.score:.2f} | Coherence: {eval_b_coh.score:.2f}")

    # Comparison
    print("\n" + "-" * 40)
    print("A/B COMPARISON")
    print("-" * 40)

    score_a = (eval_a_rel.score + eval_a_coh.score) / 2
    score_b = (eval_b_rel.score + eval_b_coh.score) / 2

    print(f"\nPrompt A: Average score = {score_a:.2f}")
    print(f"Prompt B: Average score = {score_b:.2f}")

    if score_b > score_a:
        print(f"\n→ Prompt B is {((score_b - score_a) / score_a * 100):.1f}% better in quality")
        print(f"→ But uses {tokens_b['total_tokens'] - tokens_a['total_tokens']} more tokens")
    else:
        print(f"\n→ Prompt A is more efficient with similar quality")


def main():
    print("=" * 60)
    print("PROMPT EVALUATION AND OBSERVABILITY")
    print("=" * 60)

    print("""
    Prompt evaluation is essential for:

    1. Ensuring response quality
    2. Measuring model consistency
    3. Optimizing costs (tokens)
    4. Comparing different versions
    5. Detecting regressions

    Common metrics:
    ┌─────────────────┬────────────────────────────────────┐
    │ Metric          │ What it evaluates                  │
    ├─────────────────┼────────────────────────────────────┤
    │ Relevance       │ Does response address the question?│
    │ Coherence       │ Does text have logic and flow?     │
    │ Groundedness    │ Based on facts/context?            │
    │ Accuracy        │ Correct information?               │
    │ Safety          │ Appropriate content?               │
    │ Fluency         │ Well-written text?                 │
    └─────────────────┴────────────────────────────────────┘
    """)

    # Demonstrations
    demonstrate_basic_evaluation()
    demonstrate_test_suite()
    demonstrate_ab_testing()

    print("\n" + "=" * 60)
    print("OBSERVABILITY TOOLS")
    print("=" * 60)

    print("""
    Popular tools for monitoring LLMs in production:

    1. LangSmith (LangChain)
       - Call tracing
       - Automated evaluation
       - Test playground
       - pip install langsmith

    2. LangFuse (Open Source)
       - Complete observability
       - Custom metrics
       - Self-hosted or cloud
       - pip install langfuse

    3. Weights & Biases
       - Experiment tracking
       - Model comparison
       - Prompt versioning
       - pip install wandb

    4. Promptfoo
       - CLI for testing
       - Prompt comparison
       - CI/CD integration
       - npx promptfoo@latest

    5. Phoenix (Arize)
       - OpenTelemetry tracing
       - Embedding analysis
       - Anomaly detection
       - pip install arize-phoenix
    """)

    print("\n" + "=" * 60)
    print("BEST PRACTICES")
    print("=" * 60)

    print("""
    1. Define clear metrics before optimizing
    2. Create representative test cases
    3. Automate evaluations in CI/CD
    4. Monitor latency and costs
    5. Version your prompts
    6. Document changes and results
    7. Use golden datasets for regression
    8. Consider human evaluation when needed
    """)

    print("\nEnd of Prompt Evaluation demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
