"""
LLM Cost Optimization

Cost optimization is crucial for production LLM applications.
This module demonstrates techniques to reduce API costs
while maintaining response quality.

Optimization strategies:
1. Appropriate model selection
2. Prompt optimization (fewer tokens)
3. Smart caching
4. Batch processing
5. Usage monitoring

Typical costs (Jan 2025):
- GPT-4o: $2.50/1M input, $10.00/1M output
- GPT-4o-mini: $0.15/1M input, $0.60/1M output
- Claude Sonnet: $3.00/1M input, $15.00/1M output

Requirements:
- pip install openai tiktoken
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from utils.openai_client import get_openai_client


# Pricing per model (per 1M tokens) - January 2025
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
}


@dataclass
class UsageRecord:
    """API usage record."""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency_ms: float
    endpoint: str = "chat"
    metadata: dict = field(default_factory=dict)


class TokenCounter:
    """
    Token counter.

    Uses tiktoken to count tokens before making
    API calls, enabling cost estimates.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Counts tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        # Approximate estimate if tiktoken is not available
        return len(text) // 4

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Counts tokens in a list of messages."""
        total = 0
        for message in messages:
            total += 4  # Overhead per message
            for key, value in message.items():
                total += self.count_tokens(str(value))
        total += 2  # Assistant message start
        return total

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = None
    ) -> float:
        """Estimates the cost of a call."""
        model = model or self.model
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost


class UsageTracker:
    """
    Usage and cost tracker.

    Monitors all API calls and calculates
    accumulated costs by period.
    """

    def __init__(self):
        self.records: list[UsageRecord] = []
        self.daily_budget: Optional[float] = None
        self.monthly_budget: Optional[float] = None

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        endpoint: str = "chat",
        metadata: dict = None
    ):
        """Records an API call."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
        cost = (
            (prompt_tokens / 1_000_000) * pricing["input"] +
            (completion_tokens / 1_000_000) * pricing["output"]
        )

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            endpoint=endpoint,
            metadata=metadata or {}
        )

        self.records.append(record)
        return record

    def get_usage_summary(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> dict:
        """Returns usage summary for the period."""
        filtered = self.records

        if start_date:
            filtered = [r for r in filtered if r.timestamp >= start_date]
        if end_date:
            filtered = [r for r in filtered if r.timestamp <= end_date]

        if not filtered:
            return {"total_cost": 0, "total_tokens": 0, "calls": 0}

        by_model = defaultdict(lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0
        })

        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0

        for record in filtered:
            by_model[record.model]["calls"] += 1
            by_model[record.model]["prompt_tokens"] += record.prompt_tokens
            by_model[record.model]["completion_tokens"] += record.completion_tokens
            by_model[record.model]["cost"] += record.cost

            total_cost += record.cost
            total_tokens += record.total_tokens
            total_latency += record.latency_ms

        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_calls": len(filtered),
            "avg_latency_ms": total_latency / len(filtered),
            "by_model": dict(by_model),
            "period": {
                "start": min(r.timestamp for r in filtered).isoformat(),
                "end": max(r.timestamp for r in filtered).isoformat()
            }
        }

    def check_budget(self) -> dict:
        """Checks budget status."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        daily_usage = sum(
            r.cost for r in self.records
            if r.timestamp >= today_start
        )

        monthly_usage = sum(
            r.cost for r in self.records
            if r.timestamp >= month_start
        )

        return {
            "daily": {
                "used": daily_usage,
                "budget": self.daily_budget,
                "remaining": (self.daily_budget - daily_usage) if self.daily_budget else None,
                "percent_used": (daily_usage / self.daily_budget * 100) if self.daily_budget else None
            },
            "monthly": {
                "used": monthly_usage,
                "budget": self.monthly_budget,
                "remaining": (self.monthly_budget - monthly_usage) if self.monthly_budget else None,
                "percent_used": (monthly_usage / self.monthly_budget * 100) if self.monthly_budget else None
            }
        }


class ModelSelector:
    """
    Intelligent model selector.

    Chooses the most appropriate model based on
    task complexity and budget.
    """

    MODEL_CAPABILITIES = {
        "gpt-4o": {
            "complexity": "high",
            "reasoning": "excellent",
            "coding": "excellent",
            "speed": "medium",
            "cost": "high"
        },
        "gpt-4o-mini": {
            "complexity": "medium",
            "reasoning": "good",
            "coding": "good",
            "speed": "fast",
            "cost": "low"
        },
        "gpt-3.5-turbo": {
            "complexity": "low",
            "reasoning": "basic",
            "coding": "basic",
            "speed": "very_fast",
            "cost": "very_low"
        }
    }

    def __init__(self):
        self.client = get_openai_client()

    def analyze_task(self, prompt: str) -> dict:
        """Analyzes task complexity."""
        analysis_prompt = f"""Analyze the following task and classify its complexity.

Task: {prompt[:500]}

Respond in JSON:
{{
    "complexity": "low/medium/high",
    "requires_reasoning": true/false,
    "requires_coding": true/false,
    "requires_creativity": true/false,
    "estimated_output_tokens": number
}}

JSON only:"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=200,
            temperature=0.0
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "complexity": "medium",
                "requires_reasoning": False,
                "requires_coding": False,
                "requires_creativity": False,
                "estimated_output_tokens": 500
            }

    def select_model(
        self,
        prompt: str,
        prefer_quality: bool = False,
        prefer_speed: bool = False,
        max_cost: float = None
    ) -> dict:
        """Selects the best model for the task."""
        analysis = self.analyze_task(prompt)

        # Selection rules
        if analysis["complexity"] == "high" or analysis.get("requires_reasoning"):
            recommended = "gpt-4o" if prefer_quality else "gpt-4o-mini"
        elif analysis["complexity"] == "low" and prefer_speed:
            recommended = "gpt-3.5-turbo"
        else:
            recommended = "gpt-4o-mini"

        # Check cost
        token_counter = TokenCounter()
        prompt_tokens = token_counter.count_tokens(prompt)
        estimated_output = analysis.get("estimated_output_tokens", 500)
        estimated_cost = token_counter.estimate_cost(
            prompt_tokens, estimated_output, recommended
        )

        if max_cost and estimated_cost > max_cost:
            for model in ["gpt-4o-mini", "gpt-3.5-turbo"]:
                cost = token_counter.estimate_cost(
                    prompt_tokens, estimated_output, model
                )
                if cost <= max_cost:
                    recommended = model
                    estimated_cost = cost
                    break

        return {
            "recommended_model": recommended,
            "analysis": analysis,
            "estimated_cost": estimated_cost,
            "alternatives": [
                {
                    "model": model,
                    "cost": token_counter.estimate_cost(
                        prompt_tokens, estimated_output, model
                    ),
                    "capabilities": caps
                }
                for model, caps in self.MODEL_CAPABILITIES.items()
            ]
        }


def demonstrate_token_counting():
    """Demonstrates token counting."""
    print("\n" + "=" * 60)
    print("TOKEN COUNTING")
    print("=" * 60)

    counter = TokenCounter()

    texts = [
        "Hello, world!",
        "Machine learning is a subset of artificial intelligence.",
        "Write a Python function that calculates the factorial of a number.",
    ]

    print("\nToken counts:")
    print("-" * 40)

    for text in texts:
        tokens = counter.count_tokens(text)
        cost = counter.estimate_cost(tokens, 100)
        print(f"\n\"{text[:40]}...\"")
        print(f"   Tokens: {tokens}")
        print(f"   Estimated cost (+ 100 output): ${cost:.6f}")


def demonstrate_model_selection():
    """Demonstrates model selection."""
    print("\n" + "=" * 60)
    print("INTELLIGENT MODEL SELECTION")
    print("=" * 60)

    selector = ModelSelector()

    tasks = [
        "What is 2 + 2?",
        "Write a quicksort algorithm in Python with complexity analysis.",
        "Analyze the economic impact of AI over the next 10 years.",
    ]

    for task in tasks:
        print(f"\n\nTask: \"{task[:60]}...\"")
        print("-" * 40)

        result = selector.select_model(task)

        print(f"Recommended model: {result['recommended_model']}")
        print(f"Complexity: {result['analysis']['complexity']}")
        print(f"Estimated cost: ${result['estimated_cost']:.6f}")


def demonstrate_usage_tracking():
    """Demonstrates usage tracking."""
    print("\n" + "=" * 60)
    print("USAGE AND COST TRACKING")
    print("=" * 60)

    tracker = UsageTracker()
    tracker.daily_budget = 1.00
    tracker.monthly_budget = 20.00

    # Simulate some calls
    test_calls = [
        ("gpt-4o-mini", 100, 200, 150),
        ("gpt-4o-mini", 150, 300, 180),
        ("gpt-4o", 200, 500, 500),
        ("gpt-4o-mini", 80, 150, 120),
    ]

    print("\nSimulating calls...")
    for model, prompt, completion, latency in test_calls:
        record = tracker.record(model, prompt, completion, latency)
        print(f"   {model}: {record.total_tokens} tokens, ${record.cost:.6f}")

    # Summary
    print("\n" + "-" * 40)
    print("USAGE SUMMARY")
    print("-" * 40)

    summary = tracker.get_usage_summary()
    print(f"\nTotal calls: {summary['total_calls']}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print(f"Avg latency: {summary['avg_latency_ms']:.0f}ms")

    print("\nBy model:")
    for model, data in summary["by_model"].items():
        print(f"   {model}: {data['calls']} calls, ${data['cost']:.4f}")

    # Budget status
    print("\n" + "-" * 40)
    print("BUDGET STATUS")
    print("-" * 40)

    budget = tracker.check_budget()
    print(f"\nDaily: ${budget['daily']['used']:.4f} / ${budget['daily']['budget']}")
    print(f"Monthly: ${budget['monthly']['used']:.4f} / ${budget['monthly']['budget']}")


def main():
    print("=" * 60)
    print("LLM COST OPTIMIZATION")
    print("=" * 60)

    print("""
    Cost optimization is essential for production LLM applications.

    Reference pricing (January 2025):
    ┌─────────────────┬───────────────┬───────────────┐
    │ Model           │ Input/1M tok  │ Output/1M tok │
    ├─────────────────┼───────────────┼───────────────┤
    │ GPT-4o          │ $2.50         │ $10.00        │
    │ GPT-4o-mini     │ $0.15         │ $0.60         │
    │ GPT-3.5-turbo   │ $0.50         │ $1.50         │
    │ Claude Sonnet   │ $3.00         │ $15.00        │
    └─────────────────┴───────────────┴───────────────┘

    Optimization strategies:
    1. 📊 Intelligent model selection
    2. ✂️  Prompt optimization (fewer tokens)
    3. 💾 Response caching
    4. 📦 Batch processing
    5. 📈 Continuous monitoring
    """)

    # Demonstrations
    demonstrate_token_counting()
    demonstrate_model_selection()
    demonstrate_usage_tracking()

    print("\n" + "=" * 60)
    print("OPTIMIZATION TIPS")
    print("=" * 60)

    print("""
    1. Use GPT-4o-mini as default (17x cheaper than GPT-4o)

    2. Implement caching for repeated questions

    3. Optimize prompts:
       - Remove redundant instructions
       - Use concise examples
       - Avoid repetitions

    4. Set appropriate max_tokens

    5. Use batch processing when possible

    6. Monitor costs daily

    7. Configure budget alerts

    8. Consider local models (Ollama) for testing

    9. Use smaller embeddings when possible

    10. Implement fallbacks to cheaper models
    """)

    print("\nEnd of Cost Optimization demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
