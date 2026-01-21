"""
Otimização de Custos em LLMs

A otimização de custos é crucial para aplicações LLM em produção.
Este módulo demonstra técnicas para reduzir gastos com APIs
mantendo a qualidade das respostas.

Estratégias de otimização:
1. Seleção de modelo adequado
2. Otimização de prompts (menos tokens)
3. Cache inteligente
4. Batch processing
5. Monitoramento de uso

Custos típicos (jan 2025):
- GPT-4o: $2.50/1M input, $10.00/1M output
- GPT-4o-mini: $0.15/1M input, $0.60/1M output
- Claude Sonnet: $3.00/1M input, $15.00/1M output

Requisitos:
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


# Preços por modelo (por 1M tokens) - Janeiro 2025
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
    """Registro de uso de API."""
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
    Contador de tokens.

    Usa tiktoken para contar tokens antes de fazer
    chamadas à API, permitindo estimativas de custo.
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
        """Conta tokens em um texto."""
        if self.encoding:
            return len(self.encoding.encode(text))
        # Estimativa aproximada se tiktoken não estiver disponível
        return len(text) // 4

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Conta tokens em uma lista de mensagens."""
        total = 0
        for message in messages:
            # Overhead por mensagem
            total += 4  # <|start|>role<|end|>
            for key, value in message.items():
                total += self.count_tokens(str(value))
        total += 2  # <|start|>assistant<|message|>
        return total

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = None
    ) -> float:
        """Estima o custo de uma chamada."""
        model = model or self.model
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost


class UsageTracker:
    """
    Rastreador de uso e custos.

    Monitora todas as chamadas à API e calcula
    custos acumulados por período.
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
        """Registra uma chamada à API."""
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
        """Retorna resumo de uso no período."""
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
        """Verifica status do orçamento."""
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


class PromptOptimizer:
    """
    Otimizador de prompts para reduzir tokens.

    Técnicas:
    - Remoção de redundâncias
    - Compressão de instruções
    - Uso de abreviações
    """

    def __init__(self):
        self.token_counter = TokenCounter()
        self.client = get_openai_client()

    def compress_prompt(self, prompt: str, target_reduction: float = 0.3) -> dict:
        """Comprime um prompt mantendo o significado."""
        original_tokens = self.token_counter.count_tokens(prompt)

        compression_prompt = f"""Reescreva o seguinte texto de forma mais concisa,
reduzindo o tamanho em aproximadamente {int(target_reduction * 100)}%,
mas mantendo todas as informações importantes e instruções.

Texto original:
{prompt}

Texto comprimido:"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": compression_prompt}],
            max_tokens=int(original_tokens * (1 - target_reduction)),
            temperature=0.3
        )

        compressed = response.choices[0].message.content
        compressed_tokens = self.token_counter.count_tokens(compressed)

        return {
            "original": prompt,
            "compressed": compressed,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "reduction": (original_tokens - compressed_tokens) / original_tokens,
            "savings_estimate": self.token_counter.estimate_cost(
                original_tokens - compressed_tokens, 0
            )
        }

    def suggest_optimizations(self, prompt: str) -> dict:
        """Sugere otimizações para um prompt."""
        tokens = self.token_counter.count_tokens(prompt)

        suggestions = []

        # Verificar redundâncias
        if prompt.count("por favor") > 1 or prompt.count("please") > 1:
            suggestions.append("Remova expressões de cortesia redundantes")

        # Verificar exemplos excessivos
        example_markers = ["exemplo:", "example:", "por exemplo", "for example"]
        example_count = sum(prompt.lower().count(m) for m in example_markers)
        if example_count > 3:
            suggestions.append(f"Considere reduzir os {example_count} exemplos para 2-3")

        # Verificar instruções repetidas
        lines = prompt.split("\n")
        if len(lines) > len(set(lines)):
            suggestions.append("Existem linhas duplicadas no prompt")

        # Verificar tamanho do system prompt
        if tokens > 1000:
            suggestions.append("Prompt muito longo. Considere usar resumos ou referências")

        return {
            "tokens": tokens,
            "estimated_cost": self.token_counter.estimate_cost(tokens, 500),
            "suggestions": suggestions,
            "optimization_potential": "alto" if len(suggestions) > 2 else "médio" if suggestions else "baixo"
        }


class ModelSelector:
    """
    Seletor inteligente de modelo.

    Escolhe o modelo mais adequado baseado na
    complexidade da tarefa e orçamento.
    """

    # Capacidades dos modelos
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
        """Analisa a complexidade de uma tarefa."""
        analysis_prompt = f"""Analise a seguinte tarefa e classifique sua complexidade.

Tarefa: {prompt[:500]}

Responda em JSON:
{{
    "complexity": "low/medium/high",
    "requires_reasoning": true/false,
    "requires_coding": true/false,
    "requires_creativity": true/false,
    "estimated_output_tokens": número
}}

Apenas JSON:"""

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
        """Seleciona o melhor modelo para a tarefa."""
        analysis = self.analyze_task(prompt)

        # Regras de seleção
        if analysis["complexity"] == "high" or analysis.get("requires_reasoning"):
            recommended = "gpt-4o" if prefer_quality else "gpt-4o-mini"
        elif analysis["complexity"] == "low" and prefer_speed:
            recommended = "gpt-3.5-turbo"
        else:
            recommended = "gpt-4o-mini"

        # Verificar custo
        token_counter = TokenCounter()
        prompt_tokens = token_counter.count_tokens(prompt)
        estimated_output = analysis.get("estimated_output_tokens", 500)
        estimated_cost = token_counter.estimate_cost(
            prompt_tokens, estimated_output, recommended
        )

        if max_cost and estimated_cost > max_cost:
            # Tentar modelo mais barato
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


class CostOptimizedLLM:
    """
    Wrapper LLM otimizado para custos.

    Combina todas as técnicas de otimização em
    uma interface unificada.
    """

    def __init__(
        self,
        daily_budget: float = None,
        monthly_budget: float = None,
        auto_select_model: bool = True,
        enable_caching: bool = True
    ):
        self.client = get_openai_client()
        self.tracker = UsageTracker()
        self.tracker.daily_budget = daily_budget
        self.tracker.monthly_budget = monthly_budget
        self.model_selector = ModelSelector()
        self.token_counter = TokenCounter()
        self.auto_select_model = auto_select_model
        self.cache: dict[str, str] = {} if enable_caching else None

    def chat(
        self,
        messages: list[dict],
        model: str = None,
        max_tokens: int = 1000,
        **kwargs
    ) -> dict:
        """Executa chat com otimizações de custo."""

        # Verificar cache
        if self.cache is not None:
            cache_key = json.dumps(messages, sort_keys=True)
            if cache_key in self.cache:
                return {
                    "content": self.cache[cache_key],
                    "cached": True,
                    "cost": 0.0
                }

        # Auto-seleção de modelo
        if self.auto_select_model and model is None:
            user_message = next(
                (m["content"] for m in messages if m["role"] == "user"),
                ""
            )
            selection = self.model_selector.select_model(user_message)
            model = selection["recommended_model"]
        else:
            model = model or "gpt-4o-mini"

        # Verificar orçamento
        budget_status = self.tracker.check_budget()
        if budget_status["daily"]["remaining"] is not None:
            if budget_status["daily"]["remaining"] <= 0:
                return {
                    "content": None,
                    "error": "Orçamento diário excedido",
                    "budget_status": budget_status
                }

        # Executar chamada
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
        latency_ms = (time.time() - start_time) * 1000

        content = response.choices[0].message.content

        # Registrar uso
        record = self.tracker.record(
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms
        )

        # Armazenar no cache
        if self.cache is not None:
            self.cache[cache_key] = content

        return {
            "content": content,
            "cached": False,
            "model": model,
            "cost": record.cost,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            },
            "latency_ms": latency_ms
        }

    def get_cost_report(self) -> dict:
        """Retorna relatório de custos."""
        return self.tracker.get_usage_summary()


def demonstrate_token_counting():
    """Demonstra contagem de tokens."""

    print("\n" + "=" * 60)
    print("CONTAGEM DE TOKENS")
    print("=" * 60)

    counter = TokenCounter()

    texts = [
        "Hello, world!",
        "Olá, mundo! Como você está hoje?",
        "Machine learning is a subset of artificial intelligence.",
    ]

    print("\nContagem de tokens:")
    print("-" * 40)

    for text in texts:
        tokens = counter.count_tokens(text)
        cost = counter.estimate_cost(tokens, 100)
        print(f"\n\"{text[:40]}...\"")
        print(f"   Tokens: {tokens}")
        print(f"   Custo estimado (+ 100 output): ${cost:.6f}")


def demonstrate_model_selection():
    """Demonstra seleção de modelo."""

    print("\n" + "=" * 60)
    print("SELEÇÃO INTELIGENTE DE MODELO")
    print("=" * 60)

    selector = ModelSelector()

    tasks = [
        "Qual é 2 + 2?",
        "Escreva um algoritmo de ordenação quicksort em Python com análise de complexidade.",
        "Analise o impacto econômico da inteligência artificial nos próximos 10 anos.",
    ]

    for task in tasks:
        print(f"\n\nTarefa: \"{task[:60]}...\"")
        print("-" * 40)

        result = selector.select_model(task)

        print(f"Modelo recomendado: {result['recommended_model']}")
        print(f"Complexidade: {result['analysis']['complexity']}")
        print(f"Custo estimado: ${result['estimated_cost']:.6f}")


def demonstrate_usage_tracking():
    """Demonstra rastreamento de uso."""

    print("\n" + "=" * 60)
    print("RASTREAMENTO DE USO E CUSTOS")
    print("=" * 60)

    tracker = UsageTracker()
    tracker.daily_budget = 1.00
    tracker.monthly_budget = 20.00

    # Simular algumas chamadas
    test_calls = [
        ("gpt-4o-mini", 100, 200, 150),
        ("gpt-4o-mini", 150, 300, 180),
        ("gpt-4o", 200, 500, 500),
        ("gpt-4o-mini", 80, 150, 120),
    ]

    print("\nSimulando chamadas...")
    for model, prompt, completion, latency in test_calls:
        record = tracker.record(model, prompt, completion, latency)
        print(f"   {model}: {record.total_tokens} tokens, ${record.cost:.6f}")

    # Resumo
    print("\n" + "-" * 40)
    print("RESUMO DE USO")
    print("-" * 40)

    summary = tracker.get_usage_summary()
    print(f"\nTotal de chamadas: {summary['total_calls']}")
    print(f"Total de tokens: {summary['total_tokens']:,}")
    print(f"Custo total: ${summary['total_cost']:.4f}")
    print(f"Latência média: {summary['avg_latency_ms']:.0f}ms")

    print("\nPor modelo:")
    for model, data in summary["by_model"].items():
        print(f"   {model}: {data['calls']} chamadas, ${data['cost']:.4f}")

    # Status do orçamento
    print("\n" + "-" * 40)
    print("STATUS DO ORÇAMENTO")
    print("-" * 40)

    budget = tracker.check_budget()
    print(f"\nDiário: ${budget['daily']['used']:.4f} / ${budget['daily']['budget']}")
    print(f"Mensal: ${budget['monthly']['used']:.4f} / ${budget['monthly']['budget']}")


def main():
    print("=" * 60)
    print("OTIMIZAÇÃO DE CUSTOS EM LLMs")
    print("=" * 60)

    print("""
    A otimização de custos é essencial para aplicações LLM em produção.

    Preços de referência (Janeiro 2025):
    ┌─────────────────┬───────────────┬───────────────┐
    │ Modelo          │ Input/1M tok  │ Output/1M tok │
    ├─────────────────┼───────────────┼───────────────┤
    │ GPT-4o          │ $2.50         │ $10.00        │
    │ GPT-4o-mini     │ $0.15         │ $0.60         │
    │ GPT-3.5-turbo   │ $0.50         │ $1.50         │
    │ Claude Sonnet   │ $3.00         │ $15.00        │
    └─────────────────┴───────────────┴───────────────┘

    Estratégias de otimização:
    1. 📊 Seleção inteligente de modelo
    2. ✂️  Otimização de prompts (menos tokens)
    3. 💾 Cache de respostas
    4. 📦 Batch processing
    5. 📈 Monitoramento contínuo
    """)

    # Demonstrações
    demonstrate_token_counting()
    demonstrate_model_selection()
    demonstrate_usage_tracking()

    print("\n" + "=" * 60)
    print("DICAS DE OTIMIZAÇÃO")
    print("=" * 60)

    print("""
    1. Use GPT-4o-mini como padrão (17x mais barato que GPT-4o)

    2. Implemente cache para perguntas repetidas

    3. Otimize prompts:
       - Remova instruções redundantes
       - Use exemplos concisos
       - Evite repetições

    4. Defina max_tokens apropriado

    5. Use batch processing quando possível

    6. Monitore custos diariamente

    7. Configure alertas de orçamento

    8. Considere modelos locais (Ollama) para testes

    9. Use embeddings menores quando possível

    10. Implemente fallbacks para modelos mais baratos
    """)

    print("\nFim do demo de Otimização de Custos")
    print("=" * 60)


if __name__ == "__main__":
    main()
