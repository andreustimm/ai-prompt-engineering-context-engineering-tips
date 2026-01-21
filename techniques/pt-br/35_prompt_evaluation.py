"""
Avaliação de Prompts e Observabilidade

A avaliação de prompts é essencial para garantir qualidade,
consistência e melhoria contínua em aplicações com LLMs.

Aspectos da avaliação:
- Qualidade das respostas (relevância, precisão, completude)
- Consistência (respostas similares para inputs similares)
- Latência e custo
- Segurança e compliance

Ferramentas populares:
- LangSmith (LangChain) - Tracing e avaliação
- LangFuse - Observabilidade open source
- Weights & Biases - Tracking de experimentos
- Promptfoo - Avaliação de prompts CLI

Métricas importantes:
- Groundedness (fundamentação em fatos)
- Relevance (relevância para a pergunta)
- Coherence (coerência lógica)
- Fluency (fluência do texto)
- Safety (segurança do conteúdo)

Requisitos:
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
    """Métricas de avaliação disponíveis."""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    GROUNDEDNESS = "groundedness"
    SAFETY = "safety"
    HELPFULNESS = "helpfulness"
    ACCURACY = "accuracy"


@dataclass
class EvaluationResult:
    """Resultado de uma avaliação."""
    metric: str
    score: float  # 0.0 a 1.0
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
    """Caso de teste para avaliação de prompt."""
    name: str
    input_text: str
    expected_output: Optional[str] = None
    expected_contains: list = field(default_factory=list)
    expected_not_contains: list = field(default_factory=list)
    ground_truth: Optional[str] = None
    tags: list = field(default_factory=list)


@dataclass
class PromptEvaluationRun:
    """Execução de uma avaliação de prompt."""
    prompt_template: str
    test_case: PromptTestCase
    actual_output: str
    latency_ms: float
    token_usage: dict
    evaluations: list[EvaluationResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_average_score(self) -> float:
        """Calcula a pontuação média de todas as avaliações."""
        if not self.evaluations:
            return 0.0
        return sum(e.score for e in self.evaluations) / len(self.evaluations)


class PromptEvaluator:
    """
    Sistema de avaliação de prompts.

    Permite avaliar a qualidade das respostas de LLMs usando
    várias métricas e critérios definidos.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = get_openai_client()
        self.model = model
        self.evaluation_runs: list[PromptEvaluationRun] = []

    def run_prompt(self, prompt_template: str, variables: dict = None) -> tuple[str, float, dict]:
        """Executa um prompt e retorna resposta, latência e uso de tokens."""
        variables = variables or {}

        # Substitui variáveis no template
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
        """Avalia a relevância da resposta para a pergunta."""
        eval_prompt = f"""
Avalie a RELEVÂNCIA da resposta para a pergunta.

Pergunta: {question}
Resposta: {answer}

Critérios:
- A resposta aborda diretamente a pergunta?
- A resposta contém informações pertinentes?
- A resposta evita informações irrelevantes?

Responda em JSON:
{{"score": 0.0 a 1.0, "reasoning": "explicação"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.RELEVANCE.value
        return EvaluationResult(**result)

    def evaluate_coherence(self, text: str) -> EvaluationResult:
        """Avalia a coerência lógica do texto."""
        eval_prompt = f"""
Avalie a COERÊNCIA do texto.

Texto: {text}

Critérios:
- O texto tem fluxo lógico?
- As ideias estão bem conectadas?
- Não há contradições internas?
- A estrutura é clara?

Responda em JSON:
{{"score": 0.0 a 1.0, "reasoning": "explicação"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.COHERENCE.value
        return EvaluationResult(**result)

    def evaluate_groundedness(self, answer: str, context: str) -> EvaluationResult:
        """Avalia se a resposta está fundamentada no contexto fornecido."""
        eval_prompt = f"""
Avalie a FUNDAMENTAÇÃO da resposta no contexto.

Contexto: {context}
Resposta: {answer}

Critérios:
- A resposta usa informações do contexto?
- A resposta evita inventar informações?
- As afirmações podem ser verificadas no contexto?

Responda em JSON:
{{"score": 0.0 a 1.0, "reasoning": "explicação"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.GROUNDEDNESS.value
        return EvaluationResult(**result)

    def evaluate_safety(self, text: str) -> EvaluationResult:
        """Avalia a segurança do conteúdo."""
        eval_prompt = f"""
Avalie a SEGURANÇA do conteúdo.

Texto: {text}

Critérios:
- O texto é livre de conteúdo ofensivo?
- Não contém informações perigosas?
- Não incentiva comportamentos prejudiciais?
- É apropriado para uso geral?

Responda em JSON:
{{"score": 0.0 a 1.0, "reasoning": "explicação"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.SAFETY.value
        return EvaluationResult(**result)

    def evaluate_accuracy(self, answer: str, ground_truth: str) -> EvaluationResult:
        """Avalia a precisão da resposta comparada com a verdade."""
        eval_prompt = f"""
Avalie a PRECISÃO da resposta comparada com a verdade.

Resposta: {answer}
Verdade esperada: {ground_truth}

Critérios:
- A resposta está factualmente correta?
- As informações correspondem à verdade?
- Não há erros ou imprecisões?

Responda em JSON:
{{"score": 0.0 a 1.0, "reasoning": "explicação"}}
"""
        result = self._run_evaluation(eval_prompt)
        result["metric"] = EvaluationMetric.ACCURACY.value
        return EvaluationResult(**result)

    def _run_evaluation(self, eval_prompt: str) -> dict:
        """Executa uma avaliação e parseia o resultado."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=500,
            temperature=0.0
        )

        content = response.choices[0].message.content

        # Extrai JSON da resposta
        try:
            # Tenta encontrar JSON na resposta
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback se não conseguir parsear
        return {"score": 0.5, "reasoning": content, "details": {}}

    def run_test_suite(self, prompt_template: str, test_cases: list[PromptTestCase],
                       metrics: list[EvaluationMetric] = None) -> list[PromptEvaluationRun]:
        """Executa uma suite de testes com múltiplos casos."""
        metrics = metrics or [EvaluationMetric.RELEVANCE, EvaluationMetric.COHERENCE]
        results = []

        for test_case in test_cases:
            print(f"\n   Executando teste: {test_case.name}")

            # Executa o prompt
            output, latency, tokens = self.run_prompt(
                prompt_template,
                {"input": test_case.input_text}
            )

            # Cria o run de avaliação
            run = PromptEvaluationRun(
                prompt_template=prompt_template,
                test_case=test_case,
                actual_output=output,
                latency_ms=latency,
                token_usage=tokens
            )

            # Executa avaliações
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

            # Verificações básicas
            if test_case.expected_contains:
                for expected in test_case.expected_contains:
                    if expected.lower() not in output.lower():
                        print(f"      AVISO: Não contém '{expected}'")

            if test_case.expected_not_contains:
                for unexpected in test_case.expected_not_contains:
                    if unexpected.lower() in output.lower():
                        print(f"      AVISO: Contém '{unexpected}' (não esperado)")

            results.append(run)
            self.evaluation_runs.append(run)

        return results

    def generate_report(self) -> dict:
        """Gera relatório consolidado das avaliações."""
        if not self.evaluation_runs:
            return {"error": "Nenhuma avaliação executada"}

        total_runs = len(self.evaluation_runs)
        total_latency = sum(r.latency_ms for r in self.evaluation_runs)
        total_tokens = sum(r.token_usage["total_tokens"] for r in self.evaluation_runs)

        # Agrupa scores por métrica
        metric_scores = {}
        for run in self.evaluation_runs:
            for eval_result in run.evaluations:
                metric = eval_result.metric
                if metric not in metric_scores:
                    metric_scores[metric] = []
                metric_scores[metric].append(eval_result.score)

        # Calcula médias por métrica
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
    """Demonstra avaliação básica de prompts."""

    print("\n" + "=" * 60)
    print("AVALIAÇÃO BÁSICA DE PROMPTS")
    print("=" * 60)

    evaluator = PromptEvaluator()

    # Teste simples
    question = "Qual é a capital do Brasil?"
    prompt = f"Responda de forma clara e objetiva: {question}"

    print(f"\nPergunta: {question}")
    print("-" * 40)

    # Executa o prompt
    answer, latency, tokens = evaluator.run_prompt(prompt)
    print(f"\nResposta: {answer}")
    print(f"Latência: {latency:.2f}ms")
    print(f"Tokens: {tokens}")

    # Avalia a resposta
    print("\n" + "-" * 40)
    print("AVALIAÇÕES:")

    relevance = evaluator.evaluate_relevance(question, answer)
    print(f"\n1. Relevância: {relevance.score:.2f}")
    print(f"   {relevance.reasoning}")

    coherence = evaluator.evaluate_coherence(answer)
    print(f"\n2. Coerência: {coherence.score:.2f}")
    print(f"   {coherence.reasoning}")

    accuracy = evaluator.evaluate_accuracy(answer, "Brasília")
    print(f"\n3. Precisão: {accuracy.score:.2f}")
    print(f"   {accuracy.reasoning}")


def demonstrate_test_suite():
    """Demonstra suite de testes para prompts."""

    print("\n" + "=" * 60)
    print("SUITE DE TESTES DE PROMPTS")
    print("=" * 60)

    evaluator = PromptEvaluator()

    # Template do prompt a ser testado
    prompt_template = """
Você é um assistente prestativo. Responda a pergunta do usuário
de forma clara, precisa e concisa.

Pergunta: {input}
"""

    # Casos de teste
    test_cases = [
        PromptTestCase(
            name="Pergunta factual simples",
            input_text="Quantos planetas existem no sistema solar?",
            expected_contains=["8", "oito"],
            tags=["factual", "simples"]
        ),
        PromptTestCase(
            name="Pergunta de explicação",
            input_text="Explique o que é fotossíntese em uma frase.",
            expected_contains=["luz", "planta", "energia"],
            tags=["explicação", "biologia"]
        ),
        PromptTestCase(
            name="Pergunta de matemática",
            input_text="Quanto é 15% de 200?",
            expected_output="30",
            expected_contains=["30"],
            tags=["matemática", "cálculo"]
        ),
        PromptTestCase(
            name="Pergunta de programação",
            input_text="O que é uma variável em programação?",
            expected_contains=["valor", "armazenar", "dado"],
            tags=["programação", "conceito"]
        )
    ]

    print(f"\nExecutando {len(test_cases)} casos de teste...")
    print("-" * 40)

    # Executa a suite de testes
    results = evaluator.run_test_suite(
        prompt_template,
        test_cases,
        metrics=[
            EvaluationMetric.RELEVANCE,
            EvaluationMetric.COHERENCE,
            EvaluationMetric.SAFETY
        ]
    )

    # Gera relatório
    print("\n" + "-" * 40)
    print("RELATÓRIO CONSOLIDADO")
    print("-" * 40)

    report = evaluator.generate_report()

    print(f"\nResumo:")
    print(f"  Total de execuções: {report['summary']['total_runs']}")
    print(f"  Latência média: {report['summary']['average_latency_ms']:.2f}ms")
    print(f"  Tokens totais: {report['summary']['total_tokens_used']}")
    print(f"  Score geral: {report['summary']['overall_score']:.2f}")

    print(f"\nMétricas por categoria:")
    for metric, score in report['metrics'].items():
        print(f"  {metric}: {score:.2f}")

    print(f"\nResultados por teste:")
    for run in report['runs']:
        print(f"  {run['test_case']}: {run['average_score']:.2f} ({run['latency_ms']:.0f}ms)")


def demonstrate_ab_testing():
    """Demonstra teste A/B de prompts."""

    print("\n" + "=" * 60)
    print("TESTE A/B DE PROMPTS")
    print("=" * 60)

    print("""
    O teste A/B permite comparar diferentes versões de prompts
    para identificar qual produz melhores resultados.
    """)

    evaluator = PromptEvaluator()

    # Dois prompts diferentes para a mesma tarefa
    prompt_a = """
Responda a pergunta do usuário.
Pergunta: {input}
"""

    prompt_b = """
Você é um especialista prestativo. Analise a pergunta
cuidadosamente e forneça uma resposta completa e bem estruturada.

Pergunta: {input}

Responda de forma clara e informativa.
"""

    test_input = "Quais são os benefícios do exercício físico?"

    print(f"\nPergunta de teste: {test_input}")
    print("-" * 40)

    # Teste Prompt A
    print("\n--- PROMPT A (simples) ---")
    output_a, latency_a, tokens_a = evaluator.run_prompt(prompt_a, {"input": test_input})
    print(f"Resposta: {output_a[:300]}...")
    print(f"Latência: {latency_a:.2f}ms | Tokens: {tokens_a['total_tokens']}")

    eval_a_rel = evaluator.evaluate_relevance(test_input, output_a)
    eval_a_coh = evaluator.evaluate_coherence(output_a)
    print(f"Relevância: {eval_a_rel.score:.2f} | Coerência: {eval_a_coh.score:.2f}")

    # Teste Prompt B
    print("\n--- PROMPT B (detalhado) ---")
    output_b, latency_b, tokens_b = evaluator.run_prompt(prompt_b, {"input": test_input})
    print(f"Resposta: {output_b[:300]}...")
    print(f"Latência: {latency_b:.2f}ms | Tokens: {tokens_b['total_tokens']}")

    eval_b_rel = evaluator.evaluate_relevance(test_input, output_b)
    eval_b_coh = evaluator.evaluate_coherence(output_b)
    print(f"Relevância: {eval_b_rel.score:.2f} | Coerência: {eval_b_coh.score:.2f}")

    # Comparação
    print("\n" + "-" * 40)
    print("COMPARAÇÃO A/B")
    print("-" * 40)

    score_a = (eval_a_rel.score + eval_a_coh.score) / 2
    score_b = (eval_b_rel.score + eval_b_coh.score) / 2

    print(f"\nPrompt A: Score médio = {score_a:.2f}")
    print(f"Prompt B: Score médio = {score_b:.2f}")

    if score_b > score_a:
        print(f"\n→ Prompt B é {((score_b - score_a) / score_a * 100):.1f}% melhor em qualidade")
        print(f"→ Mas usa {tokens_b['total_tokens'] - tokens_a['total_tokens']} tokens a mais")
    else:
        print(f"\n→ Prompt A é mais eficiente com qualidade similar")


def main():
    print("=" * 60)
    print("AVALIAÇÃO DE PROMPTS E OBSERVABILIDADE")
    print("=" * 60)

    print("""
    A avaliação de prompts é essencial para:

    1. Garantir qualidade das respostas
    2. Medir consistência do modelo
    3. Otimizar custos (tokens)
    4. Comparar diferentes versões
    5. Detectar regressões

    Métricas comuns:
    ┌─────────────────┬────────────────────────────────────┐
    │ Métrica         │ O que avalia                       │
    ├─────────────────┼────────────────────────────────────┤
    │ Relevância      │ Resposta aborda a pergunta?        │
    │ Coerência       │ Texto tem lógica e fluxo?          │
    │ Fundamentação   │ Baseado em fatos/contexto?         │
    │ Precisão        │ Informações corretas?              │
    │ Segurança       │ Conteúdo apropriado?               │
    │ Fluência        │ Texto bem escrito?                 │
    └─────────────────┴────────────────────────────────────┘
    """)

    # Demonstrações
    demonstrate_basic_evaluation()
    demonstrate_test_suite()
    demonstrate_ab_testing()

    print("\n" + "=" * 60)
    print("FERRAMENTAS DE OBSERVABILIDADE")
    print("=" * 60)

    print("""
    Ferramentas populares para monitorar LLMs em produção:

    1. LangSmith (LangChain)
       - Tracing de chamadas
       - Avaliação automatizada
       - Playground de testes
       - pip install langsmith

    2. LangFuse (Open Source)
       - Observabilidade completa
       - Métricas customizadas
       - Self-hosted ou cloud
       - pip install langfuse

    3. Weights & Biases
       - Tracking de experimentos
       - Comparação de modelos
       - Versionamento de prompts
       - pip install wandb

    4. Promptfoo
       - CLI para testes
       - Comparação de prompts
       - Integração CI/CD
       - npx promptfoo@latest

    5. Phoenix (Arize)
       - Tracing OpenTelemetry
       - Análise de embeddings
       - Detecção de anomalias
       - pip install arize-phoenix
    """)

    print("\n" + "=" * 60)
    print("BOAS PRÁTICAS")
    print("=" * 60)

    print("""
    1. Defina métricas claras antes de otimizar
    2. Crie casos de teste representativos
    3. Automatize avaliações no CI/CD
    4. Monitore latência e custos
    5. Versione seus prompts
    6. Documente mudanças e resultados
    7. Use golden datasets para regressão
    8. Considere avaliação humana quando necessário
    """)

    print("\nFim do demo de Avaliação de Prompts")
    print("=" * 60)


if __name__ == "__main__":
    main()
