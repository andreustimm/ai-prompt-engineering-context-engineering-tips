"""
Fine-tuning de Modelos de Linguagem

Fine-tuning permite customizar modelos para tarefas específicas,
melhorando a qualidade e reduzindo custos a longo prazo.

Quando usar fine-tuning:
1. Formato de saída muito específico
2. Estilo ou tom particular
3. Tarefas especializadas de domínio
4. Redução de custos em alto volume
5. Melhoria de consistência

Processo de fine-tuning:
1. Preparação dos dados (JSONL)
2. Upload do arquivo
3. Criação do job de fine-tuning
4. Monitoramento do treinamento
5. Avaliação e uso do modelo

Requisitos:
- pip install openai tiktoken
- Conta OpenAI com acesso a fine-tuning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from utils.openai_client import get_openai_client


@dataclass
class TrainingExample:
    """Exemplo de treinamento para fine-tuning."""
    system: str
    user: str
    assistant: str
    metadata: dict = field(default_factory=dict)

    def to_jsonl(self) -> dict:
        """Converte para formato JSONL da OpenAI."""
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant}
            ]
        }


class DatasetValidator:
    """
    Validador de dataset para fine-tuning.

    Verifica formato, tamanho e qualidade dos dados.
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
        return len(text) // 4

    def validate_example(self, example: dict) -> tuple[bool, list[str]]:
        """Valida um exemplo de treinamento."""
        errors = []

        # Verifica estrutura básica
        if "messages" not in example:
            errors.append("Campo 'messages' ausente")
            return False, errors

        messages = example["messages"]

        if not isinstance(messages, list):
            errors.append("'messages' deve ser uma lista")
            return False, errors

        if len(messages) < 2:
            errors.append("Mínimo de 2 mensagens (user + assistant)")

        # Verifica roles
        roles = [m.get("role") for m in messages]

        if "assistant" not in roles:
            errors.append("Deve ter pelo menos uma mensagem do assistant")

        # Conta mensagens do assistant para supervisão
        assistant_count = roles.count("assistant")
        if assistant_count == 0:
            errors.append("Sem mensagem do assistant para treinar")

        # Verifica conteúdo
        for i, msg in enumerate(messages):
            if "role" not in msg:
                errors.append(f"Mensagem {i}: 'role' ausente")
            if "content" not in msg:
                errors.append(f"Mensagem {i}: 'content' ausente")
            elif not msg["content"].strip():
                errors.append(f"Mensagem {i}: conteúdo vazio")

        # Verifica tamanho total
        total_tokens = sum(
            self.count_tokens(m.get("content", ""))
            for m in messages
        )

        # Limite de contexto (varia por modelo)
        max_tokens = 16385 if "gpt-4" in self.model else 4096

        if total_tokens > max_tokens:
            errors.append(f"Total de tokens ({total_tokens}) excede limite ({max_tokens})")

        return len(errors) == 0, errors

    def validate_dataset(self, examples: list[dict]) -> dict:
        """Valida um dataset completo."""
        results = {
            "total_examples": len(examples),
            "valid_examples": 0,
            "invalid_examples": 0,
            "errors": [],
            "warnings": [],
            "token_stats": {
                "total": 0,
                "min": float('inf'),
                "max": 0,
                "avg": 0
            }
        }

        token_counts = []

        for i, example in enumerate(examples):
            is_valid, errors = self.validate_example(example)

            if is_valid:
                results["valid_examples"] += 1

                # Calcula tokens
                tokens = sum(
                    self.count_tokens(m.get("content", ""))
                    for m in example.get("messages", [])
                )
                token_counts.append(tokens)
            else:
                results["invalid_examples"] += 1
                results["errors"].append({
                    "example_index": i,
                    "errors": errors
                })

        # Estatísticas de tokens
        if token_counts:
            results["token_stats"]["total"] = sum(token_counts)
            results["token_stats"]["min"] = min(token_counts)
            results["token_stats"]["max"] = max(token_counts)
            results["token_stats"]["avg"] = sum(token_counts) / len(token_counts)

        # Warnings
        if len(examples) < 10:
            results["warnings"].append("Dataset muito pequeno (mínimo recomendado: 10 exemplos)")

        if len(examples) < 50:
            results["warnings"].append("Considere mais exemplos para melhores resultados (50-100+)")

        return results


class DatasetGenerator:
    """
    Gerador de datasets para fine-tuning.

    Ajuda a criar exemplos de treinamento a partir de templates.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.examples: list[TrainingExample] = []

    def add_example(
        self,
        user_input: str,
        expected_output: str,
        metadata: dict = None
    ):
        """Adiciona um exemplo de treinamento."""
        example = TrainingExample(
            system=self.system_prompt,
            user=user_input,
            assistant=expected_output,
            metadata=metadata or {}
        )
        self.examples.append(example)

    def add_examples_from_pairs(self, pairs: list[tuple[str, str]]):
        """Adiciona múltiplos exemplos de pares (input, output)."""
        for user_input, expected_output in pairs:
            self.add_example(user_input, expected_output)

    def export_jsonl(self, filepath: str):
        """Exporta dataset para arquivo JSONL."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in self.examples:
                json.dump(example.to_jsonl(), f, ensure_ascii=False)
                f.write('\n')

    def get_dataset(self) -> list[dict]:
        """Retorna dataset como lista de dicts."""
        return [ex.to_jsonl() for ex in self.examples]


class FineTuningManager:
    """
    Gerenciador de fine-tuning.

    Facilita upload, treinamento e uso de modelos fine-tuned.
    """

    def __init__(self):
        self.client = get_openai_client()

    def upload_training_file(self, filepath: str) -> str:
        """
        Faz upload de arquivo de treinamento.

        Retorna o file_id para uso no fine-tuning.
        """
        with open(filepath, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose="fine-tune"
            )

        return response.id

    def create_fine_tuning_job(
        self,
        training_file_id: str,
        model: str = "gpt-4o-mini-2024-07-18",
        suffix: str = None,
        n_epochs: int = None,
        hyperparameters: dict = None
    ) -> dict:
        """
        Cria um job de fine-tuning.

        Args:
            training_file_id: ID do arquivo de treinamento
            model: Modelo base para fine-tuning
            suffix: Sufixo para o nome do modelo
            n_epochs: Número de épocas (auto se None)
            hyperparameters: Hiperparâmetros adicionais

        Returns:
            Informações do job criado
        """
        params = {
            "training_file": training_file_id,
            "model": model
        }

        if suffix:
            params["suffix"] = suffix

        if n_epochs or hyperparameters:
            params["hyperparameters"] = hyperparameters or {}
            if n_epochs:
                params["hyperparameters"]["n_epochs"] = n_epochs

        job = self.client.fine_tuning.jobs.create(**params)

        return {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "created_at": datetime.fromtimestamp(job.created_at).isoformat()
        }

    def get_job_status(self, job_id: str) -> dict:
        """Obtém status de um job de fine-tuning."""
        job = self.client.fine_tuning.jobs.retrieve(job_id)

        result = {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "created_at": datetime.fromtimestamp(job.created_at).isoformat(),
            "finished_at": None,
            "fine_tuned_model": job.fine_tuned_model,
            "trained_tokens": job.trained_tokens,
            "error": None
        }

        if job.finished_at:
            result["finished_at"] = datetime.fromtimestamp(job.finished_at).isoformat()

        if job.error:
            result["error"] = job.error

        return result

    def list_jobs(self, limit: int = 10) -> list[dict]:
        """Lista jobs de fine-tuning."""
        jobs = self.client.fine_tuning.jobs.list(limit=limit)

        return [
            {
                "job_id": job.id,
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model
            }
            for job in jobs.data
        ]

    def cancel_job(self, job_id: str) -> dict:
        """Cancela um job de fine-tuning."""
        job = self.client.fine_tuning.jobs.cancel(job_id)
        return {"job_id": job.id, "status": job.status}

    def wait_for_completion(
        self,
        job_id: str,
        check_interval: int = 60,
        timeout: int = 3600
    ) -> dict:
        """
        Aguarda conclusão do fine-tuning.

        Args:
            job_id: ID do job
            check_interval: Intervalo entre verificações (segundos)
            timeout: Timeout máximo (segundos)

        Returns:
            Status final do job
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if status["status"] in ["succeeded", "failed", "cancelled"]:
                return status

            print(f"Status: {status['status']} - Aguardando...")
            time.sleep(check_interval)

        return {"error": "Timeout", "job_id": job_id}

    def use_fine_tuned_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = None
    ) -> str:
        """
        Usa um modelo fine-tuned.

        Args:
            model_id: ID do modelo fine-tuned
            prompt: Prompt do usuário
            system_prompt: System prompt (opcional)

        Returns:
            Resposta do modelo
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages
        )

        return response.choices[0].message.content


def demonstrate_dataset_creation():
    """Demonstra criação de dataset."""
    print("\n" + "=" * 60)
    print("CRIAÇÃO DE DATASET")
    print("=" * 60)

    # Exemplo: Fine-tuning para classificação de sentimento
    system_prompt = """Você é um classificador de sentimento.
Classifique o texto como: POSITIVO, NEGATIVO ou NEUTRO.
Responda apenas com a classificação."""

    generator = DatasetGenerator(system_prompt)

    # Adiciona exemplos
    training_pairs = [
        ("Adorei esse produto! Muito bom!", "POSITIVO"),
        ("Péssima experiência, nunca mais compro.", "NEGATIVO"),
        ("O produto chegou no prazo.", "NEUTRO"),
        ("Excelente qualidade, recomendo!", "POSITIVO"),
        ("Não gostei, qualidade ruim.", "NEGATIVO"),
        ("Preço na média do mercado.", "NEUTRO"),
        ("Superou todas as expectativas!", "POSITIVO"),
        ("Veio com defeito, muito decepcionado.", "NEGATIVO"),
        ("Entrega normal, sem problemas.", "NEUTRO"),
        ("Melhor compra que já fiz!", "POSITIVO"),
    ]

    generator.add_examples_from_pairs(training_pairs)

    print(f"\nDataset criado com {len(generator.examples)} exemplos")

    # Mostra formato
    print("\nFormato JSONL (primeiro exemplo):")
    print("-" * 40)
    print(json.dumps(generator.examples[0].to_jsonl(), indent=2, ensure_ascii=False))

    return generator


def demonstrate_dataset_validation():
    """Demonstra validação de dataset."""
    print("\n" + "=" * 60)
    print("VALIDAÇÃO DE DATASET")
    print("=" * 60)

    validator = DatasetValidator()

    # Dataset de exemplo
    examples = [
        {
            "messages": [
                {"role": "system", "content": "Você é um assistente."},
                {"role": "user", "content": "Olá"},
                {"role": "assistant", "content": "Olá! Como posso ajudar?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Teste"},
                {"role": "assistant", "content": "Resposta"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Sem assistant"}
            ]
        },
        {
            # Exemplo inválido - sem messages
            "data": "inválido"
        }
    ]

    results = validator.validate_dataset(examples)

    print("\nResultado da validação:")
    print("-" * 40)
    print(f"Total de exemplos: {results['total_examples']}")
    print(f"Válidos: {results['valid_examples']}")
    print(f"Inválidos: {results['invalid_examples']}")

    if results['errors']:
        print("\nErros encontrados:")
        for error in results['errors']:
            print(f"  Exemplo {error['example_index']}: {error['errors']}")

    if results['warnings']:
        print("\nAvisos:")
        for warning in results['warnings']:
            print(f"  - {warning}")

    print("\nEstatísticas de tokens:")
    print(f"  Total: {results['token_stats']['total']}")
    print(f"  Média: {results['token_stats']['avg']:.1f}")
    print(f"  Min: {results['token_stats']['min']}")
    print(f"  Max: {results['token_stats']['max']}")


def demonstrate_fine_tuning_workflow():
    """Demonstra workflow de fine-tuning (sem executar)."""
    print("\n" + "=" * 60)
    print("WORKFLOW DE FINE-TUNING")
    print("=" * 60)

    print("""
    O workflow completo de fine-tuning:

    1. PREPARAÇÃO DOS DADOS
    ─────────────────────────────────
    # Criar dataset
    generator = DatasetGenerator(system_prompt)
    generator.add_examples_from_pairs(training_pairs)
    generator.export_jsonl("training_data.jsonl")

    2. VALIDAÇÃO
    ─────────────────────────────────
    validator = DatasetValidator()
    results = validator.validate_dataset(dataset)
    # Corrigir erros se necessário

    3. UPLOAD DO ARQUIVO
    ─────────────────────────────────
    manager = FineTuningManager()
    file_id = manager.upload_training_file("training_data.jsonl")
    print(f"Arquivo uploaded: {file_id}")

    4. CRIAR JOB DE FINE-TUNING
    ─────────────────────────────────
    job = manager.create_fine_tuning_job(
        training_file_id=file_id,
        model="gpt-4o-mini-2024-07-18",
        suffix="my-custom-model"
    )
    print(f"Job criado: {job['job_id']}")

    5. AGUARDAR CONCLUSÃO
    ─────────────────────────────────
    status = manager.wait_for_completion(job['job_id'])
    # ou verificar manualmente:
    status = manager.get_job_status(job['job_id'])

    6. USAR MODELO FINE-TUNED
    ─────────────────────────────────
    response = manager.use_fine_tuned_model(
        model_id=status['fine_tuned_model'],
        prompt="Texto para classificar",
        system_prompt=system_prompt
    )
    """)


def demonstrate_best_practices():
    """Demonstra boas práticas de fine-tuning."""
    print("\n" + "=" * 60)
    print("BOAS PRÁTICAS DE FINE-TUNING")
    print("=" * 60)

    print("""
    1. QUALIDADE DOS DADOS
    ─────────────────────────────────
    - Use exemplos reais e representativos
    - Varie os exemplos (evite repetições)
    - Mínimo recomendado: 50-100 exemplos
    - Ideal: 500-1000+ exemplos

    2. FORMATO CONSISTENTE
    ─────────────────────────────────
    - Mantenha o mesmo system prompt
    - Padronize formato de entrada/saída
    - Use o mesmo estilo em todos os exemplos

    3. CUSTO-BENEFÍCIO
    ─────────────────────────────────
    Custos estimados (Jan 2025):
    ┌─────────────────┬─────────────────┐
    │ Modelo          │ Custo/1K tokens │
    ├─────────────────┼─────────────────┤
    │ gpt-4o-mini     │ $0.003          │
    │ gpt-4o          │ $0.025          │
    └─────────────────┴─────────────────┘

    Considere fine-tuning quando:
    - Volume > 10K chamadas/mês
    - Precisa de consistência alta
    - Tarefas muito específicas

    4. AVALIAÇÃO
    ─────────────────────────────────
    - Reserve 10-20% dos dados para teste
    - Compare com modelo base
    - Monitore métricas após deploy

    5. ITERAÇÃO
    ─────────────────────────────────
    - Comece com poucos exemplos
    - Adicione exemplos para casos fracos
    - Re-treine periodicamente
    """)


def demonstrate_use_cases():
    """Demonstra casos de uso para fine-tuning."""
    print("\n" + "=" * 60)
    print("CASOS DE USO PARA FINE-TUNING")
    print("=" * 60)

    use_cases = [
        {
            "name": "Classificação de Texto",
            "description": "Categorizar textos em classes predefinidas",
            "example": {
                "system": "Classifique emails como: URGENTE, NORMAL, SPAM",
                "user": "Promoção imperdível! Clique agora!",
                "assistant": "SPAM"
            }
        },
        {
            "name": "Extração de Informações",
            "description": "Extrair dados estruturados de texto livre",
            "example": {
                "system": "Extraia nome, email e telefone do texto em JSON.",
                "user": "João Silva, joao@email.com, tel: 11-99999-0000",
                "assistant": '{"nome": "João Silva", "email": "joao@email.com", "telefone": "11-99999-0000"}'
            }
        },
        {
            "name": "Estilo de Escrita",
            "description": "Manter tom e estilo consistente",
            "example": {
                "system": "Responda de forma técnica e concisa.",
                "user": "O que é API?",
                "assistant": "API (Application Programming Interface): interface que define contratos de comunicação entre sistemas de software."
            }
        },
        {
            "name": "Tradução Especializada",
            "description": "Tradução com terminologia específica",
            "example": {
                "system": "Traduza termos jurídicos PT-BR para EN, mantendo precisão legal.",
                "user": "petição inicial",
                "assistant": "complaint / initial pleading"
            }
        }
    ]

    for uc in use_cases:
        print(f"\n{uc['name']}")
        print("-" * 40)
        print(f"Descrição: {uc['description']}")
        print(f"\nExemplo de treinamento:")
        print(f"  System: {uc['example']['system'][:50]}...")
        print(f"  User: {uc['example']['user']}")
        print(f"  Assistant: {uc['example']['assistant']}")


def main():
    print("=" * 60)
    print("FINE-TUNING DE MODELOS DE LINGUAGEM")
    print("=" * 60)

    print("""
    Fine-tuning permite customizar modelos LLM para:

    ✓ Formato de saída específico
    ✓ Estilo e tom consistente
    ✓ Tarefas de domínio especializado
    ✓ Melhor performance em casos específicos
    ✓ Redução de custos em alto volume

    Modelos disponíveis para fine-tuning:
    - gpt-4o-mini-2024-07-18
    - gpt-4o-2024-08-06
    - gpt-3.5-turbo
    """)

    # Demonstrações
    demonstrate_dataset_creation()
    demonstrate_dataset_validation()
    demonstrate_fine_tuning_workflow()
    demonstrate_best_practices()
    demonstrate_use_cases()

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)

    print("""
    Fine-tuning é poderoso mas requer planejamento:

    1. Comece com prompt engineering
       - Muitas vezes resolve sem fine-tuning

    2. Colete dados de qualidade
       - Mínimo 50-100 exemplos
       - Representativos do uso real

    3. Valide e itere
       - Use conjunto de teste
       - Adicione exemplos para gaps

    4. Monitore em produção
       - Compare métricas
       - Re-treine quando necessário

    5. Considere alternativas
       - Few-shot learning
       - RAG (Retrieval Augmented Generation)
       - Modelos locais (Ollama/llama.cpp)
    """)

    print("\nFim da demonstração de Fine-tuning")
    print("=" * 60)


if __name__ == "__main__":
    main()
