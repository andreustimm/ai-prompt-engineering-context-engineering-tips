"""
Fine-tuning Language Models

Fine-tuning allows customizing models for specific tasks,
improving quality and reducing costs in the long run.

When to use fine-tuning:
1. Very specific output format
2. Particular style or tone
3. Specialized domain tasks
4. Cost reduction at high volume
5. Consistency improvement

Fine-tuning process:
1. Data preparation (JSONL)
2. File upload
3. Fine-tuning job creation
4. Training monitoring
5. Model evaluation and usage

Requirements:
- pip install openai tiktoken
- OpenAI account with fine-tuning access
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
    """Training example for fine-tuning."""
    system: str
    user: str
    assistant: str
    metadata: dict = field(default_factory=dict)

    def to_jsonl(self) -> dict:
        """Converts to OpenAI JSONL format."""
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant}
            ]
        }


class DatasetValidator:
    """
    Dataset validator for fine-tuning.

    Verifies format, size, and data quality.
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
        """Counts tokens in a text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text) // 4

    def validate_example(self, example: dict) -> tuple[bool, list[str]]:
        """Validates a training example."""
        errors = []

        # Check basic structure
        if "messages" not in example:
            errors.append("'messages' field missing")
            return False, errors

        messages = example["messages"]

        if not isinstance(messages, list):
            errors.append("'messages' must be a list")
            return False, errors

        if len(messages) < 2:
            errors.append("Minimum 2 messages (user + assistant)")

        # Check roles
        roles = [m.get("role") for m in messages]

        if "assistant" not in roles:
            errors.append("Must have at least one assistant message")

        # Count assistant messages for supervision
        assistant_count = roles.count("assistant")
        if assistant_count == 0:
            errors.append("No assistant message to train on")

        # Check content
        for i, msg in enumerate(messages):
            if "role" not in msg:
                errors.append(f"Message {i}: 'role' missing")
            if "content" not in msg:
                errors.append(f"Message {i}: 'content' missing")
            elif not msg["content"].strip():
                errors.append(f"Message {i}: empty content")

        # Check total size
        total_tokens = sum(
            self.count_tokens(m.get("content", ""))
            for m in messages
        )

        # Context limit (varies by model)
        max_tokens = 16385 if "gpt-4" in self.model else 4096

        if total_tokens > max_tokens:
            errors.append(f"Total tokens ({total_tokens}) exceeds limit ({max_tokens})")

        return len(errors) == 0, errors

    def validate_dataset(self, examples: list[dict]) -> dict:
        """Validates a complete dataset."""
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

                # Calculate tokens
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

        # Token statistics
        if token_counts:
            results["token_stats"]["total"] = sum(token_counts)
            results["token_stats"]["min"] = min(token_counts)
            results["token_stats"]["max"] = max(token_counts)
            results["token_stats"]["avg"] = sum(token_counts) / len(token_counts)

        # Warnings
        if len(examples) < 10:
            results["warnings"].append("Dataset too small (minimum recommended: 10 examples)")

        if len(examples) < 50:
            results["warnings"].append("Consider more examples for better results (50-100+)")

        return results


class DatasetGenerator:
    """
    Dataset generator for fine-tuning.

    Helps create training examples from templates.
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
        """Adds a training example."""
        example = TrainingExample(
            system=self.system_prompt,
            user=user_input,
            assistant=expected_output,
            metadata=metadata or {}
        )
        self.examples.append(example)

    def add_examples_from_pairs(self, pairs: list[tuple[str, str]]):
        """Adds multiple examples from (input, output) pairs."""
        for user_input, expected_output in pairs:
            self.add_example(user_input, expected_output)

    def export_jsonl(self, filepath: str):
        """Exports dataset to JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in self.examples:
                json.dump(example.to_jsonl(), f, ensure_ascii=False)
                f.write('\n')

    def get_dataset(self) -> list[dict]:
        """Returns dataset as list of dicts."""
        return [ex.to_jsonl() for ex in self.examples]


class FineTuningManager:
    """
    Fine-tuning manager.

    Facilitates upload, training, and use of fine-tuned models.
    """

    def __init__(self):
        self.client = get_openai_client()

    def upload_training_file(self, filepath: str) -> str:
        """
        Uploads training file.

        Returns the file_id for use in fine-tuning.
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
        Creates a fine-tuning job.

        Args:
            training_file_id: Training file ID
            model: Base model for fine-tuning
            suffix: Suffix for model name
            n_epochs: Number of epochs (auto if None)
            hyperparameters: Additional hyperparameters

        Returns:
            Created job information
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
        """Gets status of a fine-tuning job."""
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
        """Lists fine-tuning jobs."""
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
        """Cancels a fine-tuning job."""
        job = self.client.fine_tuning.jobs.cancel(job_id)
        return {"job_id": job.id, "status": job.status}

    def wait_for_completion(
        self,
        job_id: str,
        check_interval: int = 60,
        timeout: int = 3600
    ) -> dict:
        """
        Waits for fine-tuning completion.

        Args:
            job_id: Job ID
            check_interval: Interval between checks (seconds)
            timeout: Maximum timeout (seconds)

        Returns:
            Final job status
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if status["status"] in ["succeeded", "failed", "cancelled"]:
                return status

            print(f"Status: {status['status']} - Waiting...")
            time.sleep(check_interval)

        return {"error": "Timeout", "job_id": job_id}

    def use_fine_tuned_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = None
    ) -> str:
        """
        Uses a fine-tuned model.

        Args:
            model_id: Fine-tuned model ID
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Model response
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
    """Demonstrates dataset creation."""
    print("\n" + "=" * 60)
    print("DATASET CREATION")
    print("=" * 60)

    # Example: Fine-tuning for sentiment classification
    system_prompt = """You are a sentiment classifier.
Classify the text as: POSITIVE, NEGATIVE, or NEUTRAL.
Respond only with the classification."""

    generator = DatasetGenerator(system_prompt)

    # Add examples
    training_pairs = [
        ("Love this product! Very good!", "POSITIVE"),
        ("Terrible experience, never buying again.", "NEGATIVE"),
        ("The product arrived on time.", "NEUTRAL"),
        ("Excellent quality, highly recommend!", "POSITIVE"),
        ("Didn't like it, poor quality.", "NEGATIVE"),
        ("Price is market average.", "NEUTRAL"),
        ("Exceeded all expectations!", "POSITIVE"),
        ("Arrived defective, very disappointed.", "NEGATIVE"),
        ("Normal delivery, no issues.", "NEUTRAL"),
        ("Best purchase I've ever made!", "POSITIVE"),
    ]

    generator.add_examples_from_pairs(training_pairs)

    print(f"\nDataset created with {len(generator.examples)} examples")

    # Show format
    print("\nJSONL format (first example):")
    print("-" * 40)
    print(json.dumps(generator.examples[0].to_jsonl(), indent=2, ensure_ascii=False))

    return generator


def demonstrate_dataset_validation():
    """Demonstrates dataset validation."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    validator = DatasetValidator()

    # Example dataset
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "No assistant"}
            ]
        },
        {
            # Invalid example - no messages
            "data": "invalid"
        }
    ]

    results = validator.validate_dataset(examples)

    print("\nValidation result:")
    print("-" * 40)
    print(f"Total examples: {results['total_examples']}")
    print(f"Valid: {results['valid_examples']}")
    print(f"Invalid: {results['invalid_examples']}")

    if results['errors']:
        print("\nErrors found:")
        for error in results['errors']:
            print(f"  Example {error['example_index']}: {error['errors']}")

    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")

    print("\nToken statistics:")
    print(f"  Total: {results['token_stats']['total']}")
    print(f"  Average: {results['token_stats']['avg']:.1f}")
    print(f"  Min: {results['token_stats']['min']}")
    print(f"  Max: {results['token_stats']['max']}")


def demonstrate_fine_tuning_workflow():
    """Demonstrates fine-tuning workflow (without executing)."""
    print("\n" + "=" * 60)
    print("FINE-TUNING WORKFLOW")
    print("=" * 60)

    print("""
    The complete fine-tuning workflow:

    1. DATA PREPARATION
    ─────────────────────────────────
    # Create dataset
    generator = DatasetGenerator(system_prompt)
    generator.add_examples_from_pairs(training_pairs)
    generator.export_jsonl("training_data.jsonl")

    2. VALIDATION
    ─────────────────────────────────
    validator = DatasetValidator()
    results = validator.validate_dataset(dataset)
    # Fix errors if necessary

    3. FILE UPLOAD
    ─────────────────────────────────
    manager = FineTuningManager()
    file_id = manager.upload_training_file("training_data.jsonl")
    print(f"File uploaded: {file_id}")

    4. CREATE FINE-TUNING JOB
    ─────────────────────────────────
    job = manager.create_fine_tuning_job(
        training_file_id=file_id,
        model="gpt-4o-mini-2024-07-18",
        suffix="my-custom-model"
    )
    print(f"Job created: {job['job_id']}")

    5. WAIT FOR COMPLETION
    ─────────────────────────────────
    status = manager.wait_for_completion(job['job_id'])
    # or check manually:
    status = manager.get_job_status(job['job_id'])

    6. USE FINE-TUNED MODEL
    ─────────────────────────────────
    response = manager.use_fine_tuned_model(
        model_id=status['fine_tuned_model'],
        prompt="Text to classify",
        system_prompt=system_prompt
    )
    """)


def demonstrate_best_practices():
    """Demonstrates fine-tuning best practices."""
    print("\n" + "=" * 60)
    print("FINE-TUNING BEST PRACTICES")
    print("=" * 60)

    print("""
    1. DATA QUALITY
    ─────────────────────────────────
    - Use real, representative examples
    - Vary examples (avoid repetitions)
    - Minimum recommended: 50-100 examples
    - Ideal: 500-1000+ examples

    2. CONSISTENT FORMAT
    ─────────────────────────────────
    - Keep the same system prompt
    - Standardize input/output format
    - Use the same style in all examples

    3. COST-BENEFIT
    ─────────────────────────────────
    Estimated costs (Jan 2025):
    ┌─────────────────┬─────────────────┐
    │ Model           │ Cost/1K tokens  │
    ├─────────────────┼─────────────────┤
    │ gpt-4o-mini     │ $0.003          │
    │ gpt-4o          │ $0.025          │
    └─────────────────┴─────────────────┘

    Consider fine-tuning when:
    - Volume > 10K calls/month
    - High consistency needed
    - Very specific tasks

    4. EVALUATION
    ─────────────────────────────────
    - Reserve 10-20% of data for testing
    - Compare with base model
    - Monitor metrics after deployment

    5. ITERATION
    ─────────────────────────────────
    - Start with few examples
    - Add examples for weak cases
    - Retrain periodically
    """)


def demonstrate_use_cases():
    """Demonstrates use cases for fine-tuning."""
    print("\n" + "=" * 60)
    print("FINE-TUNING USE CASES")
    print("=" * 60)

    use_cases = [
        {
            "name": "Text Classification",
            "description": "Categorize texts into predefined classes",
            "example": {
                "system": "Classify emails as: URGENT, NORMAL, SPAM",
                "user": "Amazing deal! Click now!",
                "assistant": "SPAM"
            }
        },
        {
            "name": "Information Extraction",
            "description": "Extract structured data from free text",
            "example": {
                "system": "Extract name, email, and phone from text in JSON.",
                "user": "John Smith, john@email.com, tel: 555-123-4567",
                "assistant": '{"name": "John Smith", "email": "john@email.com", "phone": "555-123-4567"}'
            }
        },
        {
            "name": "Writing Style",
            "description": "Maintain consistent tone and style",
            "example": {
                "system": "Respond in a technical and concise manner.",
                "user": "What is an API?",
                "assistant": "API (Application Programming Interface): interface that defines communication contracts between software systems."
            }
        },
        {
            "name": "Specialized Translation",
            "description": "Translation with specific terminology",
            "example": {
                "system": "Translate legal terms EN to ES, maintaining legal precision.",
                "user": "plaintiff",
                "assistant": "demandante"
            }
        }
    ]

    for uc in use_cases:
        print(f"\n{uc['name']}")
        print("-" * 40)
        print(f"Description: {uc['description']}")
        print(f"\nTraining example:")
        print(f"  System: {uc['example']['system'][:50]}...")
        print(f"  User: {uc['example']['user']}")
        print(f"  Assistant: {uc['example']['assistant']}")


def main():
    print("=" * 60)
    print("FINE-TUNING LANGUAGE MODELS")
    print("=" * 60)

    print("""
    Fine-tuning allows customizing LLM models for:

    ✓ Specific output format
    ✓ Consistent style and tone
    ✓ Specialized domain tasks
    ✓ Better performance on specific cases
    ✓ Cost reduction at high volume

    Models available for fine-tuning:
    - gpt-4o-mini-2024-07-18
    - gpt-4o-2024-08-06
    - gpt-3.5-turbo
    """)

    # Demonstrations
    demonstrate_dataset_creation()
    demonstrate_dataset_validation()
    demonstrate_fine_tuning_workflow()
    demonstrate_best_practices()
    demonstrate_use_cases()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("""
    Fine-tuning is powerful but requires planning:

    1. Start with prompt engineering
       - Often solves without fine-tuning

    2. Collect quality data
       - Minimum 50-100 examples
       - Representative of real usage

    3. Validate and iterate
       - Use test set
       - Add examples for gaps

    4. Monitor in production
       - Compare metrics
       - Retrain when needed

    5. Consider alternatives
       - Few-shot learning
       - RAG (Retrieval Augmented Generation)
       - Local models (Ollama/llama.cpp)
    """)

    print("\nEnd of Fine-tuning demonstration")
    print("=" * 60)


if __name__ == "__main__":
    main()
