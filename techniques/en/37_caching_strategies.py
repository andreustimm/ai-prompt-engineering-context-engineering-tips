"""
Caching Strategies for LLMs

Caching is essential to reduce costs, improve latency,
and optimize resource usage in LLM applications.

Cache types:
1. Response Cache - Caches complete responses
2. Embedding Cache - Caches embedding vectors
3. Semantic Cache - Similarity-based cache
4. Context Cache - Caches conversation context

Benefits:
- Cost reduction (fewer API calls)
- Lower latency (instant responses)
- Consistency (same responses for same questions)
- Resilience (works even when API is unavailable)

Requirements:
- pip install openai redis diskcache
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import hashlib
import time
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from abc import ABC, abstractmethod

from utils.openai_client import get_openai_client


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class CacheBackend(ABC):
    """Abstract interface for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int = None) -> None:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def stats(self) -> dict:
        pass


class InMemoryCache(CacheBackend):
    """
    Simple in-memory cache.

    Good for development and small-scale applications.
    Data is lost when the process terminates.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            entry.hit_count += 1
            self._hits += 1
            return entry
        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = None) -> None:
        # Evict oldest entries if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]

        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at
        )

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0
        }


class ResponseCache:
    """
    Complete LLM response cache.

    Stores responses based on prompt hash.
    Ideal for frequent and repetitive questions.
    """

    def __init__(
        self,
        backend: CacheBackend = None,
        ttl_seconds: int = 3600,  # 1 hour
        include_model: bool = True,
        include_temperature: bool = True
    ):
        self.backend = backend or InMemoryCache()
        self.ttl_seconds = ttl_seconds
        self.include_model = include_model
        self.include_temperature = include_temperature
        self.client = get_openai_client()

    def _generate_key(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = None
    ) -> str:
        """Generates unique key for the request."""
        key_parts = [json.dumps(messages, sort_keys=True)]

        if self.include_model and model:
            key_parts.append(f"model:{model}")
        if self.include_temperature and temperature is not None:
            key_parts.append(f"temp:{temperature}")

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        bypass_cache: bool = False,
        **kwargs
    ) -> dict:
        """Executes chat with cache."""
        cache_key = self._generate_key(messages, model, temperature)

        # Try to get from cache
        if not bypass_cache:
            cached = self.backend.get(cache_key)
            if cached:
                return {
                    "content": cached.value,
                    "cached": True,
                    "cache_key": cache_key,
                    "hit_count": cached.hit_count
                }

        # Call API
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        latency_ms = (time.time() - start_time) * 1000

        content = response.choices[0].message.content

        # Store in cache
        self.backend.set(
            cache_key,
            content,
            self.ttl_seconds
        )

        return {
            "content": content,
            "cached": False,
            "cache_key": cache_key,
            "latency_ms": latency_ms,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        }


class EmbeddingCache:
    """
    Embedding cache.

    Embeddings are computationally expensive and deterministic,
    making them ideal candidates for caching.
    """

    def __init__(
        self,
        backend: CacheBackend = None,
        model: str = "text-embedding-3-small"
    ):
        self.backend = backend or InMemoryCache(max_size=10000)
        self.model = model
        self.client = get_openai_client()

    def _generate_key(self, text: str) -> str:
        """Generates key for the text."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def get_embedding(self, text: str, bypass_cache: bool = False) -> dict:
        """Gets embedding with cache."""
        cache_key = self._generate_key(text)

        # Try to get from cache
        if not bypass_cache:
            cached = self.backend.get(cache_key)
            if cached:
                return {
                    "embedding": cached.value,
                    "cached": True,
                    "cache_key": cache_key
                }

        # Call API
        start_time = time.time()
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        latency_ms = (time.time() - start_time) * 1000

        embedding = response.data[0].embedding

        # Store in cache
        self.backend.set(cache_key, embedding)

        return {
            "embedding": embedding,
            "cached": False,
            "cache_key": cache_key,
            "latency_ms": latency_ms,
            "dimensions": len(embedding)
        }

    def get_embeddings_batch(
        self,
        texts: list[str],
        bypass_cache: bool = False
    ) -> list[dict]:
        """Gets embeddings in batch with cache."""
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._generate_key(text)

            if not bypass_cache:
                cached = self.backend.get(cache_key)
                if cached:
                    results.append({
                        "text": text,
                        "embedding": cached.value,
                        "cached": True
                    })
                    continue

            uncached_texts.append(text)
            uncached_indices.append(i)
            results.append(None)  # Placeholder

        # Fetch uncached embeddings
        if uncached_texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=uncached_texts
            )

            for j, data in enumerate(response.data):
                original_index = uncached_indices[j]
                text = uncached_texts[j]
                embedding = data.embedding

                # Store in cache
                cache_key = self._generate_key(text)
                self.backend.set(cache_key, embedding)

                results[original_index] = {
                    "text": text,
                    "embedding": embedding,
                    "cached": False
                }

        return results


class SemanticCache:
    """
    Semantic cache.

    Uses embedding similarity to find similar
    responses in cache, even if the question
    is different.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: list[dict] = []
        self.embedding_cache = EmbeddingCache()
        self.client = get_openai_client()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculates cosine similarity."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def _find_similar(self, query_embedding: list[float]) -> Optional[dict]:
        """Finds similar entry in cache."""
        best_match = None
        best_similarity = 0.0

        for entry in self.entries:
            similarity = self._cosine_similarity(
                query_embedding,
                entry["embedding"]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match and best_similarity >= self.similarity_threshold:
            return {
                **best_match,
                "similarity": best_similarity
            }
        return None

    def chat(self, query: str, system_prompt: str = None) -> dict:
        """Executes chat with semantic cache."""

        # Get query embedding
        query_result = self.embedding_cache.get_embedding(query)
        query_embedding = query_result["embedding"]

        # Search for similar in cache
        similar = self._find_similar(query_embedding)
        if similar:
            return {
                "content": similar["response"],
                "cached": True,
                "similarity": similar["similarity"],
                "original_query": similar["query"]
            }

        # Call API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        content = response.choices[0].message.content

        # Store in cache
        if len(self.entries) >= self.max_entries:
            self.entries.pop(0)  # Remove oldest

        self.entries.append({
            "query": query,
            "embedding": query_embedding,
            "response": content,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "content": content,
            "cached": False,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens
            }
        }


class ConversationCache:
    """
    Conversation context cache.

    Maintains conversation history and enables
    automatic summarization to save tokens.
    """

    def __init__(
        self,
        max_messages: int = 20,
        summarize_after: int = 10
    ):
        self.max_messages = max_messages
        self.summarize_after = summarize_after
        self.conversations: dict[str, dict] = {}
        self.client = get_openai_client()

    def get_conversation(self, session_id: str) -> dict:
        """Gets or creates a conversation."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "messages": [],
                "summary": None,
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }
        return self.conversations[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """Adds message to conversation."""
        conv = self.get_conversation(session_id)

        conv["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        conv["message_count"] += 1

        # Check if summarization is needed
        if len(conv["messages"]) > self.max_messages:
            self._summarize_conversation(session_id)

    def _summarize_conversation(self, session_id: str):
        """Summarizes conversation to save tokens."""
        conv = self.get_conversation(session_id)

        # Get old messages to summarize
        to_summarize = conv["messages"][:-self.summarize_after]

        if not to_summarize:
            return

        # Create summary
        summary_prompt = f"""Summarize the following conversation concisely, keeping the main points:

{chr(10).join(f"{m['role']}: {m['content']}" for m in to_summarize)}

Summary:"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=300,
            temperature=0.3
        )

        new_summary = response.choices[0].message.content

        # Update conversation
        if conv["summary"]:
            conv["summary"] = f"{conv['summary']}\n\n{new_summary}"
        else:
            conv["summary"] = new_summary

        # Keep only recent messages
        conv["messages"] = conv["messages"][-self.summarize_after:]

    def get_context_messages(self, session_id: str) -> list[dict]:
        """Gets formatted messages for the LLM."""
        conv = self.get_conversation(session_id)
        messages = []

        # Add summary if exists
        if conv["summary"]:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{conv['summary']}"
            })

        # Add recent messages
        for msg in conv["messages"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return messages

    def stats(self, session_id: str) -> dict:
        """Returns conversation statistics."""
        conv = self.get_conversation(session_id)
        return {
            "message_count": conv["message_count"],
            "cached_messages": len(conv["messages"]),
            "has_summary": conv["summary"] is not None,
            "created_at": conv["created_at"]
        }


def cache_decorator(ttl_seconds: int = 3600):
    """
    Decorator to add cache to functions.

    Usage:
    @cache_decorator(ttl_seconds=3600)
    def my_llm_function(prompt):
        # ...
    """
    cache = InMemoryCache()

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
            key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:32]

            # Check cache
            cached = cache.get(key)
            if cached:
                return cached.value

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl_seconds)

            return result

        wrapper.cache = cache  # Expose cache for debugging
        return wrapper

    return decorator


def demonstrate_response_cache():
    """Demonstrates response cache."""

    print("\n" + "=" * 60)
    print("RESPONSE CACHE")
    print("=" * 60)

    cache = ResponseCache(ttl_seconds=300)

    query = "What is the capital of Japan?"
    messages = [{"role": "user", "content": query}]

    print(f"\nQuestion: {query}")
    print("-" * 40)

    # First call (no cache)
    print("\n1. First call:")
    result1 = cache.chat(messages)
    print(f"   Cached: {result1['cached']}")
    print(f"   Response: {result1['content'][:100]}...")
    if not result1['cached']:
        print(f"   Latency: {result1['latency_ms']:.0f}ms")
        print(f"   Tokens: {result1['tokens']['total']}")

    # Second call (with cache)
    print("\n2. Second call (same question):")
    result2 = cache.chat(messages)
    print(f"   Cached: {result2['cached']}")
    print(f"   Hit count: {result2.get('hit_count', 1)}")

    # Statistics
    print("\n3. Cache statistics:")
    stats = cache.backend.stats()
    print(f"   Size: {stats['size']}")
    print(f"   Hits: {stats['hits']}")
    print(f"   Hit rate: {stats['hit_rate']:.2%}")


def demonstrate_embedding_cache():
    """Demonstrates embedding cache."""

    print("\n" + "=" * 60)
    print("EMBEDDING CACHE")
    print("=" * 60)

    cache = EmbeddingCache()

    texts = [
        "Machine learning is an area of artificial intelligence.",
        "Python is a programming language.",
        "Machine learning is an area of artificial intelligence.",  # Repeated
    ]

    print("\nProcessing texts:")
    print("-" * 40)

    for i, text in enumerate(texts, 1):
        result = cache.get_embedding(text)
        print(f"\n{i}. \"{text[:50]}...\"")
        print(f"   Cached: {result['cached']}")
        if not result['cached']:
            print(f"   Latency: {result['latency_ms']:.0f}ms")
            print(f"   Dimensions: {result['dimensions']}")

    # Statistics
    print("\nCache statistics:")
    stats = cache.backend.stats()
    print(f"   Hit rate: {stats['hit_rate']:.2%}")


def demonstrate_semantic_cache():
    """Demonstrates semantic cache."""

    print("\n" + "=" * 60)
    print("SEMANTIC CACHE")
    print("=" * 60)

    cache = SemanticCache(similarity_threshold=0.92)

    queries = [
        "What is the capital of France?",
        "Tell me the French capital.",  # Similar to first
        "Who discovered America?",  # Different
    ]

    print("\nProcessing queries:")
    print("-" * 40)

    for i, query in enumerate(queries, 1):
        result = cache.chat(query)
        print(f"\n{i}. \"{query}\"")
        print(f"   Cached: {result['cached']}")

        if result['cached']:
            print(f"   Similarity: {result['similarity']:.2%}")
            print(f"   Original query: \"{result['original_query']}\"")

        print(f"   Response: {result['content'][:80]}...")


def main():
    print("=" * 60)
    print("CACHING STRATEGIES FOR LLMs")
    print("=" * 60)

    print("""
    Caching is fundamental for optimizing LLM applications.

    Cache types:

    ┌────────────────────────────────────────────────────────┐
    │ Type             │ Use Case                           │
    ├────────────────────────────────────────────────────────┤
    │ Response Cache   │ Complete responses (FAQ, common)   │
    │ Embedding Cache  │ Vectors (RAG, semantic search)     │
    │ Semantic Cache   │ Similar queries (tolerance)        │
    │ Context Cache    │ Conversation history               │
    └────────────────────────────────────────────────────────┘

    Benefits:
    - 💰 Cost reduction (fewer API calls)
    - ⚡ Lower latency (instant responses)
    - 🔄 Consistency (same responses)
    - 🛡️  Resilience (works offline)
    """)

    # Demonstrations
    demonstrate_response_cache()
    demonstrate_embedding_cache()
    demonstrate_semantic_cache()

    print("\n" + "=" * 60)
    print("CACHE BACKENDS FOR PRODUCTION")
    print("=" * 60)

    print("""
    For production, consider:

    1. Redis
       - Distributed cache
       - Automatic TTL
       - Optional persistence
       - pip install redis

    2. Memcached
       - High performance
       - Distributed cache
       - Simple and efficient

    3. DiskCache
       - Disk-based cache
       - Persistent
       - Good for large embeddings
       - pip install diskcache

    4. Cloud Providers
       - AWS ElastiCache
       - Google Memorystore
       - Azure Cache for Redis

    Redis example:
    ```python
    import redis
    r = redis.Redis(host='localhost', port=6379)
    r.setex('key', 3600, 'value')  # 1 hour TTL
    ```
    """)

    print("\nEnd of Caching Strategies demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
