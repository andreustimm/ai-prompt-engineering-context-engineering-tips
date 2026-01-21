"""
Estratégias de Cache para LLMs

O cache é essencial para reduzir custos, melhorar latência
e otimizar o uso de recursos em aplicações com LLMs.

Tipos de cache:
1. Response Cache - Cacheia respostas completas
2. Embedding Cache - Cacheia vetores de embedding
3. Semantic Cache - Cache baseado em similaridade semântica
4. Context Cache - Cacheia contexto de conversação

Benefícios:
- Redução de custos (menos chamadas à API)
- Menor latência (respostas instantâneas)
- Consistência (mesmas respostas para mesmas perguntas)
- Resiliência (funciona mesmo com API indisponível)

Requisitos:
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
    """Entrada de cache com metadados."""
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
    """Interface abstrata para backends de cache."""

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
    Cache em memória simples.

    Bom para desenvolvimento e aplicações de pequena escala.
    Dados são perdidos quando o processo é encerrado.
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
    Cache de respostas completas do LLM.

    Armazena respostas baseado em hash do prompt.
    Ideal para perguntas frequentes e repetitivas.
    """

    def __init__(
        self,
        backend: CacheBackend = None,
        ttl_seconds: int = 3600,  # 1 hora
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
        """Gera chave única para a requisição."""
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
        """Executa chat com cache."""
        cache_key = self._generate_key(messages, model, temperature)

        # Tentar obter do cache
        if not bypass_cache:
            cached = self.backend.get(cache_key)
            if cached:
                return {
                    "content": cached.value,
                    "cached": True,
                    "cache_key": cache_key,
                    "hit_count": cached.hit_count
                }

        # Chamar API
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        latency_ms = (time.time() - start_time) * 1000

        content = response.choices[0].message.content

        # Armazenar no cache
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
    Cache de embeddings.

    Embeddings são computacionalmente caros e determinísticos,
    então são candidatos ideais para cache.
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
        """Gera chave para o texto."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def get_embedding(self, text: str, bypass_cache: bool = False) -> dict:
        """Obtém embedding com cache."""
        cache_key = self._generate_key(text)

        # Tentar obter do cache
        if not bypass_cache:
            cached = self.backend.get(cache_key)
            if cached:
                return {
                    "embedding": cached.value,
                    "cached": True,
                    "cache_key": cache_key
                }

        # Chamar API
        start_time = time.time()
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        latency_ms = (time.time() - start_time) * 1000

        embedding = response.data[0].embedding

        # Armazenar no cache
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
        """Obtém embeddings em lote com cache."""
        results = []
        uncached_texts = []
        uncached_indices = []

        # Verificar cache para cada texto
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

        # Buscar embeddings não cacheados
        if uncached_texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=uncached_texts
            )

            for j, data in enumerate(response.data):
                original_index = uncached_indices[j]
                text = uncached_texts[j]
                embedding = data.embedding

                # Armazenar no cache
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
    Cache semântico.

    Usa similaridade de embeddings para encontrar
    respostas similares no cache, mesmo que a pergunta
    seja diferente.
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
        """Calcula similaridade de cosseno."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def _find_similar(self, query_embedding: list[float]) -> Optional[dict]:
        """Encontra entrada similar no cache."""
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
        """Executa chat com cache semântico."""

        # Obter embedding da query
        query_result = self.embedding_cache.get_embedding(query)
        query_embedding = query_result["embedding"]

        # Buscar similar no cache
        similar = self._find_similar(query_embedding)
        if similar:
            return {
                "content": similar["response"],
                "cached": True,
                "similarity": similar["similarity"],
                "original_query": similar["query"]
            }

        # Chamar API
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

        # Armazenar no cache
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
    Cache de contexto de conversação.

    Mantém histórico de conversas e permite
    resumo automático para economizar tokens.
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
        """Obtém ou cria uma conversação."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "messages": [],
                "summary": None,
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }
        return self.conversations[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """Adiciona mensagem à conversação."""
        conv = self.get_conversation(session_id)

        conv["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        conv["message_count"] += 1

        # Verificar se precisa resumir
        if len(conv["messages"]) > self.max_messages:
            self._summarize_conversation(session_id)

    def _summarize_conversation(self, session_id: str):
        """Resume a conversação para economizar tokens."""
        conv = self.get_conversation(session_id)

        # Pegar mensagens antigas para resumir
        to_summarize = conv["messages"][:-self.summarize_after]

        if not to_summarize:
            return

        # Criar resumo
        summary_prompt = f"""Resuma a seguinte conversa de forma concisa, mantendo os pontos principais:

{chr(10).join(f"{m['role']}: {m['content']}" for m in to_summarize)}

Resumo:"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=300,
            temperature=0.3
        )

        new_summary = response.choices[0].message.content

        # Atualizar conversação
        if conv["summary"]:
            conv["summary"] = f"{conv['summary']}\n\n{new_summary}"
        else:
            conv["summary"] = new_summary

        # Manter apenas mensagens recentes
        conv["messages"] = conv["messages"][-self.summarize_after:]

    def get_context_messages(self, session_id: str) -> list[dict]:
        """Obtém mensagens formatadas para o LLM."""
        conv = self.get_conversation(session_id)
        messages = []

        # Adicionar resumo se existir
        if conv["summary"]:
            messages.append({
                "role": "system",
                "content": f"Resumo da conversa anterior:\n{conv['summary']}"
            })

        # Adicionar mensagens recentes
        for msg in conv["messages"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return messages

    def stats(self, session_id: str) -> dict:
        """Retorna estatísticas da conversação."""
        conv = self.get_conversation(session_id)
        return {
            "message_count": conv["message_count"],
            "cached_messages": len(conv["messages"]),
            "has_summary": conv["summary"] is not None,
            "created_at": conv["created_at"]
        }


def cache_decorator(ttl_seconds: int = 3600):
    """
    Decorator para adicionar cache a funções.

    Uso:
    @cache_decorator(ttl_seconds=3600)
    def my_llm_function(prompt):
        # ...
    """
    cache = InMemoryCache()

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Gerar chave do cache
            key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
            key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:32]

            # Verificar cache
            cached = cache.get(key)
            if cached:
                return cached.value

            # Executar função
            result = func(*args, **kwargs)

            # Armazenar no cache
            cache.set(key, result, ttl_seconds)

            return result

        wrapper.cache = cache  # Expor cache para debug
        return wrapper

    return decorator


def demonstrate_response_cache():
    """Demonstra cache de respostas."""

    print("\n" + "=" * 60)
    print("CACHE DE RESPOSTAS")
    print("=" * 60)

    cache = ResponseCache(ttl_seconds=300)

    query = "Qual é a capital do Japão?"
    messages = [{"role": "user", "content": query}]

    print(f"\nPergunta: {query}")
    print("-" * 40)

    # Primeira chamada (sem cache)
    print("\n1. Primeira chamada:")
    result1 = cache.chat(messages)
    print(f"   Cached: {result1['cached']}")
    print(f"   Resposta: {result1['content'][:100]}...")
    if not result1['cached']:
        print(f"   Latência: {result1['latency_ms']:.0f}ms")
        print(f"   Tokens: {result1['tokens']['total']}")

    # Segunda chamada (com cache)
    print("\n2. Segunda chamada (mesma pergunta):")
    result2 = cache.chat(messages)
    print(f"   Cached: {result2['cached']}")
    print(f"   Hit count: {result2.get('hit_count', 1)}")

    # Estatísticas
    print("\n3. Estatísticas do cache:")
    stats = cache.backend.stats()
    print(f"   Tamanho: {stats['size']}")
    print(f"   Hits: {stats['hits']}")
    print(f"   Hit rate: {stats['hit_rate']:.2%}")


def demonstrate_embedding_cache():
    """Demonstra cache de embeddings."""

    print("\n" + "=" * 60)
    print("CACHE DE EMBEDDINGS")
    print("=" * 60)

    cache = EmbeddingCache()

    texts = [
        "Machine learning é uma área da inteligência artificial.",
        "Python é uma linguagem de programação.",
        "Machine learning é uma área da inteligência artificial.",  # Repetido
    ]

    print("\nProcessando textos:")
    print("-" * 40)

    for i, text in enumerate(texts, 1):
        result = cache.get_embedding(text)
        print(f"\n{i}. \"{text[:50]}...\"")
        print(f"   Cached: {result['cached']}")
        if not result['cached']:
            print(f"   Latência: {result['latency_ms']:.0f}ms")
            print(f"   Dimensões: {result['dimensions']}")

    # Estatísticas
    print("\nEstatísticas do cache:")
    stats = cache.backend.stats()
    print(f"   Hit rate: {stats['hit_rate']:.2%}")


def demonstrate_semantic_cache():
    """Demonstra cache semântico."""

    print("\n" + "=" * 60)
    print("CACHE SEMÂNTICO")
    print("=" * 60)

    cache = SemanticCache(similarity_threshold=0.92)

    queries = [
        "Qual é a capital do Brasil?",
        "Me diga qual a capital brasileira.",  # Similar à primeira
        "Quem descobriu o Brasil?",  # Diferente
    ]

    print("\nProcessando queries:")
    print("-" * 40)

    for i, query in enumerate(queries, 1):
        result = cache.chat(query)
        print(f"\n{i}. \"{query}\"")
        print(f"   Cached: {result['cached']}")

        if result['cached']:
            print(f"   Similaridade: {result['similarity']:.2%}")
            print(f"   Query original: \"{result['original_query']}\"")

        print(f"   Resposta: {result['content'][:80]}...")


def main():
    print("=" * 60)
    print("ESTRATÉGIAS DE CACHE PARA LLMs")
    print("=" * 60)

    print("""
    O cache é fundamental para otimizar aplicações com LLMs.

    Tipos de cache:

    ┌────────────────────────────────────────────────────────┐
    │ Tipo             │ Uso                                │
    ├────────────────────────────────────────────────────────┤
    │ Response Cache   │ Respostas completas (FAQ, comum)   │
    │ Embedding Cache  │ Vetores (RAG, busca semântica)     │
    │ Semantic Cache   │ Similar queries (tolerância)       │
    │ Context Cache    │ Histórico de conversas             │
    └────────────────────────────────────────────────────────┘

    Benefícios:
    - 💰 Redução de custos (menos API calls)
    - ⚡ Menor latência (respostas instantâneas)
    - 🔄 Consistência (mesmas respostas)
    - 🛡️  Resiliência (funciona offline)
    """)

    # Demonstrações
    demonstrate_response_cache()
    demonstrate_embedding_cache()
    demonstrate_semantic_cache()

    print("\n" + "=" * 60)
    print("BACKENDS DE CACHE PARA PRODUÇÃO")
    print("=" * 60)

    print("""
    Para produção, considere:

    1. Redis
       - Cache distribuído
       - TTL automático
       - Persistência opcional
       - pip install redis

    2. Memcached
       - Alta performance
       - Cache distribuído
       - Simples e eficiente

    3. DiskCache
       - Cache em disco
       - Persistente
       - Bom para embeddings grandes
       - pip install diskcache

    4. Cloud Providers
       - AWS ElastiCache
       - Google Memorystore
       - Azure Cache for Redis

    Exemplo Redis:
    ```python
    import redis
    r = redis.Redis(host='localhost', port=6379)
    r.setex('key', 3600, 'value')  # TTL de 1 hora
    ```
    """)

    print("\nFim do demo de Estratégias de Cache")
    print("=" * 60)


if __name__ == "__main__":
    main()
