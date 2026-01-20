"""
Contextual Compression

Extracts only the relevant portions from retrieved documents,
reducing noise and focusing on the most pertinent information
for answering the user's query.

Components:
- Base Retriever: Initial document retrieval
- Document Compressor: Extracts relevant passages
- Compression Retriever: Combines retrieval with compression

Use cases:
- Reducing token usage by removing irrelevant content
- Improving answer quality by focusing on relevant passages
- Handling long documents efficiently
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Conditional imports
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not installed. Run: pip install chromadb")

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Global token tracker
token_tracker = TokenUsage()


def load_sample_documents() -> list[Document]:
    """Load sample documents for demonstration."""
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"
    documents = []

    for file_path in sample_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path.name}
                ))
        except Exception as e:
            print(f"   Warning: Could not load {file_path}: {e}")

    return documents


class LLMExtractorCompressor:
    """Compresses documents by extracting only relevant passages using an LLM."""

    def __init__(self, max_tokens: int = 500):
        """
        Initialize the LLM-based compressor.

        Args:
            max_tokens: Maximum tokens for extracted content
        """
        self.llm = get_llm(temperature=0)
        self.max_tokens = max_tokens

    def compress(self, documents: list[Document], query: str) -> list[Document]:
        """
        Extract relevant passages from documents.

        Args:
            documents: Documents to compress
            query: User query for context

        Returns:
            Compressed documents with only relevant content
        """
        compressed = []

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting relevant information.
Given a document and a query, extract ONLY the sentences or passages that are directly relevant to answering the query.

Rules:
- Include only relevant information
- Keep the original wording where possible
- If nothing is relevant, respond with "NO_RELEVANT_CONTENT"
- Do not add any commentary or explanation
- Keep the extraction under {max_tokens} tokens"""),
            ("user", """Query: {query}

Document:
{document}

Relevant extraction:""")
        ])

        chain = prompt | self.llm

        for doc in documents:
            response = chain.invoke({
                "query": query,
                "document": doc.page_content,
                "max_tokens": self.max_tokens
            })

            input_tokens, output_tokens = extract_tokens_from_response(response)
            token_tracker.add(input_tokens, output_tokens)

            extracted = response.content.strip()

            if extracted and extracted != "NO_RELEVANT_CONTENT":
                compressed.append(Document(
                    page_content=extracted,
                    metadata={
                        **doc.metadata,
                        "compression_type": "llm_extract",
                        "original_length": len(doc.page_content),
                        "compressed_length": len(extracted)
                    }
                ))

        return compressed


class EmbeddingsFilterCompressor:
    """Filters document chunks based on embedding similarity to the query."""

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize the embeddings-based filter.

        Args:
            similarity_threshold: Minimum similarity score to keep a chunk
        """
        self.embeddings = get_embeddings()
        self.threshold = similarity_threshold

    def compress(self, documents: list[Document], query: str) -> list[Document]:
        """
        Filter documents based on embedding similarity.

        Args:
            documents: Documents to filter
            query: User query

        Returns:
            Documents that meet the similarity threshold
        """
        if not documents:
            return []

        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Get document embeddings
        doc_embeddings = self.embeddings.embed_documents(
            [doc.page_content for doc in documents]
        )

        # Calculate similarities and filter
        compressed = []
        for doc, doc_emb in zip(documents, doc_embeddings):
            # Cosine similarity
            similarity = sum(a * b for a, b in zip(query_embedding, doc_emb))
            norm_q = sum(a * a for a in query_embedding) ** 0.5
            norm_d = sum(a * a for a in doc_emb) ** 0.5
            similarity = similarity / (norm_q * norm_d) if norm_q * norm_d > 0 else 0

            if similarity >= self.threshold:
                compressed.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "compression_type": "embeddings_filter",
                        "similarity_score": similarity
                    }
                ))

        return compressed


class SentenceExtractorCompressor:
    """Extracts relevant sentences based on keyword matching and scoring."""

    def __init__(self, top_n_sentences: int = 5):
        """
        Initialize sentence extractor.

        Args:
            top_n_sentences: Number of top sentences to extract per document
        """
        self.top_n = top_n_sentences

    def compress(self, documents: list[Document], query: str) -> list[Document]:
        """
        Extract top relevant sentences from documents.

        Args:
            documents: Documents to compress
            query: User query

        Returns:
            Documents with only top relevant sentences
        """
        import re

        query_terms = set(query.lower().split())
        compressed = []

        for doc in documents:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)

            # Score sentences based on query term overlap
            scored = []
            for sentence in sentences:
                sentence_terms = set(sentence.lower().split())
                overlap = len(query_terms & sentence_terms)
                scored.append((sentence, overlap))

            # Sort by score and take top N
            scored.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s for s, _ in scored[:self.top_n] if _]

            if top_sentences:
                compressed.append(Document(
                    page_content=" ".join(top_sentences),
                    metadata={
                        **doc.metadata,
                        "compression_type": "sentence_extract",
                        "original_sentences": len(sentences),
                        "extracted_sentences": len(top_sentences)
                    }
                ))

        return compressed


class ContextualCompressionRetriever:
    """Combines retrieval with compression for more focused results."""

    def __init__(
        self,
        vectorstore,
        compressor,
        base_k: int = 10,
        final_k: int = 3
    ):
        """
        Initialize the compression retriever.

        Args:
            vectorstore: Vector store for retrieval
            compressor: Document compressor
            base_k: Number of documents to retrieve before compression
            final_k: Number of documents to return after compression
        """
        self.vectorstore = vectorstore
        self.compressor = compressor
        self.base_k = base_k
        self.final_k = final_k

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve and compress documents.

        Args:
            query: User query

        Returns:
            Compressed and filtered documents
        """
        # Initial retrieval
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
        initial_docs = retriever.invoke(query)

        # Compress
        compressed_docs = self.compressor.compress(initial_docs, query)

        # Return top final_k
        return compressed_docs[:self.final_k]


def create_vectorstore(documents: list[Document], collection_name: str = "compression_demo"):
    """Create vector store from documents."""
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb is required")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks to demonstrate compression
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )

    return vectorstore


def compare_compression_methods(vectorstore, query: str):
    """Compare different compression methods."""

    print(f"\n   Query: '{query}'")
    print("   " + "=" * 50)

    # Retrieve base documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    base_docs = retriever.invoke(query)

    print(f"\n   📚 Base retrieval: {len(base_docs)} documents")
    total_chars = sum(len(d.page_content) for d in base_docs)
    print(f"   Total characters: {total_chars:,}")

    # Method 1: No compression
    print("\n   --- NO COMPRESSION ---")
    print(f"   Documents: {len(base_docs)}")
    print(f"   Characters: {total_chars:,}")
    for i, doc in enumerate(base_docs[:2], 1):
        print(f"\n   Doc {i} preview: {doc.page_content[:150]}...")

    # Method 2: LLM Extraction
    print("\n   --- LLM EXTRACTION ---")
    llm_compressor = LLMExtractorCompressor(max_tokens=200)
    llm_compressed = llm_compressor.compress(base_docs, query)
    llm_chars = sum(len(d.page_content) for d in llm_compressed)
    print(f"   Documents: {len(llm_compressed)}")
    print(f"   Characters: {llm_chars:,} ({100*llm_chars/total_chars:.1f}% of original)")
    for i, doc in enumerate(llm_compressed[:2], 1):
        print(f"\n   Doc {i}: {doc.page_content[:200]}...")

    # Method 3: Sentence Extraction
    print("\n   --- SENTENCE EXTRACTION ---")
    sentence_compressor = SentenceExtractorCompressor(top_n_sentences=3)
    sentence_compressed = sentence_compressor.compress(base_docs, query)
    sentence_chars = sum(len(d.page_content) for d in sentence_compressed)
    print(f"   Documents: {len(sentence_compressed)}")
    print(f"   Characters: {sentence_chars:,} ({100*sentence_chars/total_chars:.1f}% of original)")
    for i, doc in enumerate(sentence_compressed[:2], 1):
        print(f"\n   Doc {i}: {doc.page_content[:200]}...")


def generate_answer(query: str, documents: list[Document]) -> str:
    """Generate answer using compressed context."""
    llm = get_llm(temperature=0.3)

    context = "\n\n---\n\n".join([doc.page_content for doc in documents])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question based on the provided context.
Be concise and focused. The context has been pre-filtered to contain only relevant information."""),
        ("user", """Context:
{context}

Question: {question}

Answer:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Answer Generation")

    return response.content


def demonstrate_token_savings(vectorstore, query: str):
    """Demonstrate token savings from compression."""

    print(f"\n   Query: '{query}'")
    print("   " + "=" * 50)

    # Without compression
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    uncompressed = retriever.invoke(query)
    uncompressed_chars = sum(len(d.page_content) for d in uncompressed)

    # With LLM compression
    compressor = LLMExtractorCompressor(max_tokens=150)
    compressed = compressor.compress(uncompressed, query)
    compressed_chars = sum(len(d.page_content) for d in compressed)

    print(f"\n   Without compression:")
    print(f"      Documents: {len(uncompressed)}")
    print(f"      Total characters: {uncompressed_chars:,}")
    print(f"      Estimated tokens: ~{uncompressed_chars // 4:,}")

    print(f"\n   With compression:")
    print(f"      Documents: {len(compressed)}")
    print(f"      Total characters: {compressed_chars:,}")
    print(f"      Estimated tokens: ~{compressed_chars // 4:,}")

    savings = 100 * (1 - compressed_chars / uncompressed_chars) if uncompressed_chars > 0 else 0
    print(f"\n   💰 Token savings: {savings:.1f}%")

    # Generate answers with both
    print("\n   Generating answers...")

    print("\n   Answer (without compression):")
    answer_uncompressed = generate_answer(query, uncompressed)
    print(f"   {answer_uncompressed[:300]}...")

    print("\n   Answer (with compression):")
    answer_compressed = generate_answer(query, compressed)
    print(f"   {answer_compressed[:300]}...")


def main():
    print("=" * 60)
    print("CONTEXTUAL COMPRESSION")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb is required for this demo.")
        print("Install with: pip install chromadb")
        return

    token_tracker.reset()

    # Load documents
    print("\n📚 LOADING DOCUMENTS")
    print("-" * 40)

    documents = load_sample_documents()
    if not documents:
        print("   No documents found. Using sample documents.")
        documents = [
            Document(
                page_content="""Machine learning is a subset of artificial intelligence that enables systems to learn
                and improve from experience without being explicitly programmed. It focuses on developing computer
                programs that can access data and use it to learn for themselves. The process begins with observations
                or data, such as examples, direct experience, or instruction. Machine learning algorithms use
                computational methods to learn information directly from data without relying on a predetermined
                equation as a model.""",
                metadata={"source": "ml_intro.txt"}
            ),
        ]

    print(f"   Loaded {len(documents)} documents")

    print("\n   Creating vector store...")
    vectorstore = create_vectorstore(documents, "compression_demo")
    print("   Vector store ready!")

    # Demo 1: Compare compression methods
    print("\n\n📊 COMPARING COMPRESSION METHODS")
    print("=" * 60)

    compare_compression_methods(
        vectorstore,
        "What is machine learning and how does it work?"
    )

    # Demo 2: Token savings
    print("\n\n💰 TOKEN SAVINGS DEMONSTRATION")
    print("=" * 60)

    demonstrate_token_savings(
        vectorstore,
        "How do neural networks learn from data?"
    )

    # Demo 3: Full pipeline
    print("\n\n🎯 FULL COMPRESSION PIPELINE")
    print("=" * 60)

    query = "What are the applications of deep learning?"

    print(f"\n   Query: '{query}'")

    # Create compression retriever
    compressor = LLMExtractorCompressor(max_tokens=200)
    compression_retriever = ContextualCompressionRetriever(
        vectorstore=vectorstore,
        compressor=compressor,
        base_k=8,
        final_k=3
    )

    print("\n   Retrieving with compression...")
    compressed_docs = compression_retriever.retrieve(query)

    print(f"\n   Retrieved {len(compressed_docs)} compressed documents:")
    for i, doc in enumerate(compressed_docs, 1):
        print(f"\n   Document {i}:")
        print(f"   Original length: {doc.metadata.get('original_length', 'N/A')} chars")
        print(f"   Compressed length: {doc.metadata.get('compressed_length', len(doc.page_content))} chars")
        print(f"   Content: {doc.page_content[:200]}...")

    print("\n   Generating answer...")
    answer = generate_answer(query, compressed_docs)
    print(f"\n   Answer: {answer}")

    # Best practices
    print("\n\n💡 COMPRESSION BEST PRACTICES")
    print("-" * 40)
    print("""
   | Method              | Pros                      | Cons                    |
   |---------------------|---------------------------|-------------------------|
   | LLM Extraction      | High quality, contextual  | Higher latency & cost   |
   | Embeddings Filter   | Fast, no LLM calls        | May miss nuanced match  |
   | Sentence Extraction | Fast, preserves original  | Keyword-based only      |

   Tips:
   - Use LLM extraction for quality-critical applications
   - Use embeddings filter for high-volume, low-latency needs
   - Combine methods: filter first, then LLM extract
   - Adjust thresholds based on your precision/recall needs
   - Monitor token savings vs. answer quality tradeoff
    """)

    print_total_usage(token_tracker, "TOTAL - Contextual Compression")

    print("\nEnd of Contextual Compression demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
