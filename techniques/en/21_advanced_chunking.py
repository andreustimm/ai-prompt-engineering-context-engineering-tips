"""
Advanced Chunking Strategies

Explores different text chunking strategies for RAG systems.
The quality of chunking significantly impacts retrieval effectiveness.

Strategies implemented:
1. RecursiveCharacterTextSplitter - Hierarchical character-based splitting
2. SentenceTransformers Token Splitter - Token-based splitting
3. Semantic Chunker - Similarity-based splitting
4. Markdown Splitter - Structure-aware splitting for markdown
5. Custom sliding window - Overlapping chunks with configurable size

Use cases:
- Optimizing RAG retrieval quality
- Handling different document types
- Balancing chunk size vs. context preservation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document

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


def load_sample_document() -> str:
    """Load sample document for chunking demonstrations."""
    sample_path = Path(__file__).parent.parent.parent / "sample_data" / "documents" / "long_document.txt"

    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Fallback sample text
        return """
        # Introduction to Machine Learning

        Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

        ## Types of Machine Learning

        ### Supervised Learning
        Supervised learning uses labeled data to train models. The algorithm learns from input-output pairs.
        Common algorithms include:
        - Linear Regression
        - Decision Trees
        - Support Vector Machines
        - Neural Networks

        ### Unsupervised Learning
        Unsupervised learning finds patterns in unlabeled data. It discovers hidden structures.
        Common algorithms include:
        - K-Means Clustering
        - Hierarchical Clustering
        - Principal Component Analysis

        ### Reinforcement Learning
        Reinforcement learning trains agents through rewards and penalties.
        Applications include:
        - Game playing (AlphaGo)
        - Robotics
        - Autonomous vehicles

        ## Deep Learning

        Deep learning uses neural networks with multiple layers to model complex patterns.
        Key architectures:
        - Convolutional Neural Networks (CNNs) for images
        - Recurrent Neural Networks (RNNs) for sequences
        - Transformers for language understanding

        ## Best Practices

        1. Start with clean, quality data
        2. Choose appropriate algorithms for your problem
        3. Use cross-validation for model evaluation
        4. Monitor for overfitting and underfitting
        5. Document your experiments and results
        """


def recursive_character_chunking(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    """
    Split text using recursive character splitting with hierarchy.

    This splitter tries to split on these separators in order:
    ["\n\n", "\n", " ", ""]

    It recursively splits on smaller separators when chunks are too large.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.create_documents([text])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["strategy"] = "recursive_character"
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


def token_based_chunking(text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> list[Document]:
    """
    Split text based on token count rather than character count.

    This ensures chunks fit within model context limits.
    Token-based splitting is more accurate for LLM processing.
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.create_documents([text])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["strategy"] = "token_based"
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


def markdown_aware_chunking(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    """
    Split markdown text respecting its structure.

    This splitter understands markdown headers and code blocks,
    preserving document structure in chunks.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.create_documents([text])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["strategy"] = "markdown_aware"
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


def semantic_chunking(text: str, breakpoint_threshold: float = 0.5) -> list[Document]:
    """
    Split text based on semantic similarity between sentences.

    Groups sentences that are semantically similar together,
    creating more coherent chunks for retrieval.

    Note: This is a simplified implementation. For production,
    use langchain_experimental.text_splitter.SemanticChunker
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker

        embeddings = get_embeddings()
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold * 100
        )

        chunks = splitter.create_documents([text])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["strategy"] = "semantic"
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    except ImportError:
        print("   Warning: langchain_experimental not available. Using fallback.")
        return recursive_character_chunking(text)


def sliding_window_chunking(text: str, window_size: int = 500, step_size: int = 250) -> list[Document]:
    """
    Create overlapping chunks using a sliding window approach.

    Args:
        text: Input text to chunk
        window_size: Size of each chunk in characters
        step_size: How much to advance window (window_size - overlap)

    This creates maximum overlap between chunks, useful when
    context continuity is critical.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + window_size
        chunk_text = text[start:end]

        # Avoid cutting words in the middle
        if end < len(text) and text[end] not in ' \n\t':
            # Find last space within the chunk
            last_space = chunk_text.rfind(' ')
            if last_space > 0:
                chunk_text = chunk_text[:last_space]

        chunk = Document(
            page_content=chunk_text.strip(),
            metadata={
                "chunk_index": len(chunks),
                "strategy": "sliding_window",
                "start_char": start,
                "chunk_size": len(chunk_text.strip())
            }
        )
        chunks.append(chunk)

        start += step_size

    return chunks


def sentence_based_chunking(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> list[Document]:
    """
    Split text into chunks containing a fixed number of sentences.

    Preserves sentence boundaries, ensuring chunks are grammatically complete.
    """
    import re

    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    i = 0

    while i < len(sentences):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = ' '.join(chunk_sentences)

        chunk = Document(
            page_content=chunk_text,
            metadata={
                "chunk_index": len(chunks),
                "strategy": "sentence_based",
                "sentence_count": len(chunk_sentences),
                "chunk_size": len(chunk_text)
            }
        )
        chunks.append(chunk)

        i += sentences_per_chunk - overlap_sentences

    return chunks


def compare_chunking_strategies(text: str):
    """Compare all chunking strategies on the same text."""

    print("\n   Comparing chunking strategies on sample text...")
    print(f"   Original text length: {len(text)} characters")

    strategies = {
        "Recursive Character": recursive_character_chunking(text, 500, 100),
        "Token-Based": token_based_chunking(text, 200, 50),
        "Markdown-Aware": markdown_aware_chunking(text, 500, 100),
        "Sliding Window": sliding_window_chunking(text, 500, 250),
        "Sentence-Based": sentence_based_chunking(text, 5, 1)
    }

    print("\n   " + "=" * 50)
    print("   CHUNKING STRATEGY COMPARISON")
    print("   " + "=" * 50)

    for name, chunks in strategies.items():
        chunk_sizes = [len(c.page_content) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

        print(f"\n   {name}:")
        print(f"      Number of chunks: {len(chunks)}")
        print(f"      Avg chunk size: {avg_size:.1f} characters")
        print(f"      Min/Max size: {min(chunk_sizes)}/{max(chunk_sizes)} characters")

    return strategies


def analyze_chunk_quality(chunks: list[Document]) -> dict:
    """
    Analyze the quality of chunks using the LLM.

    Evaluates coherence, completeness, and information density.
    """
    llm = get_llm(temperature=0)

    # Sample a few chunks for analysis
    sample_chunks = chunks[:3] if len(chunks) > 3 else chunks

    results = []

    for chunk in sample_chunks:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at evaluating text chunk quality for RAG systems.
Analyze the given chunk and rate it on these criteria (1-10):
1. Coherence: Does it form a complete, understandable unit?
2. Information Density: How much useful information does it contain?
3. Context Independence: Can it be understood without surrounding text?

Return a JSON object with scores and brief explanations."""),
            ("user", "Chunk to analyze:\n\n{chunk}")
        ])

        chain = prompt | llm
        response = chain.invoke({"chunk": chunk.page_content[:1000]})

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        results.append({
            "chunk_index": chunk.metadata.get("chunk_index"),
            "analysis": response.content
        })

    return results


def demo_chunking_for_retrieval():
    """Demonstrate how different chunking affects retrieval."""

    print("\n   Demonstrating chunking impact on retrieval...")

    text = load_sample_document()

    # Create chunks with different strategies
    small_chunks = recursive_character_chunking(text, 300, 50)
    medium_chunks = recursive_character_chunking(text, 800, 150)
    large_chunks = recursive_character_chunking(text, 1500, 300)

    print("\n   Chunk size comparison:")
    print(f"      Small (300 chars):  {len(small_chunks)} chunks")
    print(f"      Medium (800 chars): {len(medium_chunks)} chunks")
    print(f"      Large (1500 chars): {len(large_chunks)} chunks")

    # Sample chunks
    print("\n   Sample small chunk (first 200 chars):")
    print(f"      '{small_chunks[0].page_content[:200]}...'")

    print("\n   Sample large chunk (first 200 chars):")
    print(f"      '{large_chunks[0].page_content[:200]}...'")

    print("\n   Trade-offs:")
    print("      - Smaller chunks: Better precision, but may lose context")
    print("      - Larger chunks: More context, but may include irrelevant info")
    print("      - Overlap: Helps preserve context between chunks")


def main():
    print("=" * 60)
    print("ADVANCED CHUNKING STRATEGIES")
    print("=" * 60)

    token_tracker.reset()

    # Load sample document
    print("\n📚 LOADING SAMPLE DOCUMENT")
    print("-" * 40)

    text = load_sample_document()
    print(f"   Loaded document with {len(text)} characters")

    # Demo 1: Compare strategies
    print("\n\n📊 STRATEGY COMPARISON")
    print("-" * 40)

    strategies = compare_chunking_strategies(text)

    # Demo 2: Show chunk examples
    print("\n\n📝 CHUNK EXAMPLES")
    print("-" * 40)

    recursive_chunks = strategies["Recursive Character"]
    print(f"\n   First 3 chunks from Recursive Character strategy:\n")

    for chunk in recursive_chunks[:3]:
        preview = chunk.page_content[:150].replace('\n', ' ')
        print(f"   Chunk {chunk.metadata['chunk_index']}:")
        print(f"   Size: {chunk.metadata['chunk_size']} chars")
        print(f"   Preview: {preview}...")
        print()

    # Demo 3: Analyze chunk quality
    print("\n\n🔍 CHUNK QUALITY ANALYSIS")
    print("-" * 40)

    print("\n   Analyzing chunk quality using LLM...")
    quality_results = analyze_chunk_quality(recursive_chunks)

    for result in quality_results:
        print(f"\n   Chunk {result['chunk_index']} Analysis:")
        print(f"   {result['analysis'][:300]}...")

    # Demo 4: Chunking impact on retrieval
    print("\n\n📈 CHUNKING IMPACT ON RETRIEVAL")
    print("-" * 40)

    demo_chunking_for_retrieval()

    # Best practices
    print("\n\n💡 CHUNKING BEST PRACTICES")
    print("-" * 40)
    print("""
   1. Choose chunk size based on content type:
      - Technical docs: 500-1000 chars (preserve context)
      - FAQs: 200-400 chars (one Q&A per chunk)
      - Articles: 800-1500 chars (paragraph-level)

   2. Use appropriate overlap:
      - 10-20% of chunk size is typical
      - More overlap for highly connected content
      - Less for distinct, independent sections

   3. Match strategy to document structure:
      - Markdown/HTML: Use structure-aware splitters
      - Plain text: Recursive or semantic splitting
      - Code: Use language-specific splitters

   4. Consider retrieval needs:
      - Semantic search: Larger, coherent chunks
      - Keyword search: Smaller, focused chunks
      - Hybrid: Medium chunks with good overlap
    """)

    print_total_usage(token_tracker, "TOTAL - Advanced Chunking")

    print("\nEnd of Advanced Chunking demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
