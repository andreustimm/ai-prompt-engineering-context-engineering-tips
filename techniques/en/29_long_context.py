"""
Long Context Strategies

Techniques for processing documents that exceed typical context windows.
Different strategies trade off between quality, cost, and completeness.

Strategies implemented:
1. Map-Reduce: Process chunks separately, combine results
2. Refine: Iteratively build answer with each chunk
3. Map-Rerank: Score each chunk, use best ones
4. Stuffing with prioritization: Fit most relevant content

Use cases:
- Processing long documents (50+ pages)
- Summarizing large documents
- Answering questions about entire books
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

token_tracker = TokenUsage()


def load_long_document() -> str:
    """Load long sample document."""
    doc_path = Path(__file__).parent.parent.parent / "sample_data" / "documents" / "long_document.txt"
    if doc_path.exists():
        with open(doc_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Sample long document content. " * 500


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]


def map_reduce_summarize(chunks: list[str]) -> str:
    """
    Map-Reduce: Summarize each chunk, then combine summaries.

    Good for: Comprehensive coverage of entire document
    Trade-off: Multiple LLM calls, may lose nuance
    """
    llm = get_llm(temperature=0.3)

    # Map: Summarize each chunk
    summaries = []
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the key points from this section in 2-3 sentences."),
        ("user", "{chunk}")
    ])
    map_chain = map_prompt | llm

    print("   Mapping (summarizing chunks)...")
    for i, chunk in enumerate(chunks[:5]):  # Limit for demo
        response = map_chain.invoke({"chunk": chunk})
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        summaries.append(response.content)
        print(f"      Chunk {i+1}/{min(5, len(chunks))} done")

    # Reduce: Combine summaries
    print("   Reducing (combining summaries)...")
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", "Combine these summaries into a coherent overview. Be comprehensive but concise."),
        ("user", "Summaries to combine:\n\n{summaries}")
    ])
    reduce_chain = reduce_prompt | llm

    response = reduce_chain.invoke({"summaries": "\n\n".join(summaries)})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Reduce")

    return response.content


def refine_summarize(chunks: list[str]) -> str:
    """
    Refine: Build summary iteratively, refining with each chunk.

    Good for: Maintaining coherence, building on context
    Trade-off: Sequential processing, slower
    """
    llm = get_llm(temperature=0.3)

    # Initial summary from first chunk
    initial_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the key points from this text."),
        ("user", "{chunk}")
    ])

    print("   Initial summary from first chunk...")
    response = (initial_prompt | llm).invoke({"chunk": chunks[0]})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    current_summary = response.content

    # Refine with subsequent chunks
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", """You have an existing summary:
{existing}

Refine it by incorporating relevant information from new text. Keep it comprehensive but concise."""),
        ("user", "New text:\n{chunk}")
    ])
    refine_chain = refine_prompt | llm

    print("   Refining with additional chunks...")
    for i, chunk in enumerate(chunks[1:4]):  # Limit for demo
        response = refine_chain.invoke({"existing": current_summary, "chunk": chunk})
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        current_summary = response.content
        print(f"      Chunk {i+2} incorporated")

    print_token_usage(input_tokens, output_tokens, "Final Refine")
    return current_summary


def map_rerank_answer(chunks: list[str], question: str) -> str:
    """
    Map-Rerank: Score each chunk for relevance, answer using best ones.

    Good for: Question answering on long documents
    Trade-off: Extra scoring step, but finds most relevant content
    """
    llm = get_llm(temperature=0)

    # Score each chunk
    score_prompt = ChatPromptTemplate.from_messages([
        ("system", """Score how relevant this text is for answering the question.
Return ONLY a number from 0-10."""),
        ("user", "Question: {question}\n\nText: {chunk}\n\nRelevance score:")
    ])
    score_chain = score_prompt | llm

    print("   Scoring chunks for relevance...")
    scored_chunks = []
    for i, chunk in enumerate(chunks[:6]):
        response = score_chain.invoke({"question": question, "chunk": chunk[:1500]})
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        try:
            score = float(response.content.strip())
        except:
            score = 5.0
        scored_chunks.append((chunk, score))
        print(f"      Chunk {i+1}: score {score}")

    # Select top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, s in scored_chunks[:3]]

    # Generate answer from top chunks
    print("   Generating answer from top chunks...")
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the provided context."),
        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    context = "\n\n---\n\n".join(top_chunks)
    response = (answer_prompt | llm).invoke({"context": context, "question": question})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Answer Generation")

    return response.content


def stuffing_with_prioritization(chunks: list[str], question: str, max_context: int = 6000) -> str:
    """
    Stuffing: Fit as much relevant content as possible into context.

    Good for: Simple approach when content fits
    Trade-off: Limited by context window
    """
    llm = get_llm(temperature=0.3)

    # Simple relevance scoring (keyword overlap)
    question_terms = set(question.lower().split())

    scored = []
    for chunk in chunks:
        chunk_terms = set(chunk.lower().split())
        overlap = len(question_terms & chunk_terms)
        scored.append((chunk, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Stuff chunks until max context
    context_parts = []
    total_len = 0

    for chunk, _ in scored:
        if total_len + len(chunk) > max_context:
            break
        context_parts.append(chunk)
        total_len += len(chunk)

    print(f"   Stuffed {len(context_parts)} chunks into context")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context provided."),
        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    response = (prompt | llm).invoke({
        "context": "\n\n".join(context_parts),
        "question": question
    })
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Stuffing Answer")

    return response.content


def main():
    print("=" * 60)
    print("LONG CONTEXT STRATEGIES")
    print("=" * 60)

    token_tracker.reset()

    print("\n📚 LOADING LONG DOCUMENT")
    print("-" * 40)
    text = load_long_document()
    print(f"   Document length: {len(text):,} characters")

    chunks = chunk_text(text, chunk_size=2000, overlap=200)
    print(f"   Created {len(chunks)} chunks")

    # Strategy 1: Map-Reduce
    print("\n\n📊 MAP-REDUCE SUMMARIZATION")
    print("=" * 60)
    map_reduce_result = map_reduce_summarize(chunks)
    print(f"\n   Result:\n   {map_reduce_result[:400]}...")

    # Strategy 2: Refine
    print("\n\n🔄 REFINE SUMMARIZATION")
    print("=" * 60)
    refine_result = refine_summarize(chunks)
    print(f"\n   Result:\n   {refine_result[:400]}...")

    # Strategy 3: Map-Rerank
    print("\n\n🎯 MAP-RERANK Q&A")
    print("=" * 60)
    question = "What are the key principles of software architecture?"
    print(f"   Question: '{question}'")
    rerank_result = map_rerank_answer(chunks, question)
    print(f"\n   Answer:\n   {rerank_result[:400]}...")

    # Strategy 4: Stuffing
    print("\n\n📦 STUFFING WITH PRIORITIZATION")
    print("=" * 60)
    stuffing_result = stuffing_with_prioritization(chunks, question)
    print(f"\n   Answer:\n   {stuffing_result[:400]}...")

    print("\n\n💡 STRATEGY COMPARISON")
    print("-" * 40)
    print("""
   | Strategy    | Best For           | Pros              | Cons              |
   |-------------|-------------------|-------------------|-------------------|
   | Map-Reduce  | Full summarization| Complete coverage | Multiple calls    |
   | Refine      | Coherent summaries| Builds context    | Sequential, slow  |
   | Map-Rerank  | Q&A on long docs  | Finds best content| Extra scoring step|
   | Stuffing    | Simple tasks      | Minimal calls     | Limited by window |
    """)

    print_total_usage(token_tracker, "TOTAL - Long Context Strategies")
    print("\nEnd of Long Context Strategies demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
