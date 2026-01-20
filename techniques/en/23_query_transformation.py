"""
Query Transformation

Transforms user queries to improve retrieval effectiveness.
Different transformation strategies help bridge the gap between
how users phrase queries and how relevant documents are written.

Techniques implemented:
1. HyDE - Hypothetical Document Embeddings
2. Multi-Query - Generate multiple query variations
3. Step-Back - Abstract to broader concepts
4. Query Decomposition - Break complex queries into sub-queries

Use cases:
- Improving retrieval recall for vague queries
- Handling complex multi-part questions
- Bridging vocabulary mismatch
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


def create_vectorstore(documents: list[Document], collection_name: str = "query_transform"):
    """Create vector store from documents."""
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb is required")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )

    return vectorstore


def hyde_transform(query: str) -> str:
    """
    HyDE - Hypothetical Document Embeddings

    Instead of embedding the query directly, generate a hypothetical
    document that would answer the query, then use that for retrieval.

    This often produces better matches because the generated document
    is more similar in style and vocabulary to actual documents.
    """
    llm = get_llm(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical writer. Given a question, write a detailed paragraph
that would appear in a document answering that question. Write as if this paragraph exists in
a real document, not as a direct answer to the question.

Be specific, factual, and comprehensive. Do not start with phrases like "Here's" or "This paragraph"."""),
        ("user", "Question: {query}\n\nWrite a document paragraph that would contain the answer:")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "HyDE Generation")

    return response.content


def multi_query_transform(query: str, num_queries: int = 3) -> list[str]:
    """
    Multi-Query Transformation

    Generate multiple variations of the original query to improve
    recall. Different phrasings may match different relevant documents.
    """
    llm = get_llm(temperature=0.8)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at reformulating search queries.
Given a user question, generate {num_queries} different versions of the question
that could be used to search for relevant information.

Each version should:
- Capture the same intent
- Use different vocabulary/phrasing
- Potentially emphasize different aspects

Output ONLY the queries, one per line, numbered 1., 2., etc."""),
        ("user", "Original question: {query}\n\nGenerate {num_queries} query variations:")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query, "num_queries": num_queries})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Multi-Query Generation")

    # Parse the response
    lines = response.content.strip().split('\n')
    queries = []
    for line in lines:
        # Remove numbering and clean
        cleaned = line.strip()
        if cleaned and cleaned[0].isdigit():
            # Remove "1.", "2." etc.
            cleaned = cleaned.split('.', 1)[-1].strip()
        if cleaned:
            queries.append(cleaned)

    return queries[:num_queries]


def step_back_transform(query: str) -> str:
    """
    Step-Back Prompting

    Transform a specific question into a more general/abstract question.
    This can help retrieve foundational information that provides
    context for answering the specific question.
    """
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at abstracting questions to their underlying concepts.
Given a specific question, generate a more general "step-back" question that addresses
the broader concept or principle behind the original question.

The step-back question should:
- Be more general/abstract
- Cover the foundational knowledge needed
- Help retrieve background information

Output ONLY the step-back question, nothing else."""),
        ("user", """Original question: {query}

Step-back question:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Step-Back Generation")

    return response.content.strip()


def decompose_query(query: str) -> list[str]:
    """
    Query Decomposition

    Break down a complex question into simpler sub-questions.
    Each sub-question can be answered independently, and the
    answers combined to address the original question.
    """
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at breaking down complex questions.
Given a complex question, decompose it into 2-4 simpler sub-questions that,
when answered together, would provide a complete answer to the original question.

Each sub-question should:
- Be self-contained and answerable independently
- Address a specific aspect of the original question
- Build towards a comprehensive answer

Output ONLY the sub-questions, one per line, numbered 1., 2., etc."""),
        ("user", """Complex question: {query}

Sub-questions:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Query Decomposition")

    # Parse the response
    lines = response.content.strip().split('\n')
    queries = []
    for line in lines:
        cleaned = line.strip()
        if cleaned and cleaned[0].isdigit():
            cleaned = cleaned.split('.', 1)[-1].strip()
        if cleaned:
            queries.append(cleaned)

    return queries


def retrieve_with_transformation(
    vectorstore,
    query: str,
    transform_type: str = "hyde",
    k: int = 3
) -> list[Document]:
    """
    Retrieve documents using a transformed query.

    Args:
        vectorstore: Vector store to search
        query: Original user query
        transform_type: Type of transformation ("hyde", "multi_query", "step_back", "decompose")
        k: Number of documents to retrieve
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    if transform_type == "hyde":
        # Generate hypothetical document and search with it
        hypothetical_doc = hyde_transform(query)
        print(f"\n   HyDE Document (first 200 chars): {hypothetical_doc[:200]}...")
        return retriever.invoke(hypothetical_doc)

    elif transform_type == "multi_query":
        # Search with multiple query variations and combine results
        queries = multi_query_transform(query)
        print(f"\n   Generated queries:")
        for q in queries:
            print(f"      - {q}")

        all_docs = []
        seen = set()
        for q in queries:
            docs = retriever.invoke(q)
            for doc in docs:
                doc_key = doc.page_content[:100]
                if doc_key not in seen:
                    seen.add(doc_key)
                    all_docs.append(doc)

        return all_docs[:k * 2]  # Return more since we deduplicated

    elif transform_type == "step_back":
        # Search with both original and step-back question
        step_back = step_back_transform(query)
        print(f"\n   Step-back question: {step_back}")

        original_docs = retriever.invoke(query)
        step_back_docs = retriever.invoke(step_back)

        # Combine, prioritizing original
        all_docs = original_docs.copy()
        seen = {doc.page_content[:100] for doc in original_docs}
        for doc in step_back_docs:
            if doc.page_content[:100] not in seen:
                all_docs.append(doc)

        return all_docs[:k * 2]

    elif transform_type == "decompose":
        # Search with sub-questions and combine results
        sub_queries = decompose_query(query)
        print(f"\n   Sub-questions:")
        for q in sub_queries:
            print(f"      - {q}")

        all_docs = []
        seen = set()
        for q in sub_queries:
            docs = retriever.invoke(q)
            for doc in docs:
                doc_key = doc.page_content[:100]
                if doc_key not in seen:
                    seen.add(doc_key)
                    all_docs.append(doc)

        return all_docs[:k * 2]

    else:
        # No transformation
        return retriever.invoke(query)


def generate_answer(query: str, documents: list[Document]) -> str:
    """Generate answer using retrieved context."""
    llm = get_llm(temperature=0.3)

    context = "\n\n---\n\n".join([doc.page_content for doc in documents])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question based on the provided context.
Be thorough but concise. If the context doesn't contain enough information, say so."""),
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


def compare_transformations(vectorstore, query: str):
    """Compare different query transformation techniques."""

    print(f"\n   Original Query: '{query}'")
    print("   " + "=" * 50)

    techniques = ["none", "hyde", "multi_query", "step_back", "decompose"]

    for technique in techniques:
        print(f"\n   📌 Technique: {technique.upper()}")
        print("   " + "-" * 30)

        if technique == "none":
            docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
        else:
            docs = retrieve_with_transformation(vectorstore, query, technique, k=3)

        print(f"\n   Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs[:3], 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   {i}. {preview}...")


def main():
    print("=" * 60)
    print("QUERY TRANSFORMATION")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb is required for this demo.")
        print("Install with: pip install chromadb")
        return

    token_tracker.reset()

    # Load documents and create vector store
    print("\n📚 LOADING DOCUMENTS")
    print("-" * 40)

    documents = load_sample_documents()
    if not documents:
        print("   No documents found. Using sample documents.")
        documents = [
            Document(page_content="Machine learning is a subset of AI that enables systems to learn from data."),
            Document(page_content="Neural networks are computing systems inspired by biological neural networks."),
            Document(page_content="Deep learning uses multiple layers to progressively extract higher-level features."),
        ]

    print(f"   Loaded {len(documents)} documents")

    print("\n   Creating vector store...")
    vectorstore = create_vectorstore(documents, "query_transform_demo")
    print("   Vector store ready!")

    # Demo 1: HyDE
    print("\n\n🔮 HYDE - HYPOTHETICAL DOCUMENT EMBEDDINGS")
    print("=" * 60)

    query1 = "How do neural networks learn?"

    print(f"\n   Query: '{query1}'")
    print("\n   Generating hypothetical document...")
    hyde_doc = hyde_transform(query1)
    print(f"\n   Generated Document:\n   {hyde_doc[:300]}...")

    docs = retrieve_with_transformation(vectorstore, query1, "hyde", k=3)
    print("\n   Retrieved documents with HyDE:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"   {i}. {doc.page_content[:100]}...")

    # Demo 2: Multi-Query
    print("\n\n🔄 MULTI-QUERY TRANSFORMATION")
    print("=" * 60)

    query2 = "What are the benefits of using transformers in NLP?"

    print(f"\n   Query: '{query2}'")
    print("\n   Generating query variations...")
    variations = multi_query_transform(query2)
    print("\n   Variations:")
    for i, v in enumerate(variations, 1):
        print(f"   {i}. {v}")

    # Demo 3: Step-Back
    print("\n\n⬅️ STEP-BACK PROMPTING")
    print("=" * 60)

    query3 = "Why does GPT-4 sometimes hallucinate facts?"

    print(f"\n   Query: '{query3}'")
    print("\n   Generating step-back question...")
    step_back = step_back_transform(query3)
    print(f"\n   Step-back question: {step_back}")

    # Demo 4: Decomposition
    print("\n\n🔨 QUERY DECOMPOSITION")
    print("=" * 60)

    query4 = "How can I build a RAG system that handles multiple document types and supports conversational memory?"

    print(f"\n   Query: '{query4}'")
    print("\n   Decomposing into sub-questions...")
    sub_queries = decompose_query(query4)
    print("\n   Sub-questions:")
    for i, q in enumerate(sub_queries, 1):
        print(f"   {i}. {q}")

    # Demo 5: Full pipeline with answer generation
    print("\n\n🎯 FULL PIPELINE DEMONSTRATION")
    print("=" * 60)

    query5 = "What is machine learning and how is it used?"

    print(f"\n   Query: '{query5}'")

    for technique in ["none", "hyde", "multi_query"]:
        print(f"\n   --- Using {technique.upper()} ---")

        if technique == "none":
            docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query5)
        else:
            docs = retrieve_with_transformation(vectorstore, query5, technique, k=3)

        answer = generate_answer(query5, docs)
        print(f"\n   Answer: {answer[:300]}...")

    # Best practices
    print("\n\n💡 WHEN TO USE EACH TECHNIQUE")
    print("-" * 40)
    print("""
   | Technique     | Best For                                  |
   |---------------|-------------------------------------------|
   | HyDE          | Queries with vocabulary mismatch          |
   | Multi-Query   | Ambiguous or vague queries                |
   | Step-Back     | Specific questions needing context        |
   | Decomposition | Complex multi-part questions              |

   Tips:
   - HyDE works best with factual, knowledge-seeking queries
   - Multi-Query helps when you're not sure of exact terminology
   - Step-Back is useful for "why" and "how" questions
   - Decomposition handles compound questions well

   Combine techniques for even better results!
    """)

    print_total_usage(token_tracker, "TOTAL - Query Transformation")

    print("\nEnd of Query Transformation demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
