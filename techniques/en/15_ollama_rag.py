"""
Ollama RAG - 100% Offline RAG

Complete RAG system running entirely locally using Ollama for both
embeddings and language model generation.

Components:
- Local Embeddings: nomic-embed-text via Ollama
- Local LLM: llama3.2, mistral, etc. via Ollama
- Vector Store: ChromaDB (local)
- No API calls: Complete privacy

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Pull models:
   - ollama pull llama3.2
   - ollama pull nomic-embed-text

Use cases:
- Air-gapped environments
- Sensitive document analysis
- GDPR/privacy compliance
- Offline deployments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from config import (
    get_ollama_llm,
    get_ollama_embeddings,
    is_ollama_available
)


def check_ollama_status():
    """Check if Ollama is running."""
    if is_ollama_available():
        print("   ✓ Ollama is running and accessible")
        return True
    else:
        print("   ✗ Ollama is not available")
        print("   Please ensure Ollama is installed and running:")
        print("   1. Install from https://ollama.ai")
        print("   2. Run: ollama serve")
        print("   3. Pull models:")
        print("      ollama pull llama3.2")
        print("      ollama pull nomic-embed-text")
        return False


def load_documents(path: str) -> list[Document]:
    """Load documents from path."""
    documents = []
    path_obj = Path(path)

    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.glob("**/*"))

    for file_path in files:
        ext = file_path.suffix.lower()
        try:
            if ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                ))
            elif ext == '.pdf':
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(file_path))
                    documents.extend(loader.load())
                except ImportError:
                    pass
        except Exception as e:
            print(f"   Warning: Could not load {file_path}: {e}")

    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


class OllamaRAG:
    """
    Fully local RAG system using Ollama.
    No external API calls - complete privacy.
    """

    def __init__(
        self,
        llm_model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        collection_name: str = "ollama_rag"
    ):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.vectorstore = None
        self.chunks = []

    def load_and_index(self, path: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """Load documents and create local vector index."""
        print(f"\n   Loading documents from: {path}")
        documents = load_documents(path)
        print(f"   Loaded {len(documents)} document(s)")

        print(f"\n   Chunking documents...")
        self.chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        print(f"   Created {len(self.chunks)} chunks")

        print(f"\n   Creating local embeddings with {self.embed_model}...")
        embeddings = get_ollama_embeddings(model=self.embed_model)

        print(f"   Building vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name=self.collection_name
        )
        print(f"   ✓ Vector store ready!")

    def query(self, question: str, k: int = 3) -> dict:
        """
        Query the local RAG system.

        Args:
            question: User's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer and metadata
        """
        if not self.vectorstore:
            raise ValueError("No documents indexed. Call load_and_index first.")

        # Retrieve relevant documents
        print(f"\n   Retrieving {k} relevant chunks...")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(question)

        # Generate answer with local LLM
        print(f"\n   Generating answer with {self.llm_model}...")
        answer = self._generate_answer(question, relevant_docs)

        return {
            "question": question,
            "answer": answer,
            "sources": relevant_docs,
            "model": self.llm_model,
            "embed_model": self.embed_model
        }

    def _generate_answer(self, question: str, context_docs: list[Document]) -> str:
        """Generate answer using local LLM."""
        llm = get_ollama_llm(model=self.llm_model, temperature=0.3)

        context = "\n\n---\n\n".join([
            f"[Chunk {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
            for doc in context_docs
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer.
If the answer is not in the context, say "I don't have enough information to answer that question."
Be concise but thorough."""),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        return response.content

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Perform similarity search without generation."""
        if not self.vectorstore:
            raise ValueError("No documents indexed.")

        return self.vectorstore.similarity_search(query, k=k)

    def show_stats(self):
        """Show system statistics."""
        print("\n📊 System Statistics:")
        print("-" * 40)
        print(f"   LLM Model: {self.llm_model}")
        print(f"   Embedding Model: {self.embed_model}")
        print(f"   Total Chunks: {len(self.chunks)}")
        print(f"   Vector Store: ChromaDB (local)")
        print(f"   External API Calls: 0 (fully offline)")


def main():
    print("=" * 60)
    print("OLLAMA RAG - 100% Offline RAG Demo")
    print("=" * 60)

    # Check prerequisites
    print("\n🔍 CHECKING PREREQUISITES")
    print("-" * 40)

    if not check_ollama_status():
        print("\nDemo cannot continue without Ollama running.")
        return

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb required. Install with: pip install chromadb")
        return

    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"

    if not sample_dir.exists():
        print(f"\nError: Sample directory not found at {sample_dir}")
        return

    # Initialize offline RAG
    print("\n📚 INITIALIZING OFFLINE RAG SYSTEM")
    print("-" * 40)

    rag = OllamaRAG(
        llm_model="llama3.2",
        embed_model="nomic-embed-text",
        collection_name="ollama_rag_demo"
    )

    try:
        rag.load_and_index(
            str(sample_dir),
            chunk_size=500,
            chunk_overlap=100
        )
    except Exception as e:
        print(f"\nError initializing RAG: {e}")
        print("\nMake sure you have pulled the required models:")
        print("   ollama pull llama3.2")
        print("   ollama pull nomic-embed-text")
        return

    # Show system stats
    rag.show_stats()

    # Test queries
    queries = [
        "What is machine learning?",
        "What are the different types of neural networks?",
        "What are ethical considerations in AI?"
    ]

    print("\n\n❓ QUERYING OFFLINE RAG")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n📌 Question {i}: {query}")
        print("-" * 40)

        result = rag.query(query, k=3)

        print(f"\n📋 Answer:")
        print(result["answer"])

        print(f"\n📎 Sources:")
        for doc in result["sources"]:
            source = Path(doc.metadata.get('source', 'Unknown')).name
            idx = doc.metadata.get('chunk_index', '?')
            print(f"   - {source} (chunk {idx})")

    # Similarity search demo
    print("\n\n🔍 SIMILARITY SEARCH (No Generation)")
    print("-" * 40)

    search_query = "neural network training"
    print(f"\nQuery: {search_query}")
    print("\nTop 3 most similar chunks:")

    similar_docs = rag.similarity_search(search_query, k=3)
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n   {i}. Chunk {doc.metadata.get('chunk_index', '?')}:")
        print(f"      {doc.page_content[:150]}...")

    print("\n\n" + "=" * 60)
    print("✓ All operations completed 100% offline")
    print("✓ No external API calls were made")
    print("✓ Complete data privacy maintained")
    print("=" * 60)

    print("\nEnd of Ollama RAG demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
