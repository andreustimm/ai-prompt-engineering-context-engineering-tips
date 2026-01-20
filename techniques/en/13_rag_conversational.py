"""
Conversational RAG

RAG system enhanced with conversation memory, allowing multi-turn
interactions where context from previous questions is maintained.

Components:
- Vector Store: ChromaDB for document retrieval
- Conversation Memory: Tracks chat history
- Question Rewriting: Reformulates questions with context
- Context-Aware Retrieval: Uses conversation for better retrieval

Use cases:
- Document-based chatbots
- Customer support over knowledge base
- Interactive research assistants
- Multi-turn Q&A systems
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Optional

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

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


class ConversationMemory:
    """Simple conversation memory that stores chat history."""

    def __init__(self, max_history: int = 10):
        self.history: list[dict] = []
        self.max_history = max_history

    def add_exchange(self, user_message: str, assistant_message: str):
        """Add a user-assistant exchange to history."""
        self.history.append({
            "user": user_message,
            "assistant": assistant_message
        })

        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history_string(self) -> str:
        """Get history as formatted string."""
        if not self.history:
            return "No previous conversation."

        lines = []
        for exchange in self.history:
            lines.append(f"User: {exchange['user']}")
            lines.append(f"Assistant: {exchange['assistant']}")

        return "\n".join(lines)

    def get_recent_context(self, n: int = 3) -> str:
        """Get the last n exchanges as context."""
        recent = self.history[-n:] if self.history else []

        if not recent:
            return ""

        lines = []
        for exchange in recent:
            lines.append(f"User: {exchange['user']}")
            lines.append(f"Assistant: {exchange['assistant']}")

        return "\n".join(lines)

    def clear(self):
        """Clear conversation history."""
        self.history = []


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
                documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
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


def chunk_documents(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> list[Document]:
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


class ConversationalRAG:
    """
    RAG system with conversation memory for multi-turn interactions.
    """

    def __init__(self, collection_name: str = "conversational_rag"):
        self.collection_name = collection_name
        self.vectorstore = None
        self.chunks = []
        self.memory = ConversationMemory(max_history=10)

    def load_and_index(self, path: str, chunk_size: int = 800, chunk_overlap: int = 150):
        """Load documents and create vector index."""
        print(f"\n   Loading documents from: {path}")
        documents = load_documents(path)
        print(f"   Loaded {len(documents)} document(s)")

        print(f"\n   Chunking documents...")
        self.chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        print(f"   Created {len(self.chunks)} chunks")

        print(f"\n   Creating vector store...")
        embeddings = get_embeddings()
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name=self.collection_name
        )
        print(f"   Vector store ready!")

    def rewrite_question_with_context(self, question: str) -> str:
        """
        Rewrite the question to be standalone using conversation history.
        This helps with retrieval when questions contain pronouns or references.
        """
        if not self.memory.history:
            return question

        llm = get_llm(temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the conversation history and a follow-up question,
rewrite the question to be a standalone question that captures the full context.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else."""),
            ("user", """Conversation History:
{history}

Follow-up Question: {question}

Standalone Question:""")
        ])

        chain = prompt | llm
        response = chain.invoke({
            "history": self.memory.get_recent_context(n=3),
            "question": question
        })

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, "Rewrite")

        return response.content.strip()

    def retrieve_documents(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve relevant documents for the query."""
        if not self.vectorstore:
            raise ValueError("No documents indexed.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def generate_response(self, question: str, context_docs: list[Document]) -> str:
        """Generate a response using retrieved context and conversation history."""
        llm = get_llm(temperature=0.3)

        # Format retrieved context
        context = "\n\n---\n\n".join([
            f"[Source: {Path(doc.metadata.get('source', 'Unknown')).name}]\n{doc.page_content}"
            for doc in context_docs
        ])

        # Get conversation history
        history = self.memory.get_recent_context(n=5)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant engaged in a conversation.
Use the provided context documents to answer questions accurately.
Consider the conversation history for context about what the user is asking.
If information is not in the provided context, say so.
Be conversational but informative."""),
            ("user", """Document Context:
{context}

Conversation History:
{history}

Current Question: {question}

Response:""")
        ])

        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "history": history if history else "No previous conversation.",
            "question": question
        })

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, "Generation")

        return response.content

    def chat(self, question: str, k: int = 4) -> dict:
        """
        Process a chat message with context from conversation history.

        Args:
            question: User's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with response and metadata
        """
        if not self.vectorstore:
            raise ValueError("No documents indexed. Call load_and_index first.")

        # Step 1: Rewrite question with conversation context
        print(f"\n   Rewriting question with context...")
        standalone_question = self.rewrite_question_with_context(question)

        if standalone_question != question:
            print(f"   Original: {question}")
            print(f"   Rewritten: {standalone_question}")
        else:
            print(f"   Question is already standalone")

        # Step 2: Retrieve relevant documents using standalone question
        print(f"\n   Retrieving {k} relevant documents...")
        relevant_docs = self.retrieve_documents(standalone_question, k)

        # Step 3: Generate response with full context
        print(f"\n   Generating response...")
        response = self.generate_response(question, relevant_docs)

        # Step 4: Save to memory
        self.memory.add_exchange(question, response)

        return {
            "question": question,
            "standalone_question": standalone_question,
            "response": response,
            "sources": relevant_docs,
            "history_length": len(self.memory.history)
        }

    def reset_conversation(self):
        """Reset the conversation history."""
        self.memory.clear()
        print("   Conversation history cleared.")

    def show_history(self):
        """Display the conversation history."""
        print("\n📜 Conversation History:")
        print("-" * 40)

        if not self.memory.history:
            print("   No conversation history.")
            return

        for i, exchange in enumerate(self.memory.history, 1):
            print(f"\n   Turn {i}:")
            print(f"   User: {exchange['user'][:100]}...")
            print(f"   Assistant: {exchange['assistant'][:100]}...")


def simulate_conversation(rag: ConversationalRAG):
    """Simulate a multi-turn conversation."""

    conversations = [
        "What is machine learning?",
        "What are its main types?",  # Refers to "machine learning" from previous
        "Can you explain the third type in more detail?",  # Refers to types mentioned
        "What are some real-world applications of it?",  # Refers to ML/the type discussed
        "What about ethical concerns?",  # General continuation
    ]

    print("\n🗣️ SIMULATING MULTI-TURN CONVERSATION")
    print("=" * 60)

    for i, question in enumerate(conversations, 1):
        print(f"\n{'='*50}")
        print(f"📌 Turn {i}: {question}")
        print("-" * 50)

        result = rag.chat(question, k=3)

        print(f"\n📋 Response:")
        print(result["response"][:800] + "..." if len(result["response"]) > 800 else result["response"])

        print(f"\n📊 Metadata:")
        print(f"   Question was rewritten: {result['question'] != result['standalone_question']}")
        print(f"   Sources used: {len(result['sources'])}")
        print(f"   History length: {result['history_length']}")


def main():
    print("=" * 60)
    print("CONVERSATIONAL RAG - Demo")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nError: chromadb required. Install with: pip install chromadb")
        return

    token_tracker.reset()

    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"

    if not sample_dir.exists():
        print(f"\nError: Sample directory not found at {sample_dir}")
        return

    # Initialize system
    print("\n📚 INITIALIZING CONVERSATIONAL RAG")
    print("-" * 40)

    rag = ConversationalRAG(collection_name="conversational_rag_demo")

    rag.load_and_index(
        str(sample_dir),
        chunk_size=600,
        chunk_overlap=100
    )

    # Run conversation simulation
    simulate_conversation(rag)

    # Show final history
    rag.show_history()

    print_total_usage(token_tracker, "TOTAL - Conversational RAG")

    print("\nEnd of Conversational RAG demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
