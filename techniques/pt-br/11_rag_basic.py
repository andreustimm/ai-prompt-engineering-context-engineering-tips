"""
RAG (Retrieval-Augmented Generation) - Básico

Técnica que aprimora respostas do LLM recuperando informações relevantes
de uma base de conhecimento antes de gerar respostas.

Componentes:
- Document Loader: Carrega documentos (PDF, TXT, MD)
- Text Splitter: Divide documentos em chunks para embedding
- Embeddings: Converte texto em vetores
- Vector Store: ChromaDB para busca por similaridade
- Retriever: Encontra chunks relevantes
- LLM: Gera respostas usando contexto recuperado

Casos de uso:
- Perguntas e respostas sobre documentos
- Chatbots de base de conhecimento
- Busca e sumarização de documentos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Imports condicionais para componentes RAG
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Aviso: chromadb não instalado. Execute: pip install chromadb")

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Rastreador global de tokens para este script
token_tracker = TokenUsage()


def carregar_arquivo_texto(caminho_arquivo: str) -> list[Document]:
    """Carrega um arquivo de texto e retorna como Document."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    return [Document(page_content=conteudo, metadata={"source": caminho_arquivo})]


def carregar_arquivo_pdf(caminho_arquivo: str) -> list[Document]:
    """Carrega um arquivo PDF e retorna como Documents (um por página)."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(caminho_arquivo)
        return loader.load()
    except ImportError:
        print("Aviso: pypdf não instalado. Execute: pip install pypdf")
        return []


def carregar_arquivo_markdown(caminho_arquivo: str) -> list[Document]:
    """Carrega um arquivo markdown e retorna como Document."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    return [Document(page_content=conteudo, metadata={"source": caminho_arquivo})]


def carregar_documentos(caminho: str) -> list[Document]:
    """
    Carrega documentos de um arquivo ou diretório.
    Suporta: arquivos .txt, .pdf, .md
    """
    documentos = []
    caminho_obj = Path(caminho)

    if caminho_obj.is_file():
        arquivos = [caminho_obj]
    else:
        arquivos = list(caminho_obj.glob("**/*"))

    for caminho_arquivo in arquivos:
        ext = caminho_arquivo.suffix.lower()
        try:
            if ext == '.txt':
                documentos.extend(carregar_arquivo_texto(str(caminho_arquivo)))
            elif ext == '.pdf':
                documentos.extend(carregar_arquivo_pdf(str(caminho_arquivo)))
            elif ext == '.md':
                documentos.extend(carregar_arquivo_markdown(str(caminho_arquivo)))
        except Exception as e:
            print(f"   Aviso: Não foi possível carregar {caminho_arquivo}: {e}")

    return documentos


def dividir_documentos(documentos: list[Document], tamanho_chunk: int = 1000, sobreposicao_chunk: int = 200) -> list[Document]:
    """
    Divide documentos em chunks menores para embedding.

    Args:
        documentos: Lista de documentos para dividir
        tamanho_chunk: Tamanho máximo de cada chunk em caracteres
        sobreposicao_chunk: Número de caracteres sobrepostos entre chunks

    Retorna:
        Lista de documentos divididos em chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao_chunk,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documentos)

    # Adiciona índice do chunk aos metadados
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def criar_vector_store(chunks: list[Document], nome_colecao: str = "rag_colecao"):
    """
    Cria um vector store ChromaDB a partir de chunks de documentos.

    Args:
        chunks: Lista de chunks de documentos
        nome_colecao: Nome para a coleção ChromaDB

    Retorna:
        Vector store ChromaDB
    """
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb é necessário. Instale com: pip install chromadb")

    embeddings = get_embeddings()

    # Cria vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=nome_colecao
    )

    return vectorstore


def recuperar_chunks_relevantes(vectorstore, consulta: str, k: int = 4) -> list[Document]:
    """
    Recupera os chunks mais relevantes para uma consulta.

    Args:
        vectorstore: Vector store ChromaDB
        consulta: Consulta de busca
        k: Número de chunks para recuperar

    Retorna:
        Lista de chunks de documentos relevantes
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(consulta)


def gerar_resposta(pergunta: str, documentos_contexto: list[Document]) -> str:
    """
    Gera uma resposta usando o LLM com contexto recuperado.

    Args:
        pergunta: Pergunta do usuário
        documentos_contexto: Documentos relevantes recuperados

    Retorna:
        Resposta gerada
    """
    llm = get_llm(temperature=0.3)

    # Formata contexto dos documentos recuperados
    contexto = "\n\n---\n\n".join([
        f"[Fonte: {doc.metadata.get('source', 'Desconhecido')}, Chunk: {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
        for doc in documentos_contexto
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente útil que responde perguntas baseado no contexto fornecido.
Use APENAS as informações do contexto para responder. Se a resposta não estiver no contexto, diga isso.
Sempre cite de qual fonte/chunk a informação vem quando possível."""),
        ("user", """Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"contexto": contexto, "pergunta": pergunta})

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Geração")

    return response.content


class RAGSimples:
    """
    Sistema RAG simples que combina carregamento de documentos, chunking,
    armazenamento vetorial, recuperação e geração.
    """

    def __init__(self, nome_colecao: str = "rag_simples"):
        self.nome_colecao = nome_colecao
        self.vectorstore = None
        self.documentos = []
        self.chunks = []

    def carregar_e_indexar(self, caminho: str, tamanho_chunk: int = 1000, sobreposicao_chunk: int = 200):
        """Carrega documentos do caminho e cria índice vetorial."""
        print(f"\n   Carregando documentos de: {caminho}")
        self.documentos = carregar_documentos(caminho)
        print(f"   Carregados {len(self.documentos)} documento(s)")

        print(f"\n   Dividindo documentos (tamanho={tamanho_chunk}, sobreposição={sobreposicao_chunk})...")
        self.chunks = dividir_documentos(self.documentos, tamanho_chunk, sobreposicao_chunk)
        print(f"   Criados {len(self.chunks)} chunks")

        print(f"\n   Criando vector store...")
        self.vectorstore = criar_vector_store(self.chunks, self.nome_colecao)
        print(f"   Vector store pronto!")

    def consultar(self, pergunta: str, k: int = 4) -> dict:
        """
        Consulta o sistema RAG.

        Args:
            pergunta: Pergunta do usuário
            k: Número de chunks para recuperar

        Retorna:
            Dicionário com resposta e chunks recuperados
        """
        if not self.vectorstore:
            raise ValueError("Nenhum documento indexado. Chame carregar_e_indexar primeiro.")

        print(f"\n   Recuperando {k} chunks relevantes...")
        chunks_relevantes = recuperar_chunks_relevantes(self.vectorstore, pergunta, k)

        print(f"   Gerando resposta...")
        resposta = gerar_resposta(pergunta, chunks_relevantes)

        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "chunks_fonte": chunks_relevantes
        }

    def mostrar_chunks(self, n: int = 5):
        """Exibe os primeiros n chunks para inspeção."""
        print(f"\n   Primeiros {min(n, len(self.chunks))} chunks:")
        for i, chunk in enumerate(self.chunks[:n]):
            print(f"\n   --- Chunk {i} ---")
            print(f"   Fonte: {chunk.metadata.get('source', 'Desconhecido')}")
            print(f"   Prévia do conteúdo: {chunk.page_content[:200]}...")


def main():
    print("=" * 60)
    print("RAG (Retrieval-Augmented Generation) - Demo Básico")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nErro: chromadb é necessário para este demo.")
        print("Instale com: pip install chromadb")
        return

    # Reseta rastreador
    token_tracker.reset()

    # Caminho para documentos de exemplo
    diretorio_exemplos = Path(__file__).parent.parent.parent / "sample_data" / "documents"

    if not diretorio_exemplos.exists():
        print(f"\nErro: Diretório de exemplos não encontrado em {diretorio_exemplos}")
        print("Por favor, certifique-se que o diretório sample_data/documents existe com arquivos de exemplo.")
        return

    # Inicializa sistema RAG
    print("\n📚 INICIALIZANDO SISTEMA RAG")
    print("-" * 40)

    rag = RAGSimples(nome_colecao="manual_ia_rag")

    # Carrega e indexa documentos
    rag.carregar_e_indexar(
        str(diretorio_exemplos),
        tamanho_chunk=800,
        sobreposicao_chunk=150
    )

    # Mostra alguns chunks
    rag.mostrar_chunks(n=3)

    # Consultas de exemplo
    consultas = [
        "Quais são os diferentes tipos de machine learning?",
        "Explique o que são redes neurais e como funcionam.",
        "Qual é a história da inteligência artificial?",
        "Quais são as considerações éticas no desenvolvimento de IA?"
    ]

    print("\n\n❓ CONSULTANDO O SISTEMA RAG")
    print("=" * 60)

    for i, consulta in enumerate(consultas, 1):
        print(f"\n📌 Pergunta {i}: {consulta}")
        print("-" * 40)

        resultado = rag.consultar(consulta, k=3)

        print(f"\n📋 Resposta:")
        print(resultado["resposta"])

        print(f"\n📎 Fontes utilizadas:")
        for chunk in resultado["chunks_fonte"]:
            fonte = Path(chunk.metadata.get('source', 'Desconhecido')).name
            idx = chunk.metadata.get('chunk_index', '?')
            print(f"   - {fonte} (chunk {idx})")

    # Exibe total de tokens
    print_total_usage(token_tracker, "TOTAL - RAG Básico")

    print("\nFim do demo RAG Básico")
    print("=" * 60)


if __name__ == "__main__":
    main()
