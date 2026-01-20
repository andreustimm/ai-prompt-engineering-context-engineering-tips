"""
Ollama RAG - RAG 100% Offline

Sistema RAG completo rodando inteiramente local usando Ollama tanto para
embeddings quanto para geração com modelo de linguagem.

Componentes:
- Embeddings Locais: nomic-embed-text via Ollama
- LLM Local: llama3.2, mistral, etc. via Ollama
- Vector Store: ChromaDB (local)
- Sem chamadas API: Privacidade completa

Pré-requisitos:
1. Instalar Ollama: https://ollama.ai
2. Baixar modelos:
   - ollama pull llama3.2
   - ollama pull nomic-embed-text

Casos de uso:
- Ambientes isolados (air-gapped)
- Análise de documentos sensíveis
- Conformidade GDPR/privacidade
- Implantações offline
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


def verificar_status_ollama():
    """Verifica se o Ollama está rodando."""
    if is_ollama_available():
        print("   ✓ Ollama está rodando e acessível")
        return True
    else:
        print("   ✗ Ollama não está disponível")
        print("   Por favor, certifique-se que o Ollama está instalado e rodando:")
        print("   1. Instale de https://ollama.ai")
        print("   2. Execute: ollama serve")
        print("   3. Baixe os modelos:")
        print("      ollama pull llama3.2")
        print("      ollama pull nomic-embed-text")
        return False


def carregar_documentos(caminho: str) -> list[Document]:
    """Carrega documentos do caminho."""
    documentos = []
    caminho_obj = Path(caminho)

    if caminho_obj.is_file():
        arquivos = [caminho_obj]
    else:
        arquivos = list(caminho_obj.glob("**/*"))

    for caminho_arquivo in arquivos:
        ext = caminho_arquivo.suffix.lower()
        try:
            if ext in ['.txt', '.md']:
                with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                    conteudo = f.read()
                documentos.append(Document(
                    page_content=conteudo,
                    metadata={"source": str(caminho_arquivo)}
                ))
            elif ext == '.pdf':
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(caminho_arquivo))
                    documentos.extend(loader.load())
                except ImportError:
                    pass
        except Exception as e:
            print(f"   Aviso: Não foi possível carregar {caminho_arquivo}: {e}")

    return documentos


def dividir_documentos(documentos: list[Document], tamanho_chunk: int = 500, sobreposicao_chunk: int = 100) -> list[Document]:
    """Divide documentos em chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao_chunk,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documentos)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


class OllamaRAG:
    """
    Sistema RAG totalmente local usando Ollama.
    Sem chamadas de API externas - privacidade completa.
    """

    def __init__(
        self,
        modelo_llm: str = "llama3.2",
        modelo_embed: str = "nomic-embed-text",
        nome_colecao: str = "ollama_rag"
    ):
        self.modelo_llm = modelo_llm
        self.modelo_embed = modelo_embed
        self.nome_colecao = nome_colecao
        self.vectorstore = None
        self.chunks = []

    def carregar_e_indexar(self, caminho: str, tamanho_chunk: int = 500, sobreposicao_chunk: int = 100):
        """Carrega documentos e cria índice vetorial local."""
        print(f"\n   Carregando documentos de: {caminho}")
        documentos = carregar_documentos(caminho)
        print(f"   Carregados {len(documentos)} documento(s)")

        print(f"\n   Dividindo documentos...")
        self.chunks = dividir_documentos(documentos, tamanho_chunk, sobreposicao_chunk)
        print(f"   Criados {len(self.chunks)} chunks")

        print(f"\n   Criando embeddings locais com {self.modelo_embed}...")
        embeddings = get_ollama_embeddings(model=self.modelo_embed)

        print(f"   Construindo vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name=self.nome_colecao
        )
        print(f"   ✓ Vector store pronto!")

    def consultar(self, pergunta: str, k: int = 3) -> dict:
        """
        Consulta o sistema RAG local.

        Args:
            pergunta: Pergunta do usuário
            k: Número de documentos para recuperar

        Retorna:
            Dicionário com resposta e metadados
        """
        if not self.vectorstore:
            raise ValueError("Nenhum documento indexado. Chame carregar_e_indexar primeiro.")

        print(f"\n   Recuperando {k} chunks relevantes...")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs_relevantes = retriever.invoke(pergunta)

        print(f"\n   Gerando resposta com {self.modelo_llm}...")
        resposta = self._gerar_resposta(pergunta, docs_relevantes)

        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "fontes": docs_relevantes,
            "modelo": self.modelo_llm,
            "modelo_embed": self.modelo_embed
        }

    def _gerar_resposta(self, pergunta: str, docs_contexto: list[Document]) -> str:
        """Gera resposta usando LLM local."""
        llm = get_ollama_llm(model=self.modelo_llm, temperature=0.3)

        contexto = "\n\n---\n\n".join([
            f"[Chunk {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
            for doc in docs_contexto
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente útil que responde perguntas baseado no contexto fornecido.
Use APENAS as informações do contexto para responder.
Se a resposta não estiver no contexto, diga "Não tenho informações suficientes para responder essa pergunta."
Seja conciso mas completo."""),
            ("user", """Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:""")
        ])

        chain = prompt | llm
        response = chain.invoke({"contexto": contexto, "pergunta": pergunta})

        return response.content

    def busca_similaridade(self, consulta: str, k: int = 5) -> list[Document]:
        """Realiza busca por similaridade sem geração."""
        if not self.vectorstore:
            raise ValueError("Nenhum documento indexado.")

        return self.vectorstore.similarity_search(consulta, k=k)

    def mostrar_estatisticas(self):
        """Mostra estatísticas do sistema."""
        print("\n📊 Estatísticas do Sistema:")
        print("-" * 40)
        print(f"   Modelo LLM: {self.modelo_llm}")
        print(f"   Modelo de Embedding: {self.modelo_embed}")
        print(f"   Total de Chunks: {len(self.chunks)}")
        print(f"   Vector Store: ChromaDB (local)")
        print(f"   Chamadas API Externas: 0 (totalmente offline)")


def main():
    print("=" * 60)
    print("OLLAMA RAG - Demo de RAG 100% Offline")
    print("=" * 60)

    # Verifica pré-requisitos
    print("\n🔍 VERIFICANDO PRÉ-REQUISITOS")
    print("-" * 40)

    if not verificar_status_ollama():
        print("\nDemo não pode continuar sem o Ollama rodando.")
        return

    if not CHROMA_AVAILABLE:
        print("\nErro: chromadb necessário. Instale com: pip install chromadb")
        return

    diretorio_exemplos = Path(__file__).parent.parent.parent / "sample_data" / "documents"

    if not diretorio_exemplos.exists():
        print(f"\nErro: Diretório de exemplos não encontrado em {diretorio_exemplos}")
        return

    # Inicializa RAG offline
    print("\n📚 INICIALIZANDO SISTEMA RAG OFFLINE")
    print("-" * 40)

    rag = OllamaRAG(
        modelo_llm="llama3.2",
        modelo_embed="nomic-embed-text",
        nome_colecao="ollama_rag_demo"
    )

    try:
        rag.carregar_e_indexar(
            str(diretorio_exemplos),
            tamanho_chunk=500,
            sobreposicao_chunk=100
        )
    except Exception as e:
        print(f"\nErro inicializando RAG: {e}")
        print("\nCertifique-se de ter baixado os modelos necessários:")
        print("   ollama pull llama3.2")
        print("   ollama pull nomic-embed-text")
        return

    # Mostra estatísticas do sistema
    rag.mostrar_estatisticas()

    # Consultas de teste
    consultas = [
        "O que é machine learning?",
        "Quais são os diferentes tipos de redes neurais?",
        "Quais são as considerações éticas em IA?"
    ]

    print("\n\n❓ CONSULTANDO RAG OFFLINE")
    print("=" * 60)

    for i, consulta in enumerate(consultas, 1):
        print(f"\n📌 Pergunta {i}: {consulta}")
        print("-" * 40)

        resultado = rag.consultar(consulta, k=3)

        print(f"\n📋 Resposta:")
        print(resultado["resposta"])

        print(f"\n📎 Fontes:")
        for doc in resultado["fontes"]:
            fonte = Path(doc.metadata.get('source', 'Desconhecido')).name
            idx = doc.metadata.get('chunk_index', '?')
            print(f"   - {fonte} (chunk {idx})")

    # Demo de busca por similaridade
    print("\n\n🔍 BUSCA POR SIMILARIDADE (Sem Geração)")
    print("-" * 40)

    consulta_busca = "treinamento de rede neural"
    print(f"\nConsulta: {consulta_busca}")
    print("\nTop 3 chunks mais similares:")

    docs_similares = rag.busca_similaridade(consulta_busca, k=3)
    for i, doc in enumerate(docs_similares, 1):
        print(f"\n   {i}. Chunk {doc.metadata.get('chunk_index', '?')}:")
        print(f"      {doc.page_content[:150]}...")

    print("\n\n" + "=" * 60)
    print("✓ Todas as operações completadas 100% offline")
    print("✓ Nenhuma chamada de API externa foi feita")
    print("✓ Privacidade completa dos dados mantida")
    print("=" * 60)

    print("\nFim do demo Ollama RAG")
    print("=" * 60)


if __name__ == "__main__":
    main()
