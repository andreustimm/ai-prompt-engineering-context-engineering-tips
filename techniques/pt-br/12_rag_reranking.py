"""
RAG com Reranking

RAG aprimorado que usa uma etapa de reranking para melhorar a qualidade da recuperação.
Após a recuperação inicial, um modelo de reranking pontua e reordena documentos
por relevância para a consulta.

Componentes:
- Recuperação Inicial: Obtém k candidatos (mais do que necessário)
- Reranking: Pontua e reordena por relevância
- Seleção Final: Usa os top n documentos reordenados

Casos de uso:
- Maior precisão para consultas complexas
- Melhor tratamento de similaridade semântica
- Sistemas RAG de produção que requerem alta precisão
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Imports condicionais
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)
import os

# Rastreador global de tokens
token_tracker = TokenUsage()


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
                documentos.append(Document(page_content=conteudo, metadata={"source": str(caminho_arquivo)}))
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


class RerankerLLM:
    """Reranker usando LLM para pontuar relevância de documentos."""

    def __init__(self):
        self.llm = get_llm(temperature=0)

    def reordenar(self, consulta: str, documentos: list[Document], top_n: int = 3) -> list[Document]:
        """
        Reordena documentos usando pontuação do LLM.

        Args:
            consulta: A consulta de busca
            documentos: Lista de documentos candidatos
            top_n: Número de documentos top para retornar

        Retorna:
            Lista reordenada de top_n documentos
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em pontuação de relevância. Dada uma consulta e um documento,
pontue a relevância do documento para a consulta em uma escala de 0-10.
Retorne APENAS um único número (a pontuação), nada mais."""),
            ("user", """Consulta: {consulta}

Documento:
{documento}

Pontuação de relevância (0-10):""")
        ])

        chain = prompt | self.llm
        docs_pontuados = []

        for doc in documentos:
            response = chain.invoke({
                "consulta": consulta,
                "documento": doc.page_content[:1000]
            })

            input_tokens, output_tokens = extract_tokens_from_response(response)
            token_tracker.add(input_tokens, output_tokens)

            try:
                pontuacao = float(response.content.strip())
            except ValueError:
                pontuacao = 5.0

            docs_pontuados.append((doc, pontuacao))

        docs_pontuados.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, pontuacao in docs_pontuados[:top_n]]


class RerankerCohere:
    """Reranker usando API de reranking da Cohere."""

    def __init__(self, api_key: str = None):
        if not COHERE_AVAILABLE:
            raise ImportError("pacote cohere necessário. Instale com: pip install cohere")

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY não encontrada.")

        self.client = cohere.Client(self.api_key)

    def reordenar(self, consulta: str, documentos: list[Document], top_n: int = 3) -> list[Document]:
        """Reordena documentos usando modelo de reranking da Cohere."""
        textos_docs = [doc.page_content for doc in documentos]

        results = self.client.rerank(
            model="rerank-multilingual-v2.0",
            query=consulta,
            documents=textos_docs,
            top_n=top_n
        )

        reordenados = []
        for result in results.results:
            idx = result.index
            reordenados.append(documentos[idx])

        return reordenados


class RerankerCrossEncoder:
    """Reranker usando CrossEncoder do sentence-transformers."""

    def __init__(self, nome_modelo: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.modelo = CrossEncoder(nome_modelo)
        except ImportError:
            raise ImportError("sentence-transformers necessário. Instale com: pip install sentence-transformers")

    def reordenar(self, consulta: str, documentos: list[Document], top_n: int = 3) -> list[Document]:
        """Reordena documentos usando modelo CrossEncoder."""
        pares = [[consulta, doc.page_content] for doc in documentos]
        pontuacoes = self.modelo.predict(pares)
        docs_pontuados = list(zip(documentos, pontuacoes))
        docs_pontuados.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, pontuacao in docs_pontuados[:top_n]]


class RAGComReranking:
    """Sistema RAG com capacidade de reranking."""

    def __init__(self, nome_colecao: str = "rag_rerank", tipo_reranker: str = "llm"):
        self.nome_colecao = nome_colecao
        self.vectorstore = None
        self.chunks = []

        if tipo_reranker == "cohere":
            try:
                self.reranker = RerankerCohere()
            except (ImportError, ValueError) as e:
                print(f"   Reranker Cohere não disponível: {e}")
                print("   Usando reranker LLM como fallback")
                self.reranker = RerankerLLM()
        elif tipo_reranker == "cross-encoder":
            try:
                self.reranker = RerankerCrossEncoder()
            except ImportError as e:
                print(f"   CrossEncoder não disponível: {e}")
                print("   Usando reranker LLM como fallback")
                self.reranker = RerankerLLM()
        else:
            self.reranker = RerankerLLM()

    def carregar_e_indexar(self, caminho: str, tamanho_chunk: int = 500, sobreposicao_chunk: int = 100):
        """Carrega documentos e cria índice vetorial."""
        print(f"\n   Carregando documentos de: {caminho}")
        documentos = carregar_documentos(caminho)
        print(f"   Carregados {len(documentos)} documento(s)")

        print(f"\n   Dividindo documentos...")
        self.chunks = dividir_documentos(documentos, tamanho_chunk, sobreposicao_chunk)
        print(f"   Criados {len(self.chunks)} chunks")

        print(f"\n   Criando vector store...")
        embeddings = get_embeddings()
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name=self.nome_colecao
        )
        print(f"   Vector store pronto!")

    def consultar(self, pergunta: str, k_inicial: int = 10, k_final: int = 3) -> dict:
        """
        Consulta com reranking.

        Args:
            pergunta: Pergunta do usuário
            k_inicial: Número de candidatos iniciais para recuperar
            k_final: Número de documentos após reranking

        Retorna:
            Dicionário com resposta e metadados
        """
        if not self.vectorstore:
            raise ValueError("Nenhum documento indexado.")

        print(f"\n   Passo 1: Recuperando {k_inicial} candidatos iniciais...")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_inicial})
        docs_iniciais = retriever.invoke(pergunta)

        print(f"\n   Passo 2: Reordenando para top {k_final}...")
        docs_reordenados = self.reranker.reordenar(pergunta, docs_iniciais, top_n=k_final)

        print(f"\n   Passo 3: Gerando resposta...")
        resposta = self._gerar_resposta(pergunta, docs_reordenados)

        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "candidatos_iniciais": len(docs_iniciais),
            "docs_reordenados": docs_reordenados
        }

    def _gerar_resposta(self, consulta: str, docs_contexto: list[Document]) -> str:
        """Gera resposta usando contexto reordenado."""
        llm = get_llm(temperature=0.3)

        contexto = "\n\n---\n\n".join([
            f"[Chunk {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
            for doc in docs_contexto
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente útil que responde perguntas baseado no contexto fornecido.
Use o contexto para fornecer respostas precisas e detalhadas. Cite números dos chunks quando possível."""),
            ("user", """Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:""")
        ])

        chain = prompt | llm
        response = chain.invoke({"contexto": contexto, "pergunta": consulta})

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, "Geração")

        return response.content


def comparar_com_sem_reranking(sistema_rag: RAGComReranking, pergunta: str):
    """Compara resultados RAG com e sem reranking."""

    print("\n" + "=" * 50)
    print("COMPARAÇÃO: Com vs Sem Reranking")
    print("=" * 50)

    print("\n📌 SEM RERANKING (top 3 da busca por similaridade):")
    print("-" * 40)

    retriever = sistema_rag.vectorstore.as_retriever(search_kwargs={"k": 3})
    docs_basicos = retriever.invoke(pergunta)

    llm = get_llm(temperature=0.3)
    contexto = "\n\n".join([doc.page_content[:500] for doc in docs_basicos])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda a pergunta baseado no contexto fornecido."),
        ("user", "Contexto:\n{contexto}\n\nPergunta: {pergunta}")
    ])

    response = (prompt | llm).invoke({"contexto": contexto, "pergunta": pergunta})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)

    print(f"Resposta: {response.content}")

    print("\n📌 COM RERANKING (top 3 após reordenar 10 candidatos):")
    print("-" * 40)

    resultado = sistema_rag.consultar(pergunta, k_inicial=10, k_final=3)
    print(f"Resposta: {resultado['resposta']}")


def main():
    print("=" * 60)
    print("RAG COM RERANKING - Demo")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nErro: chromadb necessário. Instale com: pip install chromadb")
        return

    token_tracker.reset()

    diretorio_exemplos = Path(__file__).parent.parent.parent / "sample_data" / "documents"

    if not diretorio_exemplos.exists():
        print(f"\nErro: Diretório de exemplos não encontrado em {diretorio_exemplos}")
        return

    print("\n📚 INICIALIZANDO RAG COM RERANKING")
    print("-" * 40)

    rag = RAGComReranking(nome_colecao="rag_rerank_demo", tipo_reranker="llm")

    rag.carregar_e_indexar(
        str(diretorio_exemplos),
        tamanho_chunk=500,
        sobreposicao_chunk=100
    )

    consultas = [
        "Quais são os principais componentes de uma rede neural?",
        "Como o aprendizado por reforço difere do aprendizado supervisionado?",
        "Quais são as considerações éticas no desenvolvimento de IA?"
    ]

    print("\n\n❓ CONSULTANDO COM RERANKING")
    print("=" * 60)

    for i, consulta in enumerate(consultas, 1):
        print(f"\n📌 Pergunta {i}: {consulta}")
        print("-" * 40)

        resultado = rag.consultar(consulta, k_inicial=10, k_final=3)

        print(f"\n📋 Resposta:")
        print(resultado["resposta"])

        print(f"\n📊 Estatísticas de Recuperação:")
        print(f"   Candidatos iniciais: {resultado['candidatos_iniciais']}")
        print(f"   Após reranking: {len(resultado['docs_reordenados'])}")

    print("\n\n🔄 COMPARANDO COM E SEM RERANKING")
    comparar_com_sem_reranking(rag, "Qual é a diferença entre CNN e RNN?")

    print_total_usage(token_tracker, "TOTAL - RAG com Reranking")

    print("\nFim do demo RAG com Reranking")
    print("=" * 60)


if __name__ == "__main__":
    main()
