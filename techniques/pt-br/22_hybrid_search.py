"""
Busca Híbrida (BM25 + Vetor)

Combina busca por palavras-chave (BM25) com busca semântica (vetores)
para melhor precisão de recuperação. Esta abordagem aproveita os pontos
fortes de ambos os métodos.

Componentes:
- Retriever BM25: Correspondência tradicional de palavras-chave (baseado em TF-IDF)
- Retriever de Vetores: Busca por similaridade semântica
- Retriever Ensemble: Combina resultados usando Reciprocal Rank Fusion

Casos de uso:
- Documentação técnica com terminologia específica
- Documentos legais com requisitos precisos de palavras-chave
- Busca de produtos combinando correspondências exatas com compreensão semântica
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
    CHROMA_DISPONIVEL = True
except ImportError:
    CHROMA_DISPONIVEL = False
    print("Aviso: chromadb não instalado. Execute: pip install chromadb")

try:
    from rank_bm25 import BM25Okapi
    BM25_DISPONIVEL = True
except ImportError:
    BM25_DISPONIVEL = False
    print("Aviso: rank-bm25 não instalado. Execute: pip install rank-bm25")

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Rastreador global de tokens
token_tracker = TokenUsage()


def carregar_documentos_exemplo() -> list[Document]:
    """Carrega documentos de exemplo para demonstração."""
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"
    documentos = []

    for file_path in sample_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conteudo = f.read()
                documentos.append(Document(
                    page_content=conteudo,
                    metadata={"fonte": file_path.name}
                ))
        except Exception as e:
            print(f"   Aviso: Não foi possível carregar {file_path}: {e}")

    for file_path in sample_dir.glob("*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conteudo = f.read()
                documentos.append(Document(
                    page_content=conteudo,
                    metadata={"fonte": file_path.name}
                ))
        except Exception as e:
            print(f"   Aviso: Não foi possível carregar {file_path}: {e}")

    return documentos


def dividir_documentos(documentos: list[Document], tamanho_chunk: int = 500, sobreposicao: int = 100) -> list[Document]:
    """Divide documentos em chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documentos:
        doc_chunks = text_splitter.split_documents([doc])
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["indice_chunk"] = i
        chunks.extend(doc_chunks)

    return chunks


class RetrieverBM25:
    """Retriever baseado em palavras-chave BM25."""

    def __init__(self, documentos: list[Document]):
        """Inicializa retriever BM25 com documentos."""
        if not BM25_DISPONIVEL:
            raise ImportError("rank-bm25 é necessário. Instale com: pip install rank-bm25")

        self.documentos = documentos
        # Tokeniza documentos para BM25
        self.docs_tokenizados = [doc.page_content.lower().split() for doc in documentos]
        self.bm25 = BM25Okapi(self.docs_tokenizados)

    def recuperar(self, consulta: str, k: int = 4) -> list[tuple[Document, float]]:
        """Recupera top-k documentos para consulta."""
        consulta_tokenizada = consulta.lower().split()
        scores = self.bm25.get_scores(consulta_tokenizada)

        # Obtém índices top-k
        indices_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        resultados = []
        for idx in indices_top:
            resultados.append((self.documentos[idx], scores[idx]))

        return resultados


class RetrieverVetor:
    """Retriever semântico baseado em vetores."""

    def __init__(self, documentos: list[Document], nome_colecao: str = "busca_hibrida"):
        """Inicializa retriever de vetores com documentos."""
        if not CHROMA_DISPONIVEL:
            raise ImportError("chromadb é necessário. Instale com: pip install chromadb")

        self.embeddings = get_embeddings()
        self.vectorstore = Chroma.from_documents(
            documents=documentos,
            embedding=self.embeddings,
            collection_name=nome_colecao
        )

    def recuperar(self, consulta: str, k: int = 4) -> list[tuple[Document, float]]:
        """Recupera top-k documentos com scores de similaridade."""
        resultados = self.vectorstore.similarity_search_with_score(consulta, k=k)
        # Nota: Chroma retorna distância, menor é melhor. Converte para similaridade.
        return [(doc, 1 / (1 + score)) for doc, score in resultados]


def fusao_rank_reciproco(
    lista_resultados: list[list[tuple[Document, float]]],
    pesos: list[float] = None,
    k: int = 60
) -> list[tuple[Document, float]]:
    """
    Combina resultados de múltiplos retrievers usando Reciprocal Rank Fusion.

    Score RRF = soma(peso_i / (k + rank_i)) para cada retriever

    Args:
        lista_resultados: Lista de resultados de cada retriever
        pesos: Pesos para cada retriever (padrão: pesos iguais)
        k: Parâmetro RRF (tipicamente 60)

    Returns:
        Resultados combinados e reordenados
    """
    if pesos is None:
        pesos = [1.0] * len(lista_resultados)

    # Normaliza pesos
    peso_total = sum(pesos)
    pesos = [w / peso_total for w in pesos]

    # Calcula scores RRF
    scores_docs = {}

    for idx_retriever, resultados in enumerate(lista_resultados):
        peso = pesos[idx_retriever]

        for rank, (doc, _) in enumerate(resultados, start=1):
            # Usa conteúdo do documento como chave (simplificado)
            chave_doc = doc.page_content[:100]

            if chave_doc not in scores_docs:
                scores_docs[chave_doc] = {"doc": doc, "score": 0.0}

            # Fórmula RRF
            scores_docs[chave_doc]["score"] += peso / (k + rank)

    # Ordena por score combinado
    resultados_ordenados = sorted(
        scores_docs.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [(item["doc"], item["score"]) for item in resultados_ordenados]


class RetrieverHibrido:
    """Combina retrievers BM25 e de Vetores."""

    def __init__(
        self,
        documentos: list[Document],
        peso_bm25: float = 0.4,
        peso_vetor: float = 0.6,
        nome_colecao: str = "busca_hibrida"
    ):
        """
        Inicializa retriever híbrido.

        Args:
            documentos: Documentos para indexar
            peso_bm25: Peso para resultados BM25
            peso_vetor: Peso para resultados de vetores
        """
        self.documentos = documentos
        self.peso_bm25 = peso_bm25
        self.peso_vetor = peso_vetor

        print("   Inicializando retriever BM25...")
        self.retriever_bm25 = RetrieverBM25(documentos)

        print("   Inicializando retriever de Vetores...")
        self.retriever_vetor = RetrieverVetor(documentos, nome_colecao)

    def recuperar(self, consulta: str, k: int = 4, k_inicial: int = 10) -> list[tuple[Document, float]]:
        """
        Recupera documentos usando busca híbrida.

        Args:
            consulta: Consulta de busca
            k: Número de resultados finais
            k_inicial: Número de candidatos de cada retriever

        Returns:
            Resultados combinados com scores RRF
        """
        # Obtém resultados de ambos retrievers
        resultados_bm25 = self.retriever_bm25.recuperar(consulta, k=k_inicial)
        resultados_vetor = self.retriever_vetor.recuperar(consulta, k=k_inicial)

        # Combina usando RRF
        combinados = fusao_rank_reciproco(
            [resultados_bm25, resultados_vetor],
            pesos=[self.peso_bm25, self.peso_vetor]
        )

        return combinados[:k]


def comparar_metodos_recuperacao(chunks: list[Document], consulta: str, k: int = 3):
    """Compara métodos de recuperação BM25, Vetor e Híbrido."""

    print(f"\n   Consulta: '{consulta}'")
    print("   " + "=" * 50)

    # Somente BM25
    print("\n   📚 Resultados BM25 (Palavras-chave):")
    print("   " + "-" * 30)

    retriever_bm25 = RetrieverBM25(chunks)
    resultados_bm25 = retriever_bm25.recuperar(consulta, k=k)

    for i, (doc, score) in enumerate(resultados_bm25, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. Score: {score:.4f}")
        print(f"      Fonte: {doc.metadata.get('fonte', doc.metadata.get('source', 'Desconhecida'))}")
        print(f"      Preview: {preview}...")
        print()

    # Somente Vetor
    print("\n   🔮 Resultados Vetor (Semântico):")
    print("   " + "-" * 30)

    retriever_vetor = RetrieverVetor(chunks, nome_colecao="comparar_vetor")
    resultados_vetor = retriever_vetor.recuperar(consulta, k=k)

    for i, (doc, score) in enumerate(resultados_vetor, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. Score: {score:.4f}")
        print(f"      Fonte: {doc.metadata.get('fonte', doc.metadata.get('source', 'Desconhecida'))}")
        print(f"      Preview: {preview}...")
        print()

    # Híbrido
    print("\n   🔄 Resultados Híbridos (BM25 + Vetor):")
    print("   " + "-" * 30)

    combinados = fusao_rank_reciproco(
        [resultados_bm25, resultados_vetor],
        pesos=[0.4, 0.6]
    )

    for i, (doc, score) in enumerate(combinados[:k], 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. Score RRF: {score:.4f}")
        print(f"      Fonte: {doc.metadata.get('fonte', doc.metadata.get('source', 'Desconhecida'))}")
        print(f"      Preview: {preview}...")
        print()


def gerar_resposta_com_contexto(consulta: str, documentos: list[tuple[Document, float]]) -> str:
    """Gera resposta usando contexto recuperado."""

    llm = get_llm(temperature=0.3)

    contexto = "\n\n---\n\n".join([
        f"[Fonte: {doc.metadata.get('fonte', doc.metadata.get('source', 'Desconhecida'))}]\n{doc.page_content}"
        for doc, _ in documentos
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente útil que responde perguntas baseado no contexto fornecido.
Use APENAS as informações do contexto para responder. Se a resposta não estiver no contexto, diga isso.
Seja conciso mas completo."""),
        ("user", """Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"contexto": contexto, "pergunta": consulta})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Geração de Resposta")

    return response.content


def demonstrar_vantagem_hibrida():
    """Mostra casos onde busca híbrida supera métodos individuais."""

    print("\n   Demonstrando Vantagens da Busca Híbrida...")
    print("   " + "=" * 50)

    # Cria documentos sintéticos que destacam diferenças
    documentos = [
        Document(
            page_content="A linguagem de programação Python foi criada por Guido van Rossum. Ela enfatiza legibilidade de código e usa indentação significativa.",
            metadata={"fonte": "visao_geral_python.txt"}
        ),
        Document(
            page_content="Algoritmos de machine learning podem ser implementados em Python usando bibliotecas como scikit-learn, TensorFlow e PyTorch.",
            metadata={"fonte": "ferramentas_ml.txt"}
        ),
        Document(
            page_content="Uma píton é uma grande cobra encontrada em regiões tropicais. Esses répteis são constritores, significando que apertam suas presas.",
            metadata={"fonte": "animais.txt"}
        ),
        Document(
            page_content="Desenvolvimento web com frameworks Django e Flask torna Python uma excelente escolha para construir aplicações escaláveis.",
            metadata={"fonte": "dev_web.txt"}
        ),
        Document(
            page_content="Linguagens de programação como Java, C++ e Python são amplamente usadas em desenvolvimento de software.",
            metadata={"fonte": "linguagens.txt"}
        ),
    ]

    # Consulta que se beneficia da abordagem híbrida
    consulta = "Python programação desenvolvimento web"

    print(f"\n   Consulta: '{consulta}'")
    print("\n   Esta consulta se beneficia da busca híbrida porque:")
    print("   - BM25 captura correspondências exatas da palavra 'Python'")
    print("   - Busca vetorial entende relação semântica com dev web")
    print("   - Abordagem combinada filtra o documento sobre cobra")

    comparar_metodos_recuperacao(documentos, consulta, k=3)


def main():
    print("=" * 60)
    print("BUSCA HÍBRIDA (BM25 + Vetor)")
    print("=" * 60)

    if not BM25_DISPONIVEL:
        print("\nErro: rank-bm25 é necessário para esta demonstração.")
        print("Instale com: pip install rank-bm25")
        return

    if not CHROMA_DISPONIVEL:
        print("\nErro: chromadb é necessário para esta demonstração.")
        print("Instale com: pip install chromadb")
        return

    token_tracker.reset()

    # Carrega e prepara documentos
    print("\n📚 CARREGANDO E PREPARANDO DOCUMENTOS")
    print("-" * 40)

    documentos = carregar_documentos_exemplo()

    if not documentos:
        print("   Nenhum documento encontrado. Usando documentos de exemplo.")
        documentos = [
            Document(
                page_content="Machine learning é um subconjunto de inteligência artificial que permite sistemas aprenderem com dados.",
                metadata={"fonte": "exemplo.txt"}
            )
        ]

    print(f"   Carregados {len(documentos)} documentos")

    chunks = dividir_documentos(documentos, tamanho_chunk=400, sobreposicao=80)
    print(f"   Criados {len(chunks)} chunks")

    # Inicializa retriever híbrido
    print("\n🔧 INICIALIZANDO RETRIEVER HÍBRIDO")
    print("-" * 40)

    retriever_hibrido = RetrieverHibrido(
        chunks,
        peso_bm25=0.4,
        peso_vetor=0.6,
        nome_colecao="demo_hibrido"
    )

    # Consultas de demonstração
    consultas = [
        "O que é machine learning?",
        "redes neurais deep learning",
        "Como construir aplicações de IA?"
    ]

    print("\n\n❓ CONSULTAS DE BUSCA HÍBRIDA")
    print("=" * 60)

    for consulta in consultas:
        print(f"\n📌 Consulta: {consulta}")
        print("-" * 40)

        resultados = retriever_hibrido.recuperar(consulta, k=3)

        print("\n   Documentos Recuperados:")
        for i, (doc, score) in enumerate(resultados, 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   {i}. Score RRF: {score:.4f}")
            print(f"      Fonte: {doc.metadata.get('fonte', doc.metadata.get('source', 'Desconhecida'))}")
            print(f"      Preview: {preview}...")
            print()

        print("\n   Gerando resposta...")
        resposta = gerar_resposta_com_contexto(consulta, resultados)
        print(f"\n   Resposta: {resposta}")

    # Compara métodos
    print("\n\n📊 COMPARANDO MÉTODOS DE RECUPERAÇÃO")
    print("=" * 60)

    comparar_metodos_recuperacao(
        chunks,
        "aplicações de inteligência artificial",
        k=3
    )

    # Demonstra vantagem híbrida
    print("\n\n🎯 VANTAGEM DA BUSCA HÍBRIDA")
    print("=" * 60)

    demonstrar_vantagem_hibrida()

    # Guia de ajuste de pesos
    print("\n\n💡 GUIA DE AJUSTE DE PESOS")
    print("-" * 40)
    print("""
   Ajuste os pesos baseado no seu caso de uso:

   | Caso de Uso                | Peso BM25 | Peso Vetor |
   |----------------------------|-----------|------------|
   | Docs técnicos (precisos)   | 0.6       | 0.4        |
   | Conhecimento geral         | 0.3       | 0.7        |
   | Busca de código            | 0.5       | 0.5        |
   | FAQ/Suporte                | 0.4       | 0.6        |

   Dicas:
   - Maior peso BM25 para correspondência exata de termos (IDs, códigos, nomes)
   - Maior peso Vetor para consultas conceituais/semânticas
   - Comece com 0.4/0.6 e ajuste baseado nos resultados
    """)

    print_total_usage(token_tracker, "TOTAL - Busca Híbrida")

    print("\nFim da demonstração de Busca Híbrida")
    print("=" * 60)


if __name__ == "__main__":
    main()
