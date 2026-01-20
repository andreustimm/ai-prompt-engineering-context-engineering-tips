"""
Recuperação Ensemble

Combina múltiplos retrievers usando Reciprocal Rank Fusion (RRF)
para aproveitar os pontos fortes de diferentes estratégias de recuperação.

Componentes:
- Múltiplos Retrievers: Diferentes estratégias de busca
- Algoritmo RRF: Combina resultados ranqueados
- Fusão Ponderada: Pesos ajustáveis dos retrievers

Casos de uso:
- Combinar busca semântica e por palavras-chave
- Usar múltiplos modelos de embedding
- Ensemble de retrievers especializados
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_DISPONIVEL = True
except ImportError:
    CHROMA_DISPONIVEL = False

try:
    from rank_bm25 import BM25Okapi
    BM25_DISPONIVEL = True
except ImportError:
    BM25_DISPONIVEL = False

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

token_tracker = TokenUsage()


def carregar_documentos() -> list[Document]:
    """Carrega documentos de exemplo."""
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"
    docs = []
    for fp in sample_dir.glob("*.txt"):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                docs.append(Document(page_content=f.read(), metadata={"fonte": fp.name}))
        except Exception:
            pass
    return docs


def dividir_documentos(docs: list[Document]) -> list[Document]:
    """Divide documentos para recuperação."""
    divisor = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(divisor.split_documents([doc]))
    return chunks


class RetrieverBM25:
    """Retriever de palavras-chave BM25."""
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.tokenizados = [d.page_content.lower().split() for d in docs]
        self.bm25 = BM25Okapi(self.tokenizados)

    def recuperar(self, consulta: str, k: int = 5) -> list[tuple[Document, float]]:
        scores = self.bm25.get_scores(consulta.lower().split())
        idx_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[i], scores[i]) for i in idx_top]


class RetrieverVetor:
    """Retriever de similaridade vetorial."""
    def __init__(self, docs: list[Document], colecao: str):
        self.vectorstore = Chroma.from_documents(docs, get_embeddings(), collection_name=colecao)

    def recuperar(self, consulta: str, k: int = 5) -> list[tuple[Document, float]]:
        resultados = self.vectorstore.similarity_search_with_score(consulta, k=k)
        return [(doc, 1/(1+score)) for doc, score in resultados]


def fusao_rank_reciproco(lista_resultados: list[list[tuple[Document, float]]], pesos: list[float] = None, k: int = 60) -> list[tuple[Document, float]]:
    """Combina resultados usando RRF."""
    if pesos is None:
        pesos = [1.0] * len(lista_resultados)
    pesos = [w/sum(pesos) for w in pesos]

    scores_doc = {}
    for idx, resultados in enumerate(lista_resultados):
        for rank, (doc, _) in enumerate(resultados, 1):
            chave = doc.page_content[:100]
            if chave not in scores_doc:
                scores_doc[chave] = {"doc": doc, "score": 0}
            scores_doc[chave]["score"] += pesos[idx] / (k + rank)

    ordenados = sorted(scores_doc.values(), key=lambda x: x["score"], reverse=True)
    return [(r["doc"], r["score"]) for r in ordenados]


class RetrieverEnsemble:
    """Combina múltiplos retrievers com RRF."""

    def __init__(self, retrievers: list, pesos: list[float] = None):
        self.retrievers = retrievers
        self.pesos = pesos or [1.0] * len(retrievers)

    def recuperar(self, consulta: str, k: int = 5) -> list[tuple[Document, float]]:
        todos_resultados = []
        for retriever in self.retrievers:
            resultados = retriever.recuperar(consulta, k=k*2)
            todos_resultados.append(resultados)
        return fusao_rank_reciproco(todos_resultados, self.pesos)[:k]


def gerar_resposta(consulta: str, docs: list[Document]) -> str:
    """Gera resposta dos documentos."""
    llm = get_llm(temperature=0.3)
    contexto = "\n\n---\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda baseado no contexto. Seja conciso."),
        ("user", "Contexto:\n{contexto}\n\nPergunta: {pergunta}\n\nResposta:")
    ])
    response = (prompt | llm).invoke({"contexto": contexto, "pergunta": consulta})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Resposta")
    return response.content


def main():
    print("=" * 60)
    print("RECUPERAÇÃO ENSEMBLE")
    print("=" * 60)

    if not CHROMA_DISPONIVEL or not BM25_DISPONIVEL:
        print("\nErro: chromadb e rank-bm25 necessários")
        return

    token_tracker.reset()

    print("\n📚 CARREGANDO DOCUMENTOS")
    print("-" * 40)
    docs = carregar_documentos()
    if not docs:
        docs = [Document(page_content="Machine learning é IA que aprende com dados.")]
    chunks = dividir_documentos(docs)
    print(f"   Criados {len(chunks)} chunks")

    print("\n🔧 CRIANDO RETRIEVER ENSEMBLE")
    print("-" * 40)
    retriever_bm25 = RetrieverBM25(chunks)
    retriever_vetor = RetrieverVetor(chunks, "demo_ensemble")

    ensemble = RetrieverEnsemble(
        retrievers=[retriever_bm25, retriever_vetor],
        pesos=[0.4, 0.6]
    )
    print("   Peso BM25: 0.4, Peso Vetor: 0.6")

    consultas = ["O que é machine learning?", "arquiteturas de redes neurais"]

    print("\n\n❓ CONSULTAS ENSEMBLE")
    print("=" * 60)

    for consulta in consultas:
        print(f"\n📌 Consulta: '{consulta}'")
        print("-" * 40)

        resultados_bm25 = retriever_bm25.recuperar(consulta, k=3)
        resultados_vetor = retriever_vetor.recuperar(consulta, k=3)
        resultados_ensemble = ensemble.recuperar(consulta, k=3)

        print("\n   Top Resultado BM25:")
        print(f"      {resultados_bm25[0][0].page_content[:80]}...")

        print("\n   Top Resultado Vetor:")
        print(f"      {resultados_vetor[0][0].page_content[:80]}...")

        print("\n   Top Resultado Ensemble:")
        print(f"      {resultados_ensemble[0][0].page_content[:80]}...")

        docs_resposta = [r[0] for r in resultados_ensemble]
        resposta = gerar_resposta(consulta, docs_resposta)
        print(f"\n   Resposta: {resposta[:200]}...")

    print("\n\n💡 DICAS ENSEMBLE")
    print("-" * 40)
    print("""
   Ajuste de Pesos:
   - Mais BM25 para correspondência exata de termos
   - Mais Vetor para compreensão semântica
   - Comece com 0.4/0.6 e ajuste baseado nos resultados

   Combine diferentes estratégias para melhores resultados!
    """)

    print_total_usage(token_tracker, "TOTAL - Recuperação Ensemble")
    print("\nFim da demonstração de Recuperação Ensemble")
    print("=" * 60)


if __name__ == "__main__":
    main()
