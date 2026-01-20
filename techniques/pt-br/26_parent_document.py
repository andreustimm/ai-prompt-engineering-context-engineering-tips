"""
Recuperação de Documento Pai

Usa chunks pequenos para recuperação precisa mas retorna o documento
pai maior para contexto. Isso combina os benefícios de correspondência
precisa com contexto abrangente.

Componentes:
- Divisor Filho: Cria chunks pequenos para embedding/busca
- Divisor Pai: Cria chunks maiores para contexto
- Store de Documentos: Mapeia chunks filhos para documentos pais

Casos de uso:
- Quando correspondência precisa é importante mas contexto é necessário
- Documentos longos onde chunks pequenos podem perder significado
- Melhorar qualidade de respostas com contexto mais completo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
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

    return documentos


class StoreDocumentosEmMemoria:
    """Store de documentos simples em memória."""

    def __init__(self):
        self.store = {}

    def adicionar(self, doc_id: str, documento: Document):
        """Adiciona documento ao store."""
        self.store[doc_id] = documento

    def obter(self, doc_id: str) -> Document | None:
        """Obtém documento por ID."""
        return self.store.get(doc_id)

    def obter_multiplos(self, doc_ids: list[str]) -> list[Document | None]:
        """Obtém múltiplos documentos por IDs."""
        return [self.store.get(doc_id) for doc_id in doc_ids]


class RetrieverDocumentoPai:
    """
    Retriever que usa chunks pequenos para busca mas retorna documentos pai.

    A ideia chave é que chunks pequenos permitem correspondência precisa,
    enquanto retornar o documento pai fornece contexto suficiente
    para o LLM gerar boas respostas.
    """

    def __init__(
        self,
        tamanho_chunk_pai: int = 2000,
        sobreposicao_pai: int = 400,
        tamanho_chunk_filho: int = 400,
        sobreposicao_filho: int = 100,
        nome_colecao: str = "retriever_doc_pai"
    ):
        """
        Inicializa o retriever de documento pai.

        Args:
            tamanho_chunk_pai: Tamanho dos chunks pai (retornados para contexto)
            sobreposicao_pai: Sobreposição entre chunks pai
            tamanho_chunk_filho: Tamanho dos chunks filho (usados para busca)
            sobreposicao_filho: Sobreposição entre chunks filho
            nome_colecao: Nome para a coleção do vector store
        """
        self.divisor_pai = RecursiveCharacterTextSplitter(
            chunk_size=tamanho_chunk_pai,
            chunk_overlap=sobreposicao_pai,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.divisor_filho = RecursiveCharacterTextSplitter(
            chunk_size=tamanho_chunk_filho,
            chunk_overlap=sobreposicao_filho,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.docstore = StoreDocumentosEmMemoria()
        self.vectorstore = None
        self.embeddings = get_embeddings()
        self.nome_colecao = nome_colecao

    def adicionar_documentos(self, documentos: list[Document]):
        """
        Processa e adiciona documentos ao retriever.

        1. Divide em chunks pai
        2. Divide cada pai em chunks filho
        3. Armazena chunks pai no docstore
        4. Faz embedding e armazena chunks filho no vectorstore
        """
        todos_chunks_filho = []

        for doc in documentos:
            # Divide em chunks pai
            chunks_pai = self.divisor_pai.split_documents([doc])

            for pai in chunks_pai:
                # Gera ID único para pai
                id_pai = str(uuid.uuid4())

                # Armazena pai no docstore
                self.docstore.adicionar(id_pai, pai)

                # Divide pai em chunks filho
                chunks_filho = self.divisor_filho.split_documents([pai])

                # Adiciona id_pai aos metadados de cada filho
                for filho in chunks_filho:
                    filho.metadata["id_pai"] = id_pai
                    filho.metadata["fonte"] = doc.metadata.get("fonte", doc.metadata.get("source", "Desconhecida"))
                    todos_chunks_filho.append(filho)

        # Cria vector store com chunks filho
        if not CHROMA_DISPONIVEL:
            raise ImportError("chromadb é necessário")

        self.vectorstore = Chroma.from_documents(
            documents=todos_chunks_filho,
            embedding=self.embeddings,
            collection_name=self.nome_colecao
        )

        return len(todos_chunks_filho)

    def recuperar(self, consulta: str, k: int = 3) -> list[Document]:
        """
        Recupera documentos pai baseado em correspondência de chunks filho.

        Args:
            consulta: Consulta de busca
            k: Número de documentos pai a retornar

        Returns:
            Lista de documentos pai
        """
        if not self.vectorstore:
            raise ValueError("Nenhum documento adicionado. Chame adicionar_documentos primeiro.")

        # Recupera mais chunks filho do que necessário (alguns podem compartilhar pais)
        retriever_filho = self.vectorstore.as_retriever(search_kwargs={"k": k * 3})
        chunks_filho = retriever_filho.invoke(consulta)

        # Obtém documentos pai únicos
        pais_vistos = set()
        docs_pai = []

        for filho in chunks_filho:
            id_pai = filho.metadata.get("id_pai")
            if id_pai and id_pai not in pais_vistos:
                pai = self.docstore.obter(id_pai)
                if pai:
                    pais_vistos.add(id_pai)
                    docs_pai.append(pai)

                    if len(docs_pai) >= k:
                        break

        return docs_pai

    def recuperar_com_info_filho(self, consulta: str, k: int = 3) -> list[dict]:
        """
        Recupera com informações sobre chunks filho correspondentes.

        Retorna tanto documentos pai quanto os chunks filho que corresponderam.
        """
        if not self.vectorstore:
            raise ValueError("Nenhum documento adicionado. Chame adicionar_documentos primeiro.")

        # Recupera chunks filho
        resultados = self.vectorstore.similarity_search_with_score(consulta, k=k * 3)

        # Agrupa por pai
        info_pai = {}

        for filho, score in resultados:
            id_pai = filho.metadata.get("id_pai")
            if id_pai:
                if id_pai not in info_pai:
                    pai = self.docstore.obter(id_pai)
                    if pai:
                        info_pai[id_pai] = {
                            "pai": pai,
                            "filhos_correspondentes": [],
                            "melhor_score": score
                        }

                if id_pai in info_pai:
                    info_pai[id_pai]["filhos_correspondentes"].append({
                        "conteudo": filho.page_content[:100] + "...",
                        "score": score
                    })

        # Ordena por melhor score e retorna top k
        resultados_ordenados = sorted(
            info_pai.values(),
            key=lambda x: x["melhor_score"]
        )[:k]

        return resultados_ordenados


def comparar_tamanhos_chunk(documentos: list[Document], consulta: str):
    """Compara resultados de recuperação com diferentes tamanhos de chunk."""

    print(f"\n   Consulta: '{consulta}'")
    print("   " + "=" * 50)

    # Apenas chunks pequenos (sem recuperação de pai)
    print("\n   📄 APENAS CHUNKS PEQUENOS (400 chars)")
    print("-" * 30)

    divisor_pequeno = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks_pequenos = []
    for doc in documentos:
        chunks_pequenos.extend(divisor_pequeno.split_documents([doc]))

    vectorstore_pequeno = Chroma.from_documents(
        documents=chunks_pequenos,
        embedding=get_embeddings(),
        collection_name="comparar_chunks_pequenos"
    )

    resultados_pequenos = vectorstore_pequeno.similarity_search(consulta, k=3)

    contexto_total = 0
    for i, doc in enumerate(resultados_pequenos, 1):
        print(f"\n   {i}. Tamanho: {len(doc.page_content)} chars")
        print(f"      Preview: {doc.page_content[:100]}...")
        contexto_total += len(doc.page_content)

    print(f"\n   Contexto total: {contexto_total} chars")

    # Recuperação de documento pai
    print("\n   📚 RECUPERAÇÃO DE DOCUMENTO PAI (400 -> 2000 chars)")
    print("-" * 30)

    retriever_pai = RetrieverDocumentoPai(
        tamanho_chunk_pai=2000,
        tamanho_chunk_filho=400,
        nome_colecao="comparar_pai"
    )

    retriever_pai.adicionar_documentos(documentos)
    resultados_pai = retriever_pai.recuperar(consulta, k=3)

    contexto_total = 0
    for i, doc in enumerate(resultados_pai, 1):
        print(f"\n   {i}. Tamanho: {len(doc.page_content)} chars")
        print(f"      Preview: {doc.page_content[:100]}...")
        contexto_total += len(doc.page_content)

    print(f"\n   Contexto total: {contexto_total} chars")
    print("\n   Nota: Recuperação de pai fornece ~5x mais contexto!")


def gerar_resposta(consulta: str, documentos: list[Document]) -> str:
    """Gera resposta usando contexto recuperado."""
    llm = get_llm(temperature=0.3)

    contexto = "\n\n---\n\n".join([doc.page_content for doc in documentos])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente útil. Responda a pergunta baseado no contexto fornecido.
O contexto contém informações abrangentes - use-o para fornecer respostas detalhadas e precisas."""),
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


def main():
    print("=" * 60)
    print("RECUPERAÇÃO DE DOCUMENTO PAI")
    print("=" * 60)

    if not CHROMA_DISPONIVEL:
        print("\nErro: chromadb é necessário para esta demonstração.")
        print("Instale com: pip install chromadb")
        return

    token_tracker.reset()

    # Carrega documentos
    print("\n📚 CARREGANDO DOCUMENTOS")
    print("-" * 40)

    documentos = carregar_documentos_exemplo()
    if not documentos:
        print("   Nenhum documento encontrado. Usando documentos de exemplo.")
        documentos = [
            Document(
                page_content="Machine learning é um campo de IA... " * 50,
                metadata={"fonte": "exemplo.txt"}
            )
        ]

    print(f"   Carregados {len(documentos)} documentos")
    total_chars = sum(len(d.page_content) for d in documentos)
    print(f"   Total de caracteres: {total_chars:,}")

    # Cria retriever de documento pai
    print("\n\n🔧 CRIANDO RETRIEVER DE DOCUMENTO PAI")
    print("-" * 40)

    retriever = RetrieverDocumentoPai(
        tamanho_chunk_pai=2000,
        sobreposicao_pai=400,
        tamanho_chunk_filho=400,
        sobreposicao_filho=100,
        nome_colecao="demo_doc_pai"
    )

    num_filhos = retriever.adicionar_documentos(documentos)
    print(f"   Criados {num_filhos} chunks filho")
    print(f"   Tamanho chunk pai: 2000 chars")
    print(f"   Tamanho chunk filho: 400 chars")

    # Consultas de demonstração
    consultas = [
        "O que é machine learning?",
        "Como redes neurais funcionam?",
        "Quais são as aplicações de IA?"
    ]

    print("\n\n❓ CONSULTANDO COM RECUPERAÇÃO DE DOCUMENTO PAI")
    print("=" * 60)

    for consulta in consultas:
        print(f"\n📌 Consulta: '{consulta}'")
        print("-" * 40)

        # Recupera com info detalhada
        resultados = retriever.recuperar_com_info_filho(consulta, k=2)

        print(f"\n   Recuperados {len(resultados)} documentos pai:")
        for i, info in enumerate(resultados, 1):
            pai = info["pai"]
            filhos = info["filhos_correspondentes"]

            print(f"\n   Pai {i}:")
            print(f"      Tamanho: {len(pai.page_content)} chars")
            print(f"      Fonte: {pai.metadata.get('fonte', 'Desconhecida')}")
            print(f"      Filhos correspondentes: {len(filhos)}")
            print(f"      Melhor score: {info['melhor_score']:.4f}")
            print(f"      Preview: {pai.page_content[:150]}...")

        # Gera resposta
        docs_pai = [r["pai"] for r in resultados]
        print("\n   Gerando resposta...")
        resposta = gerar_resposta(consulta, docs_pai)
        print(f"\n   Resposta: {resposta[:300]}...")

    # Compara com chunking regular
    print("\n\n📊 COMPARAÇÃO: PAI vs CHUNKS PEQUENOS")
    print("=" * 60)

    comparar_tamanhos_chunk(documentos, "Quais são as considerações éticas em IA?")

    # Melhores práticas
    print("\n\n💡 MELHORES PRÁTICAS DE DOCUMENTO PAI")
    print("-" * 40)
    print("""
   | Parâmetro            | Faixa Recomendada    | Notas                    |
   |----------------------|----------------------|--------------------------|
   | Tamanho chunk pai    | 1500-3000 chars      | Contexto completo        |
   | Tamanho chunk filho  | 300-500 chars        | Correspondência precisa  |
   | Sobreposição pai     | 200-400 chars        | Continuidade entre chunks|
   | Sobreposição filho   | 50-100 chars         | Cobertura sem desperdício|

   Quando usar Recuperação de Documento Pai:
   - Documentos longos com informações interconectadas
   - Quando chunks pequenos perdem contexto importante
   - Quando qualidade da resposta é mais importante que custo de tokens
   - Conteúdo técnico ou educacional

   Trade-offs:
   + Mais contexto leva a melhores respostas
   + Correspondência precisa com chunks pequenos
   - Maior uso de tokens por consulta
   - Implementação mais complexa
    """)

    print_total_usage(token_tracker, "TOTAL - Recuperação de Documento Pai")

    print("\nFim da demonstração de Recuperação de Documento Pai")
    print("=" * 60)


if __name__ == "__main__":
    main()
