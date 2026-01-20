"""
Compressão Contextual

Extrai apenas as porções relevantes dos documentos recuperados,
reduzindo ruído e focando nas informações mais pertinentes
para responder à consulta do usuário.

Componentes:
- Retriever Base: Recuperação inicial de documentos
- Compressor de Documentos: Extrai passagens relevantes
- Retriever de Compressão: Combina recuperação com compressão

Casos de uso:
- Reduzir uso de tokens removendo conteúdo irrelevante
- Melhorar qualidade de respostas focando em passagens relevantes
- Lidar com documentos longos eficientemente
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


class CompressorExtratorLLM:
    """Comprime documentos extraindo apenas passagens relevantes usando um LLM."""

    def __init__(self, max_tokens: int = 500):
        """
        Inicializa o compressor baseado em LLM.

        Args:
            max_tokens: Máximo de tokens para conteúdo extraído
        """
        self.llm = get_llm(temperature=0)
        self.max_tokens = max_tokens

    def comprimir(self, documentos: list[Document], consulta: str) -> list[Document]:
        """
        Extrai passagens relevantes dos documentos.

        Args:
            documentos: Documentos para comprimir
            consulta: Consulta do usuário para contexto

        Returns:
            Documentos comprimidos com apenas conteúdo relevante
        """
        comprimidos = []

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é especialista em extrair informações relevantes.
Dado um documento e uma consulta, extraia APENAS as sentenças ou passagens que são diretamente relevantes para responder a consulta.

Regras:
- Inclua apenas informações relevantes
- Mantenha a redação original quando possível
- Se nada for relevante, responda com "SEM_CONTEUDO_RELEVANTE"
- Não adicione comentários ou explicações
- Mantenha a extração com menos de {max_tokens} tokens"""),
            ("user", """Consulta: {consulta}

Documento:
{documento}

Extração relevante:""")
        ])

        chain = prompt | self.llm

        for doc in documentos:
            response = chain.invoke({
                "consulta": consulta,
                "documento": doc.page_content,
                "max_tokens": self.max_tokens
            })

            input_tokens, output_tokens = extract_tokens_from_response(response)
            token_tracker.add(input_tokens, output_tokens)

            extraido = response.content.strip()

            if extraido and extraido != "SEM_CONTEUDO_RELEVANTE":
                comprimidos.append(Document(
                    page_content=extraido,
                    metadata={
                        **doc.metadata,
                        "tipo_compressao": "extracao_llm",
                        "tamanho_original": len(doc.page_content),
                        "tamanho_comprimido": len(extraido)
                    }
                ))

        return comprimidos


class CompressorFiltroEmbeddings:
    """Filtra chunks de documentos baseado em similaridade de embedding com a consulta."""

    def __init__(self, limiar_similaridade: float = 0.75):
        """
        Inicializa o filtro baseado em embeddings.

        Args:
            limiar_similaridade: Score mínimo de similaridade para manter um chunk
        """
        self.embeddings = get_embeddings()
        self.limiar = limiar_similaridade

    def comprimir(self, documentos: list[Document], consulta: str) -> list[Document]:
        """
        Filtra documentos baseado em similaridade de embedding.

        Args:
            documentos: Documentos para filtrar
            consulta: Consulta do usuário

        Returns:
            Documentos que atendem ao limiar de similaridade
        """
        if not documentos:
            return []

        # Obtém embedding da consulta
        embedding_consulta = self.embeddings.embed_query(consulta)

        # Obtém embeddings dos documentos
        embeddings_docs = self.embeddings.embed_documents(
            [doc.page_content for doc in documentos]
        )

        # Calcula similaridades e filtra
        comprimidos = []
        for doc, emb_doc in zip(documentos, embeddings_docs):
            # Similaridade de cosseno
            similaridade = sum(a * b for a, b in zip(embedding_consulta, emb_doc))
            norma_q = sum(a * a for a in embedding_consulta) ** 0.5
            norma_d = sum(a * a for a in emb_doc) ** 0.5
            similaridade = similaridade / (norma_q * norma_d) if norma_q * norma_d > 0 else 0

            if similaridade >= self.limiar:
                comprimidos.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "tipo_compressao": "filtro_embeddings",
                        "score_similaridade": similaridade
                    }
                ))

        return comprimidos


class CompressorExtratorSentencas:
    """Extrai sentenças relevantes baseado em correspondência de palavras-chave e pontuação."""

    def __init__(self, top_n_sentencas: int = 5):
        """
        Inicializa extrator de sentenças.

        Args:
            top_n_sentencas: Número de sentenças top para extrair por documento
        """
        self.top_n = top_n_sentencas

    def comprimir(self, documentos: list[Document], consulta: str) -> list[Document]:
        """
        Extrai sentenças mais relevantes dos documentos.

        Args:
            documentos: Documentos para comprimir
            consulta: Consulta do usuário

        Returns:
            Documentos com apenas as sentenças mais relevantes
        """
        import re

        termos_consulta = set(consulta.lower().split())
        comprimidos = []

        for doc in documentos:
            # Divide em sentenças
            sentencas = re.split(r'(?<=[.!?])\s+', doc.page_content)

            # Pontua sentenças baseado em sobreposição de termos da consulta
            pontuadas = []
            for sentenca in sentencas:
                termos_sentenca = set(sentenca.lower().split())
                sobreposicao = len(termos_consulta & termos_sentenca)
                pontuadas.append((sentenca, sobreposicao))

            # Ordena por pontuação e pega top N
            pontuadas.sort(key=lambda x: x[1], reverse=True)
            sentencas_top = [s for s, _ in pontuadas[:self.top_n] if _]

            if sentencas_top:
                comprimidos.append(Document(
                    page_content=" ".join(sentencas_top),
                    metadata={
                        **doc.metadata,
                        "tipo_compressao": "extracao_sentencas",
                        "sentencas_originais": len(sentencas),
                        "sentencas_extraidas": len(sentencas_top)
                    }
                ))

        return comprimidos


class RetrieverCompressaoContextual:
    """Combina recuperação com compressão para resultados mais focados."""

    def __init__(
        self,
        vectorstore,
        compressor,
        k_base: int = 10,
        k_final: int = 3
    ):
        """
        Inicializa o retriever de compressão.

        Args:
            vectorstore: Vector store para recuperação
            compressor: Compressor de documentos
            k_base: Número de documentos a recuperar antes da compressão
            k_final: Número de documentos a retornar após compressão
        """
        self.vectorstore = vectorstore
        self.compressor = compressor
        self.k_base = k_base
        self.k_final = k_final

    def recuperar(self, consulta: str) -> list[Document]:
        """
        Recupera e comprime documentos.

        Args:
            consulta: Consulta do usuário

        Returns:
            Documentos comprimidos e filtrados
        """
        # Recuperação inicial
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_base})
        docs_iniciais = retriever.invoke(consulta)

        # Comprime
        docs_comprimidos = self.compressor.comprimir(docs_iniciais, consulta)

        # Retorna top k_final
        return docs_comprimidos[:self.k_final]


def criar_vectorstore(documentos: list[Document], nome_colecao: str = "demo_compressao"):
    """Cria vector store a partir de documentos."""
    if not CHROMA_DISPONIVEL:
        raise ImportError("chromadb é necessário")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Chunks maiores para demonstrar compressão
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documentos)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=nome_colecao
    )

    return vectorstore


def comparar_metodos_compressao(vectorstore, consulta: str):
    """Compara diferentes métodos de compressão."""

    print(f"\n   Consulta: '{consulta}'")
    print("   " + "=" * 50)

    # Recupera documentos base
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs_base = retriever.invoke(consulta)

    print(f"\n   📚 Recuperação base: {len(docs_base)} documentos")
    total_chars = sum(len(d.page_content) for d in docs_base)
    print(f"   Total de caracteres: {total_chars:,}")

    # Método 1: Sem compressão
    print("\n   --- SEM COMPRESSÃO ---")
    print(f"   Documentos: {len(docs_base)}")
    print(f"   Caracteres: {total_chars:,}")
    for i, doc in enumerate(docs_base[:2], 1):
        print(f"\n   Doc {i} preview: {doc.page_content[:150]}...")

    # Método 2: Extração LLM
    print("\n   --- EXTRAÇÃO LLM ---")
    compressor_llm = CompressorExtratorLLM(max_tokens=200)
    comprimidos_llm = compressor_llm.comprimir(docs_base, consulta)
    chars_llm = sum(len(d.page_content) for d in comprimidos_llm)
    print(f"   Documentos: {len(comprimidos_llm)}")
    print(f"   Caracteres: {chars_llm:,} ({100*chars_llm/total_chars:.1f}% do original)")
    for i, doc in enumerate(comprimidos_llm[:2], 1):
        print(f"\n   Doc {i}: {doc.page_content[:200]}...")

    # Método 3: Extração de Sentenças
    print("\n   --- EXTRAÇÃO DE SENTENÇAS ---")
    compressor_sentencas = CompressorExtratorSentencas(top_n_sentencas=3)
    comprimidos_sentencas = compressor_sentencas.comprimir(docs_base, consulta)
    chars_sentencas = sum(len(d.page_content) for d in comprimidos_sentencas)
    print(f"   Documentos: {len(comprimidos_sentencas)}")
    print(f"   Caracteres: {chars_sentencas:,} ({100*chars_sentencas/total_chars:.1f}% do original)")
    for i, doc in enumerate(comprimidos_sentencas[:2], 1):
        print(f"\n   Doc {i}: {doc.page_content[:200]}...")


def gerar_resposta(consulta: str, documentos: list[Document]) -> str:
    """Gera resposta usando contexto comprimido."""
    llm = get_llm(temperature=0.3)

    contexto = "\n\n---\n\n".join([doc.page_content for doc in documentos])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente útil. Responda a pergunta baseado no contexto fornecido.
Seja conciso e focado. O contexto foi pré-filtrado para conter apenas informações relevantes."""),
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


def demonstrar_economia_tokens(vectorstore, consulta: str):
    """Demonstra economia de tokens com compressão."""

    print(f"\n   Consulta: '{consulta}'")
    print("   " + "=" * 50)

    # Sem compressão
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    nao_comprimidos = retriever.invoke(consulta)
    chars_nao_comprimidos = sum(len(d.page_content) for d in nao_comprimidos)

    # Com compressão LLM
    compressor = CompressorExtratorLLM(max_tokens=150)
    comprimidos = compressor.comprimir(nao_comprimidos, consulta)
    chars_comprimidos = sum(len(d.page_content) for d in comprimidos)

    print(f"\n   Sem compressão:")
    print(f"      Documentos: {len(nao_comprimidos)}")
    print(f"      Total de caracteres: {chars_nao_comprimidos:,}")
    print(f"      Tokens estimados: ~{chars_nao_comprimidos // 4:,}")

    print(f"\n   Com compressão:")
    print(f"      Documentos: {len(comprimidos)}")
    print(f"      Total de caracteres: {chars_comprimidos:,}")
    print(f"      Tokens estimados: ~{chars_comprimidos // 4:,}")

    economia = 100 * (1 - chars_comprimidos / chars_nao_comprimidos) if chars_nao_comprimidos > 0 else 0
    print(f"\n   💰 Economia de tokens: {economia:.1f}%")

    # Gera respostas com ambos
    print("\n   Gerando respostas...")

    print("\n   Resposta (sem compressão):")
    resposta_nao_comprimida = gerar_resposta(consulta, nao_comprimidos)
    print(f"   {resposta_nao_comprimida[:300]}...")

    print("\n   Resposta (com compressão):")
    resposta_comprimida = gerar_resposta(consulta, comprimidos)
    print(f"   {resposta_comprimida[:300]}...")


def main():
    print("=" * 60)
    print("COMPRESSÃO CONTEXTUAL")
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
                page_content="""Machine learning é um subconjunto de inteligência artificial que permite sistemas
                aprenderem e melhorarem com a experiência sem serem explicitamente programados. Ele foca no
                desenvolvimento de programas de computador que podem acessar dados e usá-los para aprender por
                si mesmos. O processo começa com observações ou dados, como exemplos, experiência direta ou
                instrução. Algoritmos de machine learning usam métodos computacionais para aprender informações
                diretamente dos dados sem depender de uma equação predeterminada como modelo.""",
                metadata={"fonte": "intro_ml.txt"}
            ),
        ]

    print(f"   Carregados {len(documentos)} documentos")

    print("\n   Criando vector store...")
    vectorstore = criar_vectorstore(documentos, "demo_compressao")
    print("   Vector store pronto!")

    # Demo 1: Comparar métodos de compressão
    print("\n\n📊 COMPARANDO MÉTODOS DE COMPRESSÃO")
    print("=" * 60)

    comparar_metodos_compressao(
        vectorstore,
        "O que é machine learning e como funciona?"
    )

    # Demo 2: Economia de tokens
    print("\n\n💰 DEMONSTRAÇÃO DE ECONOMIA DE TOKENS")
    print("=" * 60)

    demonstrar_economia_tokens(
        vectorstore,
        "Como redes neurais aprendem com dados?"
    )

    # Demo 3: Pipeline completo
    print("\n\n🎯 PIPELINE COMPLETO DE COMPRESSÃO")
    print("=" * 60)

    consulta = "Quais são as aplicações de deep learning?"

    print(f"\n   Consulta: '{consulta}'")

    # Cria retriever de compressão
    compressor = CompressorExtratorLLM(max_tokens=200)
    retriever_compressao = RetrieverCompressaoContextual(
        vectorstore=vectorstore,
        compressor=compressor,
        k_base=8,
        k_final=3
    )

    print("\n   Recuperando com compressão...")
    docs_comprimidos = retriever_compressao.recuperar(consulta)

    print(f"\n   Recuperados {len(docs_comprimidos)} documentos comprimidos:")
    for i, doc in enumerate(docs_comprimidos, 1):
        print(f"\n   Documento {i}:")
        print(f"   Tamanho original: {doc.metadata.get('tamanho_original', 'N/A')} chars")
        print(f"   Tamanho comprimido: {doc.metadata.get('tamanho_comprimido', len(doc.page_content))} chars")
        print(f"   Conteúdo: {doc.page_content[:200]}...")

    print("\n   Gerando resposta...")
    resposta = gerar_resposta(consulta, docs_comprimidos)
    print(f"\n   Resposta: {resposta}")

    # Melhores práticas
    print("\n\n💡 MELHORES PRÁTICAS DE COMPRESSÃO")
    print("-" * 40)
    print("""
   | Método              | Prós                        | Contras                  |
   |---------------------|-----------------------------|--------------------------|
   | Extração LLM        | Alta qualidade, contextual  | Maior latência e custo   |
   | Filtro Embeddings   | Rápido, sem chamadas LLM    | Pode perder match sutil  |
   | Extração Sentenças  | Rápido, preserva original   | Apenas baseado em keyword|

   Dicas:
   - Use extração LLM para aplicações críticas de qualidade
   - Use filtro de embeddings para alto volume e baixa latência
   - Combine métodos: filtre primeiro, depois extraia com LLM
   - Ajuste limiares baseado em suas necessidades de precisão/recall
   - Monitore trade-off entre economia de tokens e qualidade de resposta
    """)

    print_total_usage(token_tracker, "TOTAL - Compressão Contextual")

    print("\nFim da demonstração de Compressão Contextual")
    print("=" * 60)


if __name__ == "__main__":
    main()
