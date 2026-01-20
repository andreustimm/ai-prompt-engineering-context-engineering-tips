"""
Transformação de Consultas

Transforma consultas do usuário para melhorar a eficácia da recuperação.
Diferentes estratégias de transformação ajudam a preencher a lacuna entre
como usuários formulam consultas e como documentos relevantes são escritos.

Técnicas implementadas:
1. HyDE - Embeddings de Documentos Hipotéticos
2. Multi-Query - Gerar múltiplas variações da consulta
3. Step-Back - Abstrair para conceitos mais amplos
4. Decomposição - Quebrar consultas complexas em sub-consultas

Casos de uso:
- Melhorar recall de recuperação para consultas vagas
- Lidar com perguntas complexas de múltiplas partes
- Resolver incompatibilidade de vocabulário
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


def criar_vectorstore(documentos: list[Document], nome_colecao: str = "transformacao_query"):
    """Cria vector store a partir de documentos."""
    if not CHROMA_DISPONIVEL:
        raise ImportError("chromadb é necessário")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documentos)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=nome_colecao
    )

    return vectorstore


def transformar_hyde(consulta: str) -> str:
    """
    HyDE - Embeddings de Documentos Hipotéticos

    Em vez de fazer embedding da consulta diretamente, gera um documento
    hipotético que responderia à consulta, então usa isso para recuperação.

    Isso frequentemente produz melhores correspondências porque o documento
    gerado é mais similar em estilo e vocabulário aos documentos reais.
    """
    llm = get_llm(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um escritor técnico especialista. Dada uma pergunta, escreva um parágrafo detalhado
que apareceria em um documento respondendo essa pergunta. Escreva como se este parágrafo existisse em
um documento real, não como uma resposta direta à pergunta.

Seja específico, factual e abrangente. Não comece com frases como "Aqui está" ou "Este parágrafo"."""),
        ("user", "Pergunta: {consulta}\n\nEscreva um parágrafo de documento que conteria a resposta:")
    ])

    chain = prompt | llm
    response = chain.invoke({"consulta": consulta})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Geração HyDE")

    return response.content


def transformar_multi_query(consulta: str, num_consultas: int = 3) -> list[str]:
    """
    Transformação Multi-Query

    Gera múltiplas variações da consulta original para melhorar
    o recall. Diferentes formulações podem corresponder a diferentes documentos relevantes.
    """
    llm = get_llm(temperature=0.8)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é especialista em reformular consultas de busca.
Dada uma pergunta do usuário, gere {num_consultas} versões diferentes da pergunta
que poderiam ser usadas para buscar informações relevantes.

Cada versão deve:
- Capturar a mesma intenção
- Usar vocabulário/formulação diferente
- Potencialmente enfatizar diferentes aspectos

Retorne APENAS as consultas, uma por linha, numeradas 1., 2., etc."""),
        ("user", "Pergunta original: {consulta}\n\nGere {num_consultas} variações da consulta:")
    ])

    chain = prompt | llm
    response = chain.invoke({"consulta": consulta, "num_consultas": num_consultas})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Geração Multi-Query")

    # Analisa a resposta
    linhas = response.content.strip().split('\n')
    consultas = []
    for linha in linhas:
        limpa = linha.strip()
        if limpa and limpa[0].isdigit():
            limpa = limpa.split('.', 1)[-1].strip()
        if limpa:
            consultas.append(limpa)

    return consultas[:num_consultas]


def transformar_step_back(consulta: str) -> str:
    """
    Step-Back Prompting

    Transforma uma pergunta específica em uma pergunta mais geral/abstrata.
    Isso pode ajudar a recuperar informações fundamentais que fornecem
    contexto para responder a pergunta específica.
    """
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é especialista em abstrair perguntas para seus conceitos subjacentes.
Dada uma pergunta específica, gere uma pergunta "step-back" mais geral que aborde
o conceito ou princípio mais amplo por trás da pergunta original.

A pergunta step-back deve:
- Ser mais geral/abstrata
- Cobrir o conhecimento fundamental necessário
- Ajudar a recuperar informações de contexto

Retorne APENAS a pergunta step-back, nada mais."""),
        ("user", """Pergunta original: {consulta}

Pergunta step-back:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"consulta": consulta})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Geração Step-Back")

    return response.content.strip()


def decompor_consulta(consulta: str) -> list[str]:
    """
    Decomposição de Consulta

    Quebra uma pergunta complexa em sub-perguntas mais simples.
    Cada sub-pergunta pode ser respondida independentemente, e as
    respostas combinadas para abordar a pergunta original.
    """
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é especialista em decompor perguntas complexas.
Dada uma pergunta complexa, decomponha-a em 2-4 sub-perguntas mais simples que,
quando respondidas juntas, forneceriam uma resposta completa à pergunta original.

Cada sub-pergunta deve:
- Ser autocontida e respondível independentemente
- Abordar um aspecto específico da pergunta original
- Contribuir para uma resposta abrangente

Retorne APENAS as sub-perguntas, uma por linha, numeradas 1., 2., etc."""),
        ("user", """Pergunta complexa: {consulta}

Sub-perguntas:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"consulta": consulta})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Decomposição de Consulta")

    # Analisa a resposta
    linhas = response.content.strip().split('\n')
    consultas = []
    for linha in linhas:
        limpa = linha.strip()
        if limpa and limpa[0].isdigit():
            limpa = limpa.split('.', 1)[-1].strip()
        if limpa:
            consultas.append(limpa)

    return consultas


def recuperar_com_transformacao(
    vectorstore,
    consulta: str,
    tipo_transformacao: str = "hyde",
    k: int = 3
) -> list[Document]:
    """
    Recupera documentos usando uma consulta transformada.

    Args:
        vectorstore: Vector store para buscar
        consulta: Consulta original do usuário
        tipo_transformacao: Tipo de transformação ("hyde", "multi_query", "step_back", "decompor")
        k: Número de documentos a recuperar
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    if tipo_transformacao == "hyde":
        # Gera documento hipotético e busca com ele
        doc_hipotetico = transformar_hyde(consulta)
        print(f"\n   Documento HyDE (primeiros 200 chars): {doc_hipotetico[:200]}...")
        return retriever.invoke(doc_hipotetico)

    elif tipo_transformacao == "multi_query":
        # Busca com múltiplas variações e combina resultados
        consultas = transformar_multi_query(consulta)
        print(f"\n   Consultas geradas:")
        for q in consultas:
            print(f"      - {q}")

        todos_docs = []
        vistos = set()
        for q in consultas:
            docs = retriever.invoke(q)
            for doc in docs:
                chave_doc = doc.page_content[:100]
                if chave_doc not in vistos:
                    vistos.add(chave_doc)
                    todos_docs.append(doc)

        return todos_docs[:k * 2]

    elif tipo_transformacao == "step_back":
        # Busca com pergunta original e step-back
        step_back = transformar_step_back(consulta)
        print(f"\n   Pergunta step-back: {step_back}")

        docs_originais = retriever.invoke(consulta)
        docs_step_back = retriever.invoke(step_back)

        # Combina, priorizando original
        todos_docs = docs_originais.copy()
        vistos = {doc.page_content[:100] for doc in docs_originais}
        for doc in docs_step_back:
            if doc.page_content[:100] not in vistos:
                todos_docs.append(doc)

        return todos_docs[:k * 2]

    elif tipo_transformacao == "decompor":
        # Busca com sub-perguntas e combina resultados
        sub_consultas = decompor_consulta(consulta)
        print(f"\n   Sub-perguntas:")
        for q in sub_consultas:
            print(f"      - {q}")

        todos_docs = []
        vistos = set()
        for q in sub_consultas:
            docs = retriever.invoke(q)
            for doc in docs:
                chave_doc = doc.page_content[:100]
                if chave_doc not in vistos:
                    vistos.add(chave_doc)
                    todos_docs.append(doc)

        return todos_docs[:k * 2]

    else:
        # Sem transformação
        return retriever.invoke(consulta)


def gerar_resposta(consulta: str, documentos: list[Document]) -> str:
    """Gera resposta usando contexto recuperado."""
    llm = get_llm(temperature=0.3)

    contexto = "\n\n---\n\n".join([doc.page_content for doc in documentos])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente útil. Responda a pergunta baseado no contexto fornecido.
Seja completo mas conciso. Se o contexto não contiver informação suficiente, diga isso."""),
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


def comparar_transformacoes(vectorstore, consulta: str):
    """Compara diferentes técnicas de transformação de consulta."""

    print(f"\n   Consulta Original: '{consulta}'")
    print("   " + "=" * 50)

    tecnicas = ["nenhuma", "hyde", "multi_query", "step_back", "decompor"]

    for tecnica in tecnicas:
        print(f"\n   📌 Técnica: {tecnica.upper()}")
        print("   " + "-" * 30)

        if tecnica == "nenhuma":
            docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(consulta)
        else:
            docs = recuperar_com_transformacao(vectorstore, consulta, tecnica, k=3)

        print(f"\n   Recuperados {len(docs)} documentos:")
        for i, doc in enumerate(docs[:3], 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   {i}. {preview}...")


def main():
    print("=" * 60)
    print("TRANSFORMAÇÃO DE CONSULTAS")
    print("=" * 60)

    if not CHROMA_DISPONIVEL:
        print("\nErro: chromadb é necessário para esta demonstração.")
        print("Instale com: pip install chromadb")
        return

    token_tracker.reset()

    # Carrega documentos e cria vector store
    print("\n📚 CARREGANDO DOCUMENTOS")
    print("-" * 40)

    documentos = carregar_documentos_exemplo()
    if not documentos:
        print("   Nenhum documento encontrado. Usando documentos de exemplo.")
        documentos = [
            Document(page_content="Machine learning é um subconjunto de IA que permite sistemas aprenderem com dados."),
            Document(page_content="Redes neurais são sistemas computacionais inspirados em redes neurais biológicas."),
            Document(page_content="Deep learning usa múltiplas camadas para extrair progressivamente features de alto nível."),
        ]

    print(f"   Carregados {len(documentos)} documentos")

    print("\n   Criando vector store...")
    vectorstore = criar_vectorstore(documentos, "demo_transformacao_query")
    print("   Vector store pronto!")

    # Demo 1: HyDE
    print("\n\n🔮 HYDE - EMBEDDINGS DE DOCUMENTOS HIPOTÉTICOS")
    print("=" * 60)

    consulta1 = "Como redes neurais aprendem?"

    print(f"\n   Consulta: '{consulta1}'")
    print("\n   Gerando documento hipotético...")
    doc_hyde = transformar_hyde(consulta1)
    print(f"\n   Documento Gerado:\n   {doc_hyde[:300]}...")

    docs = recuperar_com_transformacao(vectorstore, consulta1, "hyde", k=3)
    print("\n   Documentos recuperados com HyDE:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"   {i}. {doc.page_content[:100]}...")

    # Demo 2: Multi-Query
    print("\n\n🔄 TRANSFORMAÇÃO MULTI-QUERY")
    print("=" * 60)

    consulta2 = "Quais são os benefícios de usar transformers em NLP?"

    print(f"\n   Consulta: '{consulta2}'")
    print("\n   Gerando variações de consulta...")
    variacoes = transformar_multi_query(consulta2)
    print("\n   Variações:")
    for i, v in enumerate(variacoes, 1):
        print(f"   {i}. {v}")

    # Demo 3: Step-Back
    print("\n\n⬅️ STEP-BACK PROMPTING")
    print("=" * 60)

    consulta3 = "Por que o GPT-4 às vezes alucina fatos?"

    print(f"\n   Consulta: '{consulta3}'")
    print("\n   Gerando pergunta step-back...")
    step_back = transformar_step_back(consulta3)
    print(f"\n   Pergunta step-back: {step_back}")

    # Demo 4: Decomposição
    print("\n\n🔨 DECOMPOSIÇÃO DE CONSULTA")
    print("=" * 60)

    consulta4 = "Como posso construir um sistema RAG que lida com múltiplos tipos de documentos e suporta memória conversacional?"

    print(f"\n   Consulta: '{consulta4}'")
    print("\n   Decompondo em sub-perguntas...")
    sub_consultas = decompor_consulta(consulta4)
    print("\n   Sub-perguntas:")
    for i, q in enumerate(sub_consultas, 1):
        print(f"   {i}. {q}")

    # Demo 5: Pipeline completo com geração de resposta
    print("\n\n🎯 DEMONSTRAÇÃO DE PIPELINE COMPLETO")
    print("=" * 60)

    consulta5 = "O que é machine learning e como é usado?"

    print(f"\n   Consulta: '{consulta5}'")

    for tecnica in ["nenhuma", "hyde", "multi_query"]:
        print(f"\n   --- Usando {tecnica.upper()} ---")

        if tecnica == "nenhuma":
            docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(consulta5)
        else:
            docs = recuperar_com_transformacao(vectorstore, consulta5, tecnica, k=3)

        resposta = gerar_resposta(consulta5, docs)
        print(f"\n   Resposta: {resposta[:300]}...")

    # Melhores práticas
    print("\n\n💡 QUANDO USAR CADA TÉCNICA")
    print("-" * 40)
    print("""
   | Técnica       | Melhor Para                               |
   |---------------|-------------------------------------------|
   | HyDE          | Consultas com incompatibilidade de vocab  |
   | Multi-Query   | Consultas ambíguas ou vagas               |
   | Step-Back     | Perguntas específicas precisando contexto |
   | Decomposição  | Perguntas complexas de múltiplas partes   |

   Dicas:
   - HyDE funciona melhor com consultas factuais, buscando conhecimento
   - Multi-Query ajuda quando você não tem certeza da terminologia exata
   - Step-Back é útil para perguntas "por que" e "como"
   - Decomposição lida bem com perguntas compostas

   Combine técnicas para resultados ainda melhores!
    """)

    print_total_usage(token_tracker, "TOTAL - Transformação de Consultas")

    print("\nFim da demonstração de Transformação de Consultas")
    print("=" * 60)


if __name__ == "__main__":
    main()
