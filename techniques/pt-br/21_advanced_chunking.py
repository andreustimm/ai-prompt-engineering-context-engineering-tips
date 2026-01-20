"""
Estratégias Avançadas de Chunking

Explora diferentes estratégias de divisão de texto para sistemas RAG.
A qualidade do chunking impacta significativamente a eficácia da recuperação.

Estratégias implementadas:
1. RecursiveCharacterTextSplitter - Divisão hierárquica por caracteres
2. SentenceTransformers Token Splitter - Divisão por tokens
3. Semantic Chunker - Divisão por similaridade semântica
4. Markdown Splitter - Divisão consciente de estrutura markdown
5. Janela deslizante personalizada - Chunks sobrepostos configuráveis

Casos de uso:
- Otimização da qualidade de recuperação RAG
- Manipulação de diferentes tipos de documentos
- Equilíbrio entre tamanho do chunk e preservação de contexto
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document

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


def carregar_documento_exemplo() -> str:
    """Carrega documento de exemplo para demonstrações de chunking."""
    sample_path = Path(__file__).parent.parent.parent / "sample_data" / "documents" / "long_document.txt"

    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Texto de exemplo alternativo
        return """
        # Introdução ao Machine Learning

        Machine learning é um subconjunto de inteligência artificial que permite que sistemas aprendam e melhorem com a experiência sem serem explicitamente programados.

        ## Tipos de Machine Learning

        ### Aprendizado Supervisionado
        O aprendizado supervisionado usa dados rotulados para treinar modelos. O algoritmo aprende com pares de entrada-saída.
        Algoritmos comuns incluem:
        - Regressão Linear
        - Árvores de Decisão
        - Máquinas de Vetores de Suporte
        - Redes Neurais

        ### Aprendizado Não Supervisionado
        O aprendizado não supervisionado encontra padrões em dados não rotulados. Ele descobre estruturas ocultas.
        Algoritmos comuns incluem:
        - Clustering K-Means
        - Clustering Hierárquico
        - Análise de Componentes Principais

        ### Aprendizado por Reforço
        O aprendizado por reforço treina agentes através de recompensas e penalidades.
        Aplicações incluem:
        - Jogos (AlphaGo)
        - Robótica
        - Veículos autônomos

        ## Deep Learning

        Deep learning usa redes neurais com múltiplas camadas para modelar padrões complexos.
        Arquiteturas principais:
        - Redes Neurais Convolucionais (CNNs) para imagens
        - Redes Neurais Recorrentes (RNNs) para sequências
        - Transformers para compreensão de linguagem

        ## Melhores Práticas

        1. Comece com dados limpos e de qualidade
        2. Escolha algoritmos apropriados para seu problema
        3. Use validação cruzada para avaliação de modelos
        4. Monitore overfitting e underfitting
        5. Documente seus experimentos e resultados
        """


def chunking_caracteres_recursivo(texto: str, tamanho_chunk: int = 500, sobreposicao: int = 100) -> list[Document]:
    """
    Divide texto usando divisão recursiva de caracteres com hierarquia.

    Este splitter tenta dividir nestes separadores em ordem:
    ["\n\n", "\n", " ", ""]

    Ele divide recursivamente em separadores menores quando chunks são muito grandes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.create_documents([texto])

    for i, chunk in enumerate(chunks):
        chunk.metadata["indice_chunk"] = i
        chunk.metadata["estrategia"] = "caracteres_recursivo"
        chunk.metadata["tamanho_chunk"] = len(chunk.page_content)

    return chunks


def chunking_por_tokens(texto: str, tamanho_chunk: int = 200, sobreposicao: int = 50) -> list[Document]:
    """
    Divide texto baseado em contagem de tokens em vez de caracteres.

    Isso garante que chunks caibam nos limites de contexto do modelo.
    Divisão por tokens é mais precisa para processamento LLM.
    """
    splitter = TokenTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao
    )

    chunks = splitter.create_documents([texto])

    for i, chunk in enumerate(chunks):
        chunk.metadata["indice_chunk"] = i
        chunk.metadata["estrategia"] = "por_tokens"
        chunk.metadata["tamanho_chunk"] = len(chunk.page_content)

    return chunks


def chunking_markdown(texto: str, tamanho_chunk: int = 500, sobreposicao: int = 100) -> list[Document]:
    """
    Divide texto markdown respeitando sua estrutura.

    Este splitter entende cabeçalhos e blocos de código markdown,
    preservando a estrutura do documento nos chunks.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao
    )

    chunks = splitter.create_documents([texto])

    for i, chunk in enumerate(chunks):
        chunk.metadata["indice_chunk"] = i
        chunk.metadata["estrategia"] = "markdown"
        chunk.metadata["tamanho_chunk"] = len(chunk.page_content)

    return chunks


def chunking_semantico(texto: str, limiar_quebra: float = 0.5) -> list[Document]:
    """
    Divide texto baseado em similaridade semântica entre sentenças.

    Agrupa sentenças semanticamente similares,
    criando chunks mais coerentes para recuperação.

    Nota: Esta é uma implementação simplificada. Para produção,
    use langchain_experimental.text_splitter.SemanticChunker
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker

        embeddings = get_embeddings()
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=limiar_quebra * 100
        )

        chunks = splitter.create_documents([texto])

        for i, chunk in enumerate(chunks):
            chunk.metadata["indice_chunk"] = i
            chunk.metadata["estrategia"] = "semantico"
            chunk.metadata["tamanho_chunk"] = len(chunk.page_content)

        return chunks

    except ImportError:
        print("   Aviso: langchain_experimental não disponível. Usando fallback.")
        return chunking_caracteres_recursivo(texto)


def chunking_janela_deslizante(texto: str, tamanho_janela: int = 500, passo: int = 250) -> list[Document]:
    """
    Cria chunks sobrepostos usando abordagem de janela deslizante.

    Args:
        texto: Texto de entrada para dividir
        tamanho_janela: Tamanho de cada chunk em caracteres
        passo: Quanto avançar a janela (tamanho_janela - sobreposição)

    Isso cria máxima sobreposição entre chunks, útil quando
    continuidade de contexto é crítica.
    """
    chunks = []
    inicio = 0

    while inicio < len(texto):
        fim = inicio + tamanho_janela
        texto_chunk = texto[inicio:fim]

        # Evita cortar palavras no meio
        if fim < len(texto) and texto[fim] not in ' \n\t':
            # Encontra último espaço dentro do chunk
            ultimo_espaco = texto_chunk.rfind(' ')
            if ultimo_espaco > 0:
                texto_chunk = texto_chunk[:ultimo_espaco]

        chunk = Document(
            page_content=texto_chunk.strip(),
            metadata={
                "indice_chunk": len(chunks),
                "estrategia": "janela_deslizante",
                "char_inicio": inicio,
                "tamanho_chunk": len(texto_chunk.strip())
            }
        )
        chunks.append(chunk)

        inicio += passo

    return chunks


def chunking_por_sentencas(texto: str, sentencas_por_chunk: int = 5, sobreposicao_sentencas: int = 1) -> list[Document]:
    """
    Divide texto em chunks contendo número fixo de sentenças.

    Preserva limites de sentença, garantindo que chunks sejam gramaticalmente completos.
    """
    import re

    # Divisão simples de sentenças
    sentencas = re.split(r'(?<=[.!?])\s+', texto)
    sentencas = [s.strip() for s in sentencas if s.strip()]

    chunks = []
    i = 0

    while i < len(sentencas):
        sentencas_chunk = sentencas[i:i + sentencas_por_chunk]
        texto_chunk = ' '.join(sentencas_chunk)

        chunk = Document(
            page_content=texto_chunk,
            metadata={
                "indice_chunk": len(chunks),
                "estrategia": "por_sentencas",
                "contagem_sentencas": len(sentencas_chunk),
                "tamanho_chunk": len(texto_chunk)
            }
        )
        chunks.append(chunk)

        i += sentencas_por_chunk - sobreposicao_sentencas

    return chunks


def comparar_estrategias_chunking(texto: str):
    """Compara todas as estratégias de chunking no mesmo texto."""

    print("\n   Comparando estratégias de chunking no texto de exemplo...")
    print(f"   Tamanho do texto original: {len(texto)} caracteres")

    estrategias = {
        "Caracteres Recursivo": chunking_caracteres_recursivo(texto, 500, 100),
        "Por Tokens": chunking_por_tokens(texto, 200, 50),
        "Markdown": chunking_markdown(texto, 500, 100),
        "Janela Deslizante": chunking_janela_deslizante(texto, 500, 250),
        "Por Sentenças": chunking_por_sentencas(texto, 5, 1)
    }

    print("\n   " + "=" * 50)
    print("   COMPARAÇÃO DE ESTRATÉGIAS DE CHUNKING")
    print("   " + "=" * 50)

    for nome, chunks in estrategias.items():
        tamanhos = [len(c.page_content) for c in chunks]
        media = sum(tamanhos) / len(tamanhos) if tamanhos else 0

        print(f"\n   {nome}:")
        print(f"      Número de chunks: {len(chunks)}")
        print(f"      Tamanho médio: {media:.1f} caracteres")
        print(f"      Mín/Máx: {min(tamanhos)}/{max(tamanhos)} caracteres")

    return estrategias


def analisar_qualidade_chunks(chunks: list[Document]) -> dict:
    """
    Analisa a qualidade dos chunks usando o LLM.

    Avalia coerência, completude e densidade de informação.
    """
    llm = get_llm(temperature=0)

    # Amostra alguns chunks para análise
    chunks_amostra = chunks[:3] if len(chunks) > 3 else chunks

    resultados = []

    for chunk in chunks_amostra:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em avaliar qualidade de chunks de texto para sistemas RAG.
Analise o chunk fornecido e avalie-o nestes critérios (1-10):
1. Coerência: Forma uma unidade completa e compreensível?
2. Densidade de Informação: Quanta informação útil contém?
3. Independência de Contexto: Pode ser entendido sem o texto ao redor?

Retorne um objeto JSON com pontuações e explicações breves."""),
            ("user", "Chunk para analisar:\n\n{chunk}")
        ])

        chain = prompt | llm
        response = chain.invoke({"chunk": chunk.page_content[:1000]})

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        resultados.append({
            "indice_chunk": chunk.metadata.get("indice_chunk"),
            "analise": response.content
        })

    return resultados


def demo_chunking_para_recuperacao():
    """Demonstra como diferentes chunkings afetam a recuperação."""

    print("\n   Demonstrando impacto do chunking na recuperação...")

    texto = carregar_documento_exemplo()

    # Cria chunks com diferentes estratégias
    chunks_pequenos = chunking_caracteres_recursivo(texto, 300, 50)
    chunks_medios = chunking_caracteres_recursivo(texto, 800, 150)
    chunks_grandes = chunking_caracteres_recursivo(texto, 1500, 300)

    print("\n   Comparação de tamanhos de chunk:")
    print(f"      Pequeno (300 chars): {len(chunks_pequenos)} chunks")
    print(f"      Médio (800 chars):   {len(chunks_medios)} chunks")
    print(f"      Grande (1500 chars): {len(chunks_grandes)} chunks")

    # Chunks de exemplo
    print("\n   Exemplo de chunk pequeno (primeiros 200 chars):")
    print(f"      '{chunks_pequenos[0].page_content[:200]}...'")

    print("\n   Exemplo de chunk grande (primeiros 200 chars):")
    print(f"      '{chunks_grandes[0].page_content[:200]}...'")

    print("\n   Trade-offs:")
    print("      - Chunks menores: Melhor precisão, mas pode perder contexto")
    print("      - Chunks maiores: Mais contexto, mas pode incluir info irrelevante")
    print("      - Sobreposição: Ajuda a preservar contexto entre chunks")


def main():
    print("=" * 60)
    print("ESTRATÉGIAS AVANÇADAS DE CHUNKING")
    print("=" * 60)

    token_tracker.reset()

    # Carrega documento de exemplo
    print("\n📚 CARREGANDO DOCUMENTO DE EXEMPLO")
    print("-" * 40)

    texto = carregar_documento_exemplo()
    print(f"   Documento carregado com {len(texto)} caracteres")

    # Demo 1: Comparar estratégias
    print("\n\n📊 COMPARAÇÃO DE ESTRATÉGIAS")
    print("-" * 40)

    estrategias = comparar_estrategias_chunking(texto)

    # Demo 2: Mostrar exemplos de chunks
    print("\n\n📝 EXEMPLOS DE CHUNKS")
    print("-" * 40)

    chunks_recursivo = estrategias["Caracteres Recursivo"]
    print(f"\n   Primeiros 3 chunks da estratégia Caracteres Recursivo:\n")

    for chunk in chunks_recursivo[:3]:
        preview = chunk.page_content[:150].replace('\n', ' ')
        print(f"   Chunk {chunk.metadata['indice_chunk']}:")
        print(f"   Tamanho: {chunk.metadata['tamanho_chunk']} chars")
        print(f"   Preview: {preview}...")
        print()

    # Demo 3: Analisar qualidade dos chunks
    print("\n\n🔍 ANÁLISE DE QUALIDADE DOS CHUNKS")
    print("-" * 40)

    print("\n   Analisando qualidade dos chunks usando LLM...")
    resultados_qualidade = analisar_qualidade_chunks(chunks_recursivo)

    for resultado in resultados_qualidade:
        print(f"\n   Análise do Chunk {resultado['indice_chunk']}:")
        print(f"   {resultado['analise'][:300]}...")

    # Demo 4: Impacto do chunking na recuperação
    print("\n\n📈 IMPACTO DO CHUNKING NA RECUPERAÇÃO")
    print("-" * 40)

    demo_chunking_para_recuperacao()

    # Melhores práticas
    print("\n\n💡 MELHORES PRÁTICAS DE CHUNKING")
    print("-" * 40)
    print("""
   1. Escolha o tamanho do chunk baseado no tipo de conteúdo:
      - Docs técnicos: 500-1000 chars (preserva contexto)
      - FAQs: 200-400 chars (uma P&R por chunk)
      - Artigos: 800-1500 chars (nível de parágrafo)

   2. Use sobreposição apropriada:
      - 10-20% do tamanho do chunk é típico
      - Mais sobreposição para conteúdo altamente conectado
      - Menos para seções distintas e independentes

   3. Combine estratégia com estrutura do documento:
      - Markdown/HTML: Use splitters conscientes de estrutura
      - Texto puro: Divisão recursiva ou semântica
      - Código: Use splitters específicos por linguagem

   4. Considere as necessidades de recuperação:
      - Busca semântica: Chunks maiores e coerentes
      - Busca por palavra-chave: Chunks menores e focados
      - Híbrido: Chunks médios com boa sobreposição
    """)

    print_total_usage(token_tracker, "TOTAL - Chunking Avançado")

    print("\nFim da demonstração de Chunking Avançado")
    print("=" * 60)


if __name__ == "__main__":
    main()
