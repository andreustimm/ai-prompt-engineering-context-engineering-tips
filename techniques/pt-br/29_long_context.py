"""
Estratégias de Contexto Longo

Técnicas para processar documentos que excedem janelas de contexto típicas.
Diferentes estratégias equilibram qualidade, custo e completude.

Estratégias implementadas:
1. Map-Reduce: Processa chunks separadamente, combina resultados
2. Refine: Constrói resposta iterativamente com cada chunk
3. Map-Rerank: Pontua cada chunk, usa melhores
4. Stuffing com priorização: Encaixa conteúdo mais relevante

Casos de uso:
- Processar documentos longos (50+ páginas)
- Resumir documentos grandes
- Responder perguntas sobre livros inteiros
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

token_tracker = TokenUsage()


def carregar_documento_longo() -> str:
    """Carrega documento de exemplo longo."""
    doc_path = Path(__file__).parent.parent.parent / "sample_data" / "documents" / "long_document.txt"
    if doc_path.exists():
        with open(doc_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Conteúdo de documento longo de exemplo. " * 500


def dividir_texto(texto: str, tamanho_chunk: int = 2000, sobreposicao: int = 200) -> list[str]:
    """Divide texto em chunks."""
    divisor = RecursiveCharacterTextSplitter(chunk_size=tamanho_chunk, chunk_overlap=sobreposicao)
    docs = divisor.create_documents([texto])
    return [d.page_content for d in docs]


def sumarizar_map_reduce(chunks: list[str]) -> str:
    """
    Map-Reduce: Resume cada chunk, depois combina resumos.

    Bom para: Cobertura completa de todo o documento
    Trade-off: Múltiplas chamadas LLM, pode perder nuances
    """
    llm = get_llm(temperature=0.3)

    # Map: Resume cada chunk
    resumos = []
    prompt_map = ChatPromptTemplate.from_messages([
        ("system", "Resuma os pontos chave desta seção em 2-3 frases."),
        ("user", "{chunk}")
    ])
    chain_map = prompt_map | llm

    print("   Mapeando (resumindo chunks)...")
    for i, chunk in enumerate(chunks[:5]):  # Limite para demo
        response = chain_map.invoke({"chunk": chunk})
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        resumos.append(response.content)
        print(f"      Chunk {i+1}/{min(5, len(chunks))} feito")

    # Reduce: Combina resumos
    print("   Reduzindo (combinando resumos)...")
    prompt_reduce = ChatPromptTemplate.from_messages([
        ("system", "Combine estes resumos em uma visão geral coerente. Seja abrangente mas conciso."),
        ("user", "Resumos para combinar:\n\n{resumos}")
    ])
    chain_reduce = prompt_reduce | llm

    response = chain_reduce.invoke({"resumos": "\n\n".join(resumos)})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Reduce")

    return response.content


def sumarizar_refine(chunks: list[str]) -> str:
    """
    Refine: Constrói resumo iterativamente, refinando com cada chunk.

    Bom para: Manter coerência, construir sobre contexto
    Trade-off: Processamento sequencial, mais lento
    """
    llm = get_llm(temperature=0.3)

    # Resumo inicial do primeiro chunk
    prompt_inicial = ChatPromptTemplate.from_messages([
        ("system", "Resuma os pontos chave deste texto."),
        ("user", "{chunk}")
    ])

    print("   Resumo inicial do primeiro chunk...")
    response = (prompt_inicial | llm).invoke({"chunk": chunks[0]})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    resumo_atual = response.content

    # Refina com chunks subsequentes
    prompt_refine = ChatPromptTemplate.from_messages([
        ("system", """Você tem um resumo existente:
{existente}

Refine-o incorporando informações relevantes do novo texto. Mantenha abrangente mas conciso."""),
        ("user", "Novo texto:\n{chunk}")
    ])
    chain_refine = prompt_refine | llm

    print("   Refinando com chunks adicionais...")
    for i, chunk in enumerate(chunks[1:4]):  # Limite para demo
        response = chain_refine.invoke({"existente": resumo_atual, "chunk": chunk})
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        resumo_atual = response.content
        print(f"      Chunk {i+2} incorporado")

    print_token_usage(input_tokens, output_tokens, "Refine Final")
    return resumo_atual


def responder_map_rerank(chunks: list[str], pergunta: str) -> str:
    """
    Map-Rerank: Pontua cada chunk por relevância, responde usando melhores.

    Bom para: Perguntas e respostas em documentos longos
    Trade-off: Etapa extra de pontuação, mas encontra conteúdo mais relevante
    """
    llm = get_llm(temperature=0)

    # Pontua cada chunk
    prompt_score = ChatPromptTemplate.from_messages([
        ("system", """Pontue quão relevante este texto é para responder a pergunta.
Retorne APENAS um número de 0-10."""),
        ("user", "Pergunta: {pergunta}\n\nTexto: {chunk}\n\nScore de relevância:")
    ])
    chain_score = prompt_score | llm

    print("   Pontuando chunks por relevância...")
    chunks_pontuados = []
    for i, chunk in enumerate(chunks[:6]):
        response = chain_score.invoke({"pergunta": pergunta, "chunk": chunk[:1500]})
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        try:
            score = float(response.content.strip())
        except:
            score = 5.0
        chunks_pontuados.append((chunk, score))
        print(f"      Chunk {i+1}: score {score}")

    # Seleciona top chunks
    chunks_pontuados.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, s in chunks_pontuados[:3]]

    # Gera resposta dos top chunks
    print("   Gerando resposta dos top chunks...")
    prompt_resposta = ChatPromptTemplate.from_messages([
        ("system", "Responda a pergunta baseado no contexto fornecido."),
        ("user", "Contexto:\n{contexto}\n\nPergunta: {pergunta}\n\nResposta:")
    ])

    contexto = "\n\n---\n\n".join(top_chunks)
    response = (prompt_resposta | llm).invoke({"contexto": contexto, "pergunta": pergunta})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Geração de Resposta")

    return response.content


def stuffing_com_priorizacao(chunks: list[str], pergunta: str, max_contexto: int = 6000) -> str:
    """
    Stuffing: Encaixa o máximo de conteúdo relevante possível no contexto.

    Bom para: Abordagem simples quando conteúdo cabe
    Trade-off: Limitado pela janela de contexto
    """
    llm = get_llm(temperature=0.3)

    # Pontuação simples de relevância (sobreposição de palavras-chave)
    termos_pergunta = set(pergunta.lower().split())

    pontuados = []
    for chunk in chunks:
        termos_chunk = set(chunk.lower().split())
        sobreposicao = len(termos_pergunta & termos_chunk)
        pontuados.append((chunk, sobreposicao))

    pontuados.sort(key=lambda x: x[1], reverse=True)

    # Encaixa chunks até max contexto
    partes_contexto = []
    total_len = 0

    for chunk, _ in pontuados:
        if total_len + len(chunk) > max_contexto:
            break
        partes_contexto.append(chunk)
        total_len += len(chunk)

    print(f"   Encaixados {len(partes_contexto)} chunks no contexto")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda baseado no contexto fornecido."),
        ("user", "Contexto:\n{contexto}\n\nPergunta: {pergunta}\n\nResposta:")
    ])

    response = (prompt | llm).invoke({
        "contexto": "\n\n".join(partes_contexto),
        "pergunta": pergunta
    })
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Resposta Stuffing")

    return response.content


def main():
    print("=" * 60)
    print("ESTRATÉGIAS DE CONTEXTO LONGO")
    print("=" * 60)

    token_tracker.reset()

    print("\n📚 CARREGANDO DOCUMENTO LONGO")
    print("-" * 40)
    texto = carregar_documento_longo()
    print(f"   Tamanho do documento: {len(texto):,} caracteres")

    chunks = dividir_texto(texto, tamanho_chunk=2000, sobreposicao=200)
    print(f"   Criados {len(chunks)} chunks")

    # Estratégia 1: Map-Reduce
    print("\n\n📊 SUMARIZAÇÃO MAP-REDUCE")
    print("=" * 60)
    resultado_map_reduce = sumarizar_map_reduce(chunks)
    print(f"\n   Resultado:\n   {resultado_map_reduce[:400]}...")

    # Estratégia 2: Refine
    print("\n\n🔄 SUMARIZAÇÃO REFINE")
    print("=" * 60)
    resultado_refine = sumarizar_refine(chunks)
    print(f"\n   Resultado:\n   {resultado_refine[:400]}...")

    # Estratégia 3: Map-Rerank
    print("\n\n🎯 Q&A MAP-RERANK")
    print("=" * 60)
    pergunta = "Quais são os princípios chave de arquitetura de software?"
    print(f"   Pergunta: '{pergunta}'")
    resultado_rerank = responder_map_rerank(chunks, pergunta)
    print(f"\n   Resposta:\n   {resultado_rerank[:400]}...")

    # Estratégia 4: Stuffing
    print("\n\n📦 STUFFING COM PRIORIZAÇÃO")
    print("=" * 60)
    resultado_stuffing = stuffing_com_priorizacao(chunks, pergunta)
    print(f"\n   Resposta:\n   {resultado_stuffing[:400]}...")

    print("\n\n💡 COMPARAÇÃO DE ESTRATÉGIAS")
    print("-" * 40)
    print("""
   | Estratégia  | Melhor Para        | Prós              | Contras           |
   |-------------|-------------------|-------------------|-------------------|
   | Map-Reduce  | Sumarização total | Cobertura completa| Múltiplas chamadas|
   | Refine      | Resumos coerentes | Constrói contexto | Sequencial, lento |
   | Map-Rerank  | Q&A em docs longos| Acha melhor conteúdo| Etapa extra     |
   | Stuffing    | Tarefas simples   | Mínimas chamadas  | Limitado por janela|
    """)

    print_total_usage(token_tracker, "TOTAL - Estratégias de Contexto Longo")
    print("\nFim da demonstração de Estratégias de Contexto Longo")
    print("=" * 60)


if __name__ == "__main__":
    main()
