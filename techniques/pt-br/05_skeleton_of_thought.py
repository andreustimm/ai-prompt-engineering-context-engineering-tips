"""
Skeleton of Thought (SoT)

Técnica que primeiro gera um "esqueleto" da resposta (estrutura/tópicos)
e depois expande cada parte. Permite paralelização e respostas mais
organizadas.

Casos de uso:
- Geração de conteúdo longo
- Respostas estruturadas
- Artigos e documentação
- Análises detalhadas
"""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Tracker global de tokens para este script
token_tracker = TokenUsage()


def gerar_esqueleto(tema: str, contexto: str = "") -> list[str]:
    """
    Fase 1: Gera o esqueleto (lista de tópicos) para um tema.

    Args:
        tema: O tema principal a ser desenvolvido
        contexto: Contexto adicional opcional

    Returns:
        Lista de tópicos que formam o esqueleto
    """
    llm = get_llm(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em estruturação de conteúdo.
Dado um tema, crie um esqueleto (outline) com os principais tópicos a serem abordados.

Regras:
- Liste entre 4 e 7 tópicos principais
- Cada tópico deve ser claro e focado
- Os tópicos devem seguir uma ordem lógica
- Use títulos curtos e descritivos

Formato de saída (um tópico por linha):
1. [Primeiro tópico]
2. [Segundo tópico]
3. [Terceiro tópico]
..."""),
        ("user", "TEMA: {tema}\n{contexto_texto}")
    ])

    contexto_texto = f"CONTEXTO: {contexto}" if contexto else ""

    chain = prompt | llm
    response = chain.invoke({"tema": tema, "contexto_texto": contexto_texto})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "gerar_esqueleto")

    resultado = response.content

    # Parse dos tópicos
    linhas = resultado.strip().split("\n")
    topicos = []

    for linha in linhas:
        # Remove numeração e limpa
        linha = linha.strip()
        if linha and linha[0].isdigit():
            # Remove "1.", "2.", etc.
            partes = linha.split(".", 1)
            if len(partes) > 1:
                topico = partes[1].strip()
                if topico:
                    topicos.append(topico)
        elif linha and not linha[0].isdigit():
            # Linha sem número, mas pode ser tópico
            if linha.startswith("-"):
                topicos.append(linha[1:].strip())

    return topicos


def expandir_topico(tema: str, topico: str, contexto: str = "") -> str:
    """
    Fase 2: Expande um tópico específico do esqueleto.

    Args:
        tema: O tema principal (para contexto)
        topico: O tópico a ser expandido
        contexto: Contexto adicional opcional

    Returns:
        Texto expandido do tópico
    """
    llm = get_llm(temperature=0.6)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um escritor especializado em criar conteúdo informativo.
Expanda o tópico fornecido de forma clara e detalhada.

Regras:
- Escreva 2-3 parágrafos sobre o tópico
- Seja informativo e preciso
- Use exemplos quando apropriado
- Mantenha o foco no tema principal
- Não repita informações do título do tópico"""),
        ("user", "TEMA PRINCIPAL: {tema}\n\nTÓPICO A EXPANDIR: {topico}\n{contexto_texto}")
    ])

    contexto_texto = f"\nCONTEXTO ADICIONAL: {contexto}" if contexto else ""

    chain = prompt | llm
    response = chain.invoke({
        "tema": tema,
        "topico": topico,
        "contexto_texto": contexto_texto
    })

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, f"expandir: {topico[:20]}...")

    return response.content


async def expandir_topico_async(llm, tema: str, topico: str, contexto: str = "") -> dict:
    """Versão assíncrona da expansão de tópico para paralelização."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um escritor especializado em criar conteúdo informativo.
Expanda o tópico fornecido de forma clara e detalhada.

Regras:
- Escreva 2-3 parágrafos sobre o tópico
- Seja informativo e preciso
- Use exemplos quando apropriado
- Mantenha o foco no tema principal"""),
        ("user", "TEMA PRINCIPAL: {tema}\n\nTÓPICO A EXPANDIR: {topico}")
    ])

    chain = prompt | llm

    # Executa em thread pool para não bloquear
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: chain.invoke({"tema": tema, "topico": topico})
    )

    # Extrair tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)

    return {
        "topico": topico,
        "conteudo": response.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


async def skeleton_of_thought_async(tema: str, contexto: str = "") -> str:
    """
    Implementação assíncrona do Skeleton of Thought com expansão paralela.

    Args:
        tema: O tema a ser desenvolvido
        contexto: Contexto adicional opcional

    Returns:
        Texto completo gerado
    """
    print("\n🦴 Iniciando Skeleton of Thought (Async)")
    print("=" * 50)

    # Fase 1: Gerar esqueleto
    print("\n📋 Fase 1: Gerando esqueleto...")
    topicos = gerar_esqueleto(tema, contexto)

    print(f"  Esqueleto gerado com {len(topicos)} tópicos:")
    for i, t in enumerate(topicos, 1):
        print(f"    {i}. {t}")

    # Fase 2: Expandir tópicos em paralelo
    print("\n✍️ Fase 2: Expandindo tópicos em paralelo...")

    llm = get_llm(temperature=0.6)

    # Criar tarefas assíncronas para cada tópico
    tarefas = [
        expandir_topico_async(llm, tema, topico, contexto)
        for topico in topicos
    ]

    # Executar todas em paralelo
    resultados = await asyncio.gather(*tarefas)

    # Mostrar tokens por tópico
    for resultado in resultados:
        print_token_usage(
            resultado["input_tokens"],
            resultado["output_tokens"],
            f"expandir: {resultado['topico'][:20]}..."
        )

    # Ordenar resultados na ordem original dos tópicos
    resultados_ordenados = sorted(
        resultados,
        key=lambda x: topicos.index(x["topico"])
    )

    print(f"  {len(resultados)} seções expandidas!")

    # Fase 3: Montar documento final
    print("\n📄 Fase 3: Montando documento final...")

    documento = f"# {tema}\n\n"

    for i, resultado in enumerate(resultados_ordenados, 1):
        documento += f"## {i}. {resultado['topico']}\n\n"
        documento += f"{resultado['conteudo']}\n\n"

    return documento


def skeleton_of_thought_sync(tema: str, contexto: str = "") -> str:
    """
    Implementação síncrona do Skeleton of Thought.

    Args:
        tema: O tema a ser desenvolvido
        contexto: Contexto adicional opcional

    Returns:
        Texto completo gerado
    """
    print("\n🦴 Iniciando Skeleton of Thought (Sync)")
    print("=" * 50)

    # Fase 1: Gerar esqueleto
    print("\n📋 Fase 1: Gerando esqueleto...")
    topicos = gerar_esqueleto(tema, contexto)

    print(f"  Esqueleto gerado com {len(topicos)} tópicos:")
    for i, t in enumerate(topicos, 1):
        print(f"    {i}. {t}")

    # Fase 2: Expandir cada tópico sequencialmente
    print("\n✍️ Fase 2: Expandindo tópicos...")

    secoes = []
    for i, topico in enumerate(topicos, 1):
        print(f"  Expandindo tópico {i}/{len(topicos)}: {topico[:30]}...")
        conteudo = expandir_topico(tema, topico, contexto)
        secoes.append({"topico": topico, "conteudo": conteudo})

    # Fase 3: Montar documento final
    print("\n📄 Fase 3: Montando documento final...")

    documento = f"# {tema}\n\n"

    for i, secao in enumerate(secoes, 1):
        documento += f"## {i}. {secao['topico']}\n\n"
        documento += f"{secao['conteudo']}\n\n"

    return documento


def gerar_com_revisao(tema: str, contexto: str = "") -> str:
    """
    Skeleton of Thought com etapa adicional de revisão.

    Args:
        tema: O tema a ser desenvolvido
        contexto: Contexto adicional opcional

    Returns:
        Texto final revisado
    """
    # Gerar documento base
    documento = skeleton_of_thought_sync(tema, contexto)

    # Fase de revisão
    print("\n🔍 Fase Extra: Revisando documento...")

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um editor profissional.
Revise o documento fornecido e faça melhorias em:
- Coesão entre seções
- Clareza do texto
- Correção de repetições
- Adição de transições entre tópicos

Retorne o documento completo revisado, mantendo a estrutura de títulos."""),
        ("user", "{documento}")
    ])

    chain = prompt | llm
    response = chain.invoke({"documento": documento})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "revisao")

    return response.content


def main():
    print("=" * 60)
    print("SKELETON OF THOUGHT (SoT) - Demonstração")
    print("=" * 60)

    # Reset do tracker
    token_tracker.reset()

    # Tema para demonstração
    tema = "Inteligência Artificial na Medicina: Aplicações e Desafios"
    contexto = "Foco em aplicações práticas já em uso e desafios éticos"

    print(f"\n📌 TEMA: {tema}")
    print(f"📝 CONTEXTO: {contexto}")

    # Demonstração 1: Versão Síncrona
    print("\n" + "-" * 60)
    print("DEMONSTRAÇÃO 1: Versão Síncrona")
    print("-" * 60)

    documento_sync = skeleton_of_thought_sync(tema, contexto)

    print("\n" + "=" * 60)
    print("📄 DOCUMENTO GERADO (Versão Síncrona)")
    print("=" * 60)
    print(documento_sync)

    # Demonstração 2: Versão Assíncrona (se suportado)
    print("\n" + "-" * 60)
    print("DEMONSTRAÇÃO 2: Versão Assíncrona (Paralela)")
    print("-" * 60)

    tema2 = "Boas Práticas de Segurança em Aplicações Web"

    try:
        documento_async = asyncio.run(skeleton_of_thought_async(tema2, ""))

        print("\n" + "=" * 60)
        print("📄 DOCUMENTO GERADO (Versão Assíncrona)")
        print("=" * 60)
        print(documento_async)
    except Exception as e:
        print(f"  Erro na versão assíncrona: {e}")
        print("  Usando versão síncrona como fallback...")
        documento_async = skeleton_of_thought_sync(tema2, "")
        print(documento_async)

    # Exibir total de tokens
    print_total_usage(token_tracker, "TOTAL - Skeleton of Thought")

    print("\nFim da demonstração Skeleton of Thought")
    print("=" * 60)


if __name__ == "__main__":
    main()
