"""
Zero-Shot Prompting

Técnica onde o modelo recebe uma tarefa sem exemplos prévios.
O modelo usa apenas seu conhecimento pré-treinado para responder.

Casos de uso:
- Classificação de sentimento
- Tradução de texto
- Extração de entidades
- Perguntas e respostas simples
"""

import sys
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


def classificar_sentimento(texto: str) -> str:
    """Classifica o sentimento de um texto sem exemplos."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um analisador de sentimentos. Classifique o sentimento do texto como: POSITIVO, NEGATIVO ou NEUTRO. Responda apenas com a classificação."),
        ("user", "{texto}")
    ])

    chain = prompt | llm
    response = chain.invoke({"texto": texto})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def traduzir_texto(texto: str, idioma_destino: str) -> str:
    """Traduz texto para o idioma especificado."""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um tradutor profissional. Traduza o texto para {idioma}. Retorne apenas a tradução, sem explicações."),
        ("user", "{texto}")
    ])

    chain = prompt | llm
    response = chain.invoke({"texto": texto, "idioma": idioma_destino})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def extrair_entidades(texto: str) -> str:
    """Extrai entidades nomeadas de um texto."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um extrator de entidades nomeadas.
Extraia e liste as seguintes entidades do texto:
- PESSOA: nomes de pessoas
- LOCAL: lugares, cidades, países
- ORGANIZAÇÃO: empresas, instituições
- DATA: datas e períodos

Formato de saída:
PESSOA: [lista]
LOCAL: [lista]
ORGANIZAÇÃO: [lista]
DATA: [lista]"""),
        ("user", "{texto}")
    ])

    chain = prompt | llm
    response = chain.invoke({"texto": texto})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def resumir_texto(texto: str) -> str:
    """Resume um texto em poucas frases."""
    llm = get_llm(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em resumos. Resuma o texto a seguir em no máximo 3 frases, mantendo as informações mais importantes."),
        ("user", "{texto}")
    ])

    chain = prompt | llm
    response = chain.invoke({"texto": texto})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def main():
    print("=" * 60)
    print("ZERO-SHOT PROMPTING - Demonstração")
    print("=" * 60)

    # Reset do tracker
    token_tracker.reset()

    # Exemplo 1: Classificação de Sentimento
    print("\n📊 CLASSIFICAÇÃO DE SENTIMENTO")
    print("-" * 40)

    textos_sentimento = [
        "Este produto é incrível! Superou todas as minhas expectativas.",
        "Péssimo atendimento, nunca mais volto nessa loja.",
        "O pacote chegou hoje às 14h conforme previsto."
    ]

    for texto in textos_sentimento:
        print(f"\nTexto: {texto[:50]}...")
        resultado = classificar_sentimento(texto)
        print(f"Sentimento: {resultado}")

    # Exemplo 2: Tradução
    print("\n\n🌍 TRADUÇÃO")
    print("-" * 40)

    texto_original = "Artificial intelligence is transforming how we work and live."
    print(f"\nOriginal: {texto_original}")
    traducao = traduzir_texto(texto_original, "português brasileiro")
    print(f"Tradução: {traducao}")

    # Exemplo 3: Extração de Entidades
    print("\n\n🏷️ EXTRAÇÃO DE ENTIDADES")
    print("-" * 40)

    texto_entidades = """
    Em março de 2024, Elon Musk anunciou que a Tesla inaugurará uma nova
    fábrica em São Paulo. A parceria com o Governo do Estado prevê
    investimentos de R$ 5 bilhões nos próximos 3 anos.
    """

    print(f"\nTexto: {texto_entidades.strip()}")
    entidades = extrair_entidades(texto_entidades)
    print(f"\nEntidades extraídas:\n{entidades}")

    # Exemplo 4: Resumo
    print("\n\n📝 RESUMO DE TEXTO")
    print("-" * 40)

    texto_longo = """
    A inteligência artificial generativa está revolucionando diversos setores
    da economia global. Empresas de tecnologia investem bilhões em pesquisa
    e desenvolvimento de modelos de linguagem cada vez mais sofisticados.
    Ferramentas como ChatGPT, Claude e Gemini permitem que usuários comuns
    realizem tarefas complexas de escrita, programação e análise de dados.
    No entanto, especialistas alertam para os riscos associados ao uso
    irresponsável dessas tecnologias, incluindo a disseminação de
    desinformação e preocupações com privacidade de dados.
    """

    print(f"\nTexto original: {texto_longo.strip()}")
    resumo = resumir_texto(texto_longo)
    print(f"\nResumo:\n{resumo}")

    # Exibir total de tokens
    print_total_usage(token_tracker, "TOTAL - Zero-Shot Prompting")

    print("\nFim da demonstração Zero-Shot")
    print("=" * 60)


if __name__ == "__main__":
    main()
