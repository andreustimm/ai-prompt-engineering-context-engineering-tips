"""
Least-to-Most Prompting (Do Menor para o Maior)

Técnica que decompõe problemas complexos em sub-problemas mais simples,
resolvendo-os progressivamente do mais fácil ao mais difícil, usando
resultados anteriores para informar soluções subsequentes.

Casos de uso:
- Problemas matemáticos de múltiplas etapas
- Tarefas de raciocínio complexo
- Planejamento e cronogramas
- Explicações educacionais
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

# Rastreador global de tokens para este script
token_tracker = TokenUsage()


def decompor_problema(problema: str) -> list[str]:
    """Decompõe um problema complexo em sub-problemas mais simples."""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em dividir problemas complexos em sub-problemas mais simples.
Dado um problema, identifique os sub-problemas componentes que precisam ser resolvidos.
Ordene-os do mais simples ao mais complexo.
Retorne APENAS uma lista numerada de sub-problemas, nada mais.

Formato:
1. [sub-problema mais simples]
2. [próximo sub-problema]
3. [próximo sub-problema]
... e assim por diante"""),
        ("user", "Problema: {problema}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problema": problema})

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Decomposição")

    # Analisa a lista numerada
    linhas = response.content.strip().split('\n')
    sub_problemas = []
    for linha in linhas:
        linha = linha.strip()
        if linha and linha[0].isdigit():
            # Remove o número e o ponto
            partes = linha.split('.', 1)
            if len(partes) > 1:
                sub_problemas.append(partes[1].strip())

    return sub_problemas


def resolver_sub_problema(sub_problema: str, contexto_anterior: str = "") -> str:
    """Resolve um único sub-problema, opcionalmente usando contexto anterior."""
    llm = get_llm(temperature=0.2)

    if contexto_anterior:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em resolução de problemas.
Resolva o sub-problema dado usando o contexto fornecido das soluções anteriores.
Seja conciso mas completo. Mostre seu trabalho."""),
            ("user", """Soluções anteriores:
{contexto}

Sub-problema atual a resolver: {sub_problema}

Solução:""")
        ])
        inputs = {"sub_problema": sub_problema, "contexto": contexto_anterior}
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em resolução de problemas.
Resolva o sub-problema dado. Seja conciso mas completo. Mostre seu trabalho."""),
            ("user", """Sub-problema: {sub_problema}

Solução:""")
        ])
        inputs = {"sub_problema": sub_problema}

    chain = prompt | llm
    response = chain.invoke(inputs)

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Sub-problema")

    return response.content


def sintetizar_resposta_final(problema: str, solucoes: list[tuple[str, str]]) -> str:
    """Sintetiza todas as soluções dos sub-problemas em uma resposta final."""
    llm = get_llm(temperature=0.2)

    texto_solucoes = "\n\n".join([
        f"Sub-problema: {sp}\nSolução: {sol}"
        for sp, sol in solucoes
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em sintetizar soluções.
Dado o problema original e as soluções de seus sub-problemas,
forneça uma resposta final clara e abrangente."""),
        ("user", """Problema Original: {problema}

Soluções dos Sub-problemas:
{solucoes}

Por favor, forneça a resposta final e completa para o problema original:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"problema": problema, "solucoes": texto_solucoes})

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Síntese")

    return response.content


def resolver_least_to_most(problema: str) -> dict:
    """
    Resolve um problema complexo usando prompting do menor para o maior.

    Retorna:
        Dicionário com decomposição, sub-soluções e resposta final
    """
    print("\n   Passo 1: Decompondo problema...")
    sub_problemas = decompor_problema(problema)

    print(f"\n   Encontrados {len(sub_problemas)} sub-problemas:")
    for i, sp in enumerate(sub_problemas, 1):
        print(f"      {i}. {sp}")

    print("\n   Passo 2: Resolvendo sub-problemas progressivamente...")
    solucoes = []
    contexto = ""

    for i, sub_problema in enumerate(sub_problemas, 1):
        print(f"\n      Resolvendo sub-problema {i}/{len(sub_problemas)}...")
        solucao = resolver_sub_problema(sub_problema, contexto)
        solucoes.append((sub_problema, solucao))

        # Atualiza contexto com esta solução
        contexto += f"\nSub-problema: {sub_problema}\nSolução: {solucao}\n"

    print("\n   Passo 3: Sintetizando resposta final...")
    resposta_final = sintetizar_resposta_final(problema, solucoes)

    return {
        "sub_problemas": sub_problemas,
        "solucoes": solucoes,
        "resposta_final": resposta_final
    }


def resolver_problema_matematico(problema: str) -> dict:
    """Resolve um problema matemático usando abordagem do menor para o maior."""
    return resolver_least_to_most(problema)


def criar_trilha_aprendizado(topico: str, nivel_atual: str) -> dict:
    """Cria uma trilha de aprendizado progressiva para um tópico."""
    problema = f"Crie uma trilha de aprendizado para alguém no nível {nivel_atual} dominar {topico}"
    return resolver_least_to_most(problema)


def planejar_projeto(descricao_projeto: str) -> dict:
    """Planeja um projeto dividindo-o em marcos progressivos."""
    problema = f"Planeje a implementação deste projeto: {descricao_projeto}"
    return resolver_least_to_most(problema)


def explicar_conceito(conceito: str) -> dict:
    """Explica um conceito complexo construindo a partir dos fundamentos."""
    problema = f"Explique {conceito} começando dos conceitos básicos e construindo até o entendimento completo"
    return resolver_least_to_most(problema)


def main():
    print("=" * 60)
    print("LEAST-TO-MOST PROMPTING (DO MENOR PARA O MAIOR) - Demo")
    print("=" * 60)

    # Reseta rastreador
    token_tracker.reset()

    # Exemplo 1: Problema Matemático Complexo
    print("\n🔢 PROBLEMA MATEMÁTICO COMPLEXO")
    print("-" * 40)

    problema_matematica = """
    Um trem sai da Cidade A às 9:00 viajando a 60 km/h em direção à Cidade B.
    Outro trem sai da Cidade B às 10:00 viajando a 80 km/h em direção à Cidade A.
    As cidades estão a 280 km de distância.
    A que horas os trens se encontrarão, e a que distância da Cidade A?
    """

    print(f"\nProblema: {problema_matematica.strip()}")
    resultado = resolver_problema_matematico(problema_matematica)

    print(f"\n📋 RESPOSTA FINAL:")
    print("-" * 40)
    print(resultado["resposta_final"])

    # Exemplo 2: Trilha de Aprendizado
    print("\n\n📚 CRIAÇÃO DE TRILHA DE APRENDIZADO")
    print("-" * 40)

    topico = "Machine Learning"
    nivel = "iniciante com conhecimento básico de Python"

    print(f"\nTópico: {topico}")
    print(f"Nível Atual: {nivel}")

    resultado = criar_trilha_aprendizado(topico, nivel)

    print(f"\n📋 TRILHA DE APRENDIZADO:")
    print("-" * 40)
    print(resultado["resposta_final"])

    # Exemplo 3: Planejamento de Projeto
    print("\n\n🏗️ PLANEJAMENTO DE PROJETO")
    print("-" * 40)

    projeto = "Construir uma API REST para uma aplicação de tarefas com autenticação de usuários, operações CRUD e persistência em banco de dados"

    print(f"\nProjeto: {projeto}")

    resultado = planejar_projeto(projeto)

    print(f"\n📋 PLANO DO PROJETO:")
    print("-" * 40)
    print(resultado["resposta_final"])

    # Exemplo 4: Explicação de Conceito
    print("\n\n💡 EXPLICAÇÃO DE CONCEITO")
    print("-" * 40)

    conceito = "como redes neurais aprendem através da retropropagação"

    print(f"\nConceito: {conceito}")

    resultado = explicar_conceito(conceito)

    print(f"\n📋 EXPLICAÇÃO:")
    print("-" * 40)
    print(resultado["resposta_final"])

    # Exibe total de tokens
    print_total_usage(token_tracker, "TOTAL - Least-to-Most Prompting")

    print("\nFim do demo de Least-to-Most")
    print("=" * 60)


if __name__ == "__main__":
    main()
