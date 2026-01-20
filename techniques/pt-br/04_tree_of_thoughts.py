"""
Tree of Thoughts (ToT)

Técnica que explora múltiplos caminhos de raciocínio em paralelo,
avalia cada um e seleciona o mais promissor. Útil para problemas
que requerem exploração e backtracking.

Casos de uso:
- Planejamento estratégico
- Problemas com múltiplas soluções possíveis
- Jogos e puzzles
- Otimização de decisões
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


def gerar_pensamentos(problema: str, num_pensamentos: int = 3) -> list[str]:
    """Gera múltiplos caminhos de raciocínio inicial."""
    llm = get_llm(temperature=0.8)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um solucionador criativo de problemas.
Dado um problema, gere {num_pensamentos} abordagens DIFERENTES e INDEPENDENTES para resolvê-lo.

Cada abordagem deve:
- Ser única e distinta das outras
- Ter uma lógica clara
- Ser um primeiro passo viável para a solução

Formato de saída (use exatamente este formato):
ABORDAGEM 1: [descrição da primeira abordagem]
ABORDAGEM 2: [descrição da segunda abordagem]
ABORDAGEM 3: [descrição da terceira abordagem]"""),
        ("user", "{problema}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problema": problema, "num_pensamentos": num_pensamentos})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "gerar_pensamentos")

    resultado = response.content

    # Parse das abordagens
    linhas = resultado.strip().split("\n")
    abordagens = []
    abordagem_atual = ""

    for linha in linhas:
        if linha.startswith("ABORDAGEM"):
            if abordagem_atual:
                abordagens.append(abordagem_atual.strip())
            abordagem_atual = linha.split(":", 1)[1] if ":" in linha else linha
        else:
            abordagem_atual += " " + linha

    if abordagem_atual:
        abordagens.append(abordagem_atual.strip())

    return abordagens[:num_pensamentos]


def avaliar_pensamento(problema: str, pensamento: str) -> dict:
    """Avalia a qualidade e viabilidade de um caminho de raciocínio."""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um avaliador crítico de soluções.
Avalie a abordagem proposta para resolver o problema.

Critérios de avaliação (0-10 cada):
1. VIABILIDADE: A abordagem é realizável?
2. EFICIÊNCIA: A abordagem é eficiente?
3. COMPLETUDE: A abordagem resolve o problema completamente?
4. CRIATIVIDADE: A abordagem é inovadora?

Formato de resposta:
VIABILIDADE: [0-10]
EFICIÊNCIA: [0-10]
COMPLETUDE: [0-10]
CRIATIVIDADE: [0-10]
TOTAL: [soma/40]
JUSTIFICATIVA: [breve explicação]
PRÓXIMO_PASSO: [sugestão de como continuar esta abordagem]"""),
        ("user", "PROBLEMA: {problema}\n\nABORDAGEM PROPOSTA: {pensamento}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problema": problema, "pensamento": pensamento})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "avaliar_pensamento")

    resultado = response.content

    # Parse básico para extrair a pontuação total
    linhas = resultado.strip().split("\n")
    avaliacao = {
        "texto_completo": resultado,
        "pontuacao": 0,
        "proximo_passo": ""
    }

    for linha in linhas:
        if linha.startswith("TOTAL:"):
            try:
                # Extrai número do formato "XX/40" ou similar
                valor = linha.split(":")[1].strip().split("/")[0]
                avaliacao["pontuacao"] = float(valor)
            except (ValueError, IndexError):
                avaliacao["pontuacao"] = 0
        elif linha.startswith("PRÓXIMO_PASSO:"):
            avaliacao["proximo_passo"] = linha.split(":", 1)[1].strip()

    return avaliacao


def expandir_pensamento(problema: str, pensamento: str, proximo_passo: str) -> str:
    """Expande um pensamento promissor para o próximo nível."""
    llm = get_llm(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um solucionador metódico de problemas.
Continue desenvolvendo a abordagem proposta, seguindo a sugestão de próximo passo.

Desenvolva o raciocínio de forma detalhada, mostrando:
1. Como implementar o próximo passo
2. Possíveis obstáculos e como superá-los
3. Resultados esperados desta etapa"""),
        ("user", """PROBLEMA: {problema}

ABORDAGEM ATUAL: {pensamento}

PRÓXIMO PASSO SUGERIDO: {proximo_passo}

Continue o desenvolvimento:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "problema": problema,
        "pensamento": pensamento,
        "proximo_passo": proximo_passo
    })

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "expandir_pensamento")

    return response.content


def sintetizar_solucao(problema: str, melhor_caminho: list[str]) -> str:
    """Sintetiza a solução final a partir do melhor caminho encontrado."""
    llm = get_llm(temperature=0.3)

    caminho_texto = "\n\n".join([f"ETAPA {i+1}:\n{etapa}" for i, etapa in enumerate(melhor_caminho)])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em síntese de soluções.
Com base no caminho de raciocínio desenvolvido, apresente a solução final de forma clara e estruturada.

A solução deve incluir:
1. RESUMO: Visão geral da solução
2. PASSOS DE IMPLEMENTAÇÃO: Lista ordenada de ações
3. RECURSOS NECESSÁRIOS: O que é preciso para implementar
4. RESULTADO ESPERADO: O que será alcançado
5. RISCOS E MITIGAÇÕES: Possíveis problemas e como evitá-los"""),
        ("user", "PROBLEMA: {problema}\n\nCAMINHO DE RACIOCÍNIO:\n{caminho}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problema": problema, "caminho": caminho_texto})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "sintetizar_solucao")

    return response.content


def tree_of_thoughts(problema: str, profundidade: int = 2) -> str:
    """
    Implementa o algoritmo Tree of Thoughts completo.

    Args:
        problema: O problema a ser resolvido
        profundidade: Quantos níveis de expansão explorar

    Returns:
        A solução sintetizada
    """
    print(f"\n🌳 Iniciando Tree of Thoughts (profundidade={profundidade})")
    print("=" * 50)

    # Fase 1: Gerar pensamentos iniciais
    print("\n📝 Fase 1: Gerando abordagens iniciais...")
    pensamentos = gerar_pensamentos(problema, num_pensamentos=3)

    for i, p in enumerate(pensamentos, 1):
        print(f"\n  Abordagem {i}: {p[:100]}...")

    # Fase 2: Avaliar e selecionar o melhor
    print("\n⚖️ Fase 2: Avaliando abordagens...")
    avaliacoes = []

    for i, pensamento in enumerate(pensamentos, 1):
        avaliacao = avaliar_pensamento(problema, pensamento)
        avaliacoes.append({
            "pensamento": pensamento,
            "avaliacao": avaliacao
        })
        print(f"\n  Abordagem {i}: Pontuação = {avaliacao['pontuacao']}/40")

    # Ordenar por pontuação e selecionar o melhor
    avaliacoes.sort(key=lambda x: x["avaliacao"]["pontuacao"], reverse=True)
    melhor = avaliacoes[0]

    print(f"\n✅ Melhor abordagem selecionada (pontuação: {melhor['avaliacao']['pontuacao']}/40)")

    # Fase 3: Expandir o melhor caminho
    caminho = [melhor["pensamento"]]
    pensamento_atual = melhor["pensamento"]
    proximo_passo = melhor["avaliacao"]["proximo_passo"]

    for nivel in range(profundidade):
        print(f"\n🔍 Fase 3.{nivel+1}: Expandindo nível {nivel+1}...")

        expansao = expandir_pensamento(problema, pensamento_atual, proximo_passo)
        caminho.append(expansao)

        # Avaliar a expansão para obter próximo passo
        avaliacao = avaliar_pensamento(problema, expansao)
        proximo_passo = avaliacao["proximo_passo"]
        pensamento_atual = expansao

        print(f"  Expansão concluída (pontuação: {avaliacao['pontuacao']}/40)")

    # Fase 4: Sintetizar solução
    print("\n📋 Fase 4: Sintetizando solução final...")
    solucao = sintetizar_solucao(problema, caminho)

    return solucao


def main():
    print("=" * 60)
    print("TREE OF THOUGHTS (ToT) - Demonstração")
    print("=" * 60)

    # Reset do tracker
    token_tracker.reset()

    # Problema complexo para demonstração
    problema = """
    Uma startup de tecnologia com 15 funcionários precisa decidir
    como expandir suas operações. Atualmente opera apenas no Brasil,
    tem um produto SaaS com 500 clientes pagantes e faturamento de
    R$ 200 mil/mês. O objetivo é triplicar o faturamento em 18 meses.

    Restrições:
    - Budget disponível para investimento: R$ 1 milhão
    - Time técnico já está no limite da capacidade
    - Produto ainda tem débito técnico significativo

    Qual a melhor estratégia de crescimento?
    """

    print(f"\n📌 PROBLEMA:\n{problema.strip()}")

    # Executar Tree of Thoughts
    solucao = tree_of_thoughts(problema, profundidade=2)

    print("\n" + "=" * 60)
    print("🎯 SOLUÇÃO FINAL")
    print("=" * 60)
    print(solucao)

    # Exibir total de tokens
    print_total_usage(token_tracker, "TOTAL - Tree of Thoughts")

    print("\nFim da demonstração Tree of Thoughts")
    print("=" * 60)


if __name__ == "__main__":
    main()
