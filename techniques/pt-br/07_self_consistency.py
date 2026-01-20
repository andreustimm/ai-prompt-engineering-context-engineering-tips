"""
Self-Consistency Prompting (Auto-Consistência)

Técnica que gera múltiplas respostas com temperatura alta,
depois usa votação majoritária para selecionar a resposta mais consistente.

Casos de uso:
- Problemas matemáticos com respostas definitivas
- Questões de múltipla escolha
- Consultas baseadas em fatos
- Problemas de raciocínio lógico
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collections import Counter
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


def gerar_multiplas_respostas(prompt_template: ChatPromptTemplate, inputs: dict, n_amostras: int = 5) -> list[str]:
    """Gera múltiplas respostas com temperatura alta para diversidade."""
    llm = get_llm(temperature=0.8)  # Temperatura alta para respostas diversas
    chain = prompt_template | llm

    respostas = []
    for i in range(n_amostras):
        response = chain.invoke(inputs)

        # Extrai e registra tokens
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        respostas.append(response.content)

    return respostas


def extrair_resposta_final(resposta: str) -> str:
    """Extrai a resposta final de uma resposta."""
    # Procura padrões comuns de resposta
    linhas = resposta.strip().split('\n')

    # Tenta encontrar resposta final explícita
    for linha in reversed(linhas):
        linha_lower = linha.lower()
        if 'resposta final' in linha_lower or 'resposta:' in linha_lower or 'resultado:' in linha_lower:
            # Extrai a parte da resposta
            if ':' in linha:
                return linha.split(':', 1)[1].strip()
            return linha.strip()

    # Se não houver resposta explícita, retorna a última linha não vazia
    for linha in reversed(linhas):
        if linha.strip():
            return linha.strip()

    return resposta.strip()


def votacao_majoritaria(respostas: list[str]) -> tuple[str, int, int]:
    """
    Realiza votação majoritária em uma lista de respostas.

    Retorna:
        Tupla de (resposta_vencedora, contagem_votos, total_votos)
    """
    # Normaliza respostas para comparação
    normalizadas = [r.strip().lower() for r in respostas]
    contador = Counter(normalizadas)

    # Obtém mais comum
    mais_comum = contador.most_common(1)[0]

    # Retorna resposta original (não normalizada)
    for i, norm in enumerate(normalizadas):
        if norm == mais_comum[0]:
            return respostas[i], mais_comum[1], len(respostas)

    return respostas[0], 1, len(respostas)


def resolver_matematica_com_consistencia(problema: str, n_amostras: int = 5) -> dict:
    """Resolve um problema matemático usando auto-consistência."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em matemática. Resolva o problema dado passo a passo.
No final, declare claramente sua resposta final no formato:
Resposta Final: [sua resposta numérica]"""),
        ("user", "{problema}")
    ])

    print(f"   Gerando {n_amostras} soluções independentes...")
    respostas = gerar_multiplas_respostas(prompt, {"problema": problema}, n_amostras)

    # Extrai respostas de cada resposta
    respostas_extraidas = [extrair_resposta_final(r) for r in respostas]

    # Realiza votação majoritária
    vencedor, votos, total = votacao_majoritaria(respostas_extraidas)

    return {
        "respostas": respostas,
        "respostas_extraidas": respostas_extraidas,
        "resposta_final": vencedor,
        "confianca": f"{votos}/{total} ({votos/total*100:.0f}%)"
    }


def responder_multipla_escolha(pergunta: str, opcoes: list[str], n_amostras: int = 5) -> dict:
    """Responde uma questão de múltipla escolha usando auto-consistência."""

    texto_opcoes = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(opcoes)])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em responder questões de múltipla escolha.
Analise a pergunta cuidadosamente, considere cada opção e selecione a melhor resposta.
No final, declare claramente sua resposta final no formato:
Resposta Final: [letra]"""),
        ("user", """Pergunta: {pergunta}

Opções:
{opcoes}

Pense passo a passo e selecione a melhor resposta.""")
    ])

    print(f"   Gerando {n_amostras} análises independentes...")
    respostas = gerar_multiplas_respostas(
        prompt,
        {"pergunta": pergunta, "opcoes": texto_opcoes},
        n_amostras
    )

    # Extrai respostas
    respostas_extraidas = []
    for r in respostas:
        ans = extrair_resposta_final(r)
        # Tenta extrair apenas a letra
        for char in ans.upper():
            if char in 'ABCDEFGH':
                respostas_extraidas.append(char)
                break
        else:
            respostas_extraidas.append(ans)

    # Realiza votação majoritária
    vencedor, votos, total = votacao_majoritaria(respostas_extraidas)

    return {
        "respostas": respostas,
        "respostas_extraidas": respostas_extraidas,
        "resposta_final": vencedor,
        "confianca": f"{votos}/{total} ({votos/total*100:.0f}%)"
    }


def verificar_fato_com_consistencia(afirmacao: str, n_amostras: int = 5) -> dict:
    """Verifica uma afirmação factual usando auto-consistência."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um verificador de fatos. Analise a afirmação dada e determine se é VERDADEIRA ou FALSA.
Explique seu raciocínio, depois declare claramente seu veredito.
Resposta Final: VERDADEIRO ou FALSO"""),
        ("user", "Afirmação: {afirmacao}")
    ])

    print(f"   Gerando {n_amostras} verificações independentes...")
    respostas = gerar_multiplas_respostas(prompt, {"afirmacao": afirmacao}, n_amostras)

    # Extrai vereditos
    respostas_extraidas = []
    for r in respostas:
        ans = extrair_resposta_final(r).upper()
        if 'VERDADEIR' in ans or 'TRUE' in ans:
            respostas_extraidas.append('VERDADEIRO')
        elif 'FALS' in ans or 'FALSE' in ans:
            respostas_extraidas.append('FALSO')
        else:
            respostas_extraidas.append(ans)

    # Realiza votação majoritária
    vencedor, votos, total = votacao_majoritaria(respostas_extraidas)

    return {
        "respostas": respostas,
        "respostas_extraidas": respostas_extraidas,
        "resposta_final": vencedor,
        "confianca": f"{votos}/{total} ({votos/total*100:.0f}%)"
    }


def resolver_puzzle_logico(puzzle: str, n_amostras: int = 5) -> dict:
    """Resolve um puzzle lógico usando auto-consistência."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em resolver puzzles lógicos.
Trabalhe através do puzzle sistematicamente, considerando todas as restrições.
No final, declare claramente sua solução.
Resposta Final: [sua resposta]"""),
        ("user", "{puzzle}")
    ])

    print(f"   Gerando {n_amostras} soluções independentes...")
    respostas = gerar_multiplas_respostas(prompt, {"puzzle": puzzle}, n_amostras)

    # Extrai respostas
    respostas_extraidas = [extrair_resposta_final(r) for r in respostas]

    # Realiza votação majoritária
    vencedor, votos, total = votacao_majoritaria(respostas_extraidas)

    return {
        "respostas": respostas,
        "respostas_extraidas": respostas_extraidas,
        "resposta_final": vencedor,
        "confianca": f"{votos}/{total} ({votos/total*100:.0f}%)"
    }


def main():
    print("=" * 60)
    print("SELF-CONSISTENCY PROMPTING (AUTO-CONSISTÊNCIA) - Demo")
    print("=" * 60)

    # Reseta rastreador
    token_tracker.reset()

    # Exemplo 1: Problema Matemático
    print("\n🔢 PROBLEMA MATEMÁTICO COM AUTO-CONSISTÊNCIA")
    print("-" * 40)

    problema_matematica = """
    Uma loja vende maçãs a R$2 cada e laranjas a R$3 cada.
    João compra 5 maçãs e algumas laranjas por um total de R$22.
    Quantas laranjas João comprou?
    """

    print(f"\nProblema: {problema_matematica.strip()}")
    resultado = resolver_matematica_com_consistencia(problema_matematica, n_amostras=5)

    print(f"\n   Respostas extraídas: {resultado['respostas_extraidas']}")
    print(f"   Resposta Final: {resultado['resposta_final']}")
    print(f"   Confiança: {resultado['confianca']}")

    # Exemplo 2: Múltipla Escolha
    print("\n\n📝 MÚLTIPLA ESCOLHA COM AUTO-CONSISTÊNCIA")
    print("-" * 40)

    pergunta = "Qual é a função principal das mitocôndrias em uma célula?"
    opcoes = [
        "Armazenar informação genética",
        "Produzir energia (ATP)",
        "Sintetizar proteínas",
        "Controlar a divisão celular"
    ]

    print(f"\nPergunta: {pergunta}")
    for i, opt in enumerate(opcoes):
        print(f"   {chr(65+i)}. {opt}")

    resultado = responder_multipla_escolha(pergunta, opcoes, n_amostras=5)

    print(f"\n   Respostas extraídas: {resultado['respostas_extraidas']}")
    print(f"   Resposta Final: {resultado['resposta_final']}")
    print(f"   Confiança: {resultado['confianca']}")

    # Exemplo 3: Verificação de Fatos
    print("\n\n✓ VERIFICAÇÃO DE FATOS COM AUTO-CONSISTÊNCIA")
    print("-" * 40)

    afirmacao = "A Grande Muralha da China é visível do espaço a olho nu."

    print(f"\nAfirmação: {afirmacao}")
    resultado = verificar_fato_com_consistencia(afirmacao, n_amostras=5)

    print(f"\n   Respostas extraídas: {resultado['respostas_extraidas']}")
    print(f"   Resposta Final: {resultado['resposta_final']}")
    print(f"   Confiança: {resultado['confianca']}")

    # Exemplo 4: Puzzle Lógico
    print("\n\n🧩 PUZZLE LÓGICO COM AUTO-CONSISTÊNCIA")
    print("-" * 40)

    puzzle = """
    Três amigos - Alice, Bruno e Carol - cada um tem um animal de estimação diferente: um gato, um cachorro ou um peixe.
    - Alice não tem o cachorro.
    - Bruno não tem o gato.
    - Carol tem o peixe.
    Quem tem o cachorro?
    """

    print(f"\nPuzzle: {puzzle.strip()}")
    resultado = resolver_puzzle_logico(puzzle, n_amostras=5)

    print(f"\n   Respostas extraídas: {resultado['respostas_extraidas']}")
    print(f"   Resposta Final: {resultado['resposta_final']}")
    print(f"   Confiança: {resultado['confianca']}")

    # Exibe total de tokens
    print_total_usage(token_tracker, "TOTAL - Self-Consistency Prompting")

    print("\nFim do demo de Self-Consistency")
    print("=" * 60)


if __name__ == "__main__":
    main()
