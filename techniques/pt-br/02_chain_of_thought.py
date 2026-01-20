"""
Chain of Thought (CoT) Prompting

Técnica que instrui o modelo a "pensar passo a passo" antes de
chegar à resposta final. Melhora significativamente o desempenho
em tarefas de raciocínio.

Casos de uso:
- Problemas matemáticos
- Raciocínio lógico
- Análise de problemas complexos
- Tomada de decisões
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


def resolver_problema_matematico(problema: str) -> str:
    """Resolve problemas matemáticos usando raciocínio passo a passo."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um professor de matemática experiente.
Resolva o problema a seguir mostrando seu raciocínio passo a passo.

IMPORTANTE:
1. Primeiro, identifique o que o problema está pedindo
2. Liste os dados fornecidos
3. Mostre cada passo do cálculo com explicação
4. Chegue à resposta final de forma clara

Use o formato:
ENTENDIMENTO: [o que o problema pede]
DADOS: [informações fornecidas]
PASSO 1: [primeiro cálculo/raciocínio]
PASSO 2: [segundo cálculo/raciocínio]
...
RESPOSTA FINAL: [resultado]"""),
        ("user", "{problema}")
    ])

    chain = prompt | llm
    response = chain.invoke({"problema": problema})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def raciocinio_logico(puzzle: str) -> str:
    """Resolve puzzles lógicos com raciocínio estruturado."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em lógica e resolução de puzzles.
Vamos pensar passo a passo para resolver este problema.

Para cada etapa do raciocínio:
1. Identifique as premissas dadas
2. Faça deduções lógicas a partir das premissas
3. Elimine possibilidades impossíveis
4. Chegue à conclusão

Mostre claramente seu processo de pensamento antes de dar a resposta final."""),
        ("user", "{puzzle}")
    ])

    chain = prompt | llm
    response = chain.invoke({"puzzle": puzzle})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def analisar_decisao(situacao: str) -> str:
    """Analisa uma situação e ajuda na tomada de decisão."""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um consultor estratégico experiente.
Analise a situação apresentada pensando passo a passo:

1. CONTEXTO: Resuma a situação atual
2. PARTES INTERESSADAS: Identifique quem é afetado
3. OPÇÕES: Liste as alternativas possíveis
4. PRÓS E CONTRAS: Analise cada opção
5. RISCOS: Identifique possíveis problemas
6. RECOMENDAÇÃO: Sugira a melhor decisão com justificativa

Seja analítico e objetivo em cada etapa."""),
        ("user", "{situacao}")
    ])

    chain = prompt | llm
    response = chain.invoke({"situacao": situacao})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def debug_codigo(codigo: str, erro: str) -> str:
    """Analisa código e erro para encontrar a solução."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um desenvolvedor sênior especialista em debugging.
Analise o código e o erro reportado, pensando passo a passo:

1. ENTENDIMENTO DO CÓDIGO: O que o código deveria fazer?
2. ANÁLISE DO ERRO: O que a mensagem de erro indica?
3. LOCALIZAÇÃO: Onde está o problema no código?
4. CAUSA RAIZ: Por que o erro ocorre?
5. SOLUÇÃO: Como corrigir o problema?
6. CÓDIGO CORRIGIDO: Mostre a versão correta

Explique cada passo do seu raciocínio."""),
        ("user", "CÓDIGO:\n```\n{codigo}\n```\n\nERRO:\n{erro}")
    ])

    chain = prompt | llm
    response = chain.invoke({"codigo": codigo, "erro": erro})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def main():
    print("=" * 60)
    print("CHAIN OF THOUGHT (CoT) - Demonstração")
    print("=" * 60)

    # Reset do tracker
    token_tracker.reset()

    # Exemplo 1: Problema Matemático
    print("\n🔢 PROBLEMA MATEMÁTICO")
    print("-" * 40)

    problema_mat = """
    Uma loja vende camisetas por R$ 45,00 cada. Se um cliente comprar
    3 ou mais camisetas, recebe 15% de desconto no total. João comprou
    5 camisetas e pagou com uma nota de R$ 200,00. Quanto ele recebeu de troco?
    """

    print(f"Problema: {problema_mat.strip()}")
    print("\nResolução:")
    print(resolver_problema_matematico(problema_mat))

    # Exemplo 2: Raciocínio Lógico
    print("\n\n🧩 PUZZLE LÓGICO")
    print("-" * 40)

    puzzle = """
    Três amigos (Ana, Bruno e Carla) têm profissões diferentes:
    médica, engenheiro e advogada. Sabemos que:
    1. Ana não é médica
    2. Bruno não é advogado
    3. A médica é amiga de Carla, mas não de Ana

    Qual é a profissão de cada um?
    """

    print(f"Puzzle: {puzzle.strip()}")
    print("\nResolução:")
    print(raciocinio_logico(puzzle))

    # Exemplo 3: Tomada de Decisão
    print("\n\n💼 ANÁLISE DE DECISÃO")
    print("-" * 40)

    situacao = """
    Sou gerente de uma startup de tecnologia com 20 funcionários.
    Recebemos duas propostas:
    A) Investimento de R$ 2 milhões de um fundo de venture capital
       que quer 30% da empresa e um assento no conselho.
    B) Empréstimo bancário de R$ 1,5 milhão com juros de 12% ao ano,
       sem ceder participação.

    Nosso faturamento atual é de R$ 500 mil/mês e estamos crescendo
    15% ao mês. Qual opção devo escolher?
    """

    print(f"Situação: {situacao.strip()}")
    print("\nAnálise:")
    print(analisar_decisao(situacao))

    # Exemplo 4: Debug de Código
    print("\n\n🐛 DEBUG DE CÓDIGO")
    print("-" * 40)

    codigo = """def calcular_media(notas):
    soma = 0
    for nota in notas:
        soma += nota
    return soma / len(notas)

resultado = calcular_media([])
print(f"Média: {resultado}")"""

    erro = "ZeroDivisionError: division by zero"

    print(f"Código:\n{codigo}")
    print(f"\nErro: {erro}")
    print("\nAnálise e Solução:")
    print(debug_codigo(codigo, erro))

    # Exibir total de tokens
    print_total_usage(token_tracker, "TOTAL - Chain of Thought")

    print("\nFim da demonstração Chain of Thought")
    print("=" * 60)


if __name__ == "__main__":
    main()
