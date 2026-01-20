"""
Self-Refine Prompting (Auto-Refinamento)

Técnica onde o modelo gera uma resposta inicial, critica-a,
e melhora iterativamente baseado em seu próprio feedback até ficar satisfatório.

Casos de uso:
- Escrita e melhoria de conteúdo
- Otimização de código
- Explicações detalhadas
- Geração de conteúdo criativo
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


def gerar_resposta_inicial(tarefa: str, contexto: str = "") -> str:
    """Gera uma resposta inicial para a tarefa."""
    llm = get_llm(temperature=0.7)

    if contexto:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente habilidoso. Complete a tarefa dada da melhor forma possível.
Considere o contexto fornecido em sua resposta."""),
            ("user", """Contexto: {contexto}

Tarefa: {tarefa}

Resposta:""")
        ])
        inputs = {"tarefa": tarefa, "contexto": contexto}
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente habilidoso. Complete a tarefa dada da melhor forma possível."),
            ("user", """Tarefa: {tarefa}

Resposta:""")
        ])
        inputs = {"tarefa": tarefa}

    chain = prompt | llm
    response = chain.invoke(inputs)

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Inicial")

    return response.content


def criticar_resposta(tarefa: str, resposta: str, criterios: list[str] = None) -> str:
    """Critica uma resposta e identifica áreas para melhoria."""
    llm = get_llm(temperature=0.3)

    criterios_padrao = [
        "Precisão e correção",
        "Clareza e legibilidade",
        "Completude",
        "Estrutura e organização",
        "Relevância para a tarefa"
    ]

    criterios = criterios or criterios_padrao
    texto_criterios = "\n".join([f"- {c}" for c in criterios])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um revisor crítico. Analise a resposta dada e forneça feedback específico e acionável.
Seja construtivo mas minucioso em identificar fraquezas e áreas para melhoria.

Avalie com base nestes critérios:
{criterios}

Formate sua crítica como:
PONTOS FORTES:
- [liste pontos fortes]

PONTOS FRACOS:
- [liste pontos fracos]

MELHORIAS ESPECÍFICAS:
- [liste mudanças específicas a fazer]"""),
        ("user", """Tarefa Original: {tarefa}

Resposta a Criticar:
{resposta}

Forneça sua crítica detalhada:""")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "tarefa": tarefa,
        "resposta": resposta,
        "criterios": texto_criterios
    })

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(result)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Crítica")

    return result.content


def refinar_resposta(tarefa: str, resposta: str, critica: str) -> str:
    """Refina uma resposta baseada na crítica."""
    llm = get_llm(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente habilidoso melhorando seu trabalho anterior.
Baseado na crítica fornecida, crie uma versão melhorada da resposta.
Aborde todos os pontos fracos e incorpore as melhorias sugeridas."""),
        ("user", """Tarefa Original: {tarefa}

Resposta Anterior:
{resposta}

Crítica e Feedback:
{critica}

Resposta Melhorada:""")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "tarefa": tarefa,
        "resposta": resposta,
        "critica": critica
    })

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(result)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Refinamento")

    return result.content


def verificar_se_satisfatorio(tarefa: str, resposta: str, pontuacao_minima: int = 8) -> tuple[bool, int, str]:
    """Verifica se a resposta é satisfatória (pontuação >= pontuação_minima)."""
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um avaliador de qualidade. Avalie a resposta em uma escala de 1-10.
Seja rigoroso e objetivo em sua avaliação.

Formato de saída (exatamente):
PONTUAÇÃO: [número 1-10]
RAZÃO: [breve explicação]"""),
        ("user", """Tarefa: {tarefa}

Resposta a Avaliar:
{resposta}

Avaliação:""")
    ])

    chain = prompt | llm
    result = chain.invoke({"tarefa": tarefa, "resposta": resposta})

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(result)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Avaliação")

    # Analisa pontuação
    content = result.content
    pontuacao = 5  # Padrão
    razao = ""

    for linha in content.split('\n'):
        if 'PONTUAÇÃO:' in linha.upper() or 'SCORE:' in linha.upper():
            try:
                pontuacao_str = linha.split(':')[1].strip()
                pontuacao = int(''.join(filter(str.isdigit, pontuacao_str[:3])))
            except (ValueError, IndexError):
                pass
        elif 'RAZÃO:' in linha.upper() or 'REASON:' in linha.upper():
            razao = linha.split(':', 1)[1].strip() if ':' in linha else ""

    return pontuacao >= pontuacao_minima, pontuacao, razao


def auto_refinar(tarefa: str, max_iteracoes: int = 3, pontuacao_minima: int = 8, criterios: list[str] = None) -> dict:
    """
    Executa o loop de auto-refinamento.

    Args:
        tarefa: A tarefa a completar
        max_iteracoes: Máximo de iterações de refinamento
        pontuacao_minima: Pontuação mínima aceitável (1-10)
        criterios: Critérios de avaliação personalizados

    Retorna:
        Dicionário com iterações, resposta final e pontuação
    """
    iteracoes = []

    print("\n   Gerando resposta inicial...")
    resposta_atual = gerar_resposta_inicial(tarefa)
    iteracoes.append({"resposta": resposta_atual, "critica": None, "pontuacao": None})

    for i in range(max_iteracoes):
        print(f"\n   Verificando qualidade (iteração {i+1})...")
        satisfatorio, pontuacao, razao = verificar_se_satisfatorio(tarefa, resposta_atual, pontuacao_minima)
        iteracoes[-1]["pontuacao"] = pontuacao
        iteracoes[-1]["razao_pontuacao"] = razao

        print(f"      Pontuação: {pontuacao}/10 - {razao}")

        if satisfatorio:
            print(f"   ✓ Resposta satisfatória (pontuação >= {pontuacao_minima})")
            break

        if i < max_iteracoes - 1:
            print(f"\n   Criticando resposta...")
            critica = criticar_resposta(tarefa, resposta_atual, criterios)
            iteracoes[-1]["critica"] = critica

            print(f"\n   Refinando baseado na crítica...")
            resposta_atual = refinar_resposta(tarefa, resposta_atual, critica)
            iteracoes.append({"resposta": resposta_atual, "critica": None, "pontuacao": None})

    # Avaliação final se o loop completou
    if iteracoes[-1]["pontuacao"] is None:
        _, pontuacao, razao = verificar_se_satisfatorio(tarefa, resposta_atual, pontuacao_minima)
        iteracoes[-1]["pontuacao"] = pontuacao
        iteracoes[-1]["razao_pontuacao"] = razao

    return {
        "iteracoes": iteracoes,
        "resposta_final": resposta_atual,
        "pontuacao_final": iteracoes[-1]["pontuacao"],
        "num_iteracoes": len(iteracoes)
    }


def melhorar_escrita(texto: str, estilo: str = "profissional") -> dict:
    """Melhora um texto usando auto-refinamento."""
    tarefa = f"Reescreva o seguinte texto em estilo {estilo} mantendo o significado original:\n\n{texto}"
    criterios = [
        "Gramática e ortografia",
        "Tom corresponde ao estilo solicitado",
        "Fluidez e legibilidade",
        "Concisão",
        "Engajamento"
    ]
    return auto_refinar(tarefa, max_iteracoes=3, criterios=criterios)


def otimizar_codigo(codigo: str, linguagem: str = "Python") -> dict:
    """Otimiza código usando auto-refinamento."""
    tarefa = f"Otimize este código {linguagem} para legibilidade, eficiência e boas práticas:\n\n{codigo}"
    criterios = [
        "Correção do código",
        "Legibilidade e clareza",
        "Eficiência",
        "Segue boas práticas",
        "Tratamento de erros adequado"
    ]
    return auto_refinar(tarefa, max_iteracoes=3, criterios=criterios)


def melhorar_explicacao(topico: str, audiencia: str = "geral") -> dict:
    """Cria e melhora uma explicação usando auto-refinamento."""
    tarefa = f"Explique {topico} para uma audiência {audiencia} de forma clara e envolvente"
    criterios = [
        "Precisão",
        "Apropriado para o público-alvo",
        "Usa exemplos úteis",
        "Estrutura clara",
        "Apresentação envolvente"
    ]
    return auto_refinar(tarefa, max_iteracoes=3, criterios=criterios)


def main():
    print("=" * 60)
    print("SELF-REFINE PROMPTING (AUTO-REFINAMENTO) - Demo")
    print("=" * 60)

    # Reseta rastreador
    token_tracker.reset()

    # Exemplo 1: Melhoria de Escrita
    print("\n✍️ MELHORIA DE ESCRITA")
    print("-" * 40)

    texto_original = """
    A reunião foi boa. A gente conversou sobre algumas coisas e tomou umas decisões.
    Todo mundo pareceu concordar na maioria. Vamos fazer as coisas diferente agora.
    O projeto deve ficar pronto logo espero.
    """

    print(f"\nTexto Original:\n{texto_original.strip()}")
    print("\nMelhorando para estilo profissional...")

    resultado = melhorar_escrita(texto_original, estilo="profissional empresarial")

    print(f"\n📋 VERSÃO FINAL (Pontuação: {resultado['pontuacao_final']}/10):")
    print("-" * 40)
    print(resultado["resposta_final"])
    print(f"\n   Iterações necessárias: {resultado['num_iteracoes']}")

    # Exemplo 2: Otimização de Código
    print("\n\n💻 OTIMIZAÇÃO DE CÓDIGO")
    print("-" * 40)

    codigo_original = '''
def encontrar_duplicados(lista):
    duplicados = []
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] == lista[j]:
                if lista[i] not in duplicados:
                    duplicados.append(lista[i])
    return duplicados
'''

    print(f"\nCódigo Original:{codigo_original}")
    print("Otimizando...")

    resultado = otimizar_codigo(codigo_original, linguagem="Python")

    print(f"\n📋 CÓDIGO OTIMIZADO (Pontuação: {resultado['pontuacao_final']}/10):")
    print("-" * 40)
    print(resultado["resposta_final"])
    print(f"\n   Iterações necessárias: {resultado['num_iteracoes']}")

    # Exemplo 3: Melhoria de Explicação
    print("\n\n📚 MELHORIA DE EXPLICAÇÃO")
    print("-" * 40)

    topico = "como a tecnologia blockchain funciona"
    audiencia = "executivos de negócios não técnicos"

    print(f"\nTópico: {topico}")
    print(f"Público-Alvo: {audiencia}")
    print("\nGerando e refinando explicação...")

    resultado = melhorar_explicacao(topico, audiencia)

    print(f"\n📋 EXPLICAÇÃO FINAL (Pontuação: {resultado['pontuacao_final']}/10):")
    print("-" * 40)
    print(resultado["resposta_final"])
    print(f"\n   Iterações necessárias: {resultado['num_iteracoes']}")

    # Exibe total de tokens
    print_total_usage(token_tracker, "TOTAL - Self-Refine Prompting")

    print("\nFim do demo de Self-Refine")
    print("=" * 60)


if __name__ == "__main__":
    main()
