"""
Meta-Prompting (Meta-Prompting)

Técnica onde um LLM é usado para gerar, otimizar ou melhorar
prompts para outras tarefas de LLM. O LLM se torna um engenheiro de prompts.

Recursos:
- Geração automática de prompts
- Otimização e refinamento de prompts
- Criação de prompts específicos para tarefas
- Teste A/B de prompts

Casos de uso:
- Engenharia de prompts automatizada
- Pipelines de otimização de prompts
- Adaptação dinâmica de prompts
- Geração de templates de prompts
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

# Rastreador global de tokens
token_tracker = TokenUsage()


def gerar_prompt(descricao_tarefa: str, contexto: str = "", restricoes: list[str] = None) -> str:
    """
    Gera um prompt otimizado para uma tarefa dada.

    Args:
        descricao_tarefa: O que o prompt deve realizar
        contexto: Contexto adicional sobre o caso de uso
        restricoes: Quaisquer restrições ou requisitos

    Retorna:
        Texto do prompt gerado
    """
    llm = get_llm(temperature=0.7)

    texto_restricoes = "\n".join([f"- {r}" for r in (restricoes or [])])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um engenheiro de prompts especialista. Sua tarefa é criar prompts altamente eficazes para LLMs.

Ao criar prompts, siga estas melhores práticas:
1. Seja claro e específico sobre a tarefa
2. Inclua contexto relevante e exemplos se útil
3. Especifique o formato de saída desejado
4. Adicione restrições ou proteções conforme necessário
5. Use formatação estruturada (cabeçalhos, marcadores) para clareza
6. Inclua uma persona/função se apropriado

Crie prompts que sejam:
- Inequívocos e bem estruturados
- Focados na tarefa específica
- Projetados para obter respostas de alta qualidade"""),
        ("user", """Crie um prompt otimizado para a seguinte tarefa:

Descrição da Tarefa: {descricao_tarefa}

Contexto Adicional: {contexto}

Restrições:
{restricoes}

Gere o prompt completo:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "descricao_tarefa": descricao_tarefa,
        "contexto": contexto or "Nenhum fornecido",
        "restricoes": texto_restricoes or "Nenhuma especificada"
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Gerar")

    return response.content


def otimizar_prompt(prompt_original: str, problemas: list[str] = None, objetivos: list[str] = None) -> str:
    """
    Otimiza um prompt existente baseado em problemas identificados ou objetivos.

    Args:
        prompt_original: O prompt para otimizar
        problemas: Problemas conhecidos com o prompt
        objetivos: Objetivos de otimização

    Retorna:
        Prompt otimizado
    """
    llm = get_llm(temperature=0.5)

    texto_problemas = "\n".join([f"- {p}" for p in (problemas or [])])
    texto_objetivos = "\n".join([f"- {o}" for o in (objetivos or [])])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em otimizar prompts para LLMs.
Analise o prompt dado e melhore-o preservando seu propósito central.

Considere:
- Clareza e especificidade
- Estrutura e formatação
- Contexto ou restrições faltantes
- Ambiguidades potenciais
- Especificação do formato de saída"""),
        ("user", """Prompt Original:
{prompt_original}

Problemas Conhecidos:
{problemas}

Objetivos de Otimização:
{objetivos}

Forneça o prompt otimizado:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "prompt_original": prompt_original,
        "problemas": texto_problemas or "Nenhum identificado",
        "objetivos": texto_objetivos or "Melhoria geral"
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Otimizar")

    return response.content


def avaliar_prompt(prompt: str, descricao_tarefa: str) -> dict:
    """
    Avalia a qualidade de um prompt e fornece feedback.

    Args:
        prompt: O prompt para avaliar
        descricao_tarefa: O que o prompt deve realizar

    Retorna:
        Dicionário com pontuações e feedback
    """
    llm = get_llm(temperature=0.3)

    prompt_avaliacao = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em avaliação de prompts. Analise prompts e forneça feedback detalhado.

Avalie cada aspecto de 1-10 e explique seu raciocínio:
1. Clareza: Quão claro e inequívoco é o prompt?
2. Especificidade: Quão bem define a tarefa?
3. Estrutura: Está bem organizado?
4. Completude: Inclui contexto necessário?
5. Orientação de Saída: Especifica o formato desejado?

Forneça sugestões de melhoria acionáveis."""),
        ("user", """Tarefa que o prompt deve realizar:
{descricao_tarefa}

Prompt para avaliar:
{prompt}

Forneça sua avaliação neste formato:
CLAREZA: [pontuação]/10 - [explicação]
ESPECIFICIDADE: [pontuação]/10 - [explicação]
ESTRUTURA: [pontuação]/10 - [explicação]
COMPLETUDE: [pontuação]/10 - [explicação]
ORIENTAÇÃO DE SAÍDA: [pontuação]/10 - [explicação]

PONTUAÇÃO GERAL: [média]/10

SUGESTÕES DE MELHORIA:
[lista de sugestões específicas]""")
    ])

    chain = prompt_avaliacao | llm
    response = chain.invoke({
        "prompt": prompt,
        "descricao_tarefa": descricao_tarefa
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Avaliar")

    return {"avaliacao": response.content}


def gerar_variacoes_prompt(prompt_base: str, num_variacoes: int = 3) -> list[str]:
    """
    Gera variações de um prompt para teste A/B.

    Args:
        prompt_base: O prompt base para variar
        num_variacoes: Número de variações para gerar

    Retorna:
        Lista de variações de prompt
    """
    llm = get_llm(temperature=0.8)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um engenheiro de prompts criativo. Gere variações distintas
de um prompt dado mantendo seu propósito central.

Cada variação deve:
- Ter uma abordagem diferente para a tarefa
- Usar fraseado ou estrutura diferentes
- Potencialmente usar técnicas diferentes (few-shot, cadeia de pensamento, etc.)
- Ser claramente distinta das outras variações"""),
        ("user", """Prompt Base:
{prompt_base}

Gere {num_variacoes} variações distintas deste prompt.
Separe cada variação com "---VARIAÇÃO---"

Variações:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "prompt_base": prompt_base,
        "num_variacoes": num_variacoes
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Variações")

    # Analisa variações
    conteudo = response.content
    variacoes = [v.strip() for v in conteudo.split("---VARIAÇÃO---") if v.strip()]

    return variacoes[:num_variacoes]


def criar_template_prompt(tipo_tarefa: str, variaveis: list[str]) -> str:
    """
    Cria um template de prompt reutilizável com variáveis.

    Args:
        tipo_tarefa: Tipo de tarefa (sumarização, classificação, etc.)
        variaveis: Lista de nomes de variáveis para incluir

    Retorna:
        Template de prompt com placeholders {variavel}
    """
    llm = get_llm(temperature=0.5)

    texto_variaveis = ", ".join([f"{{{v}}}" for v in variaveis])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um designer de templates de prompts. Crie templates de prompts reutilizáveis
com placeholders de variáveis.

Use sintaxe {nome_variavel} para placeholders.
O template deve ser:
- Flexível o suficiente para lidar com diferentes entradas
- Específico o suficiente para produzir saídas consistentes
- Bem estruturado e claro"""),
        ("user", """Crie um template de prompt para o seguinte:

Tipo de Tarefa: {tipo_tarefa}
Variáveis Necessárias: {variaveis}

O template deve incluir estas variáveis como placeholders: {texto_variaveis}

Template de Prompt:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "tipo_tarefa": tipo_tarefa,
        "variaveis": ", ".join(variaveis),
        "texto_variaveis": texto_variaveis
    })

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Template")

    return response.content


def auto_melhorar_prompt(prompt: str, tarefa: str, entrada_teste: str, max_iteracoes: int = 3) -> dict:
    """
    Melhora automaticamente um prompt através de testes iterativos e refinamento.

    Args:
        prompt: Prompt inicial
        tarefa: Descrição da tarefa
        entrada_teste: Entrada de exemplo para testar
        max_iteracoes: Máximo de iterações de melhoria

    Retorna:
        Dicionário com prompt final e histórico de melhorias
    """
    llm = get_llm(temperature=0.3)
    llm_tarefa = get_llm(temperature=0.7)

    historico = []
    prompt_atual = prompt

    for i in range(max_iteracoes):
        print(f"\n   Iteração {i+1}...")

        # Testa o prompt atual
        prompt_teste = ChatPromptTemplate.from_messages([
            ("system", prompt_atual),
            ("user", "{entrada}")
        ])

        chain_teste = prompt_teste | llm_tarefa
        resposta_teste = chain_teste.invoke({"entrada": entrada_teste})

        input_tokens, output_tokens = extract_tokens_from_response(resposta_teste)
        token_tracker.add(input_tokens, output_tokens)

        # Avalia e melhora
        prompt_melhoria = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em melhoria de prompts. Analise como um prompt performou
e sugira melhorias.

Avalie:
1. A saída correspondeu à tarefa esperada?
2. A resposta foi bem estruturada?
3. Houve algum problema ou lacuna?

Forneça uma versão melhorada do prompt."""),
            ("user", """Tarefa: {tarefa}

Prompt Atual:
{prompt_atual}

Entrada de Teste: {entrada_teste}

Saída Gerada:
{saida}

Análise e Prompt Melhorado:""")
        ])

        chain_melhoria = prompt_melhoria | llm
        resposta_melhoria = chain_melhoria.invoke({
            "tarefa": tarefa,
            "prompt_atual": prompt_atual,
            "entrada_teste": entrada_teste,
            "saida": resposta_teste.content
        })

        input_tokens, output_tokens = extract_tokens_from_response(resposta_melhoria)
        token_tracker.add(input_tokens, output_tokens)

        historico.append({
            "iteracao": i + 1,
            "prompt": prompt_atual,
            "saida": resposta_teste.content[:200] + "...",
            "feedback": resposta_melhoria.content[:200] + "..."
        })

        prompt_atual = resposta_melhoria.content

    return {
        "prompt_inicial": prompt,
        "prompt_final": prompt_atual,
        "iteracoes": len(historico),
        "historico": historico
    }


def main():
    print("=" * 60)
    print("META-PROMPTING - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Exemplo 1: Gerar um Prompt
    print("\n📝 GERAÇÃO DE PROMPT")
    print("-" * 40)

    tarefa = "Extrair informações chave de emails de suporte ao cliente e categorizá-los"
    contexto = "Para uma empresa SaaS, emails podem conter relatórios de bugs, pedidos de recursos, dúvidas de cobrança"
    restricoes = ["Saída deve ser formato JSON", "Incluir nível de urgência", "Suportar múltiplos idiomas"]

    print(f"\nTarefa: {tarefa}")
    gerado = gerar_prompt(tarefa, contexto, restricoes)
    print(f"\n📋 Prompt Gerado:\n{gerado}")

    # Exemplo 2: Otimizar um Prompt
    print("\n\n🔧 OTIMIZAÇÃO DE PROMPT")
    print("-" * 40)

    original = "Resuma este artigo."
    problemas = ["Muito vago", "Sem orientação de tamanho", "Sem formato especificado"]
    objetivos = ["Tornar específico", "Adicionar formato de saída", "Incluir extração de pontos chave"]

    print(f"\nOriginal: {original}")
    otimizado = otimizar_prompt(original, problemas, objetivos)
    print(f"\n📋 Prompt Otimizado:\n{otimizado}")

    # Exemplo 3: Avaliar um Prompt
    print("\n\n📊 AVALIAÇÃO DE PROMPT")
    print("-" * 40)

    prompt_para_avaliar = """Você é um assistente útil. Responda a pergunta do usuário."""
    desc_tarefa = "Criar um chatbot de suporte ao cliente que lida com problemas técnicos"

    print(f"\nPrompt: {prompt_para_avaliar}")
    print(f"Tarefa: {desc_tarefa}")
    avaliacao = avaliar_prompt(prompt_para_avaliar, desc_tarefa)
    print(f"\n📋 Avaliação:\n{avaliacao['avaliacao']}")

    # Exemplo 4: Gerar Variações
    print("\n\n🔀 VARIAÇÕES DE PROMPT")
    print("-" * 40)

    base = "Explique {conceito} para um {publico}."
    print(f"\nBase: {base}")
    variacoes = gerar_variacoes_prompt(base, num_variacoes=3)
    print(f"\n📋 Variações:")
    for i, v in enumerate(variacoes, 1):
        print(f"\n--- Variação {i} ---")
        print(v[:300] + "..." if len(v) > 300 else v)

    # Exemplo 5: Criar Template
    print("\n\n📄 CRIAÇÃO DE TEMPLATE")
    print("-" * 40)

    tipo_tarefa = "análise de sentimento"
    variaveis = ["texto", "idioma", "formato_saida"]

    print(f"\nTipo de Tarefa: {tipo_tarefa}")
    print(f"Variáveis: {variaveis}")
    template = criar_template_prompt(tipo_tarefa, variaveis)
    print(f"\n📋 Template:\n{template}")

    print_total_usage(token_tracker, "TOTAL - Meta-Prompting")

    print("\n\n" + "=" * 60)
    print("Meta-Prompting permite que LLMs engenheirem prompts automaticamente,")
    print("habilitando otimização e adaptação automatizada de prompts.")
    print("=" * 60)

    print("\nFim do demo Meta-Prompting")
    print("=" * 60)


if __name__ == "__main__":
    main()
