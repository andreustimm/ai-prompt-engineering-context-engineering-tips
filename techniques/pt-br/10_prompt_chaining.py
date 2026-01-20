"""
Prompt Chaining (Encadeamento de Prompts)

Técnica que conecta múltiplos prompts em um pipeline, onde a saída
de um prompt se torna a entrada para o próximo, criando um fluxo de trabalho.

Casos de uso:
- Pipelines de criação de conteúdo
- Fluxos de pesquisa e análise
- Cadeias de processamento de dados
- Geração de documentos em múltiplas etapas
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


def executar_passo_cadeia(prompt_template: ChatPromptTemplate, inputs: dict, nome_passo: str, temperature: float = 0.7) -> str:
    """Executa um único passo na cadeia de prompts."""
    llm = get_llm(temperature=temperature)
    chain = prompt_template | llm
    response = chain.invoke(inputs)

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, nome_passo)

    return response.content


def pesquisar_analisar_resumir(topico: str) -> dict:
    """
    Uma cadeia de três passos: Pesquisar -> Analisar -> Resumir

    Passo 1: Reunir fatos principais sobre o tópico
    Passo 2: Analisar implicações e conexões
    Passo 3: Criar um resumo executivo
    """
    resultados = {}

    # Passo 1: Pesquisar
    print("\n   Passo 1: Pesquisando tópico...")
    prompt_pesquisa = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em pesquisa. Reúna e liste fatos principais sobre o tópico dado.
Inclua: definições, estatísticas principais, principais atores/componentes, contexto histórico e estado atual.
Formate como pontos organizados."""),
        ("user", "Tópico de pesquisa: {topico}")
    ])

    resultados["pesquisa"] = executar_passo_cadeia(
        prompt_pesquisa, {"topico": topico}, "Pesquisa", temperature=0.3
    )

    # Passo 2: Analisar
    print("\n   Passo 2: Analisando descobertas...")
    prompt_analise = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista analítico. Baseado nas descobertas da pesquisa fornecidas,
identifique padrões, implicações, oportunidades e desafios potenciais.
Forneça insights profundos que vão além dos fatos superficiais."""),
        ("user", """Descobertas da pesquisa:
{pesquisa}

Forneça sua análise:""")
    ])

    resultados["analise"] = executar_passo_cadeia(
        prompt_analise, {"pesquisa": resultados["pesquisa"]}, "Análise", temperature=0.5
    )

    # Passo 3: Resumir
    print("\n   Passo 3: Criando resumo executivo...")
    prompt_resumo = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em comunicação empresarial. Crie um resumo executivo conciso
que combine a pesquisa e análise em insights acionáveis.
Mantenha em menos de 200 palavras cobrindo os pontos mais importantes."""),
        ("user", """Pesquisa:
{pesquisa}

Análise:
{analise}

Crie o resumo executivo:""")
    ])

    resultados["resumo"] = executar_passo_cadeia(
        prompt_resumo,
        {"pesquisa": resultados["pesquisa"], "analise": resultados["analise"]},
        "Resumo",
        temperature=0.4
    )

    return resultados


def pipeline_criacao_conteudo(topico: str, publico_alvo: str, tipo_conteudo: str = "artigo de blog") -> dict:
    """
    Pipeline de criação de conteúdo: Esboço -> Rascunho -> Edição -> Polimento

    Passo 1: Criar esboço detalhado
    Passo 2: Escrever primeiro rascunho
    Passo 3: Editar para clareza e fluidez
    Passo 4: Polir e finalizar
    """
    resultados = {}

    # Passo 1: Esboço
    print("\n   Passo 1: Criando esboço...")
    prompt_esboco = ChatPromptTemplate.from_messages([
        ("system", """Você é um estrategista de conteúdo. Crie um esboço detalhado para um {tipo_conteudo}
sobre o tópico dado, otimizado para o público-alvo.
Inclua: gancho/intro, seções principais, pontos-chave para cada seção e conclusão."""),
        ("user", """Tópico: {topico}
Público-alvo: {publico}

Crie o esboço:""")
    ])

    resultados["esboco"] = executar_passo_cadeia(
        prompt_esboco,
        {"topico": topico, "publico": publico_alvo, "tipo_conteudo": tipo_conteudo},
        "Esboço",
        temperature=0.5
    )

    # Passo 2: Rascunho
    print("\n   Passo 2: Escrevendo primeiro rascunho...")
    prompt_rascunho = ChatPromptTemplate.from_messages([
        ("system", """Você é um escritor de conteúdo habilidoso. Escreva o primeiro rascunho baseado no esboço fornecido.
Foque em colocar todas as ideias com boa fluidez, mas não se preocupe com perfeição ainda.
Escreva em um estilo apropriado para um {tipo_conteudo}."""),
        ("user", """Esboço:
{esboco}

Público-alvo: {publico}

Escreva o primeiro rascunho:""")
    ])

    resultados["rascunho"] = executar_passo_cadeia(
        prompt_rascunho,
        {"esboco": resultados["esboco"], "publico": publico_alvo, "tipo_conteudo": tipo_conteudo},
        "Rascunho",
        temperature=0.7
    )

    # Passo 3: Edição
    print("\n   Passo 3: Editando para clareza...")
    prompt_edicao = ChatPromptTemplate.from_messages([
        ("system", """Você é um editor profissional. Revise e edite o rascunho para:
- Clareza e legibilidade
- Fluxo lógico entre parágrafos
- Remoção de redundâncias
- Escolhas de palavras mais fortes
- Tom consistente

Forneça a versão editada."""),
        ("user", """Rascunho para editar:
{rascunho}

Público-alvo: {publico}

Versão editada:""")
    ])

    resultados["editado"] = executar_passo_cadeia(
        prompt_edicao,
        {"rascunho": resultados["rascunho"], "publico": publico_alvo},
        "Edição",
        temperature=0.3
    )

    # Passo 4: Polimento
    print("\n   Passo 4: Polimento final...")
    prompt_polimento = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em publicação. Aplique o polimento final ao conteúdo:
- Garanta gramática e pontuação perfeitas
- Adicione abertura e fechamento envolventes
- Otimize para engajamento
- Adicione transições que estejam faltando
- Garanta que está pronto para publicação

Forneça a versão final polida."""),
        ("user", """Conteúdo para polir:
{editado}

Versão final polida:""")
    ])

    resultados["final"] = executar_passo_cadeia(
        prompt_polimento,
        {"editado": resultados["editado"]},
        "Polimento",
        temperature=0.3
    )

    return resultados


def cadeia_insight_dados(descricao_dados: str, contexto_negocio: str) -> dict:
    """
    Cadeia de análise de dados: Extrair -> Interpretar -> Recomendar

    Passo 1: Extrair pontos de dados principais
    Passo 2: Interpretar padrões e tendências
    Passo 3: Gerar recomendações de negócio
    """
    resultados = {}

    # Passo 1: Extrair
    print("\n   Passo 1: Extraindo pontos de dados principais...")
    prompt_extracao = ChatPromptTemplate.from_messages([
        ("system", """Você é um analista de dados. Da descrição de dados fornecida,
identifique e liste as métricas principais, tendências e pontos de dados notáveis.
Seja específico com números e porcentagens onde disponível."""),
        ("user", """Descrição dos dados:
{dados}

Contexto de negócio: {contexto}

Extraia os pontos de dados principais:""")
    ])

    resultados["extracao"] = executar_passo_cadeia(
        prompt_extracao,
        {"dados": descricao_dados, "contexto": contexto_negocio},
        "Extração",
        temperature=0.2
    )

    # Passo 2: Interpretar
    print("\n   Passo 2: Interpretando padrões...")
    prompt_interpretacao = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em inteligência de negócios. Baseado nos pontos de dados extraídos,
identifique padrões, correlações e o que os dados estão nos dizendo.
Considere o contexto de negócio em sua interpretação."""),
        ("user", """Pontos de dados extraídos:
{extracao}

Contexto de negócio: {contexto}

Interpretação:""")
    ])

    resultados["interpretacao"] = executar_passo_cadeia(
        prompt_interpretacao,
        {"extracao": resultados["extracao"], "contexto": contexto_negocio},
        "Interpretação",
        temperature=0.4
    )

    # Passo 3: Recomendar
    print("\n   Passo 3: Gerando recomendações...")
    prompt_recomendacao = ChatPromptTemplate.from_messages([
        ("system", """Você é um conselheiro estratégico de negócios. Baseado na análise de dados,
forneça recomendações específicas e acionáveis.
Priorize por impacto e viabilidade. Inclua tanto vitórias rápidas quanto estratégias de longo prazo."""),
        ("user", """Extração de dados:
{extracao}

Interpretação:
{interpretacao}

Contexto de negócio: {contexto}

Recomendações:""")
    ])

    resultados["recomendacoes"] = executar_passo_cadeia(
        prompt_recomendacao,
        {
            "extracao": resultados["extracao"],
            "interpretacao": resultados["interpretacao"],
            "contexto": contexto_negocio
        },
        "Recomendação",
        temperature=0.5
    )

    return resultados


def cadeia_traducao_localizacao(texto: str, idioma_origem: str, idioma_destino: str, cultura_destino: str) -> dict:
    """
    Cadeia de tradução e localização: Traduzir -> Adaptar -> Verificar

    Passo 1: Tradução direta
    Passo 2: Adaptação cultural
    Passo 3: Verificação de qualidade
    """
    resultados = {}

    # Passo 1: Traduzir
    print("\n   Passo 1: Traduzindo...")
    prompt_traducao = ChatPromptTemplate.from_messages([
        ("system", """Você é um tradutor profissional. Traduza o texto com precisão
de {idioma_origem} para {idioma_destino}. Mantenha o significado e tom originais."""),
        ("user", """Texto para traduzir:
{texto}

Tradução:""")
    ])

    resultados["traducao"] = executar_passo_cadeia(
        prompt_traducao,
        {"texto": texto, "idioma_origem": idioma_origem, "idioma_destino": idioma_destino},
        "Tradução",
        temperature=0.3
    )

    # Passo 2: Adaptar
    print("\n   Passo 2: Adaptação cultural...")
    prompt_adaptacao = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em localização para {cultura}.
Adapte a tradução para ser culturalmente apropriada:
- Ajuste expressões idiomáticas
- Considere costumes e sensibilidades locais
- Adapte referências que podem não traduzir bem
- Mantenha fluxo natural de linguagem para falantes nativos"""),
        ("user", """Tradução para adaptar:
{traducao}

Versão culturalmente adaptada:""")
    ])

    resultados["adaptacao"] = executar_passo_cadeia(
        prompt_adaptacao,
        {"traducao": resultados["traducao"], "cultura": cultura_destino},
        "Adaptação",
        temperature=0.4
    )

    # Passo 3: Verificar
    print("\n   Passo 3: Verificação de qualidade...")
    prompt_verificacao = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em garantia de qualidade para conteúdo em {idioma_destino}.
Revise o texto localizado e:
1. Verifique se há problemas de tradução remanescentes
2. Verifique apropriação cultural
3. Garanta fluxo natural de linguagem
4. Faça correções finais

Forneça a versão final verificada e um breve relatório de qualidade."""),
        ("user", """Original ({idioma_origem}):
{original}

Versão localizada:
{adaptacao}

Versão final verificada e notas de qualidade:""")
    ])

    resultados["verificado"] = executar_passo_cadeia(
        prompt_verificacao,
        {
            "original": texto,
            "adaptacao": resultados["adaptacao"],
            "idioma_origem": idioma_origem,
            "idioma_destino": idioma_destino
        },
        "Verificação",
        temperature=0.2
    )

    return resultados


def main():
    print("=" * 60)
    print("PROMPT CHAINING (ENCADEAMENTO DE PROMPTS) - Demo")
    print("=" * 60)

    # Reseta rastreador
    token_tracker.reset()

    # Exemplo 1: Pesquisar-Analisar-Resumir
    print("\n🔍 PESQUISAR -> ANALISAR -> RESUMIR")
    print("-" * 40)

    topico = "O impacto da inteligência artificial no diagnóstico médico"
    print(f"\nTópico: {topico}")

    resultados = pesquisar_analisar_resumir(topico)

    print(f"\n📋 RESUMO EXECUTIVO:")
    print("-" * 40)
    print(resultados["resumo"])

    # Exemplo 2: Pipeline de Criação de Conteúdo
    print("\n\n✍️ PIPELINE DE CRIAÇÃO DE CONTEÚDO")
    print("-" * 40)

    topico_conteudo = "5 Dicas de Produtividade para Trabalhadores Remotos"
    publico = "profissionais do conhecimento de 25-45 anos"

    print(f"\nTópico: {topico_conteudo}")
    print(f"Público: {publico}")

    resultados = pipeline_criacao_conteudo(topico_conteudo, publico, "artigo de blog")

    print(f"\n📋 CONTEÚDO FINAL:")
    print("-" * 40)
    print(resultados["final"][:1500] + "..." if len(resultados["final"]) > 1500 else resultados["final"])

    # Exemplo 3: Cadeia de Insight de Dados
    print("\n\n📊 CADEIA DE INSIGHT DE DADOS")
    print("-" * 40)

    desc_dados = """
    Relatório de Vendas Q3:
    - Receita total: R$12M (aumento de 15% do Q2)
    - Novos clientes: 340 (queda de 8% do Q2)
    - Taxa de retenção de clientes: 92%
    - Tamanho médio de negócio: R$35.290 (aumento de 25%)
    - Duração do ciclo de vendas: 45 dias (aumento de 38 dias)
    - Região com melhor desempenho: Sudeste (35% da receita)
    - Região com baixo desempenho: Norte (8% da receita)
    """
    ctx_negocio = "Empresa B2B SaaS mirando empresas de médio porte, meta é crescimento de 30% ano a ano"

    print(f"\nDados: {desc_dados.strip()}")
    print(f"\nContexto: {ctx_negocio}")

    resultados = cadeia_insight_dados(desc_dados, ctx_negocio)

    print(f"\n📋 RECOMENDAÇÕES:")
    print("-" * 40)
    print(resultados["recomendacoes"])

    # Exemplo 4: Tradução e Localização
    print("\n\n🌐 TRADUÇÃO E LOCALIZAÇÃO")
    print("-" * 40)

    texto_original = """
    Our Black Friday sale is a slam dunk! Get up to 50% off everything in the store.
    Don't miss out - these deals are going like hotcakes! Sale runs through Cyber Monday.
    """

    print(f"\nOriginal (Inglês): {texto_original.strip()}")

    resultados = cadeia_traducao_localizacao(
        texto_original,
        idioma_origem="Inglês",
        idioma_destino="Português Brasileiro",
        cultura_destino="consumidores brasileiros"
    )

    print(f"\n📋 VERSÃO LOCALIZADA:")
    print("-" * 40)
    print(resultados["verificado"])

    # Exibe total de tokens
    print_total_usage(token_tracker, "TOTAL - Prompt Chaining")

    print("\nFim do demo de Prompt Chaining")
    print("=" * 60)


if __name__ == "__main__":
    main()
