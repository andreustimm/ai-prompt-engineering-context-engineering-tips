"""
ReAct (Reasoning + Acting) Agent

Técnica que combina raciocínio (Thought) com ações (Action) e
observações (Observation) em um loop iterativo. O agente pensa,
age, observa o resultado e repete até resolver o problema.

Casos de uso:
- Pesquisa e análise de informações
- Tarefas que requerem ferramentas externas
- Problemas que precisam de múltiplas etapas
- Integração com APIs e bases de dados
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain import hub
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Tentativa de importar Wikipedia
try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("Aviso: Wikipedia não disponível. Instale com: pip install wikipedia")

# Tracker global de tokens para este script
token_tracker = TokenUsage()


class TokenCounterCallback(BaseCallbackHandler):
    """Callback para contar tokens durante execução do agente."""

    def __init__(self, tracker: TokenUsage):
        self.tracker = tracker

    def on_llm_end(self, response, **kwargs):
        """Chamado quando o LLM termina de gerar."""
        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            self.tracker.add(input_tokens, output_tokens)
            print_token_usage(input_tokens, output_tokens, "agent_step")


def criar_ferramenta_calculo() -> Tool:
    """Cria uma ferramenta simples de cálculo."""

    def calcular(expressao: str) -> str:
        """Avalia uma expressão matemática simples."""
        try:
            # Apenas operações básicas seguras
            expressao_limpa = expressao.replace("^", "**")
            # Permitir apenas caracteres seguros
            caracteres_permitidos = set("0123456789+-*/().** ")
            if not all(c in caracteres_permitidos for c in expressao_limpa):
                return "Erro: Expressão contém caracteres não permitidos"

            resultado = eval(expressao_limpa)
            return f"Resultado: {resultado}"
        except Exception as e:
            return f"Erro ao calcular: {str(e)}"

    return Tool(
        name="calculadora",
        description="Útil para fazer cálculos matemáticos. Input deve ser uma expressão matemática como '2 + 2' ou '10 * 5'.",
        func=calcular
    )


def criar_ferramenta_busca() -> Tool:
    """Cria ferramenta de busca na web usando DuckDuckGo."""
    search = DuckDuckGoSearchRun()

    return Tool(
        name="busca_web",
        description="Útil para buscar informações atuais na internet. Use para encontrar dados recentes, notícias ou informações que você não conhece.",
        func=search.run
    )


def criar_ferramenta_wikipedia() -> Tool:
    """Cria ferramenta de busca na Wikipedia."""
    if not WIKIPEDIA_AVAILABLE:
        def wikipedia_fallback(query: str) -> str:
            return "Wikipedia não está disponível. Use a busca web como alternativa."

        return Tool(
            name="wikipedia",
            description="Busca informações na Wikipedia (indisponível no momento).",
            func=wikipedia_fallback
        )

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    return Tool(
        name="wikipedia",
        description="Útil para buscar informações enciclopédicas e fatos históricos na Wikipedia. Boa para conceitos, biografias e eventos históricos.",
        func=wikipedia.run
    )


def criar_agente_react():
    """
    Cria um agente ReAct com as ferramentas configuradas.

    Returns:
        AgentExecutor configurado
    """
    llm = get_llm(temperature=0)

    # Criar ferramentas
    tools = [
        criar_ferramenta_busca(),
        criar_ferramenta_wikipedia(),
        criar_ferramenta_calculo()
    ]

    # Template ReAct padrão do LangChain Hub
    # Você também pode criar um personalizado
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception:
        # Fallback para template manual se hub não estiver disponível
        prompt = ChatPromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

    # Criar agente
    agent = create_react_agent(llm, tools, prompt)

    # Callback para contar tokens
    token_callback = TokenCounterCallback(token_tracker)

    # Criar executor com configurações
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        callbacks=[token_callback]
    )

    return agent_executor


def executar_agente(pergunta: str) -> str:
    """
    Executa o agente ReAct para responder uma pergunta.

    Args:
        pergunta: A pergunta a ser respondida

    Returns:
        Resposta do agente
    """
    agente = criar_agente_react()

    try:
        resultado = agente.invoke({"input": pergunta})
        return resultado.get("output", "Não foi possível obter uma resposta.")
    except Exception as e:
        return f"Erro ao executar agente: {str(e)}"


def demonstrar_react_manual():
    """
    Demonstra o padrão ReAct de forma manual para fins educacionais.
    Mostra explicitamente o ciclo Thought-Action-Observation.
    """
    print("\n🔄 Demonstração Manual do Padrão ReAct")
    print("=" * 50)

    llm = get_llm(temperature=0)

    pergunta = "Qual é a população atual do Brasil e quanto isso representa em porcentagem da população mundial?"

    print(f"\n❓ Pergunta: {pergunta}")

    # Template que força o padrão ReAct
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente que usa o padrão ReAct para responder perguntas.
Para cada pergunta, siga este processo:

1. THOUGHT (Pensamento): Analise o que precisa fazer
2. ACTION (Ação): Decida qual ação tomar e com qual input
3. OBSERVATION (Observação): Analise o resultado
4. Repita até ter a resposta final

Ações disponíveis:
- SEARCH: buscar na internet (input: termo de busca)
- CALCULATE: fazer cálculo (input: expressão matemática)
- ANSWER: dar a resposta final (input: resposta)

Formato de resposta:
THOUGHT: [seu raciocínio]
ACTION: [SEARCH/CALCULATE/ANSWER]
ACTION_INPUT: [input para a ação]"""),
        ("user", "{pergunta}")
    ])

    chain = prompt | llm

    # Simulação do loop ReAct
    contexto = ""
    iteracao = 1
    max_iteracoes = 4

    while iteracao <= max_iteracoes:
        print(f"\n--- Iteração {iteracao} ---")

        if contexto:
            mensagem = f"{pergunta}\n\nContexto anterior:\n{contexto}\n\nContinue o raciocínio:"
        else:
            mensagem = pergunta

        response = chain.invoke({"pergunta": mensagem})

        # Extrair e registrar tokens
        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, f"iteracao_{iteracao}")

        resposta = response.content
        print(resposta)

        # Verificar se chegou à resposta final
        if "ACTION: ANSWER" in resposta:
            print("\n✅ Agente chegou à resposta final!")
            break

        # Simular execução da ação
        if "ACTION: SEARCH" in resposta:
            # Em produção, aqui executaria a busca real
            observacao = "[Simulação] População do Brasil: ~215 milhões. População mundial: ~8 bilhões."
        elif "ACTION: CALCULATE" in resposta:
            # Em produção, aqui executaria o cálculo real
            observacao = "[Simulação] 215/8000 * 100 = 2.69%"
        else:
            observacao = "[Ação não reconhecida]"

        print(f"\nOBSERVATION: {observacao}")
        contexto += f"\n{resposta}\nOBSERVATION: {observacao}"

        iteracao += 1

    return "Demonstração concluída"


def main():
    print("=" * 60)
    print("ReAct AGENT - Demonstração")
    print("=" * 60)

    # Reset do tracker
    token_tracker.reset()

    # Demonstração 1: Padrão ReAct Manual (educacional)
    demonstrar_react_manual()

    # Demonstração 2: Agente ReAct Completo
    print("\n" + "=" * 60)
    print("🤖 AGENTE ReAct COM FERRAMENTAS REAIS")
    print("=" * 60)

    perguntas = [
        "Quem ganhou a última Copa do Mundo de futebol e em que país foi realizada?",
        "Calcule quanto é 15% de 3500 e depois multiplique por 12.",
    ]

    for i, pergunta in enumerate(perguntas, 1):
        print(f"\n{'='*60}")
        print(f"📝 Pergunta {i}: {pergunta}")
        print("=" * 60)

        try:
            resposta = executar_agente(pergunta)
            print(f"\n🎯 Resposta Final: {resposta}")
        except Exception as e:
            print(f"\n❌ Erro: {str(e)}")

        print()

    # Exibir total de tokens
    print_total_usage(token_tracker, "TOTAL - ReAct Agent")

    print("\nFim da demonstração ReAct Agent")
    print("=" * 60)


if __name__ == "__main__":
    main()
