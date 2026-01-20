"""
Tool Calling (Chamada de Ferramentas)

Técnica que permite que LLMs invoquem ferramentas/funções externas para
executar ações ou recuperar informações além de seus dados de treinamento.

Recursos:
- Definições de ferramentas customizadas
- Parsing automático de argumentos
- Execução de ferramentas e tratamento de respostas
- Fluxos de múltiplas ferramentas

Casos de uso:
- Calculadora e operações matemáticas
- Consultas de banco de dados
- Integrações de API
- Operações de sistema
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Rastreador global de tokens
token_tracker = TokenUsage()


# Define ferramentas customizadas usando o decorador @tool

@tool
def calcular(expressao: str) -> str:
    """
    Avalia uma expressão matemática.

    Args:
        expressao: Uma expressão matemática para avaliar (ex: "2 + 2 * 3")

    Returns:
        O resultado do cálculo
    """
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        resultado = eval(expressao, {"__builtins__": {}}, allowed_names)
        return f"Resultado: {resultado}"
    except Exception as e:
        return f"Erro ao calcular: {str(e)}"


@tool
def obter_hora_atual(fuso_horario: Optional[str] = None) -> str:
    """
    Obtém a data e hora atuais.

    Args:
        fuso_horario: Fuso horário opcional (não implementado, usa hora local)

    Returns:
        Data e hora atuais como string
    """
    agora = datetime.now()
    return f"Data e hora atuais: {agora.strftime('%d/%m/%Y %H:%M:%S')}"


@tool
def obter_clima(cidade: str) -> str:
    """
    Obtém informações de clima para uma cidade (simulado).

    Args:
        cidade: Nome da cidade

    Returns:
        Informações de clima para a cidade
    """
    dados_clima = {
        "são paulo": {"temp": 25, "condicao": "Parcialmente nublado", "umidade": 72},
        "rio de janeiro": {"temp": 30, "condicao": "Ensolarado", "umidade": 68},
        "brasília": {"temp": 27, "condicao": "Nublado", "umidade": 55},
        "curitiba": {"temp": 18, "condicao": "Chuvoso", "umidade": 85},
        "salvador": {"temp": 28, "condicao": "Ensolarado", "umidade": 75},
        "new york": {"temp": 22, "condicao": "Parcialmente nublado", "umidade": 65},
    }

    cidade_lower = cidade.lower()
    if cidade_lower in dados_clima:
        dados = dados_clima[cidade_lower]
        return f"Clima em {cidade}: {dados['temp']}°C, {dados['condicao']}, Umidade: {dados['umidade']}%"
    else:
        return f"Dados de clima não disponíveis para {cidade}. Tente: São Paulo, Rio de Janeiro, Brasília, Curitiba, Salvador, New York"


@tool
def buscar_banco_dados(consulta: str, tabela: str = "produtos") -> str:
    """
    Busca em um banco de dados simulado (apenas demonstração).

    Args:
        consulta: Consulta de busca
        tabela: Tabela para buscar (produtos, usuarios, pedidos)

    Returns:
        Resultados de busca simulados
    """
    bancos_dados = {
        "produtos": [
            {"id": 1, "nome": "Laptop Pro", "preco": 6499, "categoria": "Eletrônicos"},
            {"id": 2, "nome": "Mouse Sem Fio", "preco": 149, "categoria": "Eletrônicos"},
            {"id": 3, "nome": "Cadeira Escritório", "preco": 899, "categoria": "Móveis"},
        ],
        "usuarios": [
            {"id": 1, "nome": "João Silva", "email": "joao@exemplo.com"},
            {"id": 2, "nome": "Maria Santos", "email": "maria@exemplo.com"},
        ],
        "pedidos": [
            {"id": 101, "usuario_id": 1, "produto_id": 1, "status": "enviado"},
            {"id": 102, "usuario_id": 2, "produto_id": 3, "status": "processando"},
        ],
    }

    if tabela not in bancos_dados:
        return f"Tabela '{tabela}' não encontrada. Disponíveis: produtos, usuarios, pedidos"

    resultados = bancos_dados[tabela]
    consulta_lower = consulta.lower()
    correspondentes = [r for r in resultados if consulta_lower in str(r).lower()]

    if correspondentes:
        return f"Encontrados {len(correspondentes)} resultados em {tabela}:\n" + json.dumps(correspondentes, indent=2, ensure_ascii=False)
    else:
        return f"Nenhum resultado encontrado para '{consulta}' em {tabela}"


@tool
def converter_unidades(valor: float, unidade_origem: str, unidade_destino: str) -> str:
    """
    Converte entre unidades comuns.

    Args:
        valor: O valor numérico para converter
        unidade_origem: Unidade de origem (km, milhas, kg, lb, c, f)
        unidade_destino: Unidade de destino (km, milhas, kg, lb, c, f)

    Returns:
        Valor convertido com unidades
    """
    conversoes = {
        ("km", "milhas"): lambda x: x * 0.621371,
        ("milhas", "km"): lambda x: x * 1.60934,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("c", "f"): lambda x: x * 9/5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5/9,
    }

    chave = (unidade_origem.lower(), unidade_destino.lower())
    if chave in conversoes:
        resultado = conversoes[chave](valor)
        return f"{valor} {unidade_origem} = {resultado:.2f} {unidade_destino}"
    else:
        return f"Conversão de {unidade_origem} para {unidade_destino} não suportada"


def executar_agente_com_ferramentas(consulta: str, ferramentas: list, max_iteracoes: int = 5) -> str:
    """
    Executa um agente que pode chamar ferramentas para responder consultas.

    Args:
        consulta: Pergunta ou solicitação do usuário
        ferramentas: Lista de ferramentas disponíveis para o agente
        max_iteracoes: Número máximo de chamadas de ferramentas

    Retorna:
        Resposta final do agente
    """
    llm = get_llm(temperature=0)
    llm_com_ferramentas = llm.bind_tools(ferramentas)

    mensagens = [HumanMessage(content=consulta)]

    for i in range(max_iteracoes):
        print(f"\n   Iteração {i+1}...")

        resposta = llm_com_ferramentas.invoke(mensagens)
        mensagens.append(resposta)

        input_tokens, output_tokens = extract_tokens_from_response(resposta)
        token_tracker.add(input_tokens, output_tokens)

        if not resposta.tool_calls:
            print(f"   Não são necessárias mais chamadas de ferramentas")
            return resposta.content

        for chamada_ferramenta in resposta.tool_calls:
            nome_ferramenta = chamada_ferramenta["name"]
            args_ferramenta = chamada_ferramenta["args"]
            print(f"   Chamando ferramenta: {nome_ferramenta}({args_ferramenta})")

            for f in ferramentas:
                if f.name == nome_ferramenta:
                    resultado = f.invoke(args_ferramenta)
                    print(f"   Resultado da ferramenta: {resultado[:100]}...")

                    mensagens.append(ToolMessage(
                        content=str(resultado),
                        tool_call_id=chamada_ferramenta["id"]
                    ))
                    break

    return "Máximo de iterações atingido sem resposta final"


def main():
    print("=" * 60)
    print("TOOL CALLING (CHAMADA DE FERRAMENTAS) - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Define ferramentas disponíveis
    ferramentas = [calcular, obter_hora_atual, obter_clima, buscar_banco_dados, converter_unidades]

    print("\n📋 Ferramentas Disponíveis:")
    for f in ferramentas:
        print(f"   - {f.name}: {f.description.split('.')[0]}")

    # Exemplo 1: Calculadora
    print("\n\n🔢 EXEMPLO DE CALCULADORA")
    print("-" * 40)

    consulta1 = "Quanto é 15% de 250 mais 100?"
    print(f"\nConsulta: {consulta1}")
    print("\nExecutando...")
    resultado1 = executar_agente_com_ferramentas(consulta1, ferramentas)
    print(f"\n📋 Resposta Final: {resultado1}")

    # Exemplo 2: Clima
    print("\n\n🌤️ EXEMPLO DE CLIMA")
    print("-" * 40)

    consulta2 = "Como está o clima em São Paulo e Rio de Janeiro?"
    print(f"\nConsulta: {consulta2}")
    print("\nExecutando...")
    resultado2 = executar_agente_com_ferramentas(consulta2, ferramentas)
    print(f"\n📋 Resposta Final: {resultado2}")

    # Exemplo 3: Busca no Banco de Dados
    print("\n\n🔍 EXEMPLO DE BUSCA NO BANCO DE DADOS")
    print("-" * 40)

    consulta3 = "Encontre todos os produtos eletrônicos no banco de dados"
    print(f"\nConsulta: {consulta3}")
    print("\nExecutando...")
    resultado3 = executar_agente_com_ferramentas(consulta3, ferramentas)
    print(f"\n📋 Resposta Final: {resultado3}")

    # Exemplo 4: Conversão de Unidades
    print("\n\n📐 EXEMPLO DE CONVERSÃO DE UNIDADES")
    print("-" * 40)

    consulta4 = "Converta 100 quilômetros para milhas e 30 graus Celsius para Fahrenheit"
    print(f"\nConsulta: {consulta4}")
    print("\nExecutando...")
    resultado4 = executar_agente_com_ferramentas(consulta4, ferramentas)
    print(f"\n📋 Resposta Final: {resultado4}")

    # Exemplo 5: Consulta Multi-ferramenta
    print("\n\n🔄 EXEMPLO DE CONSULTA MULTI-FERRAMENTA")
    print("-" * 40)

    consulta5 = "Que horas são, como está o clima em Brasília, e calcule 20% de gorjeta em uma conta de R$85?"
    print(f"\nConsulta: {consulta5}")
    print("\nExecutando...")
    resultado5 = executar_agente_com_ferramentas(consulta5, ferramentas)
    print(f"\n📋 Resposta Final: {resultado5}")

    print_total_usage(token_tracker, "TOTAL - Tool Calling")

    print("\nFim do demo de Tool Calling")
    print("=" * 60)


if __name__ == "__main__":
    main()
