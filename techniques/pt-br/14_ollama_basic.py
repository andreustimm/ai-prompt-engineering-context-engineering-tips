"""
Ollama Básico - LLM Local

Usando modelos LLM locais via Ollama para aplicações de IA
focadas em privacidade e offline.

Modelos disponíveis:
- llama3.2: Propósito geral (padrão)
- mistral: Rápido e eficiente
- codellama: Geração de código
- phi3: Pequeno mas capaz

Pré-requisitos:
1. Instalar Ollama: https://ollama.ai
2. Iniciar servidor: ollama serve
3. Baixar modelo: ollama pull llama3.2

Casos de uso:
- Aplicações sensíveis à privacidade
- Operação offline
- Desenvolvimento/teste econômico
- Experimentação local
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from config import (
    get_ollama_llm,
    is_ollama_available,
    TokenUsage,
    print_total_usage
)

# Rastreador global de tokens (Ollama não fornece contagem de tokens)
token_tracker = TokenUsage()


def verificar_status_ollama():
    """Verifica se o Ollama está rodando e disponível."""
    if is_ollama_available():
        print("   ✓ Ollama está rodando e acessível")
        return True
    else:
        print("   ✗ Ollama não está disponível")
        print("   Por favor, certifique-se que o Ollama está instalado e rodando:")
        print("   1. Instale de https://ollama.ai")
        print("   2. Execute: ollama serve")
        print("   3. Baixe um modelo: ollama pull llama3.2")
        return False


def completar_basico(prompt: str, modelo: str = None) -> str:
    """Completar texto simples com Ollama."""
    llm = get_ollama_llm(model=modelo, temperature=0.7)

    response = llm.invoke(prompt)
    return response.content


def chat_com_prompt_sistema(mensagem_usuario: str, prompt_sistema: str, modelo: str = None) -> str:
    """Completar chat com prompt de sistema."""
    llm = get_ollama_llm(model=modelo, temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_sistema),
        ("user", "{mensagem}")
    ])

    chain = prompt | llm
    response = chain.invoke({"mensagem": mensagem_usuario})
    return response.content


def resumir_texto(texto: str, modelo: str = None) -> str:
    """Resume texto usando LLM local."""
    llm = get_ollama_llm(model=modelo, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em resumos. Forneça resumos concisos."),
        ("user", "Resuma o seguinte texto em 2-3 frases:\n\n{texto}")
    ])

    chain = prompt | llm
    response = chain.invoke({"texto": texto})
    return response.content


def traduzir_texto(texto: str, idioma_destino: str, modelo: str = None) -> str:
    """Traduz texto usando LLM local."""
    llm = get_ollama_llm(model=modelo, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um tradutor profissional. Traduza com precisão mantendo o significado original."),
        ("user", "Traduza o seguinte texto para {idioma}:\n\n{texto}")
    ])

    chain = prompt | llm
    response = chain.invoke({"texto": texto, "idioma": idioma_destino})
    return response.content


def gerar_codigo(descricao: str, linguagem: str = "Python", modelo: str = None) -> str:
    """Gera código a partir de descrição usando LLM local."""
    if modelo is None:
        modelo = "codellama"

    try:
        llm = get_ollama_llm(model=modelo, temperature=0.2)
    except Exception:
        llm = get_ollama_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Você é um programador especialista em {linguagem}. Escreva código limpo e bem documentado."),
        ("user", """Escreva código {linguagem} para o seguinte:

{descricao}

Forneça apenas o código com comentários, sem explicações fora do código.""")
    ])

    chain = prompt | llm
    response = chain.invoke({"descricao": descricao, "linguagem": linguagem})
    return response.content


def analisar_sentimento(texto: str, modelo: str = None) -> str:
    """Analisa sentimento do texto."""
    llm = get_ollama_llm(model=modelo, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analise o sentimento do texto e responda com:
SENTIMENTO: [POSITIVO/NEGATIVO/NEUTRO]
CONFIANÇA: [ALTA/MÉDIA/BAIXA]
EXPLICAÇÃO: [Breve explicação]"""),
        ("user", "{texto}")
    ])

    chain = prompt | llm
    response = chain.invoke({"texto": texto})
    return response.content


def responder_pergunta(pergunta: str, contexto: str = None, modelo: str = None) -> str:
    """Responde uma pergunta, opcionalmente com contexto."""
    llm = get_ollama_llm(model=modelo, temperature=0.5)

    if contexto:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Responda perguntas baseado no contexto fornecido. Se a resposta não estiver no contexto, diga isso."),
            ("user", """Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:""")
        ])
        inputs = {"contexto": contexto, "pergunta": pergunta}
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente útil. Forneça respostas precisas e informativas."),
            ("user", "{pergunta}")
        ])
        inputs = {"pergunta": pergunta}

    chain = prompt | llm
    response = chain.invoke(inputs)
    return response.content


def comparar_modelos(prompt: str, modelos: list[str] = None) -> dict:
    """Compara respostas de diferentes modelos."""
    if modelos is None:
        modelos = ["llama3.2", "mistral"]

    resultados = {}

    for modelo in modelos:
        try:
            print(f"\n   Testando {modelo}...")
            llm = get_ollama_llm(model=modelo, temperature=0.7)
            response = llm.invoke(prompt)
            resultados[modelo] = {
                "sucesso": True,
                "resposta": response.content
            }
        except Exception as e:
            resultados[modelo] = {
                "sucesso": False,
                "erro": str(e)
            }

    return resultados


def main():
    print("=" * 60)
    print("OLLAMA BÁSICO - Demo de LLM Local")
    print("=" * 60)

    # Verifica disponibilidade do Ollama
    print("\n🔍 VERIFICANDO STATUS DO OLLAMA")
    print("-" * 40)

    if not verificar_status_ollama():
        print("\nDemo não pode continuar sem o Ollama rodando.")
        print("Por favor, inicie o Ollama e tente novamente.")
        return

    # Reseta rastreador
    token_tracker.reset()

    # Exemplo 1: Completar Básico
    print("\n\n📝 COMPLETAR BÁSICO")
    print("-" * 40)

    prompt = "Explique o que é inteligência artificial em um parágrafo."
    print(f"\nPrompt: {prompt}")
    print("\nResposta:")
    print(completar_basico(prompt))

    # Exemplo 2: Chat com Prompt de Sistema
    print("\n\n💬 CHAT COM PROMPT DE SISTEMA")
    print("-" * 40)

    sistema = "Você é um assistente de culinária útil. Forneça dicas práticas de cozinha."
    mensagem = "Como fazer ovos mexidos fofos?"

    print(f"\nSistema: {sistema}")
    print(f"Usuário: {mensagem}")
    print("\nResposta:")
    print(chat_com_prompt_sistema(mensagem, sistema))

    # Exemplo 3: Sumarização
    print("\n\n📋 SUMARIZAÇÃO DE TEXTO")
    print("-" * 40)

    texto_longo = """
    Machine learning é um subconjunto da inteligência artificial que permite que computadores
    aprendam e melhorem a partir da experiência sem serem explicitamente programados.
    Foca no desenvolvimento de algoritmos que podem acessar dados, aprender com eles,
    e fazer previsões ou decisões. Algoritmos de machine learning são usados em
    uma grande variedade de aplicações, como filtragem de email, visão computacional,
    e reconhecimento de fala. Os três tipos principais são aprendizado supervisionado,
    aprendizado não supervisionado e aprendizado por reforço, cada um adequado para diferentes
    tipos de problemas e dados.
    """

    print(f"\nOriginal ({len(texto_longo)} caracteres)")
    print("\nResumo:")
    print(resumir_texto(texto_longo))

    # Exemplo 4: Tradução
    print("\n\n🌐 TRADUÇÃO")
    print("-" * 40)

    texto_para_traduzir = "The quick brown fox jumps over the lazy dog."
    print(f"\nOriginal: {texto_para_traduzir}")
    print("\nPortuguês:")
    print(traduzir_texto(texto_para_traduzir, "Português"))

    # Exemplo 5: Análise de Sentimento
    print("\n\n😊 ANÁLISE DE SENTIMENTO")
    print("-" * 40)

    avaliacoes = [
        "Este produto superou todas as minhas expectativas! Amei absolutamente!",
        "Experiência terrível. O item chegou quebrado e o suporte não ajudou.",
        "É ok. Faz o que deveria fazer, nada especial."
    ]

    for avaliacao in avaliacoes:
        print(f"\nAvaliação: {avaliacao[:50]}...")
        print(analisar_sentimento(avaliacao))

    # Exemplo 6: Geração de Código
    print("\n\n💻 GERAÇÃO DE CÓDIGO")
    print("-" * 40)

    pedido_codigo = "uma função que calcula a sequência de Fibonacci até n números"
    print(f"\nPedido: {pedido_codigo}")
    print("\nCódigo Gerado:")
    print(gerar_codigo(pedido_codigo, "Python"))

    # Exemplo 7: Q&A
    print("\n\n❓ PERGUNTAS E RESPOSTAS")
    print("-" * 40)

    contexto = """
    A Torre Eiffel é uma torre de treliça de ferro forjado no Champ de Mars em Paris.
    Foi construída de 1887 a 1889 como peça central da Feira Mundial de 1889.
    A torre tem 330 metros de altura e foi a estrutura feita pelo homem mais alta do mundo
    até 1930. Recebeu o nome do engenheiro Gustave Eiffel, cuja empresa projetou
    e construiu a torre.
    """

    pergunta = "Quando a Torre Eiffel foi construída e qual sua altura?"
    print(f"\nContexto fornecido: Sim")
    print(f"Pergunta: {pergunta}")
    print("\nResposta:")
    print(responder_pergunta(pergunta, contexto))

    print("\n\n" + "=" * 60)
    print("Nota: Ollama não fornece contagem de tokens como OpenAI.")
    print("Para rastreamento de custos, monitore logs do servidor Ollama ou use timing.")
    print("=" * 60)

    print("\nFim do demo Ollama Básico")
    print("=" * 60)


if __name__ == "__main__":
    main()
