"""
Memory/Conversation (Memória/Conversa)

Técnicas para manter contexto e memória de conversa através de
múltiplas interações com um LLM.

Tipos de Memória:
- Buffer Memory: Armazena histórico completo de conversa
- Window Memory: Armazena últimas N trocas
- Summary Memory: Armazena histórico resumido
- Entity Memory: Rastreia entidades mencionadas

Casos de uso:
- Chatbots e assistentes virtuais
- Diálogos de múltiplos turnos
- Interações personalizadas
- Respostas conscientes de contexto
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Rastreador global de tokens
token_tracker = TokenUsage()


class MemoriaBuffer:
    """
    Memória buffer simples que armazena histórico completo de conversa.
    Boa para conversas curtas onde contexto completo é importante.
    """

    def __init__(self, max_mensagens: int = 100):
        self.mensagens: list = []
        self.max_mensagens = max_mensagens

    def adicionar_mensagem_usuario(self, conteudo: str):
        """Adiciona uma mensagem do usuário ao histórico."""
        self.mensagens.append(HumanMessage(content=conteudo))
        self._cortar_se_necessario()

    def adicionar_mensagem_ia(self, conteudo: str):
        """Adiciona uma mensagem da IA ao histórico."""
        self.mensagens.append(AIMessage(content=conteudo))
        self._cortar_se_necessario()

    def _cortar_se_necessario(self):
        """Mantém apenas as mensagens mais recentes se limite excedido."""
        if len(self.mensagens) > self.max_mensagens:
            self.mensagens = self.mensagens[-self.max_mensagens:]

    def obter_mensagens(self) -> list:
        """Obtém todas as mensagens."""
        return self.mensagens.copy()

    def limpar(self):
        """Limpa todas as mensagens."""
        self.mensagens = []


class MemoriaJanela:
    """
    Memória de janela que armazena apenas as últimas K trocas.
    Boa para conversas longas onde contexto recente importa mais.
    """

    def __init__(self, tamanho_janela: int = 5):
        self.mensagens: list = []
        self.tamanho_janela = tamanho_janela  # Número de trocas (pares usuário + IA)

    def adicionar_troca(self, mensagem_usuario: str, mensagem_ia: str):
        """Adiciona uma troca completa."""
        self.mensagens.append(HumanMessage(content=mensagem_usuario))
        self.mensagens.append(AIMessage(content=mensagem_ia))
        self._cortar_janela()

    def _cortar_janela(self):
        """Mantém apenas mensagens dentro da janela."""
        max_mensagens = self.tamanho_janela * 2  # 2 mensagens por troca
        if len(self.mensagens) > max_mensagens:
            self.mensagens = self.mensagens[-max_mensagens:]

    def obter_mensagens(self) -> list:
        return self.mensagens.copy()

    def limpar(self):
        self.mensagens = []


class MemoriaResumo:
    """
    Memória que mantém um resumo contínuo da conversa.
    Boa para conversas muito longas para prevenir estouro de tokens.
    """

    def __init__(self):
        self.resumo: str = ""
        self.mensagens_recentes: list = []
        self.max_recentes = 4  # Mantém últimas 4 mensagens para contexto

    def adicionar_troca(self, mensagem_usuario: str, mensagem_ia: str):
        """Adiciona uma troca e potencialmente atualiza resumo."""
        self.mensagens_recentes.append(HumanMessage(content=mensagem_usuario))
        self.mensagens_recentes.append(AIMessage(content=mensagem_ia))

        # Se muitas mensagens recentes, incorporar no resumo
        if len(self.mensagens_recentes) > self.max_recentes:
            self._atualizar_resumo()

    def _atualizar_resumo(self):
        """Atualiza o resumo com mensagens antigas."""
        llm = get_llm(temperature=0)

        # Pega mensagens mais antigas para resumir
        para_resumir = self.mensagens_recentes[:-self.max_recentes]

        if not para_resumir:
            return

        contexto = "\n".join([
            f"{'Usuário' if isinstance(m, HumanMessage) else 'Assistente'}: {m.content}"
            for m in para_resumir
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Resuma a seguinte conversa, mantendo informações chave
e quaisquer detalhes ou decisões importantes. Seja conciso mas abrangente."""),
            ("user", """Resumo anterior (se houver):
{resumo_anterior}

Nova conversa para incorporar:
{conversa}

Resumo atualizado:""")
        ])

        chain = prompt | llm
        response = chain.invoke({
            "resumo_anterior": self.resumo or "Nenhum",
            "conversa": contexto
        })

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        self.resumo = response.content
        self.mensagens_recentes = self.mensagens_recentes[-self.max_recentes:]

    def obter_contexto(self) -> dict:
        """Obtém resumo e mensagens recentes."""
        return {
            "resumo": self.resumo,
            "mensagens_recentes": self.mensagens_recentes.copy()
        }

    def limpar(self):
        self.resumo = ""
        self.mensagens_recentes = []


class MemoriaEntidades:
    """
    Memória que rastreia entidades mencionadas na conversa.
    Boa para manter conhecimento sobre pessoas, lugares, coisas discutidas.
    """

    def __init__(self):
        self.entidades: dict = {}  # nome_entidade -> descrição/info

    def extrair_e_atualizar_entidades(self, texto: str):
        """Extrai entidades do texto e atualiza memória."""
        llm = get_llm(temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extraia entidades nomeadas do texto. Para cada entidade,
forneça uma breve descrição baseada no que foi mencionado.
Retorne como JSON: {"nome_entidade": "descrição", ...}
Inclua apenas entidades claramente nomeadas (pessoas, lugares, organizações, produtos)."""),
            ("user", "{texto}")
        ])

        chain = prompt | llm
        response = chain.invoke({"texto": texto})

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)

        # Analisa resposta
        try:
            import json
            conteudo = response.content
            inicio = conteudo.find('{')
            fim = conteudo.rfind('}') + 1
            if inicio >= 0 and fim > inicio:
                entidades = json.loads(conteudo[inicio:fim])
                self.entidades.update(entidades)
        except Exception:
            pass

    def obter_info_entidade(self, nome_entidade: str) -> Optional[str]:
        """Obtém informação sobre uma entidade específica."""
        return self.entidades.get(nome_entidade.lower())

    def obter_todas_entidades(self) -> dict:
        """Obtém todas as entidades rastreadas."""
        return self.entidades.copy()

    def limpar(self):
        self.entidades = {}


class CadeiaConversa:
    """
    Cadeia de conversa com memória configurável.
    """

    def __init__(self, tipo_memoria: str = "buffer", prompt_sistema: str = None):
        self.tipo_memoria = tipo_memoria
        self.prompt_sistema = prompt_sistema or "Você é um assistente útil."

        # Inicializa memória apropriada
        if tipo_memoria == "buffer":
            self.memoria = MemoriaBuffer()
        elif tipo_memoria == "janela":
            self.memoria = MemoriaJanela(tamanho_janela=5)
        elif tipo_memoria == "resumo":
            self.memoria = MemoriaResumo()
        else:
            self.memoria = MemoriaBuffer()

    def conversar(self, entrada_usuario: str) -> str:
        """Processa entrada do usuário e gera resposta."""
        llm = get_llm(temperature=0.7)

        # Constrói lista de mensagens
        mensagens = [SystemMessage(content=self.prompt_sistema)]

        if self.tipo_memoria == "resumo":
            contexto = self.memoria.obter_contexto()
            if contexto["resumo"]:
                mensagens.append(SystemMessage(content=f"Resumo da conversa: {contexto['resumo']}"))
            mensagens.extend(contexto["mensagens_recentes"])
        else:
            mensagens.extend(self.memoria.obter_mensagens())

        mensagens.append(HumanMessage(content=entrada_usuario))

        # Obtém resposta
        response = llm.invoke(mensagens)

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens)

        resposta_ia = response.content

        # Atualiza memória
        if self.tipo_memoria == "buffer":
            self.memoria.adicionar_mensagem_usuario(entrada_usuario)
            self.memoria.adicionar_mensagem_ia(resposta_ia)
        elif self.tipo_memoria == "janela":
            self.memoria.adicionar_troca(entrada_usuario, resposta_ia)
        elif self.tipo_memoria == "resumo":
            self.memoria.adicionar_troca(entrada_usuario, resposta_ia)

        return resposta_ia

    def resetar(self):
        """Reseta memória de conversa."""
        self.memoria.limpar()


def demo_memoria_buffer():
    """Demonstra memória buffer."""
    print("\n📚 DEMO MEMÓRIA BUFFER")
    print("-" * 40)

    cadeia = CadeiaConversa(
        tipo_memoria="buffer",
        prompt_sistema="Você é um conselheiro de viagens útil."
    )

    trocas = [
        "Olá! Estou planejando uma viagem ao Japão.",
        "Qual é a melhor época para visitar?",
        "Quais são os lugares imperdíveis em Tóquio?",
        "Qual foi a primeira coisa que mencionei?"
    ]

    for msg_usuario in trocas:
        print(f"\n👤 Usuário: {msg_usuario}")
        resposta = cadeia.conversar(msg_usuario)
        print(f"🤖 Assistente: {resposta[:300]}...")


def demo_memoria_janela():
    """Demonstra memória de janela."""
    print("\n📚 DEMO MEMÓRIA JANELA (janela=3)")
    print("-" * 40)

    cadeia = CadeiaConversa(
        tipo_memoria="janela",
        prompt_sistema="Você é um assistente de culinária útil."
    )

    trocas = [
        "Quero fazer macarrão.",
        "Quais ingredientes preciso para carbonara?",
        "Como cozinho o macarrão perfeitamente?",
        "E sobre o molho?",
        "O que eu originalmente queria fazer?"  # Testa se contexto inicial foi perdido
    ]

    for msg_usuario in trocas:
        print(f"\n👤 Usuário: {msg_usuario}")
        resposta = cadeia.conversar(msg_usuario)
        print(f"🤖 Assistente: {resposta[:300]}...")


def demo_memoria_entidades():
    """Demonstra memória de entidades."""
    print("\n📚 DEMO MEMÓRIA DE ENTIDADES")
    print("-" * 40)

    memoria_entidades = MemoriaEntidades()

    textos = [
        "Acabei de começar a trabalhar no Google em São Paulo. Minha gerente é Sarah Chen.",
        "Sarah mencionou que nossa equipe está trabalhando em um novo projeto de IA chamado Aurora.",
        "Estamos fazendo parceria com a USP para a pesquisa."
    ]

    for texto in textos:
        print(f"\n📝 Processando: {texto[:50]}...")
        memoria_entidades.extrair_e_atualizar_entidades(texto)

    print("\n📋 Entidades Rastreadas:")
    for entidade, info in memoria_entidades.obter_todas_entidades().items():
        print(f"   - {entidade}: {info}")


def main():
    print("=" * 60)
    print("MEMORY/CONVERSATION (MEMÓRIA/CONVERSA) - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Demo 1: Memória Buffer
    demo_memoria_buffer()

    # Demo 2: Memória Janela
    demo_memoria_janela()

    # Demo 3: Memória de Entidades
    demo_memoria_entidades()

    print_total_usage(token_tracker, "TOTAL - Memory/Conversation")

    print("\n\n" + "=" * 60)
    print("Resumo dos Tipos de Memória:")
    print("  - Buffer: Histórico completo, bom para conversas curtas")
    print("  - Janela: Últimas N trocas, bom para conversas longas")
    print("  - Resumo: Histórico comprimido, previne estouro de tokens")
    print("  - Entidades: Rastreia entidades para persistência de conhecimento")
    print("=" * 60)

    print("\nFim do demo Memory/Conversation")
    print("=" * 60)


if __name__ == "__main__":
    main()
