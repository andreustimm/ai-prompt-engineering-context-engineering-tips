"""
RAG Conversacional

Sistema RAG aprimorado com memória de conversa, permitindo interações
em múltiplos turnos onde o contexto de perguntas anteriores é mantido.

Componentes:
- Vector Store: ChromaDB para recuperação de documentos
- Memória de Conversa: Rastreia histórico do chat
- Reescrita de Pergunta: Reformula perguntas com contexto
- Recuperação Contextual: Usa conversa para melhor recuperação

Casos de uso:
- Chatbots baseados em documentos
- Suporte ao cliente sobre base de conhecimento
- Assistentes de pesquisa interativos
- Sistemas Q&A de múltiplos turnos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Optional

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Rastreador global de tokens
token_tracker = TokenUsage()


class MemoriaConversa:
    """Memória de conversa simples que armazena histórico do chat."""

    def __init__(self, max_historico: int = 10):
        self.historico: list[dict] = []
        self.max_historico = max_historico

    def adicionar_troca(self, mensagem_usuario: str, mensagem_assistente: str):
        """Adiciona uma troca usuário-assistente ao histórico."""
        self.historico.append({
            "usuario": mensagem_usuario,
            "assistente": mensagem_assistente
        })

        if len(self.historico) > self.max_historico:
            self.historico = self.historico[-self.max_historico:]

    def obter_historico_string(self) -> str:
        """Obtém histórico como string formatada."""
        if not self.historico:
            return "Sem conversa anterior."

        linhas = []
        for troca in self.historico:
            linhas.append(f"Usuário: {troca['usuario']}")
            linhas.append(f"Assistente: {troca['assistente']}")

        return "\n".join(linhas)

    def obter_contexto_recente(self, n: int = 3) -> str:
        """Obtém as últimas n trocas como contexto."""
        recentes = self.historico[-n:] if self.historico else []

        if not recentes:
            return ""

        linhas = []
        for troca in recentes:
            linhas.append(f"Usuário: {troca['usuario']}")
            linhas.append(f"Assistente: {troca['assistente']}")

        return "\n".join(linhas)

    def limpar(self):
        """Limpa histórico de conversa."""
        self.historico = []


def carregar_documentos(caminho: str) -> list[Document]:
    """Carrega documentos do caminho."""
    documentos = []
    caminho_obj = Path(caminho)

    if caminho_obj.is_file():
        arquivos = [caminho_obj]
    else:
        arquivos = list(caminho_obj.glob("**/*"))

    for caminho_arquivo in arquivos:
        ext = caminho_arquivo.suffix.lower()
        try:
            if ext in ['.txt', '.md']:
                with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                    conteudo = f.read()
                documentos.append(Document(page_content=conteudo, metadata={"source": str(caminho_arquivo)}))
            elif ext == '.pdf':
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(caminho_arquivo))
                    documentos.extend(loader.load())
                except ImportError:
                    pass
        except Exception as e:
            print(f"   Aviso: Não foi possível carregar {caminho_arquivo}: {e}")

    return documentos


def dividir_documentos(documentos: list[Document], tamanho_chunk: int = 800, sobreposicao_chunk: int = 150) -> list[Document]:
    """Divide documentos em chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao_chunk,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documentos)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


class RAGConversacional:
    """
    Sistema RAG com memória de conversa para interações de múltiplos turnos.
    """

    def __init__(self, nome_colecao: str = "rag_conversacional"):
        self.nome_colecao = nome_colecao
        self.vectorstore = None
        self.chunks = []
        self.memoria = MemoriaConversa(max_historico=10)

    def carregar_e_indexar(self, caminho: str, tamanho_chunk: int = 800, sobreposicao_chunk: int = 150):
        """Carrega documentos e cria índice vetorial."""
        print(f"\n   Carregando documentos de: {caminho}")
        documentos = carregar_documentos(caminho)
        print(f"   Carregados {len(documentos)} documento(s)")

        print(f"\n   Dividindo documentos...")
        self.chunks = dividir_documentos(documentos, tamanho_chunk, sobreposicao_chunk)
        print(f"   Criados {len(self.chunks)} chunks")

        print(f"\n   Criando vector store...")
        embeddings = get_embeddings()
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name=self.nome_colecao
        )
        print(f"   Vector store pronto!")

    def reescrever_pergunta_com_contexto(self, pergunta: str) -> str:
        """
        Reescreve a pergunta para ser autônoma usando histórico de conversa.
        Isso ajuda na recuperação quando perguntas contêm pronomes ou referências.
        """
        if not self.memoria.historico:
            return pergunta

        llm = get_llm(temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Dado o histórico de conversa e uma pergunta de acompanhamento,
reescreva a pergunta para ser uma pergunta autônoma que capture o contexto completo.
Se a pergunta já for autônoma, retorne-a sem alterações.
Retorne APENAS a pergunta reescrita, nada mais."""),
            ("user", """Histórico da Conversa:
{historico}

Pergunta de Acompanhamento: {pergunta}

Pergunta Autônoma:""")
        ])

        chain = prompt | llm
        response = chain.invoke({
            "historico": self.memoria.obter_contexto_recente(n=3),
            "pergunta": pergunta
        })

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, "Reescrita")

        return response.content.strip()

    def recuperar_documentos(self, consulta: str, k: int = 4) -> list[Document]:
        """Recupera documentos relevantes para a consulta."""
        if not self.vectorstore:
            raise ValueError("Nenhum documento indexado.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(consulta)

    def gerar_resposta(self, pergunta: str, docs_contexto: list[Document]) -> str:
        """Gera uma resposta usando contexto recuperado e histórico de conversa."""
        llm = get_llm(temperature=0.3)

        contexto = "\n\n---\n\n".join([
            f"[Fonte: {Path(doc.metadata.get('source', 'Desconhecido')).name}]\n{doc.page_content}"
            for doc in docs_contexto
        ])

        historico = self.memoria.obter_contexto_recente(n=5)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente útil engajado em uma conversa.
Use os documentos de contexto fornecidos para responder perguntas com precisão.
Considere o histórico de conversa para contexto sobre o que o usuário está perguntando.
Se a informação não estiver no contexto fornecido, diga isso.
Seja conversacional mas informativo."""),
            ("user", """Contexto dos Documentos:
{contexto}

Histórico da Conversa:
{historico}

Pergunta Atual: {pergunta}

Resposta:""")
        ])

        chain = prompt | llm
        response = chain.invoke({
            "contexto": contexto,
            "historico": historico if historico else "Sem conversa anterior.",
            "pergunta": pergunta
        })

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, "Geração")

        return response.content

    def conversar(self, pergunta: str, k: int = 4) -> dict:
        """
        Processa uma mensagem de chat com contexto do histórico de conversa.

        Args:
            pergunta: Pergunta do usuário
            k: Número de documentos para recuperar

        Retorna:
            Dicionário com resposta e metadados
        """
        if not self.vectorstore:
            raise ValueError("Nenhum documento indexado. Chame carregar_e_indexar primeiro.")

        print(f"\n   Reescrevendo pergunta com contexto...")
        pergunta_autonoma = self.reescrever_pergunta_com_contexto(pergunta)

        if pergunta_autonoma != pergunta:
            print(f"   Original: {pergunta}")
            print(f"   Reescrita: {pergunta_autonoma}")
        else:
            print(f"   Pergunta já é autônoma")

        print(f"\n   Recuperando {k} documentos relevantes...")
        docs_relevantes = self.recuperar_documentos(pergunta_autonoma, k)

        print(f"\n   Gerando resposta...")
        resposta = self.gerar_resposta(pergunta, docs_relevantes)

        self.memoria.adicionar_troca(pergunta, resposta)

        return {
            "pergunta": pergunta,
            "pergunta_autonoma": pergunta_autonoma,
            "resposta": resposta,
            "fontes": docs_relevantes,
            "tamanho_historico": len(self.memoria.historico)
        }

    def resetar_conversa(self):
        """Reseta o histórico de conversa."""
        self.memoria.limpar()
        print("   Histórico de conversa limpo.")

    def mostrar_historico(self):
        """Exibe o histórico de conversa."""
        print("\n📜 Histórico de Conversa:")
        print("-" * 40)

        if not self.memoria.historico:
            print("   Sem histórico de conversa.")
            return

        for i, troca in enumerate(self.memoria.historico, 1):
            print(f"\n   Turno {i}:")
            print(f"   Usuário: {troca['usuario'][:100]}...")
            print(f"   Assistente: {troca['assistente'][:100]}...")


def simular_conversa(rag: RAGConversacional):
    """Simula uma conversa de múltiplos turnos."""

    conversas = [
        "O que é machine learning?",
        "Quais são os principais tipos?",
        "Pode explicar o terceiro tipo em mais detalhes?",
        "Quais são algumas aplicações reais disso?",
        "E sobre preocupações éticas?",
    ]

    print("\n🗣️ SIMULANDO CONVERSA DE MÚLTIPLOS TURNOS")
    print("=" * 60)

    for i, pergunta in enumerate(conversas, 1):
        print(f"\n{'='*50}")
        print(f"📌 Turno {i}: {pergunta}")
        print("-" * 50)

        resultado = rag.conversar(pergunta, k=3)

        print(f"\n📋 Resposta:")
        print(resultado["resposta"][:800] + "..." if len(resultado["resposta"]) > 800 else resultado["resposta"])

        print(f"\n📊 Metadados:")
        print(f"   Pergunta foi reescrita: {resultado['pergunta'] != resultado['pergunta_autonoma']}")
        print(f"   Fontes usadas: {len(resultado['fontes'])}")
        print(f"   Tamanho do histórico: {resultado['tamanho_historico']}")


def main():
    print("=" * 60)
    print("RAG CONVERSACIONAL - Demo")
    print("=" * 60)

    if not CHROMA_AVAILABLE:
        print("\nErro: chromadb necessário. Instale com: pip install chromadb")
        return

    token_tracker.reset()

    diretorio_exemplos = Path(__file__).parent.parent.parent / "sample_data" / "documents"

    if not diretorio_exemplos.exists():
        print(f"\nErro: Diretório de exemplos não encontrado em {diretorio_exemplos}")
        return

    print("\n📚 INICIALIZANDO RAG CONVERSACIONAL")
    print("-" * 40)

    rag = RAGConversacional(nome_colecao="rag_conversacional_demo")

    rag.carregar_e_indexar(
        str(diretorio_exemplos),
        tamanho_chunk=600,
        sobreposicao_chunk=100
    )

    simular_conversa(rag)

    rag.mostrar_historico()

    print_total_usage(token_tracker, "TOTAL - RAG Conversacional")

    print("\nFim do demo RAG Conversacional")
    print("=" * 60)


if __name__ == "__main__":
    main()
