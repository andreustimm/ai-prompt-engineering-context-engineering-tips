"""
Recuperação Ponderada por Tempo

Incorpora relevância temporal na recuperação, dando preferência
a documentos mais recentes. Útil quando atualidade importa.

Componentes:
- Metadados de timestamp: Documentos têm horários de criação/atualização
- Função de decaimento: Documentos mais antigos recebem scores menores
- Score combinado: Similaridade semântica * peso temporal

Casos de uso:
- Notícias e eventos atuais
- Histórico de chat (mensagens recentes mais relevantes)
- Documentação (preferir versões mais recentes)
- Análise de logs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_DISPONIVEL = True
except ImportError:
    CHROMA_DISPONIVEL = False

from config import (
    get_llm,
    get_embeddings,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

token_tracker = TokenUsage()


def criar_documentos_com_tempo() -> list[Document]:
    """Cria documentos de exemplo com timestamps."""
    agora = datetime.now()

    documentos = [
        Document(
            page_content="Urgente: Empresa de IA anuncia grande avanço em capacidades de raciocínio. O novo modelo mostra desempenho sem precedentes em tarefas complexas.",
            metadata={"fonte": "noticias", "timestamp": (agora - timedelta(hours=2)).isoformat(), "titulo": "Avanço em IA"}
        ),
        Document(
            page_content="Previsão do tempo: Condições ensolaradas esperadas durante toda a semana com temperaturas em torno de 25°C.",
            metadata={"fonte": "tempo", "timestamp": (agora - timedelta(hours=6)).isoformat(), "titulo": "Atualização do Tempo"}
        ),
        Document(
            page_content="Visão histórica do desenvolvimento da inteligência artificial dos anos 1950 até 2020.",
            metadata={"fonte": "artigo", "timestamp": (agora - timedelta(days=180)).isoformat(), "titulo": "História da IA"}
        ),
        Document(
            page_content="Melhores práticas de machine learning atualizadas para 2024: Use RAG para tarefas intensivas em conhecimento.",
            metadata={"fonte": "guia", "timestamp": (agora - timedelta(days=30)).isoformat(), "titulo": "Melhores Práticas ML 2024"}
        ),
        Document(
            page_content="Atualização de política da empresa: Diretrizes de trabalho remoto foram revisadas com efeito imediato.",
            metadata={"fonte": "politica", "timestamp": (agora - timedelta(hours=12)).isoformat(), "titulo": "Atualização de Política"}
        ),
        Document(
            page_content="Documentação legada para estratégias de migração Python 2.7.",
            metadata={"fonte": "docs", "timestamp": (agora - timedelta(days=365)).isoformat(), "titulo": "Migração Python 2.7"}
        ),
        Document(
            page_content="Novo artigo de pesquisa sobre arquiteturas transformer mostra eficiência melhorada.",
            metadata={"fonte": "pesquisa", "timestamp": (agora - timedelta(days=7)).isoformat(), "titulo": "Pesquisa Transformer"}
        ),
        Document(
            page_content="Relatório financeiro trimestral mostra forte crescimento em investimentos no setor de IA.",
            metadata={"fonte": "financeiro", "timestamp": (agora - timedelta(days=45)).isoformat(), "titulo": "Relatório Financeiro Q3"}
        ),
    ]

    return documentos


def calcular_peso_temporal(timestamp_str: str, taxa_decaimento: float = 0.01, unidade_tempo: str = "horas") -> float:
    """
    Calcula peso baseado em tempo usando decaimento exponencial.

    peso = exp(-taxa_decaimento * tempo_decorrido)

    Args:
        timestamp_str: Timestamp em formato ISO
        taxa_decaimento: Quão rápido relevância decai (maior = decai mais rápido)
        unidade_tempo: "horas", "dias" ou "minutos"

    Returns:
        Peso entre 0 e 1 (1 = mais recente)
    """
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        decorrido = datetime.now() - timestamp

        if unidade_tempo == "horas":
            valor_tempo = decorrido.total_seconds() / 3600
        elif unidade_tempo == "dias":
            valor_tempo = decorrido.total_seconds() / 86400
        else:  # minutos
            valor_tempo = decorrido.total_seconds() / 60

        peso = math.exp(-taxa_decaimento * valor_tempo)
        return max(0.01, min(1.0, peso))  # Limita entre 0.01 e 1
    except:
        return 0.5  # Peso padrão se parsing falhar


class RetrieverPonderadoPorTempo:
    """Retriever que combina similaridade semântica com decaimento temporal."""

    def __init__(self, documentos: list[Document], taxa_decaimento: float = 0.01, unidade_tempo: str = "horas", nome_colecao: str = "ponderado_tempo"):
        """
        Inicializa retriever ponderado por tempo.

        Args:
            documentos: Documentos com metadados de timestamp
            taxa_decaimento: Taxa de decaimento para ponderação temporal
            unidade_tempo: Unidade de tempo para cálculo de decaimento
        """
        self.documentos = documentos
        self.taxa_decaimento = taxa_decaimento
        self.unidade_tempo = unidade_tempo
        self.embeddings = get_embeddings()

        # Cria vector store
        self.vectorstore = Chroma.from_documents(
            documents=documentos,
            embedding=self.embeddings,
            collection_name=nome_colecao
        )

    def recuperar(self, consulta: str, k: int = 5, fator_peso_tempo: float = 0.5) -> list[tuple[Document, float, float, float]]:
        """
        Recupera com pontuação ponderada por tempo.

        Args:
            consulta: Consulta de busca
            k: Número de resultados
            fator_peso_tempo: Balanço entre semântico (0) e tempo (1)

        Returns:
            Lista de (doc, score_combinado, score_semantico, peso_tempo)
        """
        # Obtém mais resultados do que necessário para reranking
        resultados = self.vectorstore.similarity_search_with_score(consulta, k=k * 2)

        # Calcula scores combinados
        resultados_pontuados = []
        for doc, distancia in resultados:
            score_semantico = 1 / (1 + distancia)  # Converte distância para similaridade
            timestamp = doc.metadata.get("timestamp", datetime.now().isoformat())
            peso_tempo = calcular_peso_temporal(timestamp, self.taxa_decaimento, self.unidade_tempo)

            # Score combinado
            combinado = (1 - fator_peso_tempo) * score_semantico + fator_peso_tempo * peso_tempo
            resultados_pontuados.append((doc, combinado, score_semantico, peso_tempo))

        # Ordena por score combinado
        resultados_pontuados.sort(key=lambda x: x[1], reverse=True)

        return resultados_pontuados[:k]


def comparar_com_sem_ponderacao_tempo(documentos: list[Document], consulta: str):
    """Compara resultados com e sem ponderação temporal."""

    print(f"\n   Consulta: '{consulta}'")
    print("   " + "=" * 50)

    # Recuperação padrão (sem ponderação temporal)
    print("\n   📚 RECUPERAÇÃO PADRÃO (Apenas Semântico)")
    print("-" * 30)

    vectorstore = Chroma.from_documents(documentos, get_embeddings(), collection_name="comparar_padrao")
    resultados = vectorstore.similarity_search_with_score(consulta, k=4)

    for i, (doc, score) in enumerate(resultados, 1):
        timestamp = doc.metadata.get("timestamp", "Desconhecido")
        idade = obter_string_idade(timestamp)
        print(f"\n   {i}. {doc.metadata.get('titulo', 'Sem título')}")
        print(f"      Idade: {idade}")
        print(f"      Score: {1/(1+score):.4f}")
        print(f"      Preview: {doc.page_content[:60]}...")

    # Recuperação ponderada por tempo
    print("\n   ⏰ RECUPERAÇÃO PONDERADA POR TEMPO")
    print("-" * 30)

    retriever = RetrieverPonderadoPorTempo(documentos, taxa_decaimento=0.05, unidade_tempo="dias", nome_colecao="comparar_tempo")
    resultados = retriever.recuperar(consulta, k=4, fator_peso_tempo=0.4)

    for i, (doc, combinado, semantico, peso_t) in enumerate(resultados, 1):
        timestamp = doc.metadata.get("timestamp", "Desconhecido")
        idade = obter_string_idade(timestamp)
        print(f"\n   {i}. {doc.metadata.get('titulo', 'Sem título')}")
        print(f"      Idade: {idade}")
        print(f"      Semântico: {semantico:.4f} | Tempo: {peso_t:.4f} | Combinado: {combinado:.4f}")
        print(f"      Preview: {doc.page_content[:60]}...")


def obter_string_idade(timestamp_str: str) -> str:
    """Converte timestamp para idade legível."""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        decorrido = datetime.now() - timestamp

        if decorrido.days > 0:
            return f"{decorrido.days} dias atrás"
        elif decorrido.seconds > 3600:
            return f"{decorrido.seconds // 3600} horas atrás"
        else:
            return f"{decorrido.seconds // 60} minutos atrás"
    except:
        return "Desconhecido"


def gerar_resposta(consulta: str, documentos: list[Document]) -> str:
    """Gera resposta dos documentos recuperados."""
    llm = get_llm(temperature=0.3)

    contexto = "\n\n".join([
        f"[{doc.metadata.get('titulo', 'Sem título')} - {obter_string_idade(doc.metadata.get('timestamp', ''))}]\n{doc.page_content}"
        for doc in documentos
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda baseado no contexto. Priorize informações recentes quando relevante."),
        ("user", "Contexto:\n{contexto}\n\nPergunta: {pergunta}\n\nResposta:")
    ])

    response = (prompt | llm).invoke({"contexto": contexto, "pergunta": consulta})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Resposta")
    return response.content


def main():
    print("=" * 60)
    print("RECUPERAÇÃO PONDERADA POR TEMPO")
    print("=" * 60)

    if not CHROMA_DISPONIVEL:
        print("\nErro: chromadb necessário. Instale: pip install chromadb")
        return

    token_tracker.reset()

    print("\n📚 CRIANDO DOCUMENTOS COM TIMESTAMP")
    print("-" * 40)
    documentos = criar_documentos_com_tempo()
    print(f"   Criados {len(documentos)} documentos com timestamps")

    for doc in documentos:
        idade = obter_string_idade(doc.metadata.get("timestamp", ""))
        print(f"   - {doc.metadata.get('titulo')}: {idade}")

    print("\n🔧 CRIANDO RETRIEVER PONDERADO POR TEMPO")
    print("-" * 40)
    retriever = RetrieverPonderadoPorTempo(
        documentos,
        taxa_decaimento=0.05,  # Decaimento por dia
        unidade_tempo="dias",
        nome_colecao="demo_ponderado_tempo"
    )
    print("   Taxa de decaimento: 0.05 por dia")
    print("   Documentos recentes terão score maior!")

    consultas = [
        "Quais são as últimas notícias sobre IA?",
        "Me conte sobre melhores práticas de machine learning",
    ]

    print("\n\n❓ CONSULTAS PONDERADAS POR TEMPO")
    print("=" * 60)

    for consulta in consultas:
        print(f"\n📌 Consulta: '{consulta}'")
        print("-" * 40)

        resultados = retriever.recuperar(consulta, k=3, fator_peso_tempo=0.4)

        print("\n   Resultados (com ponderação temporal):")
        for i, (doc, combinado, semantico, peso_t) in enumerate(resultados, 1):
            idade = obter_string_idade(doc.metadata.get("timestamp", ""))
            print(f"\n   {i}. {doc.metadata.get('titulo')}")
            print(f"      Idade: {idade}")
            print(f"      Scores: semântico={semantico:.3f}, tempo={peso_t:.3f}, combinado={combinado:.3f}")

        docs_resposta = [r[0] for r in resultados]
        resposta = gerar_resposta(consulta, docs_resposta)
        print(f"\n   Resposta: {resposta[:250]}...")

    print("\n\n📊 COMPARAÇÃO: COM vs SEM PONDERAÇÃO TEMPORAL")
    print("=" * 60)
    comparar_com_sem_ponderacao_tempo(documentos, "Quais são os últimos desenvolvimentos em IA?")

    print("\n\n💡 CONFIGURAÇÃO DE PONDERAÇÃO TEMPORAL")
    print("-" * 40)
    print("""
   | Caso de Uso       | Taxa Decaim | Unidade | Fator Peso |
   |-------------------|-------------|---------|------------|
   | Notícias/Atual    | 0.1-0.5     | horas   | 0.5-0.7    |
   | Histórico chat    | 0.05-0.1    | horas   | 0.3-0.5    |
   | Documentação      | 0.01-0.05   | dias    | 0.2-0.4    |
   | Artigos pesquisa  | 0.001-0.01  | dias    | 0.1-0.3    |

   Maior taxa_decaimento = envelhecimento mais rápido dos documentos
   Maior fator_peso = mais preferência para docs recentes
    """)

    print_total_usage(token_tracker, "TOTAL - Recuperação Ponderada por Tempo")
    print("\nFim da demonstração de Recuperação Ponderada por Tempo")
    print("=" * 60)


if __name__ == "__main__":
    main()
