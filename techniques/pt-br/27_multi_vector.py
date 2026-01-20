"""
Recuperação Multi-Vetor

Armazena múltiplas representações vetoriais para cada documento:
- Resumos: Para correspondência de alto nível
- Perguntas hipotéticas: Para correspondência de perguntas
- Conteúdo original: Para contexto completo

Isso permite correspondência baseada em diferentes representações
enquanto retorna o documento original para contexto.

Casos de uso:
- Corresponder perguntas de usuários ao conteúdo de documentos
- Melhorar recuperação para tipos diversos de consultas
- Combinar diferentes representações semânticas
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


def carregar_documentos_exemplo() -> list[Document]:
    """Carrega documentos de exemplo."""
    sample_dir = Path(__file__).parent.parent.parent / "sample_data" / "documents"
    documentos = []
    for file_path in sample_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documentos.append(Document(page_content=f.read(), metadata={"fonte": file_path.name}))
        except Exception:
            pass
    return documentos


class StoreEmMemoria:
    """Store de documentos simples em memória."""
    def __init__(self):
        self.store = {}

    def adicionar(self, doc_id: str, doc: Document):
        self.store[doc_id] = doc

    def obter(self, doc_id: str) -> Document | None:
        return self.store.get(doc_id)


def gerar_resumo(texto: str) -> str:
    """Gera um resumo do texto."""
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Resuma o seguinte texto em 2-3 frases. Foque nos conceitos chave."),
        ("user", "{texto}")
    ])
    response = (prompt | llm).invoke({"texto": texto[:3000]})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    return response.content


def gerar_perguntas(texto: str, num_perguntas: int = 3) -> list[str]:
    """Gera perguntas hipotéticas que este texto poderia responder."""
    llm = get_llm(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Gere {num} perguntas que o seguinte texto poderia responder.
Retorne APENAS as perguntas, uma por linha, numeradas."""),
        ("user", "{texto}")
    ])
    response = (prompt | llm).invoke({"texto": texto[:3000], "num": num_perguntas})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)

    linhas = response.content.strip().split('\n')
    perguntas = []
    for linha in linhas:
        limpa = linha.strip()
        if limpa and limpa[0].isdigit():
            limpa = limpa.split('.', 1)[-1].strip()
        if limpa:
            perguntas.append(limpa)
    return perguntas[:num_perguntas]


class RetrieverMultiVetor:
    """Retriever com múltiplas representações vetoriais por documento."""

    def __init__(self, nome_colecao: str = "multi_vetor"):
        self.embeddings = get_embeddings()
        self.docstore = StoreEmMemoria()
        self.vectorstore = None
        self.nome_colecao = nome_colecao
        self.todos_vetores = []

    def adicionar_documentos(self, documentos: list[Document], gerar_resumos: bool = True, gerar_hipoteticas: bool = True):
        """Adiciona documentos com múltiplas representações."""
        divisor = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

        for doc in documentos:
            chunks = divisor.split_documents([doc])

            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                self.docstore.adicionar(doc_id, chunk)

                # Vetor 1: Conteúdo original (sempre)
                self.todos_vetores.append(Document(
                    page_content=chunk.page_content,
                    metadata={"doc_id": doc_id, "tipo": "original", "fonte": doc.metadata.get("fonte")}
                ))

                # Vetor 2: Resumo
                if gerar_resumos:
                    print(f"   Gerando resumo para chunk...")
                    resumo = gerar_resumo(chunk.page_content)
                    self.todos_vetores.append(Document(
                        page_content=resumo,
                        metadata={"doc_id": doc_id, "tipo": "resumo", "fonte": doc.metadata.get("fonte")}
                    ))

                # Vetor 3: Perguntas hipotéticas
                if gerar_hipoteticas:
                    print(f"   Gerando perguntas para chunk...")
                    perguntas = gerar_perguntas(chunk.page_content, num_perguntas=2)
                    for p in perguntas:
                        self.todos_vetores.append(Document(
                            page_content=p,
                            metadata={"doc_id": doc_id, "tipo": "pergunta", "fonte": doc.metadata.get("fonte")}
                        ))

        # Cria vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.todos_vetores,
            embedding=self.embeddings,
            collection_name=self.nome_colecao
        )
        return len(self.todos_vetores)

    def recuperar(self, consulta: str, k: int = 3) -> list[Document]:
        """Recupera documentos originais via correspondência multi-vetor."""
        if not self.vectorstore:
            raise ValueError("Nenhum documento adicionado.")

        resultados = self.vectorstore.similarity_search(consulta, k=k * 3)

        ids_vistos = set()
        docs_originais = []

        for resultado in resultados:
            doc_id = resultado.metadata.get("doc_id")
            if doc_id and doc_id not in ids_vistos:
                original = self.docstore.obter(doc_id)
                if original:
                    ids_vistos.add(doc_id)
                    docs_originais.append(original)
                    if len(docs_originais) >= k:
                        break

        return docs_originais

    def recuperar_com_info_match(self, consulta: str, k: int = 3) -> list[dict]:
        """Recupera com informação sobre o que correspondeu."""
        resultados = self.vectorstore.similarity_search_with_score(consulta, k=k * 5)

        info_doc = {}
        for resultado, score in resultados:
            doc_id = resultado.metadata.get("doc_id")
            tipo_match = resultado.metadata.get("tipo")

            if doc_id not in info_doc:
                original = self.docstore.obter(doc_id)
                if original:
                    info_doc[doc_id] = {
                        "original": original,
                        "matches": [],
                        "melhor_score": score
                    }

            if doc_id in info_doc:
                info_doc[doc_id]["matches"].append({
                    "tipo": tipo_match,
                    "conteudo": resultado.page_content[:100],
                    "score": score
                })

        resultados_ordenados = sorted(info_doc.values(), key=lambda x: x["melhor_score"])[:k]
        return resultados_ordenados


def gerar_resposta(consulta: str, documentos: list[Document]) -> str:
    """Gera resposta dos documentos recuperados."""
    llm = get_llm(temperature=0.3)
    contexto = "\n\n---\n\n".join([doc.page_content for doc in documentos])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda baseado no contexto. Seja completo mas conciso."),
        ("user", "Contexto:\n{contexto}\n\nPergunta: {pergunta}\n\nResposta:")
    ])
    response = (prompt | llm).invoke({"contexto": contexto, "pergunta": consulta})
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Geração de Resposta")
    return response.content


def main():
    print("=" * 60)
    print("RECUPERAÇÃO MULTI-VETOR")
    print("=" * 60)

    if not CHROMA_DISPONIVEL:
        print("\nErro: chromadb necessário. Instale: pip install chromadb")
        return

    token_tracker.reset()

    print("\n📚 CARREGANDO DOCUMENTOS")
    print("-" * 40)
    documentos = carregar_documentos_exemplo()[:2]
    if not documentos:
        documentos = [Document(page_content="Machine learning permite sistemas aprenderem com dados." * 20)]
    print(f"   Carregados {len(documentos)} documentos")

    print("\n🔧 CRIANDO ÍNDICE MULTI-VETOR")
    print("-" * 40)
    retriever = RetrieverMultiVetor(nome_colecao="demo_multi_vetor")
    num_vetores = retriever.adicionar_documentos(documentos, gerar_resumos=True, gerar_hipoteticas=True)
    print(f"   Criadas {num_vetores} representações vetoriais")

    consultas = ["O que é machine learning?", "Como sistemas de IA aprendem?"]

    print("\n\n❓ CONSULTAS MULTI-VETOR")
    print("=" * 60)

    for consulta in consultas:
        print(f"\n📌 Consulta: '{consulta}'")
        print("-" * 40)

        resultados = retriever.recuperar_com_info_match(consulta, k=2)

        for i, info in enumerate(resultados, 1):
            print(f"\n   Resultado {i}:")
            print(f"      Correspondeu via: {[m['tipo'] for m in info['matches'][:3]]}")
            print(f"      Preview original: {info['original'].page_content[:150]}...")

        docs = [r["original"] for r in resultados]
        resposta = gerar_resposta(consulta, docs)
        print(f"\n   Resposta: {resposta[:250]}...")

    print("\n\n💡 BENEFÍCIOS MULTI-VETOR")
    print("-" * 40)
    print("""
   - Resumos: Correspondem conceitos de alto nível
   - Perguntas: Correspondem formulação de consultas do usuário
   - Original: Fornecem contexto completo para respostas

   Melhor para: Sistemas FAQ, busca em documentação, bases de conhecimento
    """)

    print_total_usage(token_tracker, "TOTAL - Recuperação Multi-Vetor")
    print("\nFim da demonstração de Recuperação Multi-Vetor")
    print("=" * 60)


if __name__ == "__main__":
    main()
