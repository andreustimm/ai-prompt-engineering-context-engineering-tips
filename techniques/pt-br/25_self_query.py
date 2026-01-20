"""
Recuperação Self-Query

Usa um LLM para gerar automaticamente filtros de metadados a partir de
consultas em linguagem natural. Isso permite filtragem estruturada sem
exigir que usuários especifiquem filtros explicitamente.

Componentes:
- Parser de Consulta: LLM analisa consulta em busca semântica + filtros
- Schema de Metadados: Define campos filtráveis e seus tipos
- Recuperação Filtrada: Aplica busca semântica e filtros

Casos de uso:
- Busca de produtos com filtros de preço/categoria
- Busca de documentos com filtros de data/autor
- Qualquer busca que requer filtragem estruturada
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Imports condicionais
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_DISPONIVEL = True
except ImportError:
    CHROMA_DISPONIVEL = False
    print("Aviso: chromadb não instalado. Execute: pip install chromadb")

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


def carregar_catalogo_produtos() -> list[Document]:
    """Carrega catálogo de produtos com metadados."""
    catalog_path = Path(__file__).parent.parent.parent / "sample_data" / "documents" / "products_catalog.json"

    if catalog_path.exists():
        with open(catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documentos = []
        for produto in data.get("products", []):
            doc = Document(
                page_content=f"{produto['name']}: {produto['description']}",
                metadata={
                    "id": produto["id"],
                    "nome": produto["name"],
                    "categoria": produto["category"],
                    "subcategoria": produto["subcategory"],
                    "preco": produto["price"],
                    "marca": produto["brand"],
                    "avaliacao": produto["rating"],
                    "em_estoque": produto["in_stock"],
                    "data_lancamento": produto["release_date"],
                    "cor": produto.get("color", "Desconhecida")
                }
            )
            documentos.append(doc)

        return documentos
    else:
        # Produtos de exemplo
        return [
            Document(
                page_content="ProBook X15 Laptop: Laptop de alto desempenho com tela 4K de 15.6 polegadas",
                metadata={"nome": "ProBook X15", "categoria": "Eletrônicos", "subcategoria": "Computadores",
                         "preco": 1299.99, "marca": "TechPro", "avaliacao": 4.5, "em_estoque": True}
            ),
            Document(
                page_content="BudgetBook 14: Laptop acessível para uso diário",
                metadata={"nome": "BudgetBook 14", "categoria": "Eletrônicos", "subcategoria": "Computadores",
                         "preco": 449.99, "marca": "ValueTech", "avaliacao": 4.2, "em_estoque": True}
            ),
            Document(
                page_content="SmartPhone Pro Max: Smartphone flagship com câmera de 108MP",
                metadata={"nome": "SmartPhone Pro Max", "categoria": "Eletrônicos", "subcategoria": "Smartphones",
                         "preco": 999.99, "marca": "TechPro", "avaliacao": 4.7, "em_estoque": True}
            ),
        ]


# Define informações dos campos de metadados para self-query
INFO_CAMPOS_METADADOS = [
    {
        "nome": "categoria",
        "descricao": "A categoria principal do produto (ex: 'Eletrônicos')",
        "tipo": "string"
    },
    {
        "nome": "subcategoria",
        "descricao": "A subcategoria (ex: 'Computadores', 'Smartphones', 'Áudio', 'Tablets', 'Vestíveis', 'Câmeras')",
        "tipo": "string"
    },
    {
        "nome": "preco",
        "descricao": "O preço do produto em USD",
        "tipo": "float"
    },
    {
        "nome": "marca",
        "descricao": "A marca/fabricante do produto",
        "tipo": "string"
    },
    {
        "nome": "avaliacao",
        "descricao": "Avaliação dos clientes de 1.0 a 5.0",
        "tipo": "float"
    },
    {
        "nome": "em_estoque",
        "descricao": "Se o produto está atualmente em estoque",
        "tipo": "boolean"
    }
]


class RetrieverSelfQuery:
    """Retriever que analisa consultas em busca semântica + filtros de metadados."""

    def __init__(self, vectorstore, campos_metadados: list[dict]):
        """
        Inicializa o retriever self-query.

        Args:
            vectorstore: Vector store com documentos
            campos_metadados: Lista de definições de campos de metadados
        """
        self.vectorstore = vectorstore
        self.campos_metadados = campos_metadados
        self.llm = get_llm(temperature=0)

    def analisar_consulta(self, consulta: str) -> dict:
        """
        Analisa consulta em linguagem natural em query de busca e filtros.

        Args:
            consulta: Consulta em linguagem natural

        Returns:
            Dicionário com 'consulta_busca' e 'filtros'
        """
        descricao_campos = "\n".join([
            f"- {f['nome']} ({f['tipo']}): {f['descricao']}"
            for f in self.campos_metadados
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um analisador de consultas para um sistema de busca de produtos.
Dada uma consulta em linguagem natural, extraia:
1. A consulta de busca semântica (o que o usuário está procurando)
2. Quaisquer filtros de metadados implícitos na consulta

Campos de metadados disponíveis:
{campos}

Retorne um objeto JSON com esta estrutura exata:
{{
    "consulta_busca": "o texto de busca semântica",
    "filtros": {{
        "nome_campo": {{"operador": "op", "valor": valor}}
    }}
}}

Operadores suportados:
- "eq": igual (para strings, números, booleanos)
- "gt": maior que (para números)
- "gte": maior ou igual (para números)
- "lt": menor que (para números)
- "lte": menor ou igual (para números)
- "contem": contém substring (para strings)

Se nenhum filtro for implícito, retorne um objeto filtros vazio: {{}}
Retorne APENAS o objeto JSON, nenhum outro texto."""),
            ("user", "Consulta: {consulta}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "consulta": consulta,
            "campos": descricao_campos
        })

        input_tokens, output_tokens = extract_tokens_from_response(response)
        token_tracker.add(input_tokens, output_tokens)
        print_token_usage(input_tokens, output_tokens, "Análise de Consulta")

        # Analisa resposta JSON
        try:
            resultado = json.loads(response.content)
        except json.JSONDecodeError:
            # Tenta extrair JSON da resposta
            conteudo = response.content
            inicio = conteudo.find('{')
            fim = conteudo.rfind('}') + 1
            if inicio >= 0 and fim > inicio:
                resultado = json.loads(conteudo[inicio:fim])
            else:
                resultado = {"consulta_busca": consulta, "filtros": {}}

        return resultado

    def aplicar_filtros(self, documentos: list[Document], filtros: dict) -> list[Document]:
        """
        Aplica filtros de metadados aos documentos.

        Args:
            documentos: Documentos para filtrar
            filtros: Especificações de filtros

        Returns:
            Documentos filtrados
        """
        if not filtros:
            return documentos

        filtrados = []

        for doc in documentos:
            passa = True

            for campo, condicao in filtros.items():
                if campo not in doc.metadata:
                    passa = False
                    break

                valor = doc.metadata[campo]
                op = condicao.get("operador", "eq")
                valor_filtro = condicao.get("valor")

                if op == "eq":
                    passa = valor == valor_filtro
                elif op == "gt":
                    passa = valor > valor_filtro
                elif op == "gte":
                    passa = valor >= valor_filtro
                elif op == "lt":
                    passa = valor < valor_filtro
                elif op == "lte":
                    passa = valor <= valor_filtro
                elif op == "contem":
                    passa = str(valor_filtro).lower() in str(valor).lower()

                if not passa:
                    break

            if passa:
                filtrados.append(doc)

        return filtrados

    def recuperar(self, consulta: str, k: int = 5) -> tuple[list[Document], dict]:
        """
        Recupera documentos usando self-query.

        Args:
            consulta: Consulta em linguagem natural
            k: Número de documentos a recuperar

        Returns:
            Tupla de (documentos filtrados, info da consulta analisada)
        """
        # Analisa a consulta
        analisado = self.analisar_consulta(consulta)
        consulta_busca = analisado.get("consulta_busca", consulta)
        filtros = analisado.get("filtros", {})

        print(f"\n   Consulta Analisada:")
        print(f"      Busca: '{consulta_busca}'")
        print(f"      Filtros: {json.dumps(filtros, indent=2, ensure_ascii=False) if filtros else 'Nenhum'}")

        # Recupera mais documentos do que necessário (vai filtrar)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k * 3})
        documentos = retriever.invoke(consulta_busca)

        # Aplica filtros
        filtrados = self.aplicar_filtros(documentos, filtros)

        return filtrados[:k], analisado


def criar_vectorstore(documentos: list[Document], nome_colecao: str = "demo_self_query"):
    """Cria vector store a partir de documentos."""
    if not CHROMA_DISPONIVEL:
        raise ImportError("chromadb é necessário")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        collection_name=nome_colecao
    )

    return vectorstore


def gerar_resposta(consulta: str, documentos: list[Document]) -> str:
    """Gera resposta usando produtos recuperados."""
    llm = get_llm(temperature=0.3)

    contexto = "\n\n".join([
        f"Produto: {doc.metadata.get('nome', 'Desconhecido')}\n"
        f"Preço: ${doc.metadata.get('preco', 'N/A')}\n"
        f"Marca: {doc.metadata.get('marca', 'N/A')}\n"
        f"Avaliação: {doc.metadata.get('avaliacao', 'N/A')}/5\n"
        f"Em Estoque: {'Sim' if doc.metadata.get('em_estoque') else 'Não'}\n"
        f"Descrição: {doc.page_content}"
        for doc in documentos
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente de compras útil. Baseado nos produtos disponíveis,
ajude o usuário a encontrar o que está procurando. Mencione produtos específicos, preços e características."""),
        ("user", """Produtos Disponíveis:
{contexto}

Consulta do Cliente: {consulta}

Resposta:""")
    ])

    chain = prompt | llm
    response = chain.invoke({"contexto": contexto, "consulta": consulta})

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens, "Geração de Resposta")

    return response.content


def demonstrar_self_query():
    """Demonstra recuperação self-query com várias consultas."""

    print("\n   Carregando catálogo de produtos...")
    documentos = carregar_catalogo_produtos()
    print(f"   Carregados {len(documentos)} produtos")

    print("\n   Criando vector store...")
    vectorstore = criar_vectorstore(documentos, "produtos_self_query")
    print("   Vector store pronto!")

    # Cria retriever self-query
    retriever = RetrieverSelfQuery(vectorstore, INFO_CAMPOS_METADADOS)

    # Consultas de teste
    consultas = [
        "laptops baratos abaixo de $500",
        "smartphones mais bem avaliados",
        "produtos TechPro em estoque",
        "fones de ouvido sem fio com boas avaliações",
        "câmeras para fotografia profissional"
    ]

    print("\n" + "=" * 60)
    print("DEMONSTRAÇÕES DE RECUPERAÇÃO SELF-QUERY")
    print("=" * 60)

    for consulta in consultas:
        print(f"\n📌 Consulta: '{consulta}'")
        print("-" * 40)

        documentos, analisado = retriever.recuperar(consulta, k=3)

        print(f"\n   Recuperados {len(documentos)} produtos:")
        for i, doc in enumerate(documentos, 1):
            print(f"\n   {i}. {doc.metadata.get('nome', 'Desconhecido')}")
            print(f"      Preço: ${doc.metadata.get('preco', 'N/A')}")
            print(f"      Avaliação: {doc.metadata.get('avaliacao', 'N/A')}/5")
            print(f"      Marca: {doc.metadata.get('marca', 'N/A')}")
            print(f"      Em Estoque: {'Sim' if doc.metadata.get('em_estoque') else 'Não'}")

        print("\n   Gerando recomendação...")
        resposta = gerar_resposta(consulta, documentos)
        print(f"\n   Resposta: {resposta[:300]}...")


def comparar_com_sem_filtros():
    """Compara resultados com e sem filtragem automática."""

    print("\n   Carregando catálogo de produtos...")
    documentos = carregar_catalogo_produtos()

    print("\n   Criando vector store...")
    vectorstore = criar_vectorstore(documentos, "comparacao_filtros")

    consulta = "laptops acessíveis abaixo de $600 com boas avaliações"

    print(f"\n   Consulta: '{consulta}'")
    print("=" * 50)

    # Sem self-query (busca semântica simples)
    print("\n   📚 SEM SELF-QUERY (Apenas Semântico):")
    print("-" * 30)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    resultados_simples = retriever.invoke(consulta)

    for i, doc in enumerate(resultados_simples[:3], 1):
        print(f"\n   {i}. {doc.metadata.get('nome', 'Desconhecido')}")
        print(f"      Preço: ${doc.metadata.get('preco', 'N/A')}")
        print(f"      Avaliação: {doc.metadata.get('avaliacao', 'N/A')}/5")

    # Com self-query
    print("\n   🔍 COM SELF-QUERY (Semântico + Filtros):")
    print("-" * 30)

    retriever_self_query = RetrieverSelfQuery(vectorstore, INFO_CAMPOS_METADADOS)
    resultados_filtrados, analisado = retriever_self_query.recuperar(consulta, k=5)

    for i, doc in enumerate(resultados_filtrados[:3], 1):
        print(f"\n   {i}. {doc.metadata.get('nome', 'Desconhecido')}")
        print(f"      Preço: ${doc.metadata.get('preco', 'N/A')}")
        print(f"      Avaliação: {doc.metadata.get('avaliacao', 'N/A')}/5")

    print("\n   Nota: Self-query filtra corretamente por preço < $600")


def main():
    print("=" * 60)
    print("RECUPERAÇÃO SELF-QUERY")
    print("=" * 60)

    if not CHROMA_DISPONIVEL:
        print("\nErro: chromadb é necessário para esta demonstração.")
        print("Instale com: pip install chromadb")
        return

    token_tracker.reset()

    # Demo 1: Demonstração básica de self-query
    print("\n\n🛒 BUSCA DE PRODUTOS SELF-QUERY")
    print("=" * 60)

    demonstrar_self_query()

    # Demo 2: Comparação
    print("\n\n📊 COMPARAÇÃO: COM vs SEM FILTROS")
    print("=" * 60)

    comparar_com_sem_filtros()

    # Melhores práticas
    print("\n\n💡 MELHORES PRÁTICAS SELF-QUERY")
    print("-" * 40)
    print("""
   | Consideração          | Recomendação                            |
   |-----------------------|-----------------------------------------|
   | Schema de Metadados   | Defina campos claros e bem documentados |
   | Tipos de Campos       | Use tipos apropriados (string, float)   |
   | Faixas de Valores     | Documente valores válidos para cada campo|
   | Fallback              | Trate casos onde análise falha          |

   Dicas:
   - Mantenha campos de metadados simples e não ambíguos
   - Forneça boas descrições de campos para o LLM
   - Teste com várias formulações de consulta
   - Considere combinar com outros métodos de recuperação
   - Cache consultas analisadas para buscas repetidas

   Casos de Uso Comuns:
   - Busca de produtos e-commerce
   - Gerenciamento de documentos com metadados
   - Sistemas de busca de vagas/candidatos
   - Busca de imóveis
    """)

    print_total_usage(token_tracker, "TOTAL - Recuperação Self-Query")

    print("\nFim da demonstração de Recuperação Self-Query")
    print("=" * 60)


if __name__ == "__main__":
    main()
