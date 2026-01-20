"""
Few-Shot Prompting (One-Shot / Few-Shot)

Técnica onde fornecemos exemplos ao modelo antes de fazer a pergunta.
Isso ajuda o modelo a entender o formato e o tipo de resposta esperada.

Casos de uso:
- Conversão de formatos
- Geração de código seguindo padrões
- Classificação com categorias específicas
- Tarefas com formato de saída específico
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Tracker global de tokens para este script
token_tracker = TokenUsage()


def classificar_ticket_suporte(ticket: str) -> str:
    """Classifica tickets de suporte usando exemplos."""
    llm = get_llm(temperature=0)

    # Exemplos para few-shot
    examples = [
        {
            "ticket": "Não consigo fazer login na minha conta, a senha está correta mas aparece erro",
            "classificacao": "CATEGORIA: Autenticação\nPRIORIDADE: Alta\nAÇÃO: Verificar bloqueio de conta e logs de acesso"
        },
        {
            "ticket": "Gostaria de saber como exportar relatórios em PDF",
            "classificacao": "CATEGORIA: Dúvida de Uso\nPRIORIDADE: Baixa\nAÇÃO: Encaminhar documentação e tutorial"
        },
        {
            "ticket": "O sistema está muito lento desde ontem, demora 30 segundos para carregar cada página",
            "classificacao": "CATEGORIA: Performance\nPRIORIDADE: Crítica\nAÇÃO: Escalar para equipe de infraestrutura"
        },
        {
            "ticket": "Preciso adicionar mais 5 usuários na minha conta empresarial",
            "classificacao": "CATEGORIA: Comercial\nPRIORIDADE: Média\nAÇÃO: Encaminhar para equipe de vendas"
        }
    ]

    # Template para cada exemplo
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{ticket}"),
        ("assistant", "{classificacao}")
    ])

    # Few-shot prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    # Prompt final
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um sistema de classificação de tickets de suporte. Classifique cada ticket com categoria, prioridade e ação recomendada."),
        few_shot_prompt,
        ("user", "{ticket}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"ticket": ticket})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def converter_para_sql(descricao: str) -> str:
    """Converte descrições em linguagem natural para SQL."""
    llm = get_llm(temperature=0)

    examples = [
        {
            "descricao": "Listar todos os clientes do Brasil",
            "sql": "SELECT * FROM clientes WHERE pais = 'Brasil';"
        },
        {
            "descricao": "Contar quantos pedidos foram feitos em janeiro de 2024",
            "sql": "SELECT COUNT(*) FROM pedidos WHERE data_pedido BETWEEN '2024-01-01' AND '2024-01-31';"
        },
        {
            "descricao": "Mostrar os 10 produtos mais vendidos com nome e quantidade",
            "sql": "SELECT p.nome, SUM(ip.quantidade) as total_vendido\nFROM produtos p\nJOIN itens_pedido ip ON p.id = ip.produto_id\nGROUP BY p.id, p.nome\nORDER BY total_vendido DESC\nLIMIT 10;"
        },
        {
            "descricao": "Atualizar o preço de todos os produtos da categoria 'Eletrônicos' aumentando 10%",
            "sql": "UPDATE produtos SET preco = preco * 1.10 WHERE categoria = 'Eletrônicos';"
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{descricao}"),
        ("assistant", "```sql\n{sql}\n```")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em SQL. Converta a descrição em linguagem natural para uma query SQL válida.
Considere as seguintes tabelas disponíveis:
- clientes (id, nome, email, pais, data_cadastro)
- produtos (id, nome, preco, categoria, estoque)
- pedidos (id, cliente_id, data_pedido, status, total)
- itens_pedido (id, pedido_id, produto_id, quantidade, preco_unitario)"""),
        few_shot_prompt,
        ("user", "{descricao}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"descricao": descricao})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def gerar_docstring(codigo: str) -> str:
    """Gera docstrings no padrão Google para funções Python."""
    llm = get_llm(temperature=0.3)

    examples = [
        {
            "codigo": "def somar(a, b):\n    return a + b",
            "documentado": '''def somar(a, b):
    """Soma dois números.

    Args:
        a: Primeiro número a ser somado.
        b: Segundo número a ser somado.

    Returns:
        A soma de a e b.
    """
    return a + b'''
        },
        {
            "codigo": "def buscar_usuario(user_id, incluir_inativos=False):\n    usuarios = db.query(User).filter(User.id == user_id)\n    if not incluir_inativos:\n        usuarios = usuarios.filter(User.ativo == True)\n    return usuarios.first()",
            "documentado": '''def buscar_usuario(user_id, incluir_inativos=False):
    """Busca um usuário pelo ID no banco de dados.

    Args:
        user_id: ID único do usuário a ser buscado.
        incluir_inativos: Se True, inclui usuários inativos na busca.
            Defaults to False.

    Returns:
        O objeto User se encontrado, None caso contrário.
    """
    usuarios = db.query(User).filter(User.id == user_id)
    if not incluir_inativos:
        usuarios = usuarios.filter(User.ativo == True)
    return usuarios.first()'''
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{codigo}"),
        ("assistant", "{documentado}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um desenvolvedor Python sênior. Adicione docstrings no padrão Google Style ao código fornecido. Mantenha o código original, apenas adicione a documentação."),
        few_shot_prompt,
        ("user", "{codigo}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"codigo": codigo})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def extrair_dados_estruturados(texto: str) -> str:
    """Extrai dados estruturados de texto livre em formato JSON."""
    llm = get_llm(temperature=0)

    examples = [
        {
            "texto": "João Silva, 35 anos, mora em São Paulo e trabalha como engenheiro de software na empresa TechCorp. Seu email é joao.silva@email.com",
            "json": '{\n  "nome": "João Silva",\n  "idade": 35,\n  "cidade": "São Paulo",\n  "profissao": "engenheiro de software",\n  "empresa": "TechCorp",\n  "email": "joao.silva@email.com"\n}'
        },
        {
            "texto": "Produto: Notebook Dell XPS 15, preço de R$ 8.500,00, em estoque (23 unidades), categoria: Eletrônicos/Computadores",
            "json": '{\n  "produto": "Notebook Dell XPS 15",\n  "marca": "Dell",\n  "modelo": "XPS 15",\n  "preco": 8500.00,\n  "moeda": "BRL",\n  "estoque": 23,\n  "disponivel": true,\n  "categoria": ["Eletrônicos", "Computadores"]\n}'
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{texto}"),
        ("assistant", "```json\n{json}\n```")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um sistema de extração de dados. Extraia informações do texto e retorne em formato JSON estruturado. Infira campos adicionais quando apropriado."),
        few_shot_prompt,
        ("user", "{texto}")
    ])

    chain = final_prompt | llm
    response = chain.invoke({"texto": texto})

    # Extrair e registrar tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def main():
    print("=" * 60)
    print("FEW-SHOT PROMPTING - Demonstração")
    print("=" * 60)

    # Reset do tracker
    token_tracker.reset()

    # Exemplo 1: Classificação de Tickets
    print("\n🎫 CLASSIFICAÇÃO DE TICKETS DE SUPORTE")
    print("-" * 40)

    ticket = "Meu cartão foi cobrado duas vezes pelo mesmo pedido #12345, preciso do estorno urgente"

    print(f"\nTicket: {ticket}")
    print(f"\nClassificação:\n{classificar_ticket_suporte(ticket)}")

    # Exemplo 2: Conversão para SQL
    print("\n\n💾 CONVERSÃO PARA SQL")
    print("-" * 40)

    descricoes = [
        "Mostrar o total de vendas por mês em 2024",
        "Encontrar clientes que não fizeram pedidos nos últimos 6 meses"
    ]

    for desc in descricoes:
        print(f"\nDescrição: {desc}")
        print(f"SQL: {converter_para_sql(desc)}")

    # Exemplo 3: Geração de Docstrings
    print("\n\n📝 GERAÇÃO DE DOCSTRINGS")
    print("-" * 40)

    codigo = """def calcular_desconto(valor_total, cupom=None, cliente_vip=False):
    desconto = 0
    if cupom and cupom in CUPONS_VALIDOS:
        desconto += CUPONS_VALIDOS[cupom]
    if cliente_vip:
        desconto += 0.1
    return valor_total * (1 - min(desconto, 0.5))"""

    print(f"\nCódigo original:\n{codigo}")
    print(f"\nCom docstring:\n{gerar_docstring(codigo)}")

    # Exemplo 4: Extração de Dados Estruturados
    print("\n\n📊 EXTRAÇÃO DE DADOS ESTRUTURADOS")
    print("-" * 40)

    texto = """
    Reserva confirmada: Hotel Marriott São Paulo, check-in dia 15/03/2024 às 14h,
    check-out 18/03/2024 às 12h. Quarto duplo superior, 3 noites, valor total
    R$ 1.890,00 (já incluso café da manhã). Hóspede: Maria Santos, CPF: 123.456.789-00
    """

    print(f"\nTexto: {texto.strip()}")
    print(f"\nDados extraídos:\n{extrair_dados_estruturados(texto)}")

    # Exibir total de tokens
    print_total_usage(token_tracker, "TOTAL - Few-Shot Prompting")

    print("\nFim da demonstração Few-Shot")
    print("=" * 60)


if __name__ == "__main__":
    main()
