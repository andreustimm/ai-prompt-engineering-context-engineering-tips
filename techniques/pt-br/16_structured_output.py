"""
Structured Output (Saída Estruturada)

Técnica para extrair dados estruturados de respostas do LLM usando
modelos Pydantic, modo JSON e validação de schema.

Recursos:
- Validação com modelo Pydantic
- Saída em modo JSON
- Extração type-safe
- Aplicação de schema

Casos de uso:
- Extração de dados de texto
- Automação de preenchimento de formulários
- Geração de respostas de API
- Parsing de documentos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Rastreador global de tokens
token_tracker = TokenUsage()


# Define modelos Pydantic para saída estruturada

class Pessoa(BaseModel):
    """Modelo para extrair informações de pessoa."""
    nome: str = Field(description="Nome completo da pessoa")
    idade: Optional[int] = Field(default=None, description="Idade em anos")
    ocupacao: Optional[str] = Field(default=None, description="Trabalho ou profissão")
    localizacao: Optional[str] = Field(default=None, description="Cidade ou país")


class Produto(BaseModel):
    """Modelo para informações de produto."""
    nome: str = Field(description="Nome do produto")
    preco: float = Field(description="Preço em reais")
    categoria: str = Field(description="Categoria do produto")
    recursos: list[str] = Field(default_factory=list, description="Recursos principais")
    avaliacao: Optional[float] = Field(default=None, ge=0, le=5, description="Avaliação de 0 a 5")


class InfoContato(BaseModel):
    """Modelo para informações de contato."""
    email: Optional[str] = Field(default=None, description="Endereço de email")
    telefone: Optional[str] = Field(default=None, description="Número de telefone")
    website: Optional[str] = Field(default=None, description="URL do website")
    endereco: Optional[str] = Field(default=None, description="Endereço físico")


class Empresa(BaseModel):
    """Modelo para informações de empresa."""
    nome: str = Field(description="Nome da empresa")
    setor: str = Field(description="Setor de negócio")
    fundacao: Optional[int] = Field(default=None, description="Ano de fundação")
    funcionarios: Optional[str] = Field(default=None, description="Faixa de número de funcionários")
    descricao: Optional[str] = Field(default=None, description="Breve descrição da empresa")
    contato: Optional[InfoContato] = Field(default=None, description="Informações de contato")


class AnaliseSentimento(BaseModel):
    """Modelo para resultados de análise de sentimento."""
    sentimento: str = Field(description="POSITIVO, NEGATIVO ou NEUTRO")
    confianca: float = Field(ge=0, le=1, description="Score de confiança 0-1")
    frases_chave: list[str] = Field(default_factory=list, description="Frases chave que indicam o sentimento")
    resumo: str = Field(description="Breve resumo da análise de sentimento")


class AtaReuniao(BaseModel):
    """Modelo para extração de ata de reunião."""
    titulo: str = Field(description="Título ou tópico da reunião")
    data: Optional[str] = Field(default=None, description="Data da reunião")
    participantes: list[str] = Field(default_factory=list, description="Lista de participantes")
    itens_pauta: list[str] = Field(default_factory=list, description="Itens da pauta discutidos")
    acoes: list[str] = Field(default_factory=list, description="Itens de ação e tarefas")
    decisoes: list[str] = Field(default_factory=list, description="Decisões tomadas")
    proximos_passos: Optional[str] = Field(default=None, description="Próximos passos ou acompanhamento")


def extrair_com_saida_estruturada(texto: str, modelo: type[BaseModel], instrucoes: str = "") -> BaseModel:
    """
    Extrai dados estruturados do texto usando modelo Pydantic.

    Args:
        texto: Texto de entrada para extrair
        modelo: Classe do modelo Pydantic definindo a estrutura
        instrucoes: Instruções adicionais de extração

    Retorna:
        Instância do modelo Pydantic populada
    """
    llm = get_llm(temperature=0)

    # Usa with_structured_output para conformidade garantida com schema
    structured_llm = llm.with_structured_output(modelo)

    prompt_sistema = f"""Você é um especialista em extrair informações estruturadas de texto.
Extraia as informações solicitadas e retorne no formato especificado.
{instrucoes}
Se a informação não estiver disponível, use null/None para campos opcionais."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_sistema),
        ("user", "{texto}")
    ])

    chain = prompt | structured_llm
    resultado = chain.invoke({"texto": texto})

    return resultado


def extrair_pessoa(texto: str) -> Pessoa:
    """Extrai informações de pessoa do texto."""
    return extrair_com_saida_estruturada(
        texto,
        Pessoa,
        "Foque em extrair nome, idade, ocupação e localização."
    )


def extrair_produto(texto: str) -> Produto:
    """Extrai informações de produto do texto."""
    return extrair_com_saida_estruturada(
        texto,
        Produto,
        "Extraia detalhes do produto incluindo nome, preço, categoria, recursos e avaliação."
    )


def extrair_empresa(texto: str) -> Empresa:
    """Extrai informações de empresa do texto."""
    return extrair_com_saida_estruturada(
        texto,
        Empresa,
        "Extraia detalhes da empresa incluindo nome, setor, ano de fundação, número de funcionários e contato."
    )


def analisar_sentimento_estruturado(texto: str) -> AnaliseSentimento:
    """Analisa sentimento e retorna resultado estruturado."""
    return extrair_com_saida_estruturada(
        texto,
        AnaliseSentimento,
        "Analise o sentimento do texto e forneça score de confiança e frases chave."
    )


def extrair_ata_reuniao(texto: str) -> AtaReuniao:
    """Extrai ata de reunião de transcrição ou resumo."""
    return extrair_com_saida_estruturada(
        texto,
        AtaReuniao,
        "Extraia informações da reunião incluindo participantes, pauta, ações e decisões."
    )


def main():
    print("=" * 60)
    print("STRUCTURED OUTPUT (SAÍDA ESTRUTURADA) - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Exemplo 1: Extração de Pessoa
    print("\n👤 EXTRAÇÃO DE PESSOA")
    print("-" * 40)

    texto_pessoa = """
    João Silva é um engenheiro de software de 35 anos que mora em São Paulo.
    Ele trabalha na indústria de tecnologia há mais de 10 anos e atualmente
    lidera uma equipe em uma grande empresa de tecnologia.
    """

    print(f"\nTexto de entrada:\n{texto_pessoa.strip()}")
    pessoa = extrair_pessoa(texto_pessoa)

    print(f"\n📋 Pessoa Extraída:")
    print(f"   Nome: {pessoa.nome}")
    print(f"   Idade: {pessoa.idade}")
    print(f"   Ocupação: {pessoa.ocupacao}")
    print(f"   Localização: {pessoa.localizacao}")

    # Exemplo 2: Extração de Produto
    print("\n\n📦 EXTRAÇÃO DE PRODUTO")
    print("-" * 40)

    texto_produto = """
    O novo iPhone 15 Pro tem preço de R$9.999 e está na categoria de smartphones.
    Ele possui design em titânio, chip A17 Pro, sistema de câmera de 48MP e porta USB-C.
    Os clientes deram uma avaliação média de 4.7 de 5 estrelas.
    """

    print(f"\nTexto de entrada:\n{texto_produto.strip()}")
    produto = extrair_produto(texto_produto)

    print(f"\n📋 Produto Extraído:")
    print(f"   Nome: {produto.nome}")
    print(f"   Preço: R${produto.preco}")
    print(f"   Categoria: {produto.categoria}")
    print(f"   Recursos: {', '.join(produto.recursos)}")
    print(f"   Avaliação: {produto.avaliacao}/5")

    # Exemplo 3: Extração de Empresa
    print("\n\n🏢 EXTRAÇÃO DE EMPRESA")
    print("-" * 40)

    texto_empresa = """
    A OpenAI é uma empresa de pesquisa em inteligência artificial fundada em 2015.
    Sediada em San Francisco, a empresa tem cerca de 500-1000 funcionários e foca
    no desenvolvimento de IA segura e benéfica. Eles podem ser contactados em openai.com
    e o email é support@openai.com.
    """

    print(f"\nTexto de entrada:\n{texto_empresa.strip()}")
    empresa = extrair_empresa(texto_empresa)

    print(f"\n📋 Empresa Extraída:")
    print(f"   Nome: {empresa.nome}")
    print(f"   Setor: {empresa.setor}")
    print(f"   Fundação: {empresa.fundacao}")
    print(f"   Funcionários: {empresa.funcionarios}")
    if empresa.contato:
        print(f"   Email: {empresa.contato.email}")
        print(f"   Website: {empresa.contato.website}")

    # Exemplo 4: Análise de Sentimento
    print("\n\n😊 ANÁLISE DE SENTIMENTO (Estruturada)")
    print("-" * 40)

    texto_sentimento = """
    Eu absolutamente amei esse novo restaurante! A comida estava incrível, o serviço
    foi excelente, e a atmosfera era perfeita para um jantar romântico. O único
    problema menor foi o tempo de espera, mas valeu totalmente a pena. Recomendo muito!
    """

    print(f"\nTexto de entrada:\n{texto_sentimento.strip()}")
    sentimento = analisar_sentimento_estruturado(texto_sentimento)

    print(f"\n📋 Análise de Sentimento:")
    print(f"   Sentimento: {sentimento.sentimento}")
    print(f"   Confiança: {sentimento.confianca:.1%}")
    print(f"   Frases Chave: {', '.join(sentimento.frases_chave)}")
    print(f"   Resumo: {sentimento.resumo}")

    # Exemplo 5: Extração de Ata de Reunião
    print("\n\n📝 EXTRAÇÃO DE ATA DE REUNIÃO")
    print("-" * 40)

    texto_reuniao = """
    Reunião de Planejamento Q3 - 15 de Outubro de 2024

    Participantes: Sarah (PM), João (Tech Lead), Emily (Design), Mike (QA)

    Discutimos o roadmap para Q3 e decidimos priorizar o redesign do app mobile.
    A equipe concordou em lançar o beta até 1º de dezembro.

    Itens de Ação:
    - João criar especificação técnica até 20/10
    - Emily finalizar mockups até 25/10
    - Mike configurar ambiente de testes
    - Sarah coordenar com marketing

    Decisão: Usaremos React Native para o app mobile para compartilhar código com web.

    Próxima reunião agendada para 22 de outubro para revisar progresso.
    """

    print(f"\nTexto de entrada:\n{texto_reuniao.strip()}")
    reuniao = extrair_ata_reuniao(texto_reuniao)

    print(f"\n📋 Ata de Reunião Extraída:")
    print(f"   Título: {reuniao.titulo}")
    print(f"   Data: {reuniao.data}")
    print(f"   Participantes: {', '.join(reuniao.participantes)}")
    print(f"   Itens de Ação:")
    for item in reuniao.acoes:
        print(f"      - {item}")
    print(f"   Decisões:")
    for decisao in reuniao.decisoes:
        print(f"      - {decisao}")
    print(f"   Próximos Passos: {reuniao.proximos_passos}")

    print("\n\n" + "=" * 60)
    print("Nota: Saída estruturada usa with_structured_output() para")
    print("conformidade garantida com schema usando modelos Pydantic.")
    print("=" * 60)

    print("\nFim do demo de Saída Estruturada")
    print("=" * 60)


if __name__ == "__main__":
    main()
