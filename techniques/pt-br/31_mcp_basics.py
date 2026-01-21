"""
MCP (Model Context Protocol) - Fundamentos

O MCP (Model Context Protocol) é um protocolo aberto criado pela Anthropic para
conectar assistentes de IA a fontes de dados e ferramentas externas de forma
padronizada e segura.

Componentes principais:
- MCP Host: Aplicação que hospeda o cliente MCP (ex: Claude Desktop, IDEs)
- MCP Client: Componente que se conecta aos servidores MCP
- MCP Server: Serviço que expõe recursos, ferramentas e prompts

Recursos do MCP:
- Resources: Dados que o servidor expõe (arquivos, banco de dados, APIs)
- Tools: Funções que o LLM pode invocar
- Prompts: Templates de prompts reutilizáveis

Casos de uso:
- Integração com banco de dados
- Acesso a sistemas de arquivos
- Conexão com APIs externas
- Integração com GitHub, Slack, etc.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Simulação de estruturas MCP (em produção, use mcp-python-sdk)

@dataclass
class MCPResource:
    """Representa um recurso exposto pelo servidor MCP."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


@dataclass
class MCPTool:
    """Representa uma ferramenta exposta pelo servidor MCP."""
    name: str
    description: str
    input_schema: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class MCPPrompt:
    """Representa um prompt template exposto pelo servidor MCP."""
    name: str
    description: str
    arguments: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments
        }


class MCPServerSimulator:
    """
    Simulador de servidor MCP para demonstração.

    Em produção, use o SDK oficial: pip install mcp
    Documentação: https://modelcontextprotocol.io/
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.resources: list[MCPResource] = []
        self.tools: list[MCPTool] = []
        self.prompts: list[MCPPrompt] = []
        self._tool_handlers: dict[str, callable] = {}

    def add_resource(self, uri: str, name: str, description: str, mime_type: str = "text/plain"):
        """Adiciona um recurso ao servidor."""
        resource = MCPResource(uri=uri, name=name, description=description, mime_type=mime_type)
        self.resources.append(resource)
        print(f"   Recurso adicionado: {name} ({uri})")
        return resource

    def add_tool(self, name: str, description: str, input_schema: dict, handler: callable):
        """Adiciona uma ferramenta ao servidor."""
        tool = MCPTool(name=name, description=description, input_schema=input_schema)
        self.tools.append(tool)
        self._tool_handlers[name] = handler
        print(f"   Ferramenta adicionada: {name}")
        return tool

    def add_prompt(self, name: str, description: str, arguments: list = None):
        """Adiciona um prompt template ao servidor."""
        prompt = MCPPrompt(name=name, description=description, arguments=arguments or [])
        self.prompts.append(prompt)
        print(f"   Prompt adicionado: {name}")
        return prompt

    def list_resources(self) -> list[dict]:
        """Lista todos os recursos disponíveis."""
        return [r.to_dict() for r in self.resources]

    def list_tools(self) -> list[dict]:
        """Lista todas as ferramentas disponíveis."""
        return [t.to_dict() for t in self.tools]

    def list_prompts(self) -> list[dict]:
        """Lista todos os prompts disponíveis."""
        return [p.to_dict() for p in self.prompts]

    def call_tool(self, name: str, arguments: dict) -> Any:
        """Executa uma ferramenta."""
        if name not in self._tool_handlers:
            raise ValueError(f"Ferramenta '{name}' não encontrada")

        handler = self._tool_handlers[name]
        return handler(**arguments)

    def get_server_info(self) -> dict:
        """Retorna informações do servidor."""
        return {
            "name": self.name,
            "version": self.version,
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "resources": {"listChanged": True},
                "tools": {},
                "prompts": {"listChanged": True}
            }
        }


# Exemplo de implementação de ferramentas MCP

def buscar_no_banco(query: str, tabela: str = "usuarios") -> dict:
    """Simula busca em banco de dados."""
    dados_simulados = {
        "usuarios": [
            {"id": 1, "nome": "Ana Silva", "email": "ana@email.com", "cargo": "Desenvolvedora"},
            {"id": 2, "nome": "Carlos Santos", "email": "carlos@email.com", "cargo": "Designer"},
            {"id": 3, "nome": "Maria Oliveira", "email": "maria@email.com", "cargo": "Gerente"},
        ],
        "produtos": [
            {"id": 1, "nome": "Notebook", "preco": 3500.00, "estoque": 15},
            {"id": 2, "nome": "Mouse", "preco": 89.90, "estoque": 150},
            {"id": 3, "nome": "Teclado", "preco": 199.90, "estoque": 80},
        ]
    }

    if tabela not in dados_simulados:
        return {"error": f"Tabela '{tabela}' não encontrada"}

    resultados = [
        item for item in dados_simulados[tabela]
        if query.lower() in str(item).lower()
    ]

    return {
        "tabela": tabela,
        "query": query,
        "resultados": resultados,
        "total": len(resultados)
    }


def ler_arquivo(caminho: str) -> dict:
    """Simula leitura de arquivo."""
    arquivos_simulados = {
        "config.json": '{"app": "demo", "version": "1.0"}',
        "readme.md": "# Projeto Demo\n\nEste é um projeto de demonstração.",
        "dados.csv": "nome,valor\nitem1,100\nitem2,200"
    }

    if caminho in arquivos_simulados:
        return {
            "caminho": caminho,
            "conteudo": arquivos_simulados[caminho],
            "tamanho": len(arquivos_simulados[caminho])
        }
    return {"error": f"Arquivo '{caminho}' não encontrado"}


def executar_comando(comando: str) -> dict:
    """Simula execução de comando (apenas para demo)."""
    comandos_seguros = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "whoami": "usuario_demo",
        "pwd": "/home/usuario/projeto"
    }

    if comando in comandos_seguros:
        return {"comando": comando, "saida": comandos_seguros[comando], "codigo": 0}
    return {"comando": comando, "error": "Comando não permitido", "codigo": 1}


def criar_servidor_exemplo() -> MCPServerSimulator:
    """Cria um servidor MCP de exemplo com recursos, ferramentas e prompts."""

    servidor = MCPServerSimulator(name="servidor-demo", version="1.0.0")

    # Adicionar recursos
    servidor.add_resource(
        uri="file:///config.json",
        name="Configuração",
        description="Arquivo de configuração da aplicação",
        mime_type="application/json"
    )

    servidor.add_resource(
        uri="db://usuarios",
        name="Tabela de Usuários",
        description="Dados dos usuários do sistema",
        mime_type="application/json"
    )

    # Adicionar ferramentas
    servidor.add_tool(
        name="buscar_banco",
        description="Busca dados no banco de dados",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Termo de busca"},
                "tabela": {"type": "string", "description": "Nome da tabela", "default": "usuarios"}
            },
            "required": ["query"]
        },
        handler=buscar_no_banco
    )

    servidor.add_tool(
        name="ler_arquivo",
        description="Lê conteúdo de um arquivo",
        input_schema={
            "type": "object",
            "properties": {
                "caminho": {"type": "string", "description": "Caminho do arquivo"}
            },
            "required": ["caminho"]
        },
        handler=ler_arquivo
    )

    servidor.add_tool(
        name="executar_comando",
        description="Executa um comando do sistema (limitado)",
        input_schema={
            "type": "object",
            "properties": {
                "comando": {"type": "string", "description": "Comando a executar"}
            },
            "required": ["comando"]
        },
        handler=executar_comando
    )

    # Adicionar prompts
    servidor.add_prompt(
        name="analise_dados",
        description="Template para análise de dados",
        arguments=[
            {"name": "dados", "description": "Dados a serem analisados", "required": True},
            {"name": "formato", "description": "Formato de saída", "required": False}
        ]
    )

    servidor.add_prompt(
        name="resumo_documento",
        description="Template para resumir documentos",
        arguments=[
            {"name": "documento", "description": "Texto do documento", "required": True},
            {"name": "tamanho", "description": "Tamanho do resumo (curto/medio/longo)", "required": False}
        ]
    )

    return servidor


def demonstrar_uso_mcp():
    """Demonstra o uso básico do protocolo MCP."""

    print("\n" + "=" * 60)
    print("CRIANDO SERVIDOR MCP")
    print("=" * 60)

    servidor = criar_servidor_exemplo()

    # Informações do servidor
    print("\n" + "-" * 40)
    print("INFORMAÇÕES DO SERVIDOR")
    print("-" * 40)
    info = servidor.get_server_info()
    print(json.dumps(info, indent=2))

    # Listar recursos
    print("\n" + "-" * 40)
    print("RECURSOS DISPONÍVEIS")
    print("-" * 40)
    for recurso in servidor.list_resources():
        print(f"  - {recurso['name']}: {recurso['uri']}")

    # Listar ferramentas
    print("\n" + "-" * 40)
    print("FERRAMENTAS DISPONÍVEIS")
    print("-" * 40)
    for ferramenta in servidor.list_tools():
        print(f"  - {ferramenta['name']}: {ferramenta['description']}")

    # Listar prompts
    print("\n" + "-" * 40)
    print("PROMPTS DISPONÍVEIS")
    print("-" * 40)
    for prompt in servidor.list_prompts():
        print(f"  - {prompt['name']}: {prompt['description']}")

    # Demonstrar chamada de ferramentas
    print("\n" + "=" * 60)
    print("DEMONSTRAÇÃO DE CHAMADAS DE FERRAMENTAS")
    print("=" * 60)

    # Busca no banco
    print("\n1. Buscar no banco de dados:")
    resultado = servidor.call_tool("buscar_banco", {"query": "Ana", "tabela": "usuarios"})
    print(f"   Resultado: {json.dumps(resultado, indent=2, ensure_ascii=False)}")

    # Ler arquivo
    print("\n2. Ler arquivo de configuração:")
    resultado = servidor.call_tool("ler_arquivo", {"caminho": "config.json"})
    print(f"   Resultado: {json.dumps(resultado, indent=2, ensure_ascii=False)}")

    # Executar comando
    print("\n3. Executar comando:")
    resultado = servidor.call_tool("executar_comando", {"comando": "date"})
    print(f"   Resultado: {json.dumps(resultado, indent=2, ensure_ascii=False)}")


def main():
    print("=" * 60)
    print("MCP (MODEL CONTEXT PROTOCOL) - Fundamentos")
    print("=" * 60)

    print("""
    O MCP é um protocolo que permite conectar assistentes de IA
    a fontes de dados e ferramentas externas de forma padronizada.

    Principais conceitos:
    1. Resources - Dados expostos pelo servidor (arquivos, BD, APIs)
    2. Tools - Funções que o LLM pode invocar
    3. Prompts - Templates de prompts reutilizáveis

    Arquitetura:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  MCP Host   │────▶│ MCP Client  │────▶│ MCP Server  │
    │ (Claude,    │     │             │     │ (BD, API,   │
    │  VS Code)   │◀────│             │◀────│  Arquivos)  │
    └─────────────┘     └─────────────┘     └─────────────┘
    """)

    demonstrar_uso_mcp()

    print("\n" + "=" * 60)
    print("PRÓXIMOS PASSOS")
    print("=" * 60)
    print("""
    Para usar MCP em produção:

    1. Instale o SDK: pip install mcp

    2. Crie um servidor MCP real:
       - Use STDIO para comunicação local
       - Use HTTP/SSE para comunicação remota

    3. Configure no Claude Desktop ou sua IDE:
       - Adicione a configuração em ~/.config/claude/claude_desktop_config.json

    Documentação oficial: https://modelcontextprotocol.io/
    """)

    print("\nFim do demo de MCP Basics")
    print("=" * 60)


if __name__ == "__main__":
    main()
