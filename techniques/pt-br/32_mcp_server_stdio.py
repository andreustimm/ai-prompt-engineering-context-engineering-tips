"""
Servidor MCP com STDIO (Standard Input/Output)

STDIO é o método de transporte mais comum para servidores MCP locais.
A comunicação ocorre através de stdin/stdout, ideal para ferramentas CLI
e integrações com IDEs como Claude Desktop, VS Code, etc.

Características do STDIO:
- Comunicação local (mesmo computador)
- Baixa latência
- Fácil configuração
- Ideal para ferramentas de desenvolvimento

Casos de uso:
- Integração com Claude Desktop
- Ferramentas de linha de comando
- Acesso a arquivos locais
- Execução de scripts

Requisitos:
- pip install mcp
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import asyncio
from typing import Any
from datetime import datetime

# Simulação do protocolo MCP STDIO
# Em produção, use: from mcp.server import Server
# from mcp.server.stdio import stdio_server

class MCPMessage:
    """Representa uma mensagem do protocolo MCP."""

    def __init__(self, jsonrpc: str = "2.0", id: int = None, method: str = None,
                 params: dict = None, result: Any = None, error: dict = None):
        self.jsonrpc = jsonrpc
        self.id = id
        self.method = method
        self.params = params or {}
        self.result = result
        self.error = error

    def to_dict(self) -> dict:
        msg = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            msg["id"] = self.id
        if self.method:
            msg["method"] = self.method
        if self.params:
            msg["params"] = self.params
        if self.result is not None:
            msg["result"] = self.result
        if self.error:
            msg["error"] = self.error
        return msg

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> 'MCPMessage':
        parsed = json.loads(data)
        return cls(**parsed)


class MCPSTDIOServer:
    """
    Servidor MCP que comunica via STDIO.

    Este é um simulador para demonstração. Em produção, use:
    ```python
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server = Server("meu-servidor")

    @server.list_tools()
    async def listar_ferramentas():
        return [...]

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)
    ```
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        self._running = False

    def register_tool(self, name: str, description: str, schema: dict, handler: callable):
        """Registra uma ferramenta no servidor."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": schema,
            "handler": handler
        }
        print(f"[SERVIDOR] Ferramenta registrada: {name}")

    def register_resource(self, uri: str, name: str, description: str):
        """Registra um recurso no servidor."""
        self.resources[uri] = {
            "uri": uri,
            "name": name,
            "description": description
        }
        print(f"[SERVIDOR] Recurso registrado: {name}")

    def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Processa uma mensagem MCP e retorna a resposta."""

        method = message.method

        if method == "initialize":
            return self._handle_initialize(message)
        elif method == "tools/list":
            return self._handle_list_tools(message)
        elif method == "tools/call":
            return self._handle_call_tool(message)
        elif method == "resources/list":
            return self._handle_list_resources(message)
        elif method == "resources/read":
            return self._handle_read_resource(message)
        elif method == "ping":
            return MCPMessage(id=message.id, result={})
        else:
            return MCPMessage(
                id=message.id,
                error={"code": -32601, "message": f"Método desconhecido: {method}"}
            )

    def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Trata requisição de inicialização."""
        return MCPMessage(
            id=message.id,
            result={
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                },
                "capabilities": {
                    "tools": {},
                    "resources": {"listChanged": True}
                }
            }
        )

    def _handle_list_tools(self, message: MCPMessage) -> MCPMessage:
        """Lista todas as ferramentas disponíveis."""
        tools_list = [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"]
            }
            for t in self.tools.values()
        ]
        return MCPMessage(id=message.id, result={"tools": tools_list})

    def _handle_call_tool(self, message: MCPMessage) -> MCPMessage:
        """Executa uma ferramenta."""
        tool_name = message.params.get("name")
        arguments = message.params.get("arguments", {})

        if tool_name not in self.tools:
            return MCPMessage(
                id=message.id,
                error={"code": -32602, "message": f"Ferramenta não encontrada: {tool_name}"}
            )

        try:
            tool = self.tools[tool_name]
            result = tool["handler"](**arguments)
            return MCPMessage(
                id=message.id,
                result={
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                    "isError": False
                }
            )
        except Exception as e:
            return MCPMessage(
                id=message.id,
                result={
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True
                }
            )

    def _handle_list_resources(self, message: MCPMessage) -> MCPMessage:
        """Lista todos os recursos disponíveis."""
        resources_list = list(self.resources.values())
        return MCPMessage(id=message.id, result={"resources": resources_list})

    def _handle_read_resource(self, message: MCPMessage) -> MCPMessage:
        """Lê um recurso específico."""
        uri = message.params.get("uri")

        if uri not in self.resources:
            return MCPMessage(
                id=message.id,
                error={"code": -32602, "message": f"Recurso não encontrado: {uri}"}
            )

        # Simula leitura do recurso
        return MCPMessage(
            id=message.id,
            result={
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": f"Conteúdo do recurso {uri}"
                }]
            }
        )

    def simulate_stdio_communication(self, messages: list[dict]):
        """Simula comunicação STDIO com uma lista de mensagens."""
        print("\n" + "=" * 60)
        print("SIMULAÇÃO DE COMUNICAÇÃO STDIO")
        print("=" * 60)

        for i, msg_data in enumerate(messages, 1):
            print(f"\n--- Mensagem {i} ---")

            # Simula recebimento via stdin
            message = MCPMessage(**msg_data)
            print(f"[STDIN]  Recebido: {message.to_json()}")

            # Processa mensagem
            response = self.handle_message(message)

            # Simula envio via stdout
            print(f"[STDOUT] Enviado: {response.to_json()}")


# Ferramentas de exemplo

def obter_informacoes_sistema() -> dict:
    """Retorna informações do sistema."""
    return {
        "plataforma": sys.platform,
        "versao_python": sys.version,
        "diretorio_atual": str(Path.cwd()),
        "data_hora": datetime.now().isoformat()
    }


def listar_arquivos(diretorio: str = ".") -> dict:
    """Lista arquivos em um diretório."""
    try:
        path = Path(diretorio)
        if not path.exists():
            return {"erro": f"Diretório não encontrado: {diretorio}"}

        arquivos = []
        for item in path.iterdir():
            arquivos.append({
                "nome": item.name,
                "tipo": "diretorio" if item.is_dir() else "arquivo",
                "tamanho": item.stat().st_size if item.is_file() else None
            })

        return {
            "diretorio": str(path.absolute()),
            "total": len(arquivos),
            "itens": arquivos[:20]  # Limita a 20 itens
        }
    except Exception as e:
        return {"erro": str(e)}


def ler_arquivo_texto(caminho: str) -> dict:
    """Lê conteúdo de um arquivo texto."""
    try:
        path = Path(caminho)
        if not path.exists():
            return {"erro": f"Arquivo não encontrado: {caminho}"}

        if path.stat().st_size > 100000:  # Limite de 100KB
            return {"erro": "Arquivo muito grande (máximo 100KB)"}

        conteudo = path.read_text(encoding='utf-8')
        return {
            "caminho": str(path.absolute()),
            "tamanho": len(conteudo),
            "conteudo": conteudo[:5000]  # Limita a 5000 caracteres
        }
    except Exception as e:
        return {"erro": str(e)}


def executar_calculo(expressao: str) -> dict:
    """Executa um cálculo matemático seguro."""
    try:
        # Apenas operações básicas permitidas
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        resultado = eval(expressao, {"__builtins__": {}}, allowed)
        return {"expressao": expressao, "resultado": resultado}
    except Exception as e:
        return {"erro": f"Erro no cálculo: {str(e)}"}


def criar_servidor_exemplo() -> MCPSTDIOServer:
    """Cria um servidor MCP STDIO de exemplo."""

    servidor = MCPSTDIOServer(name="servidor-stdio-demo", version="1.0.0")

    # Registrar ferramentas
    servidor.register_tool(
        name="system_info",
        description="Obtém informações do sistema",
        schema={"type": "object", "properties": {}},
        handler=obter_informacoes_sistema
    )

    servidor.register_tool(
        name="list_files",
        description="Lista arquivos em um diretório",
        schema={
            "type": "object",
            "properties": {
                "diretorio": {"type": "string", "description": "Caminho do diretório"}
            }
        },
        handler=listar_arquivos
    )

    servidor.register_tool(
        name="read_file",
        description="Lê conteúdo de um arquivo texto",
        schema={
            "type": "object",
            "properties": {
                "caminho": {"type": "string", "description": "Caminho do arquivo"}
            },
            "required": ["caminho"]
        },
        handler=ler_arquivo_texto
    )

    servidor.register_tool(
        name="calculate",
        description="Executa cálculo matemático",
        schema={
            "type": "object",
            "properties": {
                "expressao": {"type": "string", "description": "Expressão matemática"}
            },
            "required": ["expressao"]
        },
        handler=executar_calculo
    )

    # Registrar recursos
    servidor.register_resource(
        uri="file:///config",
        name="Configurações",
        description="Arquivo de configuração do sistema"
    )

    return servidor


def main():
    print("=" * 60)
    print("SERVIDOR MCP COM STDIO")
    print("=" * 60)

    print("""
    STDIO (Standard Input/Output) é o método de transporte mais
    comum para servidores MCP locais.

    Fluxo de comunicação:
    ┌──────────────┐    stdin     ┌──────────────┐
    │  MCP Client  │ ──────────▶ │  MCP Server  │
    │  (Claude)    │ ◀────────── │  (Python)    │
    └──────────────┘    stdout    └──────────────┘

    Formato das mensagens: JSON-RPC 2.0
    """)

    # Criar servidor
    print("\n" + "=" * 60)
    print("CRIANDO SERVIDOR MCP STDIO")
    print("=" * 60)

    servidor = criar_servidor_exemplo()

    # Simular comunicação
    mensagens_teste = [
        # 1. Inicialização
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "cliente-teste", "version": "1.0.0"}
            }
        },
        # 2. Listar ferramentas
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        },
        # 3. Chamar ferramenta - Info do sistema
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "system_info",
                "arguments": {}
            }
        },
        # 4. Chamar ferramenta - Cálculo
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "calculate",
                "arguments": {"expressao": "2 ** 10 + 100"}
            }
        },
        # 5. Listar recursos
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/list"
        }
    ]

    servidor.simulate_stdio_communication(mensagens_teste)

    # Configuração para Claude Desktop
    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO PARA CLAUDE DESKTOP")
    print("=" * 60)

    config_exemplo = {
        "mcpServers": {
            "meu-servidor": {
                "command": "python",
                "args": ["/caminho/para/meu_servidor_mcp.py"],
                "env": {
                    "PYTHONPATH": "/caminho/para/projeto"
                }
            }
        }
    }

    print("\nAdicione ao arquivo ~/.config/claude/claude_desktop_config.json:")
    print(json.dumps(config_exemplo, indent=2))

    print("\n" + "=" * 60)
    print("CÓDIGO DE SERVIDOR MCP REAL")
    print("=" * 60)

    print("""
    Para criar um servidor MCP real com STDIO:

    ```python
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    server = Server("meu-servidor")

    @server.list_tools()
    async def listar_ferramentas():
        return [
            Tool(
                name="minha_ferramenta",
                description="Descrição da ferramenta",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"}
                    }
                }
            )
        ]

    @server.call_tool()
    async def chamar_ferramenta(name: str, arguments: dict):
        if name == "minha_ferramenta":
            return [TextContent(type="text", text="Resultado")]

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )

    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())
    ```

    Instale: pip install mcp
    Documentação: https://modelcontextprotocol.io/
    """)

    print("\nFim do demo de MCP Server STDIO")
    print("=" * 60)


if __name__ == "__main__":
    main()
