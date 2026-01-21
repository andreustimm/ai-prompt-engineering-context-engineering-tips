"""
Servidor MCP com HTTP/SSE (Server-Sent Events)

HTTP/SSE é o método de transporte para servidores MCP remotos.
Permite comunicação através da rede, ideal para serviços em nuvem
e integrações com múltiplos clientes.

Características do HTTP/SSE:
- Comunicação remota (rede/internet)
- Suporte a múltiplos clientes simultâneos
- Ideal para serviços em produção
- Streaming de respostas com SSE

Casos de uso:
- APIs de IA como serviço
- Integrações empresariais
- Servidores centralizados
- Microserviços de IA

Requisitos:
- pip install mcp fastapi uvicorn
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import asyncio
from typing import Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

# Simulação do protocolo MCP HTTP/SSE
# Em produção, use FastAPI com mcp.server.sse

@dataclass
class SSEEvent:
    """Representa um evento Server-Sent Events."""
    event: str
    data: str
    id: str = None

    def to_sse_format(self) -> str:
        """Converte para formato SSE."""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event}")
        lines.append(f"data: {self.data}")
        lines.append("")  # Linha vazia para finalizar evento
        return "\n".join(lines)


class MCPHTTPServer:
    """
    Simulador de servidor MCP com HTTP/SSE.

    Em produção, use FastAPI com o SDK MCP:
    ```python
    from fastapi import FastAPI
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport

    app = FastAPI()
    server = Server("meu-servidor")
    sse = SseServerTransport("/messages")

    @app.get("/sse")
    async def sse_endpoint(request: Request):
        async with sse.connect_sse(request) as streams:
            await server.run(streams[0], streams[1])
    ```
    """

    def __init__(self, name: str, host: str = "localhost", port: int = 8000):
        self.name = name
        self.host = host
        self.port = port
        self.tools = {}
        self.resources = {}
        self.sessions = {}

    def register_tool(self, name: str, description: str, schema: dict, handler: callable):
        """Registra uma ferramenta no servidor."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": schema,
            "handler": handler
        }

    async def handle_request(self, method: str, params: dict = None) -> dict:
        """Processa uma requisição HTTP."""
        params = params or {}

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
        }

        if method in handlers:
            return await handlers[method](params)
        return {"error": {"code": -32601, "message": f"Método desconhecido: {method}"}}

    async def _handle_initialize(self, params: dict) -> dict:
        """Trata inicialização."""
        session_id = f"session_{datetime.now().timestamp()}"
        self.sessions[session_id] = {"created": datetime.now().isoformat()}

        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.name, "version": "1.0.0"},
            "sessionId": session_id,
            "capabilities": {"tools": {}, "resources": {}}
        }

    async def _handle_list_tools(self, params: dict) -> dict:
        """Lista ferramentas."""
        return {
            "tools": [
                {"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]}
                for t in self.tools.values()
            ]
        }

    async def _handle_call_tool(self, params: dict) -> dict:
        """Executa ferramenta."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            return {"error": {"code": -32602, "message": f"Ferramenta não encontrada: {tool_name}"}}

        try:
            result = self.tools[tool_name]["handler"](**arguments)
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}
        except Exception as e:
            return {"error": {"code": -32000, "message": str(e)}}

    async def _handle_list_resources(self, params: dict) -> dict:
        """Lista recursos."""
        return {"resources": list(self.resources.values())}

    async def stream_response(self, tool_name: str, arguments: dict) -> AsyncGenerator[SSEEvent, None]:
        """Simula streaming de resposta via SSE."""

        # Evento de início
        yield SSEEvent(
            event="start",
            data=json.dumps({"tool": tool_name, "status": "processing"})
        )

        await asyncio.sleep(0.1)  # Simula processamento

        # Executa ferramenta
        if tool_name in self.tools:
            result = self.tools[tool_name]["handler"](**arguments)

            # Evento de progresso
            yield SSEEvent(
                event="progress",
                data=json.dumps({"progress": 50, "message": "Processando..."})
            )

            await asyncio.sleep(0.1)

            # Evento de resultado
            yield SSEEvent(
                event="result",
                data=json.dumps(result, ensure_ascii=False)
            )
        else:
            yield SSEEvent(
                event="error",
                data=json.dumps({"error": f"Ferramenta não encontrada: {tool_name}"})
            )

        # Evento de conclusão
        yield SSEEvent(
            event="done",
            data=json.dumps({"status": "completed"})
        )


# Ferramentas de exemplo

def consultar_api_externa(endpoint: str, metodo: str = "GET") -> dict:
    """Simula consulta a API externa."""
    apis_simuladas = {
        "/usuarios": [
            {"id": 1, "nome": "João", "ativo": True},
            {"id": 2, "nome": "Maria", "ativo": True},
        ],
        "/produtos": [
            {"id": 1, "nome": "Laptop", "preco": 2999.00},
            {"id": 2, "nome": "Mouse", "preco": 79.90},
        ],
        "/status": {"status": "online", "versao": "1.0.0", "uptime": "99.9%"}
    }

    if endpoint in apis_simuladas:
        return {
            "endpoint": endpoint,
            "metodo": metodo,
            "status_code": 200,
            "dados": apis_simuladas[endpoint]
        }
    return {"endpoint": endpoint, "status_code": 404, "erro": "Endpoint não encontrado"}


def processar_dados(dados: list, operacao: str = "soma") -> dict:
    """Processa dados numéricos."""
    try:
        numeros = [float(x) for x in dados]
        operacoes = {
            "soma": sum(numeros),
            "media": sum(numeros) / len(numeros),
            "maximo": max(numeros),
            "minimo": min(numeros),
            "contagem": len(numeros)
        }

        if operacao not in operacoes:
            return {"erro": f"Operação desconhecida: {operacao}"}

        return {
            "operacao": operacao,
            "entrada": dados,
            "resultado": operacoes[operacao]
        }
    except Exception as e:
        return {"erro": str(e)}


def gerar_relatorio(tipo: str, formato: str = "json") -> dict:
    """Gera relatório simulado."""
    relatorios = {
        "vendas": {
            "titulo": "Relatório de Vendas",
            "periodo": "Janeiro 2024",
            "total": 150000.00,
            "itens": 342,
            "crescimento": "+15%"
        },
        "usuarios": {
            "titulo": "Relatório de Usuários",
            "total_usuarios": 5420,
            "ativos": 4890,
            "novos_mes": 234
        },
        "sistema": {
            "titulo": "Relatório do Sistema",
            "cpu_uso": "45%",
            "memoria_uso": "62%",
            "disco_uso": "38%"
        }
    }

    if tipo not in relatorios:
        return {"erro": f"Tipo de relatório desconhecido: {tipo}"}

    return {
        "tipo": tipo,
        "formato": formato,
        "dados": relatorios[tipo],
        "gerado_em": datetime.now().isoformat()
    }


def criar_servidor_http() -> MCPHTTPServer:
    """Cria servidor MCP HTTP de exemplo."""

    servidor = MCPHTTPServer(name="servidor-http-demo", port=8000)

    servidor.register_tool(
        name="consultar_api",
        description="Consulta uma API externa",
        schema={
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "Endpoint da API"},
                "metodo": {"type": "string", "enum": ["GET", "POST"], "default": "GET"}
            },
            "required": ["endpoint"]
        },
        handler=consultar_api_externa
    )

    servidor.register_tool(
        name="processar_dados",
        description="Processa dados numéricos",
        schema={
            "type": "object",
            "properties": {
                "dados": {"type": "array", "items": {"type": "number"}},
                "operacao": {"type": "string", "enum": ["soma", "media", "maximo", "minimo", "contagem"]}
            },
            "required": ["dados"]
        },
        handler=processar_dados
    )

    servidor.register_tool(
        name="gerar_relatorio",
        description="Gera relatório do sistema",
        schema={
            "type": "object",
            "properties": {
                "tipo": {"type": "string", "enum": ["vendas", "usuarios", "sistema"]},
                "formato": {"type": "string", "default": "json"}
            },
            "required": ["tipo"]
        },
        handler=gerar_relatorio
    )

    return servidor


async def demonstrar_http_server():
    """Demonstra o servidor HTTP/SSE."""

    print("\n" + "=" * 60)
    print("CRIANDO SERVIDOR MCP HTTP")
    print("=" * 60)

    servidor = criar_servidor_http()

    # Simular requisições HTTP
    print("\n" + "-" * 40)
    print("SIMULAÇÃO DE REQUISIÇÕES HTTP")
    print("-" * 40)

    # 1. Inicialização
    print("\n1. POST /initialize")
    resultado = await servidor.handle_request("initialize", {})
    print(f"   Resposta: {json.dumps(resultado, indent=2)}")

    # 2. Listar ferramentas
    print("\n2. GET /tools/list")
    resultado = await servidor.handle_request("tools/list", {})
    print(f"   Ferramentas disponíveis: {len(resultado['tools'])}")
    for tool in resultado['tools']:
        print(f"     - {tool['name']}: {tool['description']}")

    # 3. Chamar ferramenta - API
    print("\n3. POST /tools/call (consultar_api)")
    resultado = await servidor.handle_request("tools/call", {
        "name": "consultar_api",
        "arguments": {"endpoint": "/usuarios"}
    })
    print(f"   Resposta: {resultado['content'][0]['text'][:200]}...")

    # 4. Chamar ferramenta - Processar dados
    print("\n4. POST /tools/call (processar_dados)")
    resultado = await servidor.handle_request("tools/call", {
        "name": "processar_dados",
        "arguments": {"dados": [10, 20, 30, 40, 50], "operacao": "media"}
    })
    print(f"   Resposta: {resultado['content'][0]['text']}")

    # Simular streaming SSE
    print("\n" + "-" * 40)
    print("SIMULAÇÃO DE STREAMING SSE")
    print("-" * 40)

    print("\nGET /sse/stream?tool=gerar_relatorio&tipo=vendas")
    async for evento in servidor.stream_response("gerar_relatorio", {"tipo": "vendas"}):
        print(f"   {evento.to_sse_format()}")


def main():
    print("=" * 60)
    print("SERVIDOR MCP COM HTTP/SSE")
    print("=" * 60)

    print("""
    HTTP/SSE é o método de transporte para servidores MCP remotos.

    Arquitetura:
    ┌──────────────┐    HTTP POST   ┌──────────────┐
    │  MCP Client  │ ─────────────▶ │  MCP Server  │
    │  (Claude)    │                │  (FastAPI)   │
    │              │ ◀───────────── │              │
    └──────────────┘    SSE Stream  └──────────────┘

    Endpoints típicos:
    - POST /initialize - Inicializa sessão
    - GET  /tools/list - Lista ferramentas
    - POST /tools/call - Executa ferramenta
    - GET  /sse        - Stream de eventos
    """)

    # Executar demonstração assíncrona
    asyncio.run(demonstrar_http_server())

    # Código de exemplo real
    print("\n" + "=" * 60)
    print("CÓDIGO DE SERVIDOR MCP HTTP REAL")
    print("=" * 60)

    print("""
    Exemplo com FastAPI e MCP SDK:

    ```python
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp.types import Tool, TextContent

    app = FastAPI()
    server = Server("meu-servidor-http")

    @server.list_tools()
    async def listar_ferramentas():
        return [
            Tool(
                name="minha_ferramenta",
                description="Descrição",
                inputSchema={"type": "object", "properties": {}}
            )
        ]

    @server.call_tool()
    async def chamar_ferramenta(name: str, arguments: dict):
        return [TextContent(type="text", text="Resultado")]

    # Endpoint SSE
    @app.get("/sse")
    async def sse_endpoint(request: Request):
        sse = SseServerTransport("/messages")
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send
        ) as streams:
            await server.run(streams[0], streams[1])

    # POST endpoint para mensagens
    @app.post("/messages")
    async def messages_endpoint(request: Request):
        # Processa mensagens MCP
        body = await request.json()
        # ... processa requisição
        return {"result": "..."}

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

    Instale: pip install mcp fastapi uvicorn
    Execute: uvicorn servidor:app --reload
    """)

    print("\nFim do demo de MCP Server HTTP/SSE")
    print("=" * 60)


if __name__ == "__main__":
    main()
