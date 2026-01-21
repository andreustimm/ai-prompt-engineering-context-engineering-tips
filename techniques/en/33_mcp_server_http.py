"""
MCP Server with HTTP/SSE (Server-Sent Events)

HTTP/SSE is the transport method for remote MCP servers.
Enables communication over the network, ideal for cloud services
and integrations with multiple clients.

HTTP/SSE characteristics:
- Remote communication (network/internet)
- Support for multiple simultaneous clients
- Ideal for production services
- Response streaming with SSE

Use cases:
- AI-as-a-service APIs
- Enterprise integrations
- Centralized servers
- AI microservices

Requirements:
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

# MCP HTTP/SSE protocol simulation
# In production, use FastAPI with mcp.server.sse

@dataclass
class SSEEvent:
    """Represents a Server-Sent Events event."""
    event: str
    data: str
    id: str = None

    def to_sse_format(self) -> str:
        """Converts to SSE format."""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event}")
        lines.append(f"data: {self.data}")
        lines.append("")  # Empty line to end event
        return "\n".join(lines)


class MCPHTTPServer:
    """
    MCP server simulator with HTTP/SSE.

    In production, use FastAPI with MCP SDK:
    ```python
    from fastapi import FastAPI
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport

    app = FastAPI()
    server = Server("my-server")
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
        """Registers a tool on the server."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": schema,
            "handler": handler
        }

    async def handle_request(self, method: str, params: dict = None) -> dict:
        """Processes an HTTP request."""
        params = params or {}

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
        }

        if method in handlers:
            return await handlers[method](params)
        return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

    async def _handle_initialize(self, params: dict) -> dict:
        """Handles initialization."""
        session_id = f"session_{datetime.now().timestamp()}"
        self.sessions[session_id] = {"created": datetime.now().isoformat()}

        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.name, "version": "1.0.0"},
            "sessionId": session_id,
            "capabilities": {"tools": {}, "resources": {}}
        }

    async def _handle_list_tools(self, params: dict) -> dict:
        """Lists tools."""
        return {
            "tools": [
                {"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]}
                for t in self.tools.values()
            ]
        }

    async def _handle_call_tool(self, params: dict) -> dict:
        """Executes tool."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            return {"error": {"code": -32602, "message": f"Tool not found: {tool_name}"}}

        try:
            result = self.tools[tool_name]["handler"](**arguments)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as e:
            return {"error": {"code": -32000, "message": str(e)}}

    async def _handle_list_resources(self, params: dict) -> dict:
        """Lists resources."""
        return {"resources": list(self.resources.values())}

    async def stream_response(self, tool_name: str, arguments: dict) -> AsyncGenerator[SSEEvent, None]:
        """Simulates SSE response streaming."""

        # Start event
        yield SSEEvent(
            event="start",
            data=json.dumps({"tool": tool_name, "status": "processing"})
        )

        await asyncio.sleep(0.1)  # Simulates processing

        # Execute tool
        if tool_name in self.tools:
            result = self.tools[tool_name]["handler"](**arguments)

            # Progress event
            yield SSEEvent(
                event="progress",
                data=json.dumps({"progress": 50, "message": "Processing..."})
            )

            await asyncio.sleep(0.1)

            # Result event
            yield SSEEvent(
                event="result",
                data=json.dumps(result)
            )
        else:
            yield SSEEvent(
                event="error",
                data=json.dumps({"error": f"Tool not found: {tool_name}"})
            )

        # Completion event
        yield SSEEvent(
            event="done",
            data=json.dumps({"status": "completed"})
        )


# Example tools

def query_external_api(endpoint: str, method: str = "GET") -> dict:
    """Simulates external API query."""
    simulated_apis = {
        "/users": [
            {"id": 1, "name": "John", "active": True},
            {"id": 2, "name": "Mary", "active": True},
        ],
        "/products": [
            {"id": 1, "name": "Laptop", "price": 999.00},
            {"id": 2, "name": "Mouse", "price": 29.90},
        ],
        "/status": {"status": "online", "version": "1.0.0", "uptime": "99.9%"}
    }

    if endpoint in simulated_apis:
        return {
            "endpoint": endpoint,
            "method": method,
            "status_code": 200,
            "data": simulated_apis[endpoint]
        }
    return {"endpoint": endpoint, "status_code": 404, "error": "Endpoint not found"}


def process_data(data: list, operation: str = "sum") -> dict:
    """Processes numeric data."""
    try:
        numbers = [float(x) for x in data]
        operations = {
            "sum": sum(numbers),
            "average": sum(numbers) / len(numbers),
            "max": max(numbers),
            "min": min(numbers),
            "count": len(numbers)
        }

        if operation not in operations:
            return {"error": f"Unknown operation: {operation}"}

        return {
            "operation": operation,
            "input": data,
            "result": operations[operation]
        }
    except Exception as e:
        return {"error": str(e)}


def generate_report(type: str, format: str = "json") -> dict:
    """Generates simulated report."""
    reports = {
        "sales": {
            "title": "Sales Report",
            "period": "January 2024",
            "total": 150000.00,
            "items": 342,
            "growth": "+15%"
        },
        "users": {
            "title": "Users Report",
            "total_users": 5420,
            "active": 4890,
            "new_this_month": 234
        },
        "system": {
            "title": "System Report",
            "cpu_usage": "45%",
            "memory_usage": "62%",
            "disk_usage": "38%"
        }
    }

    if type not in reports:
        return {"error": f"Unknown report type: {type}"}

    return {
        "type": type,
        "format": format,
        "data": reports[type],
        "generated_at": datetime.now().isoformat()
    }


def create_http_server() -> MCPHTTPServer:
    """Creates example MCP HTTP server."""

    server = MCPHTTPServer(name="http-demo-server", port=8000)

    server.register_tool(
        name="query_api",
        description="Queries an external API",
        schema={
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API endpoint"},
                "method": {"type": "string", "enum": ["GET", "POST"], "default": "GET"}
            },
            "required": ["endpoint"]
        },
        handler=query_external_api
    )

    server.register_tool(
        name="process_data",
        description="Processes numeric data",
        schema={
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "number"}},
                "operation": {"type": "string", "enum": ["sum", "average", "max", "min", "count"]}
            },
            "required": ["data"]
        },
        handler=process_data
    )

    server.register_tool(
        name="generate_report",
        description="Generates system report",
        schema={
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["sales", "users", "system"]},
                "format": {"type": "string", "default": "json"}
            },
            "required": ["type"]
        },
        handler=generate_report
    )

    return server


async def demonstrate_http_server():
    """Demonstrates the HTTP/SSE server."""

    print("\n" + "=" * 60)
    print("CREATING MCP HTTP SERVER")
    print("=" * 60)

    server = create_http_server()

    # Simulate HTTP requests
    print("\n" + "-" * 40)
    print("HTTP REQUEST SIMULATION")
    print("-" * 40)

    # 1. Initialization
    print("\n1. POST /initialize")
    result = await server.handle_request("initialize", {})
    print(f"   Response: {json.dumps(result, indent=2)}")

    # 2. List tools
    print("\n2. GET /tools/list")
    result = await server.handle_request("tools/list", {})
    print(f"   Available tools: {len(result['tools'])}")
    for tool in result['tools']:
        print(f"     - {tool['name']}: {tool['description']}")

    # 3. Call tool - API
    print("\n3. POST /tools/call (query_api)")
    result = await server.handle_request("tools/call", {
        "name": "query_api",
        "arguments": {"endpoint": "/users"}
    })
    print(f"   Response: {result['content'][0]['text'][:200]}...")

    # 4. Call tool - Process data
    print("\n4. POST /tools/call (process_data)")
    result = await server.handle_request("tools/call", {
        "name": "process_data",
        "arguments": {"data": [10, 20, 30, 40, 50], "operation": "average"}
    })
    print(f"   Response: {result['content'][0]['text']}")

    # Simulate SSE streaming
    print("\n" + "-" * 40)
    print("SSE STREAMING SIMULATION")
    print("-" * 40)

    print("\nGET /sse/stream?tool=generate_report&type=sales")
    async for event in server.stream_response("generate_report", {"type": "sales"}):
        print(f"   {event.to_sse_format()}")


def main():
    print("=" * 60)
    print("MCP SERVER WITH HTTP/SSE")
    print("=" * 60)

    print("""
    HTTP/SSE is the transport method for remote MCP servers.

    Architecture:
    ┌──────────────┐    HTTP POST   ┌──────────────┐
    │  MCP Client  │ ─────────────▶ │  MCP Server  │
    │  (Claude)    │                │  (FastAPI)   │
    │              │ ◀───────────── │              │
    └──────────────┘    SSE Stream  └──────────────┘

    Typical endpoints:
    - POST /initialize - Initializes session
    - GET  /tools/list - Lists tools
    - POST /tools/call - Executes tool
    - GET  /sse        - Event stream
    """)

    # Run async demonstration
    asyncio.run(demonstrate_http_server())

    # Real example code
    print("\n" + "=" * 60)
    print("REAL MCP HTTP SERVER CODE")
    print("=" * 60)

    print("""
    Example with FastAPI and MCP SDK:

    ```python
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp.types import Tool, TextContent

    app = FastAPI()
    server = Server("my-http-server")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="my_tool",
                description="Description",
                inputSchema={"type": "object", "properties": {}}
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        return [TextContent(type="text", text="Result")]

    # SSE endpoint
    @app.get("/sse")
    async def sse_endpoint(request: Request):
        sse = SseServerTransport("/messages")
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send
        ) as streams:
            await server.run(streams[0], streams[1])

    # POST endpoint for messages
    @app.post("/messages")
    async def messages_endpoint(request: Request):
        # Process MCP messages
        body = await request.json()
        # ... process request
        return {"result": "..."}

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

    Install: pip install mcp fastapi uvicorn
    Run: uvicorn server:app --reload
    """)

    print("\nEnd of MCP Server HTTP/SSE demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
