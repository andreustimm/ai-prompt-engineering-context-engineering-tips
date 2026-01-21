"""
MCP Server with STDIO (Standard Input/Output)

STDIO is the most common transport method for local MCP servers.
Communication occurs through stdin/stdout, ideal for CLI tools
and integrations with IDEs like Claude Desktop, VS Code, etc.

STDIO characteristics:
- Local communication (same computer)
- Low latency
- Easy configuration
- Ideal for development tools

Use cases:
- Claude Desktop integration
- Command line tools
- Local file access
- Script execution

Requirements:
- pip install mcp
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import asyncio
from typing import Any
from datetime import datetime

# MCP STDIO protocol simulation
# In production, use: from mcp.server import Server
# from mcp.server.stdio import stdio_server

class MCPMessage:
    """Represents an MCP protocol message."""

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
    MCP server that communicates via STDIO.

    This is a simulator for demonstration. In production, use:
    ```python
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server = Server("my-server")

    @server.list_tools()
    async def list_tools():
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
        """Registers a tool on the server."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": schema,
            "handler": handler
        }
        print(f"[SERVER] Tool registered: {name}")

    def register_resource(self, uri: str, name: str, description: str):
        """Registers a resource on the server."""
        self.resources[uri] = {
            "uri": uri,
            "name": name,
            "description": description
        }
        print(f"[SERVER] Resource registered: {name}")

    def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Processes an MCP message and returns the response."""

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
                error={"code": -32601, "message": f"Unknown method: {method}"}
            )

    def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Handles initialization request."""
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
        """Lists all available tools."""
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
        """Executes a tool."""
        tool_name = message.params.get("name")
        arguments = message.params.get("arguments", {})

        if tool_name not in self.tools:
            return MCPMessage(
                id=message.id,
                error={"code": -32602, "message": f"Tool not found: {tool_name}"}
            )

        try:
            tool = self.tools[tool_name]
            result = tool["handler"](**arguments)
            return MCPMessage(
                id=message.id,
                result={
                    "content": [{"type": "text", "text": json.dumps(result)}],
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
        """Lists all available resources."""
        resources_list = list(self.resources.values())
        return MCPMessage(id=message.id, result={"resources": resources_list})

    def _handle_read_resource(self, message: MCPMessage) -> MCPMessage:
        """Reads a specific resource."""
        uri = message.params.get("uri")

        if uri not in self.resources:
            return MCPMessage(
                id=message.id,
                error={"code": -32602, "message": f"Resource not found: {uri}"}
            )

        return MCPMessage(
            id=message.id,
            result={
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": f"Content of resource {uri}"
                }]
            }
        )

    def simulate_stdio_communication(self, messages: list[dict]):
        """Simulates STDIO communication with a list of messages."""
        print("\n" + "=" * 60)
        print("STDIO COMMUNICATION SIMULATION")
        print("=" * 60)

        for i, msg_data in enumerate(messages, 1):
            print(f"\n--- Message {i} ---")

            # Simulates receiving via stdin
            message = MCPMessage(**msg_data)
            print(f"[STDIN]  Received: {message.to_json()}")

            # Process message
            response = self.handle_message(message)

            # Simulates sending via stdout
            print(f"[STDOUT] Sent: {response.to_json()}")


# Example tools

def get_system_info() -> dict:
    """Returns system information."""
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "current_directory": str(Path.cwd()),
        "datetime": datetime.now().isoformat()
    }


def list_files(directory: str = ".") -> dict:
    """Lists files in a directory."""
    try:
        path = Path(directory)
        if not path.exists():
            return {"error": f"Directory not found: {directory}"}

        files = []
        for item in path.iterdir():
            files.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })

        return {
            "directory": str(path.absolute()),
            "total": len(files),
            "items": files[:20]  # Limit to 20 items
        }
    except Exception as e:
        return {"error": str(e)}


def read_text_file(path: str) -> dict:
    """Reads text file content."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {"error": f"File not found: {path}"}

        if file_path.stat().st_size > 100000:  # 100KB limit
            return {"error": "File too large (max 100KB)"}

        content = file_path.read_text(encoding='utf-8')
        return {
            "path": str(file_path.absolute()),
            "size": len(content),
            "content": content[:5000]  # Limit to 5000 characters
        }
    except Exception as e:
        return {"error": str(e)}


def execute_calculation(expression: str) -> dict:
    """Executes a safe mathematical calculation."""
    try:
        # Only basic operations allowed
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}


def create_example_server() -> MCPSTDIOServer:
    """Creates an example MCP STDIO server."""

    server = MCPSTDIOServer(name="stdio-demo-server", version="1.0.0")

    # Register tools
    server.register_tool(
        name="system_info",
        description="Gets system information",
        schema={"type": "object", "properties": {}},
        handler=get_system_info
    )

    server.register_tool(
        name="list_files",
        description="Lists files in a directory",
        schema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path"}
            }
        },
        handler=list_files
    )

    server.register_tool(
        name="read_file",
        description="Reads text file content",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        },
        handler=read_text_file
    )

    server.register_tool(
        name="calculate",
        description="Executes mathematical calculation",
        schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression"}
            },
            "required": ["expression"]
        },
        handler=execute_calculation
    )

    # Register resources
    server.register_resource(
        uri="file:///config",
        name="Configuration",
        description="System configuration file"
    )

    return server


def main():
    print("=" * 60)
    print("MCP SERVER WITH STDIO")
    print("=" * 60)

    print("""
    STDIO (Standard Input/Output) is the most common transport
    method for local MCP servers.

    Communication flow:
    ┌──────────────┐    stdin     ┌──────────────┐
    │  MCP Client  │ ──────────▶ │  MCP Server  │
    │  (Claude)    │ ◀────────── │  (Python)    │
    └──────────────┘    stdout    └──────────────┘

    Message format: JSON-RPC 2.0
    """)

    # Create server
    print("\n" + "=" * 60)
    print("CREATING MCP STDIO SERVER")
    print("=" * 60)

    server = create_example_server()

    # Simulate communication
    test_messages = [
        # 1. Initialization
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        },
        # 2. List tools
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        },
        # 3. Call tool - System info
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "system_info",
                "arguments": {}
            }
        },
        # 4. Call tool - Calculation
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "calculate",
                "arguments": {"expression": "2 ** 10 + 100"}
            }
        },
        # 5. List resources
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/list"
        }
    ]

    server.simulate_stdio_communication(test_messages)

    # Claude Desktop configuration
    print("\n" + "=" * 60)
    print("CLAUDE DESKTOP CONFIGURATION")
    print("=" * 60)

    config_example = {
        "mcpServers": {
            "my-server": {
                "command": "python",
                "args": ["/path/to/my_mcp_server.py"],
                "env": {
                    "PYTHONPATH": "/path/to/project"
                }
            }
        }
    }

    print("\nAdd to ~/.config/claude/claude_desktop_config.json:")
    print(json.dumps(config_example, indent=2))

    print("\n" + "=" * 60)
    print("REAL MCP SERVER CODE")
    print("=" * 60)

    print("""
    To create a real MCP server with STDIO:

    ```python
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    server = Server("my-server")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="my_tool",
                description="Tool description",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"}
                    }
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "my_tool":
            return [TextContent(type="text", text="Result")]

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

    Install: pip install mcp
    Documentation: https://modelcontextprotocol.io/
    """)

    print("\nEnd of MCP Server STDIO demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
