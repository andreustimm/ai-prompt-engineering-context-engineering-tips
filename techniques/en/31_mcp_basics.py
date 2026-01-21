"""
MCP (Model Context Protocol) - Fundamentals

MCP (Model Context Protocol) is an open protocol created by Anthropic to
connect AI assistants to external data sources and tools in a standardized
and secure way.

Main components:
- MCP Host: Application hosting the MCP client (e.g., Claude Desktop, IDEs)
- MCP Client: Component that connects to MCP servers
- MCP Server: Service that exposes resources, tools, and prompts

MCP Features:
- Resources: Data exposed by the server (files, databases, APIs)
- Tools: Functions that the LLM can invoke
- Prompts: Reusable prompt templates

Use cases:
- Database integration
- File system access
- External API connections
- Integration with GitHub, Slack, etc.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# MCP structure simulation (in production, use mcp-python-sdk)

@dataclass
class MCPResource:
    """Represents a resource exposed by the MCP server."""
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
    """Represents a tool exposed by the MCP server."""
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
    """Represents a prompt template exposed by the MCP server."""
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
    MCP server simulator for demonstration.

    In production, use the official SDK: pip install mcp
    Documentation: https://modelcontextprotocol.io/
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.resources: list[MCPResource] = []
        self.tools: list[MCPTool] = []
        self.prompts: list[MCPPrompt] = []
        self._tool_handlers: dict[str, callable] = {}

    def add_resource(self, uri: str, name: str, description: str, mime_type: str = "text/plain"):
        """Adds a resource to the server."""
        resource = MCPResource(uri=uri, name=name, description=description, mime_type=mime_type)
        self.resources.append(resource)
        print(f"   Resource added: {name} ({uri})")
        return resource

    def add_tool(self, name: str, description: str, input_schema: dict, handler: callable):
        """Adds a tool to the server."""
        tool = MCPTool(name=name, description=description, input_schema=input_schema)
        self.tools.append(tool)
        self._tool_handlers[name] = handler
        print(f"   Tool added: {name}")
        return tool

    def add_prompt(self, name: str, description: str, arguments: list = None):
        """Adds a prompt template to the server."""
        prompt = MCPPrompt(name=name, description=description, arguments=arguments or [])
        self.prompts.append(prompt)
        print(f"   Prompt added: {name}")
        return prompt

    def list_resources(self) -> list[dict]:
        """Lists all available resources."""
        return [r.to_dict() for r in self.resources]

    def list_tools(self) -> list[dict]:
        """Lists all available tools."""
        return [t.to_dict() for t in self.tools]

    def list_prompts(self) -> list[dict]:
        """Lists all available prompts."""
        return [p.to_dict() for p in self.prompts]

    def call_tool(self, name: str, arguments: dict) -> Any:
        """Executes a tool."""
        if name not in self._tool_handlers:
            raise ValueError(f"Tool '{name}' not found")

        handler = self._tool_handlers[name]
        return handler(**arguments)

    def get_server_info(self) -> dict:
        """Returns server information."""
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


# Example MCP tool implementations

def search_database(query: str, table: str = "users") -> dict:
    """Simulates database search."""
    simulated_data = {
        "users": [
            {"id": 1, "name": "Ana Silva", "email": "ana@email.com", "role": "Developer"},
            {"id": 2, "name": "Carlos Santos", "email": "carlos@email.com", "role": "Designer"},
            {"id": 3, "name": "Maria Oliveira", "email": "maria@email.com", "role": "Manager"},
        ],
        "products": [
            {"id": 1, "name": "Laptop", "price": 1299.00, "stock": 15},
            {"id": 2, "name": "Mouse", "price": 29.90, "stock": 150},
            {"id": 3, "name": "Keyboard", "price": 79.90, "stock": 80},
        ]
    }

    if table not in simulated_data:
        return {"error": f"Table '{table}' not found"}

    results = [
        item for item in simulated_data[table]
        if query.lower() in str(item).lower()
    ]

    return {
        "table": table,
        "query": query,
        "results": results,
        "total": len(results)
    }


def read_file(path: str) -> dict:
    """Simulates file reading."""
    simulated_files = {
        "config.json": '{"app": "demo", "version": "1.0"}',
        "readme.md": "# Demo Project\n\nThis is a demonstration project.",
        "data.csv": "name,value\nitem1,100\nitem2,200"
    }

    if path in simulated_files:
        return {
            "path": path,
            "content": simulated_files[path],
            "size": len(simulated_files[path])
        }
    return {"error": f"File '{path}' not found"}


def execute_command(command: str) -> dict:
    """Simulates command execution (demo only)."""
    safe_commands = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "whoami": "demo_user",
        "pwd": "/home/user/project"
    }

    if command in safe_commands:
        return {"command": command, "output": safe_commands[command], "code": 0}
    return {"command": command, "error": "Command not allowed", "code": 1}


def create_example_server() -> MCPServerSimulator:
    """Creates an example MCP server with resources, tools, and prompts."""

    server = MCPServerSimulator(name="demo-server", version="1.0.0")

    # Add resources
    server.add_resource(
        uri="file:///config.json",
        name="Configuration",
        description="Application configuration file",
        mime_type="application/json"
    )

    server.add_resource(
        uri="db://users",
        name="Users Table",
        description="System user data",
        mime_type="application/json"
    )

    # Add tools
    server.add_tool(
        name="search_database",
        description="Search data in the database",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
                "table": {"type": "string", "description": "Table name", "default": "users"}
            },
            "required": ["query"]
        },
        handler=search_database
    )

    server.add_tool(
        name="read_file",
        description="Read file content",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        },
        handler=read_file
    )

    server.add_tool(
        name="execute_command",
        description="Execute a system command (limited)",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"}
            },
            "required": ["command"]
        },
        handler=execute_command
    )

    # Add prompts
    server.add_prompt(
        name="data_analysis",
        description="Template for data analysis",
        arguments=[
            {"name": "data", "description": "Data to analyze", "required": True},
            {"name": "format", "description": "Output format", "required": False}
        ]
    )

    server.add_prompt(
        name="document_summary",
        description="Template for document summarization",
        arguments=[
            {"name": "document", "description": "Document text", "required": True},
            {"name": "length", "description": "Summary length (short/medium/long)", "required": False}
        ]
    )

    return server


def demonstrate_mcp_usage():
    """Demonstrates basic MCP protocol usage."""

    print("\n" + "=" * 60)
    print("CREATING MCP SERVER")
    print("=" * 60)

    server = create_example_server()

    # Server information
    print("\n" + "-" * 40)
    print("SERVER INFORMATION")
    print("-" * 40)
    info = server.get_server_info()
    print(json.dumps(info, indent=2))

    # List resources
    print("\n" + "-" * 40)
    print("AVAILABLE RESOURCES")
    print("-" * 40)
    for resource in server.list_resources():
        print(f"  - {resource['name']}: {resource['uri']}")

    # List tools
    print("\n" + "-" * 40)
    print("AVAILABLE TOOLS")
    print("-" * 40)
    for tool in server.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # List prompts
    print("\n" + "-" * 40)
    print("AVAILABLE PROMPTS")
    print("-" * 40)
    for prompt in server.list_prompts():
        print(f"  - {prompt['name']}: {prompt['description']}")

    # Demonstrate tool calls
    print("\n" + "=" * 60)
    print("TOOL CALL DEMONSTRATION")
    print("=" * 60)

    # Database search
    print("\n1. Search database:")
    result = server.call_tool("search_database", {"query": "Ana", "table": "users"})
    print(f"   Result: {json.dumps(result, indent=2)}")

    # Read file
    print("\n2. Read configuration file:")
    result = server.call_tool("read_file", {"path": "config.json"})
    print(f"   Result: {json.dumps(result, indent=2)}")

    # Execute command
    print("\n3. Execute command:")
    result = server.call_tool("execute_command", {"command": "date"})
    print(f"   Result: {json.dumps(result, indent=2)}")


def main():
    print("=" * 60)
    print("MCP (MODEL CONTEXT PROTOCOL) - Fundamentals")
    print("=" * 60)

    print("""
    MCP is a protocol that allows connecting AI assistants
    to external data sources and tools in a standardized way.

    Key concepts:
    1. Resources - Data exposed by the server (files, DB, APIs)
    2. Tools - Functions that the LLM can invoke
    3. Prompts - Reusable prompt templates

    Architecture:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  MCP Host   │────▶│ MCP Client  │────▶│ MCP Server  │
    │ (Claude,    │     │             │     │ (DB, API,   │
    │  VS Code)   │◀────│             │◀────│  Files)     │
    └─────────────┘     └─────────────┘     └─────────────┘
    """)

    demonstrate_mcp_usage()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
    To use MCP in production:

    1. Install the SDK: pip install mcp

    2. Create a real MCP server:
       - Use STDIO for local communication
       - Use HTTP/SSE for remote communication

    3. Configure in Claude Desktop or your IDE:
       - Add configuration in ~/.config/claude/claude_desktop_config.json

    Official documentation: https://modelcontextprotocol.io/
    """)

    print("\nEnd of MCP Basics demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
