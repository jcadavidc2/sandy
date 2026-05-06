"""Sandy MCP server — exposes prediction tools to OpenCLAW via stdio.

Phase 2, Task 13.2: MCP server using stdio transport. Reads JSON-RPC from
stdin, writes responses to stdout. Registers tool definitions with JSON
Schema for inputs/outputs.

Entry point: python -m sandy.mcp.server

Requirements: 3.2, 3.4, 11.1, 11.4, 11.5
"""
from __future__ import annotations

import json
import sys
from typing import Any

from sandy.logging import configure_logging, get_logger
from sandy.mcp.tools import TOOL_DEFINITIONS, handle_tool_call

logger = get_logger("mcp.server")


def main() -> None:
    """MCP server main loop — stdio transport."""
    configure_logging("ERROR")  # suppress logs on stdout (MCP uses stdout)

    # Write server capabilities on startup
    # MCP protocol: server sends nothing until client sends initialize

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _send_error(None, -32700, "Parse error")
            continue

        request_id = request.get("id")
        method = request.get("method", "")

        try:
            if method == "initialize":
                _send_result(request_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "sandy",
                        "version": "0.2.0",
                    },
                })

            elif method == "notifications/initialized":
                # Client acknowledgment — no response needed
                pass

            elif method == "tools/list":
                _send_result(request_id, {"tools": TOOL_DEFINITIONS})

            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})

                result = handle_tool_call(tool_name, tool_args)
                _send_result(request_id, {
                    "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                })

            else:
                _send_error(request_id, -32601, f"Method not found: {method}")

        except Exception as exc:
            _send_error(request_id, -32603, f"Internal error: {exc}")


def _send_result(request_id: Any, result: Any) -> None:
    """Send a JSON-RPC success response."""
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _send_error(request_id: Any, code: int, message: str) -> None:
    """Send a JSON-RPC error response."""
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
