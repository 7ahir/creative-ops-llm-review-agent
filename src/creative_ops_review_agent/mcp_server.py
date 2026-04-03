from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response

from .config import Settings
from .knowledge import KnowledgeStore
from .tool_runtime import ConstraintToolRuntime


def create_mcp_app(settings: Optional[Settings] = None) -> FastAPI:
    active_settings = settings or Settings.load()
    runtime = ConstraintToolRuntime(KnowledgeStore(active_settings))
    app = FastAPI(title="Creative Ops MCP Tool Server", version="0.1.0")

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "name": "creative-ops-mcp",
            "transport": "streamable-http-like-jsonrpc",
            "endpoint": active_settings.mcp_path,
            "tools": [tool["name"] for tool in runtime.mcp_tools()],
        }

    @app.post(active_settings.mcp_path)
    async def mcp(request: Request) -> Response:
        payload = await request.json()
        if isinstance(payload, list):
            responses = [_handle_message(message, runtime) for message in payload]
            return Response(
                content=json.dumps([item for item in responses if item is not None]),
                media_type="application/json",
            )

        response_payload = _handle_message(payload, runtime)
        if response_payload is None:
            return Response(status_code=202)
        return Response(content=json.dumps(response_payload), media_type="application/json")

    return app


def _handle_message(message: Dict[str, Any], runtime: ConstraintToolRuntime) -> Optional[Dict[str, Any]]:
    jsonrpc_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})

    if method == "notifications/initialized":
        return None
    if method == "ping":
        return _result(jsonrpc_id, {})
    if method == "initialize":
        return _result(
            jsonrpc_id,
            {
                "protocolVersion": "2025-03-26",
                "serverInfo": {
                    "name": "creative-ops-mcp",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "tools": {},
                },
            },
        )
    if method == "tools/list":
        return _result(jsonrpc_id, {"tools": runtime.mcp_tools()})
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            output = runtime.execute(tool_name, arguments)
        except Exception as exc:
            return _result(
                jsonrpc_id,
                {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            )
        return _result(
            jsonrpc_id,
            {
                "content": [{"type": "text", "text": json.dumps(output)}],
                "structuredContent": output,
                "isError": False,
            },
        )

    if jsonrpc_id is None:
        return None
    return _error(jsonrpc_id, -32601, "Method not found: %s" % method)


def _result(jsonrpc_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": jsonrpc_id, "result": result}


def _error(jsonrpc_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": jsonrpc_id, "error": {"code": code, "message": message}}


def serve() -> None:
    settings = Settings.load()
    uvicorn.run(
        create_mcp_app(settings),
        host=settings.mcp_host,
        port=settings.mcp_port,
        reload=False,
    )
