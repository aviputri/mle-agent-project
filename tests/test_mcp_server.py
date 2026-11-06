import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from src.mcp_server import build_server


def test_server_exposes_typed_tools(settings):
    server, tools = build_server(settings)

    async def go():
        names = {t.name for t in await server.list_tools()}
        resources = {str(r.uri) for r in await server.list_resources()}
        ticket = await server.call_tool("get_ticket", {"ticket_id": "T-1001"})
        with pytest.raises(ToolError, match="Unknown ticket_id"):  # in-process; a client sees isError=True
            await server.call_tool("get_ticket", {"ticket_id": "T-9999"})
        return names, resources, ticket

    names, resources, ticket = asyncio.run(go())
    assert names == {"get_ticket", "get_customer_context", "search_policy", "check_availability",
                     "draft_ticket_update", "draft_escalation"}
    assert {"northstar://policies", "northstar://trace"} <= resources
    assert ticket.structured_content["customer_id"] == "C-001"
    assert [c.name for c in tools.calls] == ["get_ticket", "get_ticket"]