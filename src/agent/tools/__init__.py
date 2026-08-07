"""Tool registry and built-in tools."""

from agent.tools.registry import (
    ToolSpec,
    capability_report,
    get_tools,
    load_builtin_tools,
    register,
    registered,
)

load_builtin_tools()

__all__ = [
    "ToolSpec",
    "capability_report",
    "get_tools",
    "load_builtin_tools",
    "register",
    "registered",
]
