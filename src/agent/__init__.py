"""3MagicLabs agent — a supervisor multi-agent system.

Public API::

    from agent import answer_question
    answer_question("How many moons does Mars have?")
"""

from agent.config import Settings, get_settings, load_settings, set_settings
from agent.core import Orchestrator, answer_question, get_orchestrator

__version__ = "0.1.0"

__all__ = [
    "Orchestrator",
    "Settings",
    "__version__",
    "answer_question",
    "get_orchestrator",
    "get_settings",
    "load_settings",
    "set_settings",
]
