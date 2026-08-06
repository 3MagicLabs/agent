# agent.py
from typing import Literal
from pydantic import BaseModel, Field

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# 1. Import your Manager state schema and compiled specalists subgraphs
from state import ManagerAgentState
from agents.code_agent import code_agent_subgraph
from agents.web_agent import web_agent_subgraph

# 2. Define the Pydantic schema for structured routing decisions
class SupervisorRoute(BaseModel):
    """Select the next specialist agent to delegate to, or FINISH if the task is completely solved."""
    next_agent: Literal["code_agent", "web_agent", "FINISH"] = Field(
            description="The specialist to call next, or 'FINISH' if you have all information to answer the user."
        )
    reasoning: str = Field(
            description="Brief explanation of why this routing choice was made."
            )

llm = ChatOpenAI(
    base_url="https://router.huggingface.co/hf-inference/v1",
    api_key=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("OPENAI_API_KEY") or "hf_dummy",
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0
)
supervisor_router = llm.with_structured_output(SupervisorRoute)

supervisor_sys_msg = SystemMessage(content="""You are the Executive Supervisor of a multi-agent system solving GAIA benchmark tasks.
Your job is to route incoming requests to the appropriate specialist agent:
- Use 'web_agent' when you need to search the internet, look up facts, or read specific webpage/document URLs.
- Use 'code_agent' when you need to write and execute Python code for calculation, data processing, or algorithmic logic.
- Select 'FINISH' when the accumulated information in the conversation completely answers the user's prompt.

Do not attempt to write code or browse the web yourself. Delegate strictly to your specialists.""")

# 3. Define the Supervisor Decisions Node
def supervisor_node(state: ManagerAgentState):
    """Analyzes conversation history and decides which sub-agent should act next."""
    messages = [supervisor_sys_msg] + state["messages"]

    # Force LLM to output our structured SupervisorRoute schema
    decision: SupervisorRoute = supervisor_router.invoke(messages)

    print(f"\n[Supervisor]: Routing to -> {decision.next_agent} (Reason: {decision.reasoning})")

    return {"next_agent": decision.next_agent}


# 4. Define Subgraph Execution Nodes
def call_web_agent(state: ManagerAgentState):
    """Wraps web_agent_subgraph execution and feeds the result back to the Manager state."""
    # Run the web agent with initial sub-state values
    result = web_agent_subgraph.invoke({
        "messages": state["messages"],
        "iterations": 0
        })

    last_msg = result["messages"][-1]

    # Return a labeled AIMessage so the supervisor knows where the answer came from 
    return {
        "messages": [AIMessage(content=f"[Web Agent Output]:\n{last_msg.content}", name="web_agent")]
    }

def call_code_agent(state: ManagerAgentState):
    """Wraps code_agent_subgraph execution and feeds the result back to the Manager state."""

    result = code_agent_subgraph.invoke({
        "messages": state["messages"],
        "iterations": 0,
        "last_error": ""
        })
    last_msg = result["messages"][-1]

    return {
            "messages": [AIMessage(content=f"[Code Agent Output]:\n{last_msg.content}", name="code_agent")]
            }

def route_supervisor(state: ManagerAgentState):
    """Reads the selected next_agent from state and directs graph execution."""
    target = state.get("next_agent", "FINISH")

    if target == "FINISH":
        return END

    return target

builder = StateGraph(ManagerAgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("web_agent", call_web_agent)
builder.add_node("code_agent", call_code_agent)

# Set entry point
builder.add_edge(START, "supervisor")

# Set conditional edge out of the supervisor node
builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "web_agent": "web_agent",
            "code_agent": "code_agent",
            "FINISH": END
            }
        )
builder.add_edge("web_agent", "supervisor")
builder.add_edge("code_agent", "supervisor")

compiled_orchestrator = builder.compile()
