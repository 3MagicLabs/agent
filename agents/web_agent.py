# agents/web_agent.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 1. Import your WebAgentState and pre-packaged tools
from state import WebAgentState
from tools.web_tools import web_tools_list

from agent import get_llm

# Initialize the LLM and bind the web tools
llm = get_llm()
llm_with_tools = llm.bind_tools(web_tools_list)

sys_msg = SystemMessage(content="""You are the Web Research Specialist.
Your job is to search the internet and scrape webpages to find exact facts, numbers, datasets, or context needed to answer the user's query.
ALWAYS use your tools to verify information before answering.
Do not guess. If a URL is provided, scrape it to read the contents.
Synthesize the information clearly once you have found the necessary details.""")

# 2. Define the reasoning node
def web_reasoning_node(state: WebAgentState):
    """The LLM reads the state, decides what to search/scrape, and increments the loop counter."""
    messages = [sys_msg] + state["messages"]

    response = llm_with_tools.invoke(messages)


    # Just like the code agent, we return {"iterations": 1} to increment the counter
    return {"messages": [response], "iterations": 1}


# 3. Define the routing logic with your safety kill-swtich
def route_web_agent(state: WebAgentState):
    """Decides whether to run a search tool, finish, or force-quit due to infinite loops."""
    last_message = state["messages"][-1]

    # Limit web searchers to 5 iterations.
    if state["iterations"] >= 5:
        print("\n[Web Agent]: Reached max iterations. Force quitting to prevent infinite loop.")
        return END

    # If the LLM decides 
    if last_message.tool_calls:
        return "tools"

    # if there are not tool calls, LLM finished synthesizing report
    return END

# 4. Build subgraph

builder = StateGraph(WebAgentState)

builder.add_node("agent", web_reasoning_node)
builder.add_node("tools", ToolNode(web_tools_list))

builder.add_edge(START, "agent")

builder.add_conditional_edges("agent", route_web_agent)

builder.add_edge("tools", "agent")

web_agent_subgraph = builder.compile()
