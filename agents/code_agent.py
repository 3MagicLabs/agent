# code_agent.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 1. Import your specific CodeAgentState and your REPL tool
from state import CodeAgentState
from tools.code_tools import python_repl 

llm = ChatOpenAI(
    base_url="https://router.huggingface.co/hf-inference/v1",
    api_key=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("OPENAI_API_KEY") or "hf_dummy",
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0
)
tools = [python_repl]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="""You are the Code Execution Specialist.
Write and run Python code to solve the user's problem.
ALWAYS use the python_repl tool to execute your code.
ALWAYS print() your final variables so you can see the result.
If you receive an error, analyze it and rewrite the code.""")

# 2. Define the reasoning node
def code_reasoning_node(state: CodeAgentState):
    """The LLM reads the state, decides what to do, and increments the loop counter."""
    messages = [sys_msg] + state["messages"]
    
    # Optional: If the system passed a last_error, remind the LLM
    if state.get("last_error"):
        messages.append(SystemMessage(content=f"Previous error to fix: {state['last_error']}"))
        
    response = llm_with_tools.invoke(messages)
    
    # Notice we return {"iterations": 1}. Because you used operator.add in your state,
    # LangGraph will automatically add 1 to the current count!
    return {"messages": [response], "iterations": 1, "last_error": ""}

# 3. Define the routing logic with your safety kill-switch
def route_code_agent(state: CodeAgentState):
    """Decides whether to run a tool, finish, or force-quit due to infinite loops."""
    last_message = state["messages"][-1]
    
    # SAFEGUARD: If the agent has tried to fix its code 5 times and failed, kill it.
    if state["iterations"] >= 5:
        print("\n[Code Agent]: Reached max iterations. Force quitting to prevent infinite loop.")
        return END
        
    # If the LLM decided to call the python_repl tool, route to the tools node
    if last_message.tool_calls:
        return "tools"
        
    # If there are no tool calls, the LLM thinks it has the final answer
    return END

# 4. Build the Subgraph
builder = StateGraph(CodeAgentState)

builder.add_node("agent", code_reasoning_node)
builder.add_node("tools", ToolNode(tools)) # ToolNode automatically handles the tool execution

builder.add_edge(START, "agent")

# Use our custom routing function instead of the default tools_condition
builder.add_conditional_edges("agent", route_code_agent)

# After the tool runs, loop back to the agent to read the output
builder.add_edge("tools", "agent")

# Compile this specific sub-agent
code_agent_subgraph = builder.compile()
     

    
        
