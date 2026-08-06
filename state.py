import os

# Explicitly enable LangSmith tracing if API key is set
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "default"

from typing import TypedDict, Annotated, List, Sequence, Optional
import operator
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

def get_llm():
    if os.getenv("GROQ_API_KEY"):
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0
        )
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
    else:
        return ChatOpenAI(
            base_url="https://router.huggingface.co/hf-inference/v1",
            api_key=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "",
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            temperature=0
        )

class ManagerAgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], operator.add]

    next_agent: str
    
class CodeAgentState(TypedDict):

    messages: Annotated[List[BaseMessage], operator.add]

    iterations: Annotated[int, operator.add]

    last_error: str

class ImageAgentState(TypedDict):

    input_file: Optional[str]

    iterations: Annotated[int, operator.add]

    messages: Annotated[List[BaseMessage], operator.add]

class WebAgentState(TypedDict):
    iterations: Annotated[int, operator.add]

    messages: Annotated[List[BaseMessage], operator.add]

    
class DatabaseAgentState(TypedDict):

    db_path: str

    schema_info: Optional[str]

    iterations: Annotated[int, operator.add]

    messages: Annotated[List[BaseMessage], operator.add]


class RagAgentState(TypedDict):

    iterations: Annotated[int, operator.add]

    messages: Annotated[List[BaseMessage], operator.add]

    retrieved_contexts: Annotated[List[str], operator.add]

    document_paths: List[str]



