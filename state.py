# state.py
from typing import TypedDict, Annotated, List, Sequence, Optional
import operator
from langchain_core.messages import BaseMessage

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



