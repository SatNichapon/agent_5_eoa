import operator
import logging
from typing import Annotated, List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent_eoa.config import settings
from agent_eoa.tools import check_calendar_availability, book_venue, send_email_invitation

# Setup Logger
logger = logging.getLogger(__name__)

# 1. State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 2. Tools & LLM
tools = [check_calendar_availability, book_venue, send_email_invitation]
llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 3. Nodes
def planner_node(state: AgentState):
    logger.debug("Entering Planner Node")
    system_prompt = SystemMessage(content="""
    You are Agent 5 (EOA). Organize events by checking availability, booking venues, and sending invites.
    RULES: 
    1. Check availability FIRST. 
    2. If booking fails, retry with a new time.
    """)
    messages = [system_prompt] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

# 4. Router
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# 5. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "planner")

# Export the compiled app
graph = workflow.compile()