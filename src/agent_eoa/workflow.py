import operator
import logging
from typing import Annotated, List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent_eoa.config import settings
from agent_eoa.tools import check_budget, check_calendar_availability, book_venue, send_email_invitation

# Setup Logger
logger = logging.getLogger(__name__)

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 2. Setup Model and Tools
# Ensure you are using a model capable of tool calling
llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY, temperature=0)

tools = [check_budget, check_calendar_availability, book_venue, send_email_invitation]
llm_with_tools = llm.bind_tools(tools)

# 3. Define Nodes
def planner_node(state: AgentState):
    """The Reasoning Engine"""
    # System Prompt enforces the 'Error Recovery' behavior
    system_prompt = SystemMessage(content="""
    You are the Autonomous Event Planner (Agent 5).
    GOAL: Organize events by checking availability, booking venues, and sending invites.
    
    CRITICAL RULES:
    1. Check Budget (check_budget) FIRST. If budget fails, Stop.
    2. ALWAYS check availability (check_calendar_availability) before booking.
    3. If a booking fails (Error or Busy), DO NOT GIVE UP. Propose/Try/Suggest the next available time slot.
    4. Only end when the venue is booked and emails are sent.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 4. Router
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("tools", ToolNode(tools)) # Prebuilt node to run our @tool functions

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "planner") # Loop back to observe result

graph = workflow.compile()